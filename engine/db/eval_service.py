"""Evaluation persistence service — write path for the unified pipeline.

Handles create/start/complete/fail lifecycle for evaluations,
badge computation, and Redis cache invalidation.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from engine.config.settings import settings
from engine.core.badges import compute_badges
from engine.db.models import AgentProfile, Evaluation, FrontierModel

logger = logging.getLogger(__name__)


async def create_evaluation(
    session: AsyncSession,
    *,
    tenant_id: str,
    eval_type: str,
    protocol: str,
    seed: int,
    target_model: Optional[str] = None,
    target_provider: Optional[str] = None,
    target_endpoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    sample_size: int = 300,
    concurrency: int = 50,
    visibility: str = "private",
    batch_config: Optional[dict] = None,
) -> uuid.UUID:
    """Insert a new evaluation in 'queued' status. Returns the evaluation ID."""
    eval_id = uuid.uuid4()
    evaluation = Evaluation(
        id=eval_id,
        tenant_id=tenant_id,
        eval_type=eval_type,
        target_model=target_model,
        target_provider=target_provider,
        target_endpoint=target_endpoint,
        protocol=protocol,
        agent_name=agent_name,
        sample_size=sample_size,
        seed=seed,
        concurrency=concurrency,
        visibility=visibility,
        batch_config=batch_config,
        status="queued",
    )
    session.add(evaluation)
    await session.flush()
    logger.info("Created evaluation %s (type=%s, model=%s)", eval_id, eval_type, target_model)
    return eval_id


async def start_evaluation(session: AsyncSession, eval_id: uuid.UUID) -> None:
    """Mark evaluation as running."""
    await session.execute(
        update(Evaluation)
        .where(Evaluation.id == eval_id)
        .values(status="running", started_at=datetime.now(timezone.utc))
    )
    logger.info("Started evaluation %s", eval_id)


async def complete_evaluation(
    session: AsyncSession,
    eval_id: uuid.UUID,
    *,
    accuracy: float,
    total_scenarios: int,
    correct: int,
    errors: int,
    categories: dict[str, Any],
    scenario_results: Optional[list] = None,
    avg_latency_ms: Optional[float] = None,
    processing_ms: Optional[int] = None,
    trace_id: Optional[str] = None,
    trace_binding: Optional[dict] = None,
    model_version: Optional[str] = None,
) -> list[str]:
    """Mark evaluation as completed, compute badges, store results, invalidate cache.

    Returns the computed badge list.
    """
    badges = compute_badges(accuracy, categories)

    await session.execute(
        update(Evaluation)
        .where(Evaluation.id == eval_id)
        .values(
            status="completed",
            accuracy=accuracy,
            total_scenarios=total_scenarios,
            correct=correct,
            errors=errors,
            categories=categories,
            scenario_results=scenario_results,
            avg_latency_ms=avg_latency_ms,
            processing_ms=processing_ms,
            trace_id=trace_id,
            trace_binding=trace_binding,
            model_version=model_version,
            badges=badges,
            completed_scenario_count=total_scenarios,
            completed_at=datetime.now(timezone.utc),
        )
    )
    logger.info(
        "Completed evaluation %s: %.2f%% accuracy, badges=%s",
        eval_id, accuracy * 100, badges,
    )

    # Fetch eval_type and target_model for cache invalidation
    result = await session.execute(
        select(Evaluation.eval_type, Evaluation.target_model, Evaluation.visibility)
        .where(Evaluation.id == eval_id)
    )
    row = result.one_or_none()
    if row:
        await invalidate_cache(row.eval_type, row.target_model, row.visibility)

    return badges


async def fail_evaluation(
    session: AsyncSession,
    eval_id: uuid.UUID,
    error_msg: str = "",
) -> None:
    """Mark evaluation as failed."""
    await session.execute(
        update(Evaluation)
        .where(Evaluation.id == eval_id)
        .values(
            status="failed",
            completed_at=datetime.now(timezone.utc),
        )
    )
    logger.warning("Failed evaluation %s: %s", eval_id, error_msg)


async def checkpoint_scenario_results(
    session: AsyncSession,
    eval_id: uuid.UUID,
    new_results: list[dict],
) -> None:
    """Append scenario results to the JSONB array and bump checkpoint count.

    Uses PostgreSQL ``||`` to atomically append without rewriting the
    entire column from the application side.
    """
    import json as _json
    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    count = len(new_results)
    await session.execute(
        text(
            "UPDATE evaluations "
            "SET scenario_results = COALESCE(scenario_results, CAST('[]' AS jsonb)) || CAST(:new_results AS jsonb), "
            "    completed_scenario_count = completed_scenario_count + :count, "
            "    checkpoint_at = :now "
            "WHERE id = CAST(:eval_id AS uuid)"
        ),
        {
            "new_results": _json.dumps(new_results, default=str),
            "count": count,
            "now": now,
            "eval_id": str(eval_id),
        },
    )
    logger.info("Checkpointed %d results for eval %s", count, eval_id)


async def get_running_evaluations(session: AsyncSession) -> list[Evaluation]:
    """Return all evaluations stuck in 'running' status (crash recovery)."""
    result = await session.execute(
        select(Evaluation).where(Evaluation.status == "running")
    )
    return list(result.scalars().all())


async def get_frontier_models(session: AsyncSession) -> list[FrontierModel]:
    """Return all active frontier models."""
    result = await session.execute(
        select(FrontierModel).where(FrontierModel.active.is_(True))
    )
    return list(result.scalars().all())


async def invalidate_cache(
    eval_type: str,
    target_model: Optional[str] = None,
    visibility: Optional[str] = None,
) -> None:
    """Delete relevant Redis cache keys after eval completion.

    Both CIRISBench and CIRISNode share the same Redis instance,
    so direct key deletion is the simplest invalidation strategy.
    """
    try:
        r = aioredis.from_url(settings.redis_url)
        keys_to_delete = []

        if eval_type == "frontier":
            keys_to_delete.append("cache:scores:frontier")
            keys_to_delete.append("cache:embed:scores")
            if target_model:
                keys_to_delete.append(f"cache:scores:model:{target_model}")

        if visibility == "public":
            keys_to_delete.append("cache:leaderboard")

        if keys_to_delete:
            await r.delete(*keys_to_delete)
            logger.info("Invalidated cache keys: %s", keys_to_delete)
        await r.aclose()
    except Exception as e:
        logger.warning("Cache invalidation failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Agent Profile CRUD
# ---------------------------------------------------------------------------

async def create_agent_profile(
    session: AsyncSession,
    *,
    tenant_id: str,
    name: str,
    spec: dict,
    is_default: bool = False,
) -> uuid.UUID:
    """Create a saved agent profile. Returns the profile ID."""
    profile_id = uuid.uuid4()
    profile = AgentProfile(
        id=profile_id,
        tenant_id=tenant_id,
        name=name,
        spec=spec,
        is_default=is_default,
    )
    session.add(profile)
    await session.flush()
    logger.info("Created agent profile %s '%s' for %s", profile_id, name, tenant_id)
    return profile_id


async def get_agent_profiles(
    session: AsyncSession, tenant_id: str
) -> list[AgentProfile]:
    """Return all agent profiles for a tenant, ordered by name."""
    result = await session.execute(
        select(AgentProfile)
        .where(AgentProfile.tenant_id == tenant_id)
        .order_by(AgentProfile.name)
    )
    return list(result.scalars().all())


async def get_agent_profile(
    session: AsyncSession, profile_id: uuid.UUID, tenant_id: str
) -> Optional[AgentProfile]:
    """Return a single agent profile, scoped to tenant."""
    result = await session.execute(
        select(AgentProfile)
        .where(AgentProfile.id == profile_id, AgentProfile.tenant_id == tenant_id)
    )
    return result.scalar_one_or_none()


async def update_agent_profile(
    session: AsyncSession,
    profile_id: uuid.UUID,
    tenant_id: str,
    *,
    name: Optional[str] = None,
    spec: Optional[dict] = None,
    is_default: Optional[bool] = None,
) -> bool:
    """Update an agent profile. Returns True if a row was updated."""
    values: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
    if name is not None:
        values["name"] = name
    if spec is not None:
        values["spec"] = spec
    if is_default is not None:
        values["is_default"] = is_default

    result = await session.execute(
        update(AgentProfile)
        .where(AgentProfile.id == profile_id, AgentProfile.tenant_id == tenant_id)
        .values(**values)
    )
    return result.rowcount > 0


async def delete_agent_profile(
    session: AsyncSession, profile_id: uuid.UUID, tenant_id: str
) -> bool:
    """Delete an agent profile. Returns True if a row was deleted."""
    result = await session.execute(
        delete(AgentProfile)
        .where(AgentProfile.id == profile_id, AgentProfile.tenant_id == tenant_id)
    )
    return result.rowcount > 0
