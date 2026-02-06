"""Evaluation persistence service — write path for the unified pipeline.

Handles create/start/complete/fail lifecycle for evaluations,
badge computation, and Redis cache invalidation.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from engine.config.settings import settings
from engine.core.badges import compute_badges
from engine.db.models import Evaluation, FrontierModel

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
