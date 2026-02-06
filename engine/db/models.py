"""SQLAlchemy ORM models for the unified evaluation pipeline."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Evaluation(Base):
    """Single evaluations table — all eval types, one visibility flag."""

    __tablename__ = "evaluations"

    # Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(64), nullable=False)

    # What was evaluated
    eval_type = Column(String(16), nullable=False)  # 'frontier' | 'client'
    target_model = Column(String(128))  # e.g. 'anthropic/claude-4-opus'
    target_provider = Column(String(64))  # e.g. 'anthropic', 'openai'
    target_endpoint = Column(Text)  # agent URL (client evals only)
    model_version = Column(String(64))
    protocol = Column(String(16), nullable=False)  # 'direct' | 'a2a' | 'mcp'
    agent_name = Column(String(128))  # display name for leaderboard

    # Configuration
    sample_size = Column(Integer, nullable=False, default=300)
    seed = Column(Integer, nullable=False)
    concurrency = Column(Integer, nullable=False, default=50)
    batch_config = Column(JSONB)  # full BatchConfig snapshot

    # Results (mirrors HE300BatchSummary)
    status = Column(String(16), nullable=False, default="queued")
    accuracy = Column(Float)
    total_scenarios = Column(Integer)
    correct = Column(Integer)
    errors = Column(Integer)
    categories = Column(JSONB)  # {commonsense: {accuracy, correct, total}, ...}
    avg_latency_ms = Column(Float)
    processing_ms = Column(Integer)

    # Scenario-level detail
    scenario_results = Column(JSONB)  # List[HE300ScenarioResult]

    # Audit
    trace_id = Column(String(128))
    trace_binding = Column(JSONB)  # cryptographic binding (seed, hash, sig)

    # Visibility & Display
    visibility = Column(String(8), nullable=False, default="private")
    badges = Column(JSONB)  # computed on write

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "visibility IN ('public', 'private')", name="valid_visibility"
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name="valid_status",
        ),
        CheckConstraint(
            "eval_type IN ('frontier', 'client')", name="valid_eval_type"
        ),
        # Public leaderboard / frontier scores: filter public + completed, sort by accuracy
        Index(
            "idx_eval_public",
            "visibility",
            "status",
            accuracy.desc(),
            postgresql_where=(
                Column("visibility") == "public"
            )
            & (Column("status") == "completed"),
        ),
        # Frontier model history: latest per model
        Index(
            "idx_eval_frontier_model",
            "target_model",
            completed_at.desc(),
            postgresql_where=(
                Column("eval_type") == "frontier"
            )
            & (Column("status") == "completed"),
        ),
        # Client dashboard: tenant's own evals
        Index("idx_eval_tenant", "tenant_id", created_at.desc()),
        # Per-model SEO pages: all public evals for a specific model over time
        Index(
            "idx_eval_model_history",
            "target_model",
            completed_at.desc(),
            postgresql_where=(
                Column("visibility") == "public"
            )
            & (Column("status") == "completed"),
        ),
    )


class FrontierModel(Base):
    """Registry of frontier models to sweep on cron schedule."""

    __tablename__ = "frontier_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(128), nullable=False, unique=True)  # 'openai/gpt-4o'
    provider = Column(String(64), nullable=False)
    display_name = Column(String(128), nullable=False)  # 'GPT-4o'
    provider_label = Column(String(64))  # 'OpenAI'
    active = Column(Boolean, nullable=False, default=True)
    proxy_route = Column(String(256))  # CIRISProxy model string
    eval_config = Column(JSONB)  # model-specific overrides
    added_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
