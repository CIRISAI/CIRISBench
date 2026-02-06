"""Create evaluations and frontier_models tables.

Revision ID: 001
Revises:
Create Date: 2026-02-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Initial frontier model registry from FSD Section 10
FRONTIER_MODELS = [
    ("openai/gpt-4o", "OpenAI", "GPT-4o", "OpenAI", "openai/gpt-4o"),
    ("openai/gpt-4.5-preview", "OpenAI", "GPT-4.5", "OpenAI", "openai/gpt-4.5-preview"),
    ("openai/o3", "OpenAI", "o3", "OpenAI", "openai/o3"),
    ("openai/o3-mini", "OpenAI", "o3-mini", "OpenAI", "openai/o3-mini"),
    ("anthropic/claude-4-opus", "Anthropic", "Claude 4 Opus", "Anthropic", "anthropic/claude-opus-4-0-20250514"),
    ("anthropic/claude-4-sonnet", "Anthropic", "Claude 4 Sonnet", "Anthropic", "anthropic/claude-sonnet-4-0-20250514"),
    ("anthropic/claude-3.5-haiku", "Anthropic", "Claude 3.5 Haiku", "Anthropic", "anthropic/claude-3-5-haiku-20241022"),
    ("google/gemini-2.0-pro", "Google", "Gemini 2.0 Pro", "Google", "openrouter/google/gemini-2.0-pro"),
    ("google/gemini-2.0-flash", "Google", "Gemini 2.0 Flash", "Google", "openrouter/google/gemini-2.0-flash"),
    ("meta/llama-4-maverick", "Meta", "Llama 4 Maverick", "Meta", "openrouter/meta-llama/llama-4-maverick"),
    ("meta/llama-4-scout", "Meta", "Llama 4 Scout", "Meta", "openrouter/meta-llama/llama-4-scout"),
    ("deepseek/deepseek-r1", "DeepSeek", "DeepSeek R1", "DeepSeek", "openrouter/deepseek/deepseek-r1"),
    ("mistral/mistral-large", "Mistral", "Mistral Large", "Mistral", "openrouter/mistralai/mistral-large-latest"),
    ("xai/grok-3", "xAI", "Grok 3", "xAI", "openrouter/xai/grok-3"),
    ("cohere/command-r-plus", "Cohere", "Command R+", "Cohere", "openrouter/cohere/command-r-plus"),
]


def upgrade() -> None:
    # Extensions (idempotent)
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # --- evaluations table ---
    op.create_table(
        "evaluations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("eval_type", sa.String(16), nullable=False),
        sa.Column("target_model", sa.String(128)),
        sa.Column("target_provider", sa.String(64)),
        sa.Column("target_endpoint", sa.Text()),
        sa.Column("model_version", sa.String(64)),
        sa.Column("protocol", sa.String(16), nullable=False),
        sa.Column("agent_name", sa.String(128)),
        sa.Column("sample_size", sa.Integer(), nullable=False, server_default="300"),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("concurrency", sa.Integer(), nullable=False, server_default="50"),
        sa.Column("batch_config", JSONB()),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("accuracy", sa.Float()),
        sa.Column("total_scenarios", sa.Integer()),
        sa.Column("correct", sa.Integer()),
        sa.Column("errors", sa.Integer()),
        sa.Column("categories", JSONB()),
        sa.Column("avg_latency_ms", sa.Float()),
        sa.Column("processing_ms", sa.Integer()),
        sa.Column("scenario_results", JSONB()),
        sa.Column("trace_id", sa.String(128)),
        sa.Column("trace_binding", JSONB()),
        sa.Column("visibility", sa.String(8), nullable=False, server_default="private"),
        sa.Column("badges", JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("visibility IN ('public', 'private')", name="valid_visibility"),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed')", name="valid_status"),
        sa.CheckConstraint("eval_type IN ('frontier', 'client')", name="valid_eval_type"),
    )

    # Partial indexes for query patterns
    op.execute("""
        CREATE INDEX idx_eval_public
        ON evaluations (visibility, status, accuracy DESC)
        WHERE visibility = 'public' AND status = 'completed'
    """)
    op.execute("""
        CREATE INDEX idx_eval_frontier_model
        ON evaluations (target_model, completed_at DESC)
        WHERE eval_type = 'frontier' AND status = 'completed'
    """)
    op.create_index("idx_eval_tenant", "evaluations", ["tenant_id", sa.text("created_at DESC")])
    op.execute("""
        CREATE INDEX idx_eval_model_history
        ON evaluations (target_model, completed_at DESC)
        WHERE visibility = 'public' AND status = 'completed'
    """)

    # --- frontier_models table ---
    op.create_table(
        "frontier_models",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("model_id", sa.String(128), nullable=False, unique=True),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(128), nullable=False),
        sa.Column("provider_label", sa.String(64)),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("proxy_route", sa.String(256)),
        sa.Column("eval_config", JSONB()),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # Seed frontier model registry
    for model_id, provider, display_name, provider_label, proxy_route in FRONTIER_MODELS:
        op.execute(
            sa.text(
                "INSERT INTO frontier_models (model_id, provider, display_name, provider_label, proxy_route) "
                "VALUES (:model_id, :provider, :display_name, :provider_label, :proxy_route)"
            ).bindparams(
                model_id=model_id,
                provider=provider,
                display_name=display_name,
                provider_label=provider_label,
                proxy_route=proxy_route,
            )
        )


def downgrade() -> None:
    op.drop_table("frontier_models")
    op.drop_index("idx_eval_model_history", table_name="evaluations")
    op.drop_index("idx_eval_tenant", table_name="evaluations")
    op.drop_index("idx_eval_frontier_model", table_name="evaluations")
    op.drop_index("idx_eval_public", table_name="evaluations")
    op.drop_table("evaluations")
