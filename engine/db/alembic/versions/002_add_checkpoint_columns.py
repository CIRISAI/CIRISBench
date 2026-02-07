"""Add checkpoint columns for incremental persistence.

Revision ID: 002
Revises: 001
Create Date: 2026-02-06
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluations",
        sa.Column(
            "completed_scenario_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "evaluations",
        sa.Column("checkpoint_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "idx_eval_running",
        "evaluations",
        ["status"],
        postgresql_where=sa.text("status = 'running'"),
    )


def downgrade() -> None:
    op.drop_index("idx_eval_running", table_name="evaluations")
    op.drop_column("evaluations", "checkpoint_at")
    op.drop_column("evaluations", "completed_scenario_count")
