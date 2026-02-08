"""Add tenant_tiers table for Stripe subscription tracking.

Revision ID: 004
Revises: 003
Create Date: 2026-02-08

Stores the subscription tier per tenant (community/pro/enterprise).
Written by Stripe webhooks, read by CIRISNode quota enforcement.
"""

revision = "004"
down_revision = "003"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    op.create_table(
        "tenant_tiers",
        sa.Column("tenant_id", sa.String(128), primary_key=True),
        sa.Column("tier", sa.String(32), nullable=False, server_default="community"),
        sa.Column("stripe_customer_id", sa.String(128), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(128), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    # Index for Stripe webhook lookups
    op.create_index(
        "idx_tenant_tiers_stripe_customer",
        "tenant_tiers",
        ["stripe_customer_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_tenant_tiers_stripe_customer")
    op.drop_table("tenant_tiers")
