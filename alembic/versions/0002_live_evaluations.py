"""live evaluations: automatic scoring of real customer questions

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-10 00:10:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "live_evaluations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("experiment_id", sa.UUID(), nullable=True),
        sa.Column("experiment_name", sa.String(length=160), nullable=False),
        sa.Column("config_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("contexts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("relevant_dish_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("retrieved_dish_ids", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("retrieval_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("ragas_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("judge_model", sa.String(length=160), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["experiment_id"], ["rag_experiments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("message_id"),
    )
    op.create_index(op.f("ix_live_evaluations_session_id"), "live_evaluations", ["session_id"], unique=False)
    op.create_index(op.f("ix_live_evaluations_experiment_id"), "live_evaluations", ["experiment_id"], unique=False)
    op.create_index("ix_live_evaluations_status", "live_evaluations", ["status"], unique=False)
    op.create_index("ix_live_evaluations_created", "live_evaluations", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_live_evaluations_created", table_name="live_evaluations")
    op.drop_index("ix_live_evaluations_status", table_name="live_evaluations")
    op.drop_index(op.f("ix_live_evaluations_experiment_id"), table_name="live_evaluations")
    op.drop_index(op.f("ix_live_evaluations_session_id"), table_name="live_evaluations")
    op.drop_table("live_evaluations")
