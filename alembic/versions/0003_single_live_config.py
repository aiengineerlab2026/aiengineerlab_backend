"""one live RAG configuration instead of named experiments

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-10 01:00:00.000000

``rag_experiments`` becomes ``rag_config`` (a single row, fixed id). Evaluation runs and live evaluations no
longer point at an experiment: each already stores its own ``config_snapshot``.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONFIG_ID = "00000000-0000-0000-0000-000000000001"


def upgrade() -> None:
    op.drop_column("evaluation_runs", "experiment_id")  # also drops its FK and index
    op.drop_column("live_evaluations", "experiment_id")
    op.drop_column("live_evaluations", "experiment_name")

    # Keep the active experiment (else the newest) as THE configuration.
    op.execute(
        "DELETE FROM rag_experiments WHERE id <> "
        "(SELECT id FROM rag_experiments ORDER BY is_active DESC, updated_at DESC LIMIT 1)"
    )
    op.execute(f"UPDATE rag_experiments SET id = '{CONFIG_ID}'")
    op.drop_column("rag_experiments", "name")
    op.drop_column("rag_experiments", "is_active")
    op.rename_table("rag_experiments", "rag_config")
    op.execute("ALTER INDEX ix_rag_experiments_index_fingerprint RENAME TO ix_rag_config_index_fingerprint")


def downgrade() -> None:
    op.execute("ALTER INDEX ix_rag_config_index_fingerprint RENAME TO ix_rag_experiments_index_fingerprint")
    op.rename_table("rag_config", "rag_experiments")
    op.add_column("rag_experiments", sa.Column("name", sa.String(length=160), nullable=False, server_default="Live"))
    op.add_column("rag_experiments", sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.alter_column("rag_experiments", "name", server_default=None)
    op.alter_column("rag_experiments", "is_active", server_default=None)

    op.add_column("live_evaluations", sa.Column("experiment_name", sa.String(length=160), nullable=False, server_default="Live"))
    op.alter_column("live_evaluations", "experiment_name", server_default=None)
    op.add_column("live_evaluations", sa.Column("experiment_id", sa.UUID(), nullable=True))
    op.create_foreign_key(None, "live_evaluations", "rag_experiments", ["experiment_id"], ["id"], ondelete="SET NULL")
    op.create_index(op.f("ix_live_evaluations_experiment_id"), "live_evaluations", ["experiment_id"], unique=False)
    op.add_column("evaluation_runs", sa.Column("experiment_id", sa.UUID(), nullable=True))
    op.create_foreign_key(None, "evaluation_runs", "rag_experiments", ["experiment_id"], ["id"], ondelete="SET NULL")
    op.create_index(op.f("ix_evaluation_runs_experiment_id"), "evaluation_runs", ["experiment_id"], unique=False)
