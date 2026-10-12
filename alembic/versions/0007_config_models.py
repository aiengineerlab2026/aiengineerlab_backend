"""the live configuration remembers the chosen chat / evaluation / embedding models

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-11 01:00:00.000000

``models`` is {} until a model is picked in the Configuration tab; then the backend .env defaults are used.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE rag_config ADD COLUMN IF NOT EXISTS models JSONB NOT NULL DEFAULT '{}'::jsonb")


def downgrade() -> None:
    op.execute("ALTER TABLE rag_config DROP COLUMN IF EXISTS models")
