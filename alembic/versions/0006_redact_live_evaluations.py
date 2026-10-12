"""scored live evaluations no longer keep the customer's text

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-10 18:30:00.000000

The conversation lives in the customer's browser. Already finished evaluations keep their scores only.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "UPDATE live_evaluations SET question = '', query = '', answer = '', contexts = '[]'::jsonb "
        "WHERE status IN ('completed', 'failed')"
    )


def downgrade() -> None:
    pass  # the removed text cannot be restored
