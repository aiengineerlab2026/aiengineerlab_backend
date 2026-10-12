"""prices are in US dollars

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-10 18:00:00.000000

Only the currency labels of live rows change here; the new dollar prices come from ``python -m app.cli seed``
(menu fixture ``menu-v2``). Past orders keep the currency and amounts they were placed with.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("restaurants", "menu_items", "carts")


def upgrade() -> None:
    for t in _TABLES:
        op.execute(f"UPDATE {t} SET currency = 'USD' WHERE currency = 'EGP'")


def downgrade() -> None:
    for t in _TABLES:
        op.execute(f"UPDATE {t} SET currency = 'EGP' WHERE currency = 'USD'")
