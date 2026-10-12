"""a second vector column for 1024-d embeddings (Cloudflare bge-m3)

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-11 20:00:00.000000

``embedding`` keeps the EMBEDDING_DIMENSIONS-wide vectors (Gemini, cut to 384). ``embedding_1024`` holds the
native vectors of bge-m3. Each chunk fills exactly one of them, depending on the model of its index.
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS embedding_1024 vector(1024)")
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding DROP NOT NULL")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_rag_chunks_embedding_1024_hnsw "
        "ON rag_chunks USING hnsw (embedding_1024 vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_rag_chunks_embedding_1024_hnsw")
    op.execute("DELETE FROM rag_chunks WHERE embedding IS NULL")
    op.execute("ALTER TABLE rag_chunks ALTER COLUMN embedding SET NOT NULL")
    op.execute("ALTER TABLE rag_chunks DROP COLUMN IF EXISTS embedding_1024")
