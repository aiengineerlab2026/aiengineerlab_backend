"""Country Matcher: labelled images, per-model embeddings (pgvector + HNSW) and evaluation runs

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-19 09:00:00.000000

CLIP is 512-d and ResNet50 2048-d, so each gets its own table. HNSW on ``vector`` is limited to 2000 dimensions, hence
``halfvec(2048)`` for ResNet50.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "country_matcher_images",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False),
        sa.Column("country_name", sa.String(length=100), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("original_url", sa.Text(), nullable=False),
        sa.Column("source_page_url", sa.Text(), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("source_license", sa.String(length=100), nullable=False),
        sa.Column("source_attribution", sa.Text(), nullable=False),
        sa.Column("image_hash", sa.String(length=64), nullable=False),
        sa.Column("perceptual_hash", sa.String(length=16), nullable=False),
        sa.Column("scene_group", sa.String(length=300), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("original_url"),
        sa.UniqueConstraint("storage_path"),
        sa.UniqueConstraint("image_hash"),
        sa.CheckConstraint("split in ('index', 'test')", name="ck_country_matcher_images_split"),
    )
    op.create_index("ix_country_matcher_images_country_code", "country_matcher_images", ["country_code"])
    op.create_index("ix_country_matcher_images_scene_group", "country_matcher_images", ["scene_group"])
    op.create_index("ix_country_matcher_images_split", "country_matcher_images", ["split"])

    for table, column_type, ops in (
        ("country_matcher_embeddings_clip", "vector(512)", "vector_cosine_ops"),
        ("country_matcher_embeddings_resnet50", "halfvec(2048)", "halfvec_cosine_ops"),
    ):
        op.execute(
            f"""
            CREATE TABLE {table} (
                id uuid PRIMARY KEY,
                image_id uuid NOT NULL UNIQUE REFERENCES country_matcher_images(id) ON DELETE CASCADE,
                model_name varchar(100) NOT NULL,
                model_version varchar(64) NOT NULL,
                embedding {column_type} NOT NULL,
                embedding_dimension integer NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        op.execute(f"CREATE INDEX ix_{table}_embedding_hnsw ON {table} USING hnsw (embedding {ops})")

    op.create_table(
        "country_matcher_evaluation_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("results", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_country_matcher_evaluation_runs_created_at", "country_matcher_evaluation_runs", ["created_at"])


def downgrade() -> None:
    op.drop_table("country_matcher_evaluation_runs")
    op.execute("DROP TABLE IF EXISTS country_matcher_embeddings_resnet50")
    op.execute("DROP TABLE IF EXISTS country_matcher_embeddings_clip")
    op.drop_table("country_matcher_images")
