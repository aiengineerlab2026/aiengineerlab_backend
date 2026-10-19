"""Tables of Country Matcher.

CLIP vectors are 512-d and ResNet50 vectors 2048-d, so each model has its own embedding table with a column of exactly
its width (a column cannot hold both). pgvector's HNSW index on the ``vector`` type stops at 2000 dimensions, so the
2048-d ResNet50 vectors use ``halfvec`` (16-bit floats, indexable up to 4000 dims).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.country_matcher.domain.models import ModelName
from app.db import Base


class CountryImageModel(Base):
    __tablename__ = "country_matcher_images"
    __table_args__ = (CheckConstraint("split in ('index', 'test')", name="ck_country_matcher_images_split"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code: Mapped[str] = mapped_column(String(2), index=True)
    country_name: Mapped[str] = mapped_column(String(100))
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    original_url: Mapped[str] = mapped_column(Text, unique=True)
    source_page_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_path: Mapped[str] = mapped_column(Text, unique=True)
    source_license: Mapped[str] = mapped_column(String(100))
    source_attribution: Mapped[str] = mapped_column(Text)
    image_hash: Mapped[str] = mapped_column(String(64), unique=True)
    perceptual_hash: Mapped[str] = mapped_column(String(16))
    scene_group: Mapped[str] = mapped_column(String(300), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    split: Mapped[str] = mapped_column(String(8), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class _EmbeddingColumns:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(64))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ClipEmbeddingModel(_EmbeddingColumns, Base):
    __tablename__ = "country_matcher_embeddings_clip"

    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("country_matcher_images.id", ondelete="CASCADE"), unique=True
    )
    embedding = mapped_column(Vector(512), nullable=False)


class Resnet50EmbeddingModel(_EmbeddingColumns, Base):
    __tablename__ = "country_matcher_embeddings_resnet50"

    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("country_matcher_images.id", ondelete="CASCADE"), unique=True
    )
    embedding = mapped_column(HALFVEC(2048), nullable=False)


EMBEDDING_TABLES = {ModelName.CLIP: ClipEmbeddingModel, ModelName.RESNET50: Resnet50EmbeddingModel}


class EvaluationRunModel(Base):
    """One run of the evaluation script. The API only ever reads the latest row, it never recomputes."""

    __tablename__ = "country_matcher_evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    results = mapped_column(JSONB, nullable=False)
