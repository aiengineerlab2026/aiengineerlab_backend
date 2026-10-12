"""SQLAlchemy 2.0 ORM models (persistence concern only; domain entities live in app.domain)."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base

WIDE_EMBEDDING_DIMENSIONS = 1024  # bge-m3


def _embedding_dim() -> int:
    return get_settings().embedding_dimensions


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


# ------------------------------------ Restaurant & Menu ------------------------------------------


class RestaurantModel(Base):
    __tablename__ = "restaurants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    tagline: Mapped[str] = mapped_column(String(255), default="")
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MenuCategoryModel(Base):
    __tablename__ = "menu_categories"
    __table_args__ = (UniqueConstraint("restaurant_id", "name"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    restaurant_id: Mapped[str] = mapped_column(ForeignKey("restaurants.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    emoji: Mapped[str] = mapped_column(String(8), default="")
    sort_order: Mapped[int] = mapped_column(Integer, default=0)


class MenuItemModel(Base):
    __tablename__ = "menu_items"
    __table_args__ = (
        Index("ix_menu_items_allergens", "allergens", postgresql_using="gin"),
        Index("ix_menu_items_category_available", "category_id", "available"),
        Index("ix_menu_items_price", "price"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    restaurant_id: Mapped[str] = mapped_column(ForeignKey("restaurants.id", ondelete="CASCADE"))
    category_id: Mapped[str] = mapped_column(ForeignKey("menu_categories.id"))
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    ingredients: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    allergens: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    calories: Mapped[int] = mapped_column(Integer, default=0)
    protein_grams: Mapped[int] = mapped_column(Integer, default=0)
    spice_level: Mapped[int] = mapped_column(Integer, default=0)
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    category: Mapped[MenuCategoryModel] = relationship(lazy="joined")


# ------------------------------------ Chat -------------------------------------------------------


class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    state: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ------------------------------------ Cart & Orders ----------------------------------------------


class CartModel(Base):
    __tablename__ = "carts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"), unique=True
    )
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list[CartItemModel]] = relationship(
        back_populates="cart", cascade="all, delete-orphan", lazy="selectin", order_by="CartItemModel.created_at"
    )


class CartItemModel(Base):
    __tablename__ = "cart_items"
    __table_args__ = (UniqueConstraint("cart_id", "dish_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    cart_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("carts.id", ondelete="CASCADE"), index=True)
    dish_id: Mapped[str] = mapped_column(ForeignKey("menu_items.id"))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cart: Mapped[CartModel] = relationship(back_populates="items")
    dish: Mapped[MenuItemModel] = relationship(lazy="joined")


class OrderModel(Base):
    __tablename__ = "orders"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="confirmed")
    total: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    items: Mapped[list[OrderItemModel]] = relationship(
        back_populates="order", cascade="all, delete-orphan", lazy="selectin"
    )


class OrderItemModel(Base):
    __tablename__ = "order_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    order_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("orders.id", ondelete="CASCADE"), index=True)
    dish_id: Mapped[str] = mapped_column(ForeignKey("menu_items.id"))
    name: Mapped[str] = mapped_column(String(160))
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    quantity: Mapped[int] = mapped_column(Integer)

    order: Mapped[OrderModel] = relationship(back_populates="items")


# ------------------------------------ RAG (indexes, chunks, live config) -------------------------


class RagIndexModel(Base):
    __tablename__ = "rag_indexes"
    __table_args__ = (Index("ix_rag_indexes_status", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    chunking: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RagChunkModel(Base):
    __tablename__ = "rag_chunks"
    # The HNSW (cosine) index on `embedding` is created with raw SQL in the migration.
    __table_args__ = (
        Index("ix_rag_chunks_dish_ids", "dish_ids", postgresql_using="gin"),
        Index("ix_rag_chunks_index_position", "index_id", "position"),
        Index("ix_rag_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    index_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag_indexes.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(64))
    dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)
    # A chunk fills one of the two vector columns, by the width of its index's embedding model.
    embedding: Mapped[Any] = mapped_column(Vector(_embedding_dim()), nullable=True)
    embedding_1024: Mapped[Any] = mapped_column(Vector(WIDE_EMBEDDING_DIMENSIONS), nullable=True)
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RagConfigModel(Base):
    """The single live RAG configuration (one row, fixed id)."""

    __tablename__ = "rag_config"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    chunking: Mapped[dict[str, Any]] = mapped_column(JSONB)
    retrieval: Mapped[dict[str, Any]] = mapped_column(JSONB)
    embedding: Mapped[dict[str, Any]] = mapped_column(JSONB)
    llm_model: Mapped[str] = mapped_column(String(160))
    dataset_version: Mapped[str] = mapped_column(String(64))
    index_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    models: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ------------------------------------ Evaluation --------------------------------------------------


class EvaluationDatasetModel(Base):
    __tablename__ = "evaluation_datasets"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(32))
    menu_dataset_version: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvaluationQuestionModel(Base):
    __tablename__ = "evaluation_questions"
    __table_args__ = (UniqueConstraint("dataset_id", "key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_datasets.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(16))
    position: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(48))
    question: Mapped[str] = mapped_column(Text)
    history: Mapped[list[dict[str, str]]] = mapped_column(JSONB, default=list)
    relevant_dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    reference_answer: Mapped[str] = mapped_column(Text)
    required_constraints: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    expected_behavior: Mapped[str] = mapped_column(String(64))


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (Index("ix_evaluation_runs_status", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evaluation_datasets.id"))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB)
    fingerprint: Mapped[str] = mapped_column(String(64))
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)


class EvaluationQueryResultModel(Base):
    __tablename__ = "evaluation_query_results"
    __table_args__ = (UniqueConstraint("run_id", "question_key"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    question_key: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(48))
    question: Mapped[str] = mapped_column(Text)
    query: Mapped[str] = mapped_column(Text)
    relevant_dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    retrieved_dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    contexts: Mapped[list[str]] = mapped_column(JSONB, default=list)
    answer: Mapped[str] = mapped_column(Text, default="")
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    ragas_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    trace: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class LiveEvaluationModel(Base):
    """A real customer question scored automatically (no pre-written ground truth)."""

    __tablename__ = "live_evaluations"
    __table_args__ = (
        Index("ix_live_evaluations_status", "status"),
        Index("ix_live_evaluations_created", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[uuid.UUID] = mapped_column(unique=True)  # the assistant chat message that was scored
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    question: Mapped[str] = mapped_column(Text)
    query: Mapped[str] = mapped_column(Text)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    answer: Mapped[str] = mapped_column(Text, default="")
    contexts: Mapped[list[str]] = mapped_column(JSONB, default=list)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    relevant_dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    retrieved_dish_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    ragas_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    judge_model: Mapped[str] = mapped_column(String(160), default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
