"""Request/response models of the restaurant API: menu, chat, cart, orders, RAG and evaluation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.restaurant.application.chat.parser import SearchConstraints
from app.restaurant.domain.chat.entities import ChatMessage, ChatSession
from app.restaurant.domain.entities import Category, Dish, Restaurant
from app.restaurant.domain.evaluation.entities import EvaluationRun, LiveEvaluation, QueryResult
from app.restaurant.domain.ordering.entities import Cart, Order
from app.restaurant.domain.rag.entities import RagConfig, RagIndex, RetrievalTrace
from app.restaurant.domain.rag.value_objects import (
    ChunkingConfig,
    EmbeddingSpec,
    ModelRef,
    ModelSelection,
    RetrievalConfig,
)

# ------------------------------------------------------------------ menu


class DishOut(BaseModel):
    id: str
    name: str
    category: str
    description: str
    ingredients: list[str]
    allergens: list[str]
    price: float
    currency: str
    calories: int
    protein_grams: int
    spice_level: int
    tags: list[str]
    available: bool
    image_url: str | None = None

    @classmethod
    def of(cls, d: Dish) -> DishOut:
        return cls(
            id=d.id, name=d.name, category=d.category, description=d.description, ingredients=list(d.ingredients),
            allergens=list(d.allergens), price=float(d.price), currency=d.currency, calories=d.calories,
            protein_grams=d.protein_grams, spice_level=d.spice_level, tags=list(d.tags), available=d.available,
            image_url=d.image_url,
        )


class CategoryOut(BaseModel):
    id: str
    name: str
    emoji: str
    dish_count: int

    @classmethod
    def of(cls, c: Category) -> CategoryOut:
        return cls(id=c.id, name=c.name, emoji=c.emoji, dish_count=c.dish_count)


class RestaurantOut(BaseModel):
    id: str
    name: str
    tagline: str
    currency: str

    @classmethod
    def of(cls, r: Restaurant) -> RestaurantOut:
        return cls(id=r.id, name=r.name, tagline=r.tagline, currency=r.currency)


class MenuOut(BaseModel):
    restaurant: RestaurantOut
    dishes: list[DishOut]


# ------------------------------------------------------------------ cart & orders


class CartItemOut(BaseModel):
    id: str
    dish_id: str
    name: str
    category: str
    unit_price: float
    quantity: int
    line_total: float


class CartOut(BaseModel):
    session_id: str
    currency: str
    items: list[CartItemOut]
    item_count: int
    total: float

    @classmethod
    def of(cls, c: Cart) -> CartOut:
        return cls(
            session_id=c.session_id, currency=c.currency, item_count=c.item_count, total=float(c.total),
            items=[
                CartItemOut(id=i.id, dish_id=i.dish_id, name=i.name, category=i.category, unit_price=float(i.unit_price),
                            quantity=i.quantity, line_total=float(i.line_total))
                for i in c.items
            ],
        )


class AddCartItemIn(BaseModel):
    dish_id: str = Field(min_length=1, max_length=32)
    quantity: int = Field(1, ge=1, le=20)


class UpdateCartItemIn(BaseModel):
    quantity: int = Field(ge=1, le=20)


class CreateOrderIn(BaseModel):
    session_id: str
    confirm: bool = Field(description="Must be true: the customer explicitly confirmed the order.")


class OrderItemOut(BaseModel):
    dish_id: str
    name: str
    unit_price: float
    quantity: int
    line_total: float


class OrderOut(BaseModel):
    id: str
    session_id: str
    status: str
    currency: str
    total: float
    items: list[OrderItemOut]
    created_at: datetime
    demo: bool = True

    @classmethod
    def of(cls, o: Order) -> OrderOut:
        return cls(
            id=o.id, session_id=o.session_id, status=o.status, currency=o.currency, total=float(o.total),
            created_at=o.created_at,
            items=[OrderItemOut(dish_id=i.dish_id, name=i.name, unit_price=float(i.unit_price), quantity=i.quantity,
                                line_total=float(i.line_total)) for i in o.items],
        )


# ------------------------------------------------------------------ chat


class SessionOut(BaseModel):
    id: str
    created_at: datetime
    pending_order_confirmation: bool

    @classmethod
    def of(cls, s: ChatSession) -> SessionOut:
        return cls(id=s.id, created_at=s.created_at, pending_order_confirmation=s.pending_order_confirmation)


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def of(cls, m: ChatMessage) -> MessageOut:
        return cls(id=m.id, role=m.role, content=m.content, metadata=m.metadata, created_at=m.created_at)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class SendMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=2000)
    # The recent conversation, kept by the browser (messages are not stored on the server).
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)
    # The embedding model to search with. The request points to that model's pre-built index; the app default if omitted.
    embedding: ModelRef | None = None


class SendMessageOut(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
    cart: CartOut
    recommended_dishes: list[DishOut]
    order: OrderOut | None = None


# ------------------------------------------------------------------ usage limits


class ChatUsageOut(BaseModel):
    """How much of the per-IP message budget is left. ``enabled=False`` means there is no limit."""

    enabled: bool
    limit: int
    used: int
    remaining: int
    window_seconds: int
    reset_in_seconds: int


# ------------------------------------------------------------------ RAG


class IndexOut(BaseModel):
    id: str
    fingerprint: str
    status: str
    chunk_count: int
    progress: dict[str, Any]
    error: str | None
    chunking: ChunkingConfig
    embedding: EmbeddingSpec
    dataset_version: str
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, i: RagIndex) -> IndexOut:
        return cls(
            id=i.id, fingerprint=i.fingerprint, status=i.status.value, chunk_count=i.chunk_count, progress=i.progress,
            error=i.error, chunking=i.chunking, embedding=i.embedding, dataset_version=i.dataset_version,
            created_at=i.created_at, completed_at=i.completed_at,
        )


class ConfigOut(BaseModel):
    """The live RAG configuration and the state of its index."""

    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingSpec
    llm_model: str
    dataset_version: str
    index_fingerprint: str
    updated_at: datetime
    index: IndexOut | None = None

    @classmethod
    def of(cls, c: RagConfig, index: RagIndex | None = None) -> ConfigOut:
        return cls(
            chunking=c.chunking, retrieval=c.retrieval, embedding=c.embedding, llm_model=c.llm_model,
            dataset_version=c.dataset_version, index_fingerprint=c.index_fingerprint, updated_at=c.updated_at,
            index=IndexOut.of(index) if index else None,
        )


class ModelOptionOut(BaseModel):
    provider: str
    model: str
    label: str
    note: str = ""
    dimensions: int = 0  # embedding models: vector width
    quota_exhausted_for_seconds: int = 0  # > 0: the provider said its free quota is used up; usable again after this


class ProviderOut(BaseModel):
    id: str
    name: str
    configured: bool  # credentials are present in backend/.env
    hint: str | None = None  # what to set when they are not


class ModelsOut(BaseModel):
    """What can be picked in the Configuration tab, and what is picked now."""

    providers: list[ProviderOut]
    chat: list[ModelOptionOut]
    embedding: list[ModelOptionOut]
    selection: ModelSelection  # the language models (answers, evaluation judge)
    default_embedding: ModelRef  # used by requests that do not name an embedding model


class ConfigIn(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class RetrieveIn(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    filters: SearchConstraints | None = None
    embedding: ModelRef | None = None  # as for chat messages


class RetrieveOut(BaseModel):
    trace: RetrievalTrace
    dish_ids: list[str]


class StrategyParam(BaseModel):
    key: str
    label: str
    description: str


class StrategyInfo(BaseModel):
    id: str
    name: str
    description: str
    uses: list[str]  # config fields relevant to this strategy (others should be disabled in the UI)
    requires_llm: bool = False


class StrategiesOut(BaseModel):
    chunking: list[StrategyInfo]
    retrieval: list[StrategyInfo]
    rerankers: list[str]
    defaults: dict[str, Any]
    embedding: EmbeddingSpec
    llm: dict[str, Any]


# ------------------------------------------------------------------ evaluation


class CreateEvaluationIn(BaseModel):
    question_limit: int | None = Field(None, ge=1, le=200)


class EvaluationRunOut(BaseModel):
    id: str
    status: str
    fingerprint: str
    config_snapshot: dict[str, Any]
    settings: dict[str, Any]
    progress: dict[str, Any]
    metrics: dict[str, float | None]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None

    @classmethod
    def of(cls, r: EvaluationRun) -> EvaluationRunOut:
        return cls(
            id=r.id, status=r.status.value, fingerprint=r.fingerprint,
            config_snapshot=r.config_snapshot, settings=r.settings, progress=r.progress, metrics=r.metrics,
            error=r.error, created_at=r.created_at, started_at=r.started_at, completed_at=r.completed_at,
            duration_seconds=r.duration_seconds,
        )


class QueryResultOut(BaseModel):
    question_key: str
    category: str
    question: str
    query: str
    relevant_dish_ids: list[str]
    retrieved_dish_ids: list[str]
    contexts: list[str]
    answer: str
    retrieval_metrics: dict[str, float | None]
    ragas_scores: dict[str, float | None]
    latency_ms: float
    trace: dict[str, Any]
    error: str | None

    @classmethod
    def of(cls, r: QueryResult) -> QueryResultOut:
        return cls(**{k: getattr(r, k) for k in cls.model_fields})


class EvaluationResultsOut(BaseModel):
    run: EvaluationRunOut
    results: list[QueryResultOut]
    by_category: dict[str, dict[str, float | None]]


class LiveEvaluationOut(BaseModel):
    id: str
    session_id: str
    message_id: str  # the assistant message in the customer's browser that was scored
    config_snapshot: dict[str, Any]
    question: str
    query: str
    filters: dict[str, Any]
    answer: str
    contexts: list[str]
    status: str
    stage: str
    relevant_dish_ids: list[str]
    retrieved_dish_ids: list[str]
    retrieval_metrics: dict[str, float | None]
    ragas_scores: dict[str, float | None]
    latency_ms: float
    judge_model: str
    error: str | None
    created_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, i: LiveEvaluation) -> LiveEvaluationOut:
        return cls(**{k: getattr(i, k) for k in cls.model_fields if k != "status"}, status=i.status.value)


class LiveGroupOut(BaseModel):
    config_snapshot: dict[str, Any]  # the configuration these questions were answered with
    count: int
    metrics: dict[str, float | None]


class LiveSummaryOut(BaseModel):
    pending: int
    running: int
    completed: int
    failed: int
    metrics: dict[str, float | None]  # means over every question that has the metric
    by_config: list[LiveGroupOut]  # same means per configuration used, to compare settings on real traffic
    judge_model: str


class RunLiveIn(BaseModel):
    message_ids: list[uuid.UUID] = Field(min_length=1, max_length=10, description="Assistant messages to evaluate")


class RunLiveOut(BaseModel):
    queued: int  # questions handed to the worker
    usage: ChatUsageOut  # evaluations left after this one


class LiveEvaluationsOut(BaseModel):
    enabled: bool
    items: list[LiveEvaluationOut]  # newest first
    summary: LiveSummaryOut
    usage: ChatUsageOut  # how many evaluations this client (IP) may still trigger


class CompareOut(BaseModel):
    comparable: bool
    warnings: list[str]
    dataset_id: str
    runs: list[EvaluationRunOut]
    best_by_metric: dict[str, str]


class DatasetOut(BaseModel):
    id: str
    name: str
    version: str
    menu_dataset_version: str
    content_hash: str
    description: str
    question_count: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorOut(BaseModel):
    error: ErrorBody
