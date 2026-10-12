"""Shared test doubles.

Everything here is a small, *working* in-memory implementation of a domain port (not a mock of the code under
test): a bag-of-words embedder, a vector/lexical store, repositories and a scripted LLM. The production
services (IndexingService, RetrievalService, CartService, ChatService, ...) run unmodified on top of them.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.restaurant.application.chat.parser import TurnParser
from app.restaurant.application.chat.service import ChatService
from app.restaurant.application.ordering.services import CartService, OrderService
from app.restaurant.application.rag.config import ConfigService
from app.restaurant.application.rag.indexing import IndexingService
from app.restaurant.application.rag.retrieval import RetrievalService
from app.restaurant.domain.chat.entities import ChatSession
from app.restaurant.domain.chat.repositories import ChatRepository
from app.restaurant.domain.entities import Category, Dish, Restaurant
from app.restaurant.domain.ordering.entities import Cart, Order
from app.restaurant.domain.ordering.repositories import CartRepository, OrderRepository
from app.restaurant.domain.rag.entities import (
    Chunk,
    ChunkDraft,
    IndexStatus,
    RagConfig,
    RagIndex,
    ScoredChunk,
)
from app.restaurant.domain.rag.ports import (
    ConfigRepository,
    EmbeddingModel,
    IndexRepository,
    LLMClient,
    Reranker,
    VectorStore,
)
from app.restaurant.domain.rag.value_objects import ChunkingStrategy, EmbeddingSpec
from app.restaurant.domain.repositories import MenuRepository
from app.restaurant.infrastructure.rag.chunking.strategies import ChunkerFactory
from app.restaurant.infrastructure.seed.menu_data import CATEGORIES, build_menu

DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _vec(text: str) -> list[float]:
    """Deterministic hashed bag-of-words, L2-normalised (cosine similarity == shared-vocabulary overlap)."""
    v = [0.0] * DIM
    for tok in tokenize(text):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


class FakeEmbedder(EmbeddingModel):
    def __init__(self) -> None:
        self.document_calls = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return EmbeddingSpec(provider="fake", model="bow-256", dimensions=DIM, version="1")

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += len(texts)
        return [_vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _vec(text)


class FakeReranker(Reranker):
    """Scores a document by how many query tokens it contains (deterministic)."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-cross-encoder"

    async def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls += 1
        q = set(tokenize(query))
        return [float(len(q & set(tokenize(d)))) for d in documents]


class ScriptedLLM(LLMClient):
    """Returns canned structured outputs / text. ``configured=False`` simulates a missing API key."""

    def __init__(self, *, configured: bool = True, structured: dict | None = None, text: str = "ok") -> None:
        self._configured = configured
        self.structured = structured or {}
        self.text = text
        self.calls: list[tuple[str, str]] = []

    provider = "fake"  # type: ignore[assignment]
    model_name = "fake-llm"  # type: ignore[assignment]

    @property
    def is_configured(self) -> bool:
        return self._configured

    async def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.text

    async def generate_structured(self, system: str, user: str, schema):
        self.calls.append((system, user))
        if schema.__name__ not in self.structured:
            raise AssertionError(f"unexpected structured call for {schema.__name__}")
        value = self.structured[schema.__name__]
        return value if isinstance(value, schema) else schema.model_validate(value)


# ------------------------------------------------------------------------------ repositories


class MemMenu(MenuRepository):
    def __init__(self, dishes: list[Dish]) -> None:
        self.dishes = dishes

    async def get_restaurant(self):
        return Restaurant("rest_001", "AI Waiter", "tagline")

    async def list_categories(self):
        return [Category(n, n, e, i) for i, (n, e) in enumerate(CATEGORIES)]

    async def list_dishes(self, *, category=None, available_only=False):
        out = [d for d in self.dishes if (category is None or d.category == category) and (d.available or not available_only)]
        return sorted(out, key=lambda d: d.sort_order)

    async def get_dish(self, dish_id):
        return next((d for d in self.dishes if d.id == dish_id), None)

    async def get_dishes(self, dish_ids):
        by_id = {d.id: d for d in self.dishes}
        return [by_id[i] for i in dish_ids if i in by_id]


class MemChats(ChatRepository):
    def __init__(self) -> None:
        self.sessions: dict[str, ChatSession] = {}

    async def create_session(self):
        s = ChatSession(str(uuid.uuid4()))
        self.sessions[s.id] = s
        return s

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def save_session(self, session):
        self.sessions[session.id] = session


class MemCarts(CartRepository):
    def __init__(self) -> None:
        self.carts: dict[str, Cart] = {}

    async def get(self, session_id):
        c = self.carts.get(session_id)
        # return a copy so services must call save() (like the DB-backed repository)
        return Cart(session_id, c.currency, [replace(i) for i in c.items]) if c else Cart(session_id)

    async def save(self, cart):
        self.carts[cart.session_id] = Cart(cart.session_id, cart.currency, [replace(i) for i in cart.items])


class MemOrders(OrderRepository):
    def __init__(self) -> None:
        self.orders: dict[str, Order] = {}

    async def add(self, order):
        self.orders[order.id] = order

    async def get(self, order_id):
        return self.orders.get(order_id)


class MemIndexes(IndexRepository):
    def __init__(self) -> None:
        self.items: dict[str, RagIndex] = {}

    async def get(self, index_id):
        return self.items.get(index_id)

    async def get_by_fingerprint(self, fingerprint):
        return next((i for i in self.items.values() if i.fingerprint == fingerprint), None)

    async def add(self, index):
        self.items[index.id] = index

    async def update_status(self, index_id, status, *, error=None, chunk_count=None, progress=None):
        i = self.items[index_id]
        i.status, i.error = status, error
        if chunk_count is not None:
            i.chunk_count = chunk_count
        if progress is not None:
            i.progress = progress

    async def latest_ready(self, embedding_model, dimensions):
        ready = [
            i for i in self.items.values()
            if i.status == IndexStatus.READY and i.embedding.model == embedding_model and i.embedding.dimensions == dimensions
        ]
        return max(ready, key=lambda i: i.created_at, default=None)

    async def claim_pending(self):
        return None

    async def requeue_interrupted(self):
        return 0


class MemConfig(ConfigRepository):
    def __init__(self) -> None:
        self.item: RagConfig | None = None

    async def get(self):
        return self.item

    async def save(self, config):
        self.item = replace(config, updated_at=datetime.now(UTC))


class MemVectorStore(VectorStore):
    def __init__(self) -> None:
        self.chunks: dict[str, list[tuple[Chunk, list[float]]]] = {}
        self.replace_calls = 0

    async def replace_chunks(self, index_id, chunks: Sequence[tuple[ChunkDraft, list[float]]]):
        self.replace_calls += 1
        self.chunks[index_id] = [
            (Chunk(str(uuid.uuid4()), index_id, pos, d.source_type, d.dish_ids[0], d.dish_ids, d.content, d.metadata), emb)
            for pos, (d, emb) in enumerate(chunks)
        ]
        return len(chunks)

    async def count_chunks(self, index_id):
        return len(self.chunks.get(index_id, []))

    def _eligible(self, index_id, allowed):
        for chunk, emb in self.chunks.get(index_id, []):
            if allowed is None or set(chunk.dish_ids) <= set(allowed):
                yield chunk, emb

    async def vector_search(self, index_id, embedding, limit, allowed_dish_ids=None):
        scored = [
            ScoredChunk(c, 1.0 - sum(a * b for a, b in zip(embedding, e, strict=True)), "cosine_distance")
            for c, e in self._eligible(index_id, allowed_dish_ids)
        ]
        scored.sort(key=lambda s: (s.score, s.chunk.position))
        return scored[:limit]

    async def lexical_search(self, index_id, query, limit, allowed_dish_ids=None):
        q = set(tokenize(query))
        scored = []
        for c, _ in self._eligible(index_id, allowed_dish_ids):
            overlap = len(q & set(tokenize(c.content)))
            if overlap:
                scored.append(ScoredChunk(c, float(overlap), "ts_rank_cd"))
        scored.sort(key=lambda s: (-s.score, s.chunk.position))
        return scored[:limit]

    async def get_chunks(self, chunk_ids):
        return [c for cs in self.chunks.values() for c, _ in cs if c.id in set(chunk_ids)]

    async def chunks_covering(self, index_id, dish_ids):
        wanted = set(dish_ids)
        return [c for c, _ in self.chunks.get(index_id, []) if wanted & set(c.dish_ids)]


# ------------------------------------------------------------------------------ fixtures


class World:
    """A fully wired application layer over in-memory adapters."""

    def __init__(self, llm: LLMClient | None = None, live_evals=None) -> None:
        self.dishes = build_menu()
        self.menu = MemMenu(self.dishes)
        self.chats, self.carts_repo, self.orders_repo = MemChats(), MemCarts(), MemOrders()
        self.indexes, self.config_repo, self.store = MemIndexes(), MemConfig(), MemVectorStore()
        self.embedder, self.reranker = FakeEmbedder(), FakeReranker()
        self.llm = llm or ScriptedLLM(configured=False)
        self.chunkers = ChunkerFactory(lambda: _SemanticEmbeddings(self.embedder))
        self.indexing = IndexingService(self.indexes, self.store, self.menu, self.embedder, self.chunkers.create)
        self.config = ConfigService(self.config_repo, self.indexing, self.embedder, "fake-llm")
        self.retrieval = RetrievalService(self.indexing, self.store, self.embedder, lambda _m: self.reranker, self.llm, self.menu)
        self.cart_service = CartService(self.carts_repo, self.menu, self.chats)
        self.order_service = OrderService(self.carts_repo, self.orders_repo, self.menu, self.chats)
        self.chat = ChatService(
            self.chats, self.menu, self.cart_service, self.order_service, self.config, self.retrieval,
            self.llm, TurnParser(self.llm), live_evals=live_evals,
        )

    async def build_index(self, chunking) -> RagIndex:
        index, _ = await self.indexing.request_index(chunking)
        if index.status != IndexStatus.READY:
            async def _noop(_p):
                return None

            await self.indexing.build(index, _noop)
        return (await self.indexing.get_by_fingerprint(index.fingerprint))  # type: ignore[return-value]


class _SemanticEmbeddings:
    """Minimal LangChain ``Embeddings`` adapter around the fake embedder (for SemanticChunker)."""

    def __init__(self, embedder: FakeEmbedder) -> None:
        self._e = embedder

    def embed_documents(self, texts):
        return [_vec(t) for t in texts]

    def embed_query(self, text):
        return _vec(text)


@pytest.fixture
def dishes() -> list[Dish]:
    return build_menu()


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
async def active_world() -> World:
    """World with an entity+vector configuration built and live (rule parser, no LLM)."""
    from app.restaurant.domain.rag.value_objects import ChunkingConfig, RetrievalConfig

    w = World()
    chunking = ChunkingConfig(strategy=ChunkingStrategy.ENTITY)
    cfg, _ = await w.config.update(chunking, RetrievalConfig(top_k=5))
    await w.build_index(chunking)
    w.cfg = cfg  # type: ignore[attr-defined]
    return w
