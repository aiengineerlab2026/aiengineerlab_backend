from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.rag.entities import (
    CONFIG_ID,
    Chunk,
    ChunkDraft,
    IndexStatus,
    RagConfig,
    RagIndex,
    ScoredChunk,
)
from app.domain.rag.ports import ConfigRepository, IndexRepository, VectorStore
from app.domain.rag.value_objects import ChunkingConfig, EmbeddingSpec, RetrievalConfig
from app.infrastructure.db.models import (
    WIDE_EMBEDDING_DIMENSIONS,
    RagChunkModel,
    RagConfigModel,
    RagIndexModel,
)

_INSERT_BATCH = 200


def _now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------- live configuration --------------------------------------


def _cfg(m: RagConfigModel) -> RagConfig:
    return RagConfig(
        id=str(m.id),
        chunking=ChunkingConfig.model_validate(m.chunking),
        retrieval=RetrievalConfig.model_validate(m.retrieval),
        embedding=EmbeddingSpec.model_validate(m.embedding),
        llm_model=m.llm_model,
        dataset_version=m.dataset_version,
        index_fingerprint=m.index_fingerprint,
        models=m.models or {},
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


def _dump(model) -> dict:
    return model.model_dump(mode="json")


class SqlConfigRepository(ConfigRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def get(self) -> RagConfig | None:
        m = await self._s.get(RagConfigModel, uuid.UUID(CONFIG_ID))
        if m is not None:
            await self._s.refresh(m)  # always see the latest edit, even from another transaction
        return _cfg(m) if m else None

    async def save(self, c: RagConfig) -> None:
        values = dict(
            chunking=_dump(c.chunking), retrieval=_dump(c.retrieval), embedding=_dump(c.embedding),
            llm_model=c.llm_model, dataset_version=c.dataset_version, index_fingerprint=c.index_fingerprint,
            models=c.models,
        )
        stmt = pg_insert(RagConfigModel).values(id=uuid.UUID(CONFIG_ID), **values)
        await self._s.execute(
            stmt.on_conflict_do_update(index_elements=["id"], set_={**values, "updated_at": _now()})
        )


# --------------------------------------- indexes -------------------------------------------------


def _idx(m: RagIndexModel) -> RagIndex:
    return RagIndex(
        id=str(m.id),
        fingerprint=m.fingerprint,
        dataset_version=m.dataset_version,
        chunking=ChunkingConfig.model_validate(m.chunking),
        embedding=EmbeddingSpec.model_validate(m.embedding),
        status=IndexStatus(m.status),
        chunk_count=m.chunk_count,
        progress=dict(m.progress or {}),
        error=m.error,
        created_at=m.created_at,
        completed_at=m.completed_at,
    )


class SqlIndexRepository(IndexRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def get(self, index_id: str) -> RagIndex | None:
        m = await self._s.get(RagIndexModel, uuid.UUID(index_id))
        return _idx(m) if m else None

    async def get_by_fingerprint(self, fingerprint: str) -> RagIndex | None:
        m = (
            await self._s.execute(select(RagIndexModel).where(RagIndexModel.fingerprint == fingerprint))
        ).scalar_one_or_none()
        if m is not None:
            await self._s.refresh(m)  # make sure we see status changes made by other transactions
        return _idx(m) if m else None

    async def add(self, index: RagIndex) -> None:
        """Insert; a concurrent insert of the same fingerprint is a no-op (callers re-fetch)."""
        stmt = (
            pg_insert(RagIndexModel)
            .values(
                id=uuid.UUID(index.id), fingerprint=index.fingerprint, dataset_version=index.dataset_version,
                chunking=_dump(index.chunking), embedding=_dump(index.embedding),
                embedding_model=index.embedding.model, embedding_dimensions=index.embedding.dimensions,
                status=index.status.value, chunk_count=0, progress={},
            )
            .on_conflict_do_nothing(index_elements=["fingerprint"])
        )
        await self._s.execute(stmt)

    async def update_status(
        self, index_id: str, status: IndexStatus, *, error: str | None = None,
        chunk_count: int | None = None, progress: dict | None = None,
    ) -> None:
        values: dict = {"status": status.value, "error": error}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if progress is not None:
            values["progress"] = progress
        if status == IndexStatus.READY:
            values["completed_at"] = _now()
        await self._s.execute(update(RagIndexModel).where(RagIndexModel.id == uuid.UUID(index_id)).values(**values))

    async def latest_ready(self, embedding_model: str, dimensions: int) -> RagIndex | None:
        m = (
            await self._s.execute(
                select(RagIndexModel)
                .where(
                    RagIndexModel.status == IndexStatus.READY.value,
                    RagIndexModel.embedding_model == embedding_model,
                    RagIndexModel.embedding_dimensions == dimensions,
                )
                .order_by(RagIndexModel.completed_at.desc().nulls_last())
                .limit(1)
            )
        ).scalar_one_or_none()
        return _idx(m) if m else None

    async def claim_pending(self) -> RagIndex | None:
        m = (
            await self._s.execute(
                select(RagIndexModel)
                .where(RagIndexModel.status == IndexStatus.PENDING.value)
                .order_by(RagIndexModel.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if m is None:
            return None
        m.status = IndexStatus.BUILDING.value
        m.progress = {"stage": "starting", "done": 0, "total": 0}
        await self._s.flush()
        return _idx(m)

    async def requeue_interrupted(self) -> int:
        res = await self._s.execute(
            update(RagIndexModel)
            .where(RagIndexModel.status == IndexStatus.BUILDING.value)
            .values(status=IndexStatus.PENDING.value)
        )
        return res.rowcount or 0


# --------------------------------------- vector store --------------------------------------------

_STOP = {
    "a", "an", "the", "i", "me", "my", "we", "you", "is", "are", "am", "do", "does", "have", "has", "want",
    "with", "without", "for", "of", "to", "in", "on", "and", "or", "any", "anything", "some", "something",
    "what", "which", "can", "show", "find", "please", "that", "this", "it", "be", "there", "your", "than",
}


def build_or_tsquery(query: str) -> str:
    """'high protein chicken, no peanuts' -> 'high | protein | chicken | peanuts' (OR semantics)."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOP and len(t) > 1]
    return " | ".join(dict.fromkeys(tokens))


def _vector_column(dimensions: int):
    """The ``rag_chunks`` column that holds vectors of this width."""
    return RagChunkModel.embedding_1024 if dimensions == WIDE_EMBEDDING_DIMENSIONS else RagChunkModel.embedding


def _chunk(m: RagChunkModel) -> Chunk:
    return Chunk(
        id=str(m.id), index_id=str(m.index_id), position=m.position, source_type=m.source_type,
        source_id=m.source_id, dish_ids=list(m.dish_ids or []), content=m.content, metadata=dict(m.meta or {}),
    )


class PgVectorStore(VectorStore):
    """pgvector (cosine distance) + PostgreSQL full-text search over ``rag_chunks``."""

    def __init__(self, session: AsyncSession):
        self._s = session

    async def replace_chunks(self, index_id: str, chunks: Sequence[tuple[ChunkDraft, list[float]]]) -> int:
        iid = uuid.UUID(index_id)
        await self._s.execute(delete(RagChunkModel).where(RagChunkModel.index_id == iid))
        for start in range(0, len(chunks), _INSERT_BATCH):
            batch = chunks[start : start + _INSERT_BATCH]
            self._s.add_all(
                RagChunkModel(
                    index_id=iid, position=start + i, source_type=d.source_type,
                    source_id=d.dish_ids[0] if d.dish_ids else "", dish_ids=d.dish_ids, content=d.content,
                    meta=d.metadata, **{_vector_column(len(emb)).key: emb},
                )
                for i, (d, emb) in enumerate(batch)
            )
            await self._s.flush()
        return len(chunks)

    async def count_chunks(self, index_id: str) -> int:
        return (
            await self._s.execute(
                select(func.count()).select_from(RagChunkModel).where(RagChunkModel.index_id == uuid.UUID(index_id))
            )
        ).scalar_one()

    async def vector_search(
        self, index_id: str, embedding: Sequence[float], limit: int, allowed_dish_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        # Make filtered HNSW scans return `limit` rows when pgvector >= 0.8 (harmless placeholder before).
        await self._s.execute(
            text("SELECT set_config('hnsw.iterative_scan', 'strict_order', true), set_config('hnsw.ef_search', '100', true)")
        )
        dist = _vector_column(len(embedding)).cosine_distance(list(embedding))
        stmt = select(RagChunkModel, dist.label("distance")).where(RagChunkModel.index_id == uuid.UUID(index_id))
        if allowed_dish_ids is not None:
            stmt = stmt.where(RagChunkModel.dish_ids.overlap(list(allowed_dish_ids)))
        stmt = stmt.order_by(dist, RagChunkModel.position).limit(limit)
        rows = (await self._s.execute(stmt)).all()
        return [ScoredChunk(_chunk(m), float(d), "cosine_distance") for m, d in rows]

    async def lexical_search(
        self, index_id: str, query: str, limit: int, allowed_dish_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        tsq_text = build_or_tsquery(query)
        if not tsq_text:
            return []
        tsq = func.to_tsquery("english", tsq_text)
        rank = func.ts_rank_cd(RagChunkModel.tsv, tsq)
        stmt = (
            select(RagChunkModel, rank.label("rank"))
            .where(RagChunkModel.index_id == uuid.UUID(index_id), RagChunkModel.tsv.op("@@")(tsq))
        )
        if allowed_dish_ids is not None:
            stmt = stmt.where(RagChunkModel.dish_ids.overlap(list(allowed_dish_ids)))
        stmt = stmt.order_by(rank.desc(), RagChunkModel.position).limit(limit)
        rows = (await self._s.execute(stmt)).all()
        return [ScoredChunk(_chunk(m), float(r), "ts_rank_cd") for m, r in rows]

    async def chunks_covering(self, index_id: str, dish_ids: Sequence[str]) -> list[Chunk]:
        if not dish_ids:
            return []
        stmt = (
            select(RagChunkModel)
            .where(RagChunkModel.index_id == uuid.UUID(index_id), RagChunkModel.dish_ids.overlap(list(dish_ids)))
            .order_by(RagChunkModel.position)
        )
        return [_chunk(m) for m in (await self._s.execute(stmt)).scalars()]

    async def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = (
            await self._s.execute(select(RagChunkModel).where(RagChunkModel.id.in_([uuid.UUID(c) for c in chunk_ids])))
        ).scalars()
        by_id = {str(m.id): _chunk(m) for m in rows}
        return [by_id[c] for c in chunk_ids if c in by_id]
