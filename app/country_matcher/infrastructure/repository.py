"""PostgreSQL (pgvector) implementation of ``CountryStore``."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.country_matcher.application.ports import EvaluationRun, IndexedImage, SearchHit
from app.country_matcher.domain.models import EmbeddingInfo, ImageRecord, ModelName, NewImage, Split
from app.country_matcher.infrastructure.models import EMBEDDING_TABLES, CountryImageModel, EvaluationRunModel


def to_record(row: CountryImageModel) -> ImageRecord:
    return ImageRecord(
        id=row.id, country_code=row.country_code, country_name=row.country_name, city=row.city,
        original_url=row.original_url, source_page_url=row.source_page_url, storage_path=row.storage_path,
        source_license=row.source_license, source_attribution=row.source_attribution, image_hash=row.image_hash,
        perceptual_hash=row.perceptual_hash, scene_group=row.scene_group, width=row.width, height=row.height,
        split=Split(row.split), created_at=row.created_at,
    )


class SqlCountryStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_hash(self, image_hash: str) -> ImageRecord | None:
        row = await self.session.scalar(select(CountryImageModel).where(CountryImageModel.image_hash == image_hash))
        return to_record(row) if row else None

    async def has_original_url(self, original_url: str) -> bool:
        found = await self.session.scalar(select(CountryImageModel.id).where(CountryImageModel.original_url == original_url))
        return found is not None

    async def add_image(self, image: NewImage) -> ImageRecord:
        row = CountryImageModel(**{**image.__dict__, "split": image.split.value})
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row)
        return to_record(row)

    async def get_image(self, image_id: uuid.UUID) -> ImageRecord | None:
        row = await self.session.get(CountryImageModel, image_id)
        return to_record(row) if row else None

    async def list_images(self, *, split: Split | None = None) -> list[ImageRecord]:
        stmt = select(CountryImageModel).order_by(CountryImageModel.country_code, CountryImageModel.created_at, CountryImageModel.id)
        if split is not None:
            stmt = stmt.where(CountryImageModel.split == split.value)
        return [to_record(r) for r in await self.session.scalars(stmt)]

    async def indexed_images(self, *, country_code: str | None, limit: int, offset: int) -> tuple[list[IndexedImage], int]:
        img = CountryImageModel
        has = {m: select(t.image_id).where(t.image_id == img.id).exists() for m, t in EMBEDDING_TABLES.items()}
        where = [img.split == Split.INDEX.value] + ([img.country_code == country_code] if country_code else [])
        total = int(await self.session.scalar(select(func.count()).select_from(img).where(*where)) or 0)
        stmt = (
            select(img, *(has[m] for m in has)).where(*where)
            .order_by(img.country_name, img.created_at, img.id).limit(limit).offset(offset)
        )
        rows = (await self.session.execute(stmt)).all()
        names = list(has)
        return [IndexedImage(to_record(r[0]), tuple(m for m, present in zip(names, r[1:], strict=True) if present)) for r in rows], total

    async def indexed_countries(self) -> list[tuple[str, str, int]]:
        img = CountryImageModel
        stmt = (
            select(img.country_code, img.country_name, func.count()).where(img.split == Split.INDEX.value)
            .group_by(img.country_code, img.country_name).order_by(img.country_name)
        )
        return [(c, n, int(k)) for c, n, k in (await self.session.execute(stmt)).all()]

    async def count_images(self, *, split: Split | None = None) -> int:
        stmt = select(func.count()).select_from(CountryImageModel)
        if split is not None:
            stmt = stmt.where(CountryImageModel.split == split.value)
        return int(await self.session.scalar(stmt) or 0)

    async def count_countries(self) -> int:
        return int(await self.session.scalar(select(func.count(func.distinct(CountryImageModel.country_code)))) or 0)

    async def images_missing_embedding(self, info: EmbeddingInfo) -> list[ImageRecord]:
        emb = EMBEDDING_TABLES[info.model]
        done = select(emb.image_id).where(emb.model_version == info.model_version)
        stmt = (
            select(CountryImageModel)
            .where(CountryImageModel.split == Split.INDEX.value, CountryImageModel.id.not_in(done))
            .order_by(CountryImageModel.created_at, CountryImageModel.id)
        )
        return [to_record(r) for r in await self.session.scalars(stmt)]

    async def save_embeddings(self, info: EmbeddingInfo, items: Sequence[tuple[uuid.UUID, np.ndarray]]) -> None:
        if not items:
            return
        emb = EMBEDDING_TABLES[info.model]
        rows = []
        for image_id, vector in items:
            if vector.shape != (info.dimension,):
                raise ValueError(f"expected a {info.dimension}-d vector, got {vector.shape}")
            rows.append(
                {
                    "id": uuid.uuid4(), "image_id": image_id, "model_name": info.model_name,
                    "model_version": info.model_version, "embedding": vector.astype(np.float32).tolist(),
                    "embedding_dimension": info.dimension,
                }
            )
        stmt = insert(emb).values(rows)
        # Re-embedding with a new model version replaces the old vector of the same image.
        stmt = stmt.on_conflict_do_update(
            index_elements=[emb.image_id],
            set_={c: getattr(stmt.excluded, c) for c in ("model_name", "model_version", "embedding", "embedding_dimension")}
            | {"created_at": func.now()},
        )
        await self.session.execute(stmt)

    async def search(self, info: EmbeddingInfo, query: np.ndarray, k: int) -> list[SearchHit]:
        emb = EMBEDDING_TABLES[info.model]
        distance = emb.embedding.cosine_distance(query.astype(np.float32).tolist())
        stmt = (
            select(CountryImageModel, distance.label("distance"))
            .join(emb, emb.image_id == CountryImageModel.id)
            .where(CountryImageModel.split == Split.INDEX.value, emb.model_version == info.model_version)
            .order_by(distance, CountryImageModel.id)
            .limit(k)
        )
        rows = (await self.session.execute(stmt)).all()
        return [SearchHit(to_record(img), 1.0 - float(dist)) for img, dist in rows]

    async def embedding_stats(self, model: ModelName) -> dict[str, Any]:
        emb = EMBEDDING_TABLES[model]
        table = emb.__tablename__  # a constant from our own mapping, never user input
        count = int(await self.session.scalar(select(func.count()).select_from(emb)) or 0)
        versions = [v for v in await self.session.scalars(select(emb.model_version).distinct().order_by(emb.model_version))]
        sizes = (
            await self.session.execute(
                text(f"SELECT coalesce(sum(pg_column_size(embedding)), 0), pg_total_relation_size('{table}') FROM {table}")
            )
        ).one()
        return {
            "count": count, "versions": versions, "dimension": emb.embedding.type.dim,
            "vector_bytes": int(sizes[0]), "total_bytes": int(sizes[1]),
        }

    async def save_evaluation(self, results: dict[str, Any]) -> EvaluationRun:
        row = EvaluationRunModel(results=results)
        self.session.add(row)
        await self.session.flush()
        await self.session.refresh(row)
        return EvaluationRun(row.id, row.created_at, row.results)

    async def latest_evaluation(self) -> EvaluationRun | None:
        row = await self.session.scalar(select(EvaluationRunModel).order_by(EvaluationRunModel.created_at.desc()).limit(1))
        return EvaluationRun(row.id, row.created_at, row.results) if row else None
