"""Idempotent seeding of the menu, evaluation dataset."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.restaurant.domain.canonical import MENU_DATASET_VERSION
from app.restaurant.infrastructure.models import (
    EvaluationDatasetModel,
    EvaluationQuestionModel,
    MenuCategoryModel,
    MenuItemModel,
    RestaurantModel,
)
from app.restaurant.infrastructure.seed import evaluation_data as ev
from app.restaurant.infrastructure.seed import menu_data as md

log = logging.getLogger(__name__)


def _cat_id(name: str) -> str:
    return "cat_" + name.lower().replace(" ", "_")


async def seed_menu(session: AsyncSession) -> int:
    await session.execute(
        pg_insert(RestaurantModel)
        .values(id=md.RESTAURANT_ID, name=md.RESTAURANT_NAME, tagline=md.RESTAURANT_TAGLINE, currency="USD")
        .on_conflict_do_update(
            index_elements=["id"], set_={"name": md.RESTAURANT_NAME, "tagline": md.RESTAURANT_TAGLINE, "currency": "USD"}
        )
    )
    for order, (name, emoji) in enumerate(md.CATEGORIES):
        await session.execute(
            pg_insert(MenuCategoryModel)
            .values(id=_cat_id(name), restaurant_id=md.RESTAURANT_ID, name=name, emoji=emoji, sort_order=order)
            .on_conflict_do_update(index_elements=["id"], set_={"name": name, "emoji": emoji, "sort_order": order})
        )
    dishes = md.build_menu()
    for d in dishes:
        values = dict(
            id=d.id, restaurant_id=md.RESTAURANT_ID, category_id=_cat_id(d.category), name=d.name,
            description=d.description, ingredients=list(d.ingredients), allergens=list(d.allergens),
            tags=list(d.tags), price=d.price, currency=d.currency, calories=d.calories,
            protein_grams=d.protein_grams, spice_level=d.spice_level, available=d.available,
            image_url=d.image_url, sort_order=d.sort_order,
        )
        stmt = pg_insert(MenuItemModel).values(**values)
        await session.execute(
            stmt.on_conflict_do_update(index_elements=["id"], set_={k: v for k, v in values.items() if k != "id"})
        )
    log.info("menu seeded", extra={"dishes": len(dishes), "dataset_version": MENU_DATASET_VERSION})
    return len(dishes)


async def seed_evaluation_dataset(session: AsyncSession) -> tuple[str, int]:
    rows = ev.build_dataset(md.build_menu())
    content_hash = ev.dataset_content_hash(rows)
    existing = (
        await session.execute(
            select(EvaluationDatasetModel).where(
                EvaluationDatasetModel.name == ev.EVAL_DATASET_NAME,
                EvaluationDatasetModel.version == ev.EVAL_DATASET_VERSION,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        if existing.content_hash != content_hash:
            raise RuntimeError(
                f"Evaluation dataset {ev.EVAL_DATASET_NAME}:{ev.EVAL_DATASET_VERSION} already exists with different "
                "content. Datasets are immutable once seeded: bump EVAL_DATASET_VERSION in evaluation_data.py."
            )
        return str(existing.id), len(rows)
    ds = EvaluationDatasetModel(
        id=uuid.uuid4(), name=ev.EVAL_DATASET_NAME, version=ev.EVAL_DATASET_VERSION,
        menu_dataset_version=MENU_DATASET_VERSION, content_hash=content_hash, description=ev.EVAL_DATASET_DESCRIPTION,
    )
    session.add(ds)
    await session.flush()
    session.add_all(
        EvaluationQuestionModel(
            dataset_id=ds.id, key=r["key"], position=r["position"], category=r["category"], question=r["question"],
            history=r["history"], relevant_dish_ids=r["relevant_dish_ids"], reference_answer=r["reference_answer"],
            required_constraints=r["required_constraints"], expected_behavior=r["expected_behavior"],
        )
        for r in rows
    )
    await session.flush()
    log.info("evaluation dataset seeded", extra={"questions": len(rows)})
    return str(ds.id), len(rows)
