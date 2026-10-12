from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.restaurant.entities import Category, Dish, Restaurant
from app.domain.restaurant.repositories import MenuRepository
from app.infrastructure.db.models import MenuCategoryModel, MenuItemModel, RestaurantModel


def to_dish(m: MenuItemModel) -> Dish:
    return Dish(
        id=m.id,
        name=m.name,
        category=m.category.name,
        description=m.description,
        ingredients=tuple(m.ingredients),
        allergens=tuple(m.allergens or ()),
        price=m.price,
        currency=m.currency,
        calories=m.calories,
        protein_grams=m.protein_grams,
        spice_level=m.spice_level,
        tags=tuple(m.tags or ()),
        available=m.available,
        image_url=m.image_url,
        sort_order=m.sort_order,
    )


class SqlMenuRepository(MenuRepository):
    def __init__(self, session: AsyncSession):
        self._s = session

    async def get_restaurant(self) -> Restaurant | None:
        r = (await self._s.execute(select(RestaurantModel).limit(1))).scalar_one_or_none()
        return Restaurant(r.id, r.name, r.tagline, r.currency) if r else None

    async def list_categories(self) -> list[Category]:
        counts = dict(
            (
                await self._s.execute(
                    select(MenuItemModel.category_id, func.count()).group_by(MenuItemModel.category_id)
                )
            ).all()
        )
        rows = (await self._s.execute(select(MenuCategoryModel).order_by(MenuCategoryModel.sort_order))).scalars()
        return [Category(c.id, c.name, c.emoji, c.sort_order, counts.get(c.id, 0)) for c in rows]

    async def list_dishes(self, *, category: str | None = None, available_only: bool = False) -> list[Dish]:
        stmt = select(MenuItemModel).join(MenuCategoryModel, MenuItemModel.category_id == MenuCategoryModel.id)
        if category:
            stmt = stmt.where(func.lower(MenuCategoryModel.name) == category.lower())
        if available_only:
            stmt = stmt.where(MenuItemModel.available.is_(True))
        stmt = stmt.order_by(MenuItemModel.sort_order)
        return [to_dish(m) for m in (await self._s.execute(stmt)).unique().scalars()]

    async def get_dish(self, dish_id: str) -> Dish | None:
        m = await self._s.get(MenuItemModel, dish_id)
        return to_dish(m) if m else None

    async def get_dishes(self, dish_ids: Sequence[str]) -> list[Dish]:
        if not dish_ids:
            return []
        rows = (
            (await self._s.execute(select(MenuItemModel).where(MenuItemModel.id.in_(list(dish_ids)))))
            .unique()
            .scalars()
        )
        by_id = {m.id: to_dish(m) for m in rows}
        return [by_id[i] for i in dish_ids if i in by_id]
