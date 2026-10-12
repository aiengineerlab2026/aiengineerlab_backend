from __future__ import annotations

from app.errors import NotFoundError
from app.restaurant.domain.entities import Category, Dish, Restaurant
from app.restaurant.domain.repositories import MenuRepository


class MenuService:
    def __init__(self, menu: MenuRepository):
        self._menu = menu

    async def restaurant(self) -> Restaurant:
        r = await self._menu.get_restaurant()
        if r is None:
            raise NotFoundError("Menu is not seeded. Run `python -m app.cli seed`.")
        return r

    async def categories(self) -> list[Category]:
        return await self._menu.list_categories()

    async def list_dishes(self, category: str | None = None, available_only: bool = False) -> list[Dish]:
        return await self._menu.list_dishes(category=category, available_only=available_only)

    async def get_dish(self, dish_id: str) -> Dish:
        dish = await self._menu.get_dish(dish_id)
        if dish is None:
            raise NotFoundError(f"Dish '{dish_id}' not found.")
        return dish
