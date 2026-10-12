from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.restaurant.domain.entities import Category, Dish, Restaurant


class MenuRepository(ABC):
    @abstractmethod
    async def get_restaurant(self) -> Restaurant | None: ...

    @abstractmethod
    async def list_categories(self) -> list[Category]: ...

    @abstractmethod
    async def list_dishes(
        self, *, category: str | None = None, available_only: bool = False
    ) -> list[Dish]: ...

    @abstractmethod
    async def get_dish(self, dish_id: str) -> Dish | None: ...

    @abstractmethod
    async def get_dishes(self, dish_ids: Sequence[str]) -> list[Dish]:
        """Returns dishes in the order of ``dish_ids`` (unknown ids are skipped)."""
