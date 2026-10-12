"""Restaurant & Menu bounded context: entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class Restaurant:
    id: str
    name: str
    tagline: str
    currency: str = "USD"


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    emoji: str
    sort_order: int
    dish_count: int = 0


@dataclass(frozen=True)
class Dish:
    id: str
    name: str
    category: str
    description: str
    ingredients: tuple[str, ...]
    allergens: tuple[str, ...]
    price: Decimal
    currency: str
    calories: int
    protein_grams: int
    spice_level: int = 0  # 0 none, 1 mild, 2 medium, 3 hot
    tags: tuple[str, ...] = field(default_factory=tuple)  # vegetarian, vegan, gluten_free ...
    available: bool = True
    image_url: str | None = None
    sort_order: int = 0

    @property
    def is_vegetarian(self) -> bool:
        return "vegetarian" in self.tags or "vegan" in self.tags
