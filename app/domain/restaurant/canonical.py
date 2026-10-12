"""Canonical text representation of the menu.

All text-based chunking strategies operate on this single representation so that strategies are
compared on identical source content. Every dish block carries a character span so chunks can be
traced back to the dish(es) they cover.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.domain.restaurant.entities import Dish

CANONICAL_FORMAT_VERSION = "canon-v1"
# Version of the seeded menu fixture (bump whenever menu_data changes; part of index fingerprints).
MENU_DATASET_VERSION = "menu-v2"
DISH_SEPARATOR = "\n\n"
_SPICE_LABELS = {0: "none", 1: "mild", 2: "medium", 3: "hot"}


def render_dish(dish: Dish) -> str:
    """One dish -> one deterministic text block (each line is a full sentence ending in '.')."""
    allergens = ", ".join(a.replace("_", " ") for a in dish.allergens) or "none"
    lines = [
        f"Dish: {dish.name} ({dish.id}).",
        f"Category: {dish.category}.",
        f"Description: {dish.description.rstrip('.')}.",
        f"Ingredients: {', '.join(dish.ingredients)}.",
        f"Allergens: {allergens}.",
    ]
    if dish.tags:
        lines.append(f"Dietary: {', '.join(t.replace('_', ' ') for t in dish.tags)}.")
    lines.append(f"Spice level: {_SPICE_LABELS.get(dish.spice_level, 'none')}.")
    lines.append(f"Price: {float(dish.price):.2f} {dish.currency}.")
    lines.append(f"Nutrition: {dish.calories} kcal, {dish.protein_grams} g protein.")
    lines.append(f"Availability: {'available' if dish.available else 'currently unavailable'}.")
    return "\n".join(lines)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class DishSpan:
    dish_id: str
    dish_name: str
    category: str
    start: int  # offsets into CanonicalCorpus.normalized_text
    end: int


@dataclass(frozen=True)
class CanonicalCorpus:
    dishes: tuple[Dish, ...]
    text: str  # full canonical document (blocks joined by DISH_SEPARATOR)
    normalized_text: str  # whitespace-collapsed copy used for provenance matching
    spans: tuple[DishSpan, ...]
    content_hash: str

    def blocks(self) -> list[tuple[Dish, str]]:
        return [(d, render_dish(d)) for d in self.dishes]

    def dishes_in_range(self, start: int, end: int) -> list[DishSpan]:
        """Dish spans overlapping [start, end) of the normalized text (document order)."""
        return [s for s in self.spans if s.start < end and s.end > start]


def build_corpus(dishes: list[Dish]) -> CanonicalCorpus:
    ordered = tuple(sorted(dishes, key=lambda d: d.id))
    blocks = [render_dish(d) for d in ordered]
    text = DISH_SEPARATOR.join(blocks)
    normalized_parts: list[str] = []
    spans: list[DishSpan] = []
    cursor = 0
    for dish, block in zip(ordered, blocks, strict=True):
        norm = normalize_ws(block)
        start = cursor
        end = start + len(norm)
        spans.append(DishSpan(dish.id, dish.name, dish.category, start, end))
        normalized_parts.append(norm)
        cursor = end + 1  # joining space
    normalized = " ".join(normalized_parts)
    digest = hashlib.sha256((CANONICAL_FORMAT_VERSION + "\n" + text).encode()).hexdigest()
    return CanonicalCorpus(ordered, text, normalized, tuple(spans), digest)
