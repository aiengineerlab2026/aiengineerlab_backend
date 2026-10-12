"""Deterministic menu filtering.

Food-safety relevant rules (allergens, ingredients, dietary tags) are enforced here in plain
code, never by an LLM. Retrieval may *suggest* dishes; this module decides what is *allowed*.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.domain.restaurant.entities import Dish

ALLERGEN_VOCABULARY: tuple[str, ...] = (
    "gluten",
    "dairy",
    "eggs",
    "peanuts",
    "tree_nuts",
    "soy",
    "fish",
    "shellfish",
    "sesame",
    "mustard",
)

# Free-text -> canonical allergen(s). Deliberately conservative: "nuts" excludes both nut kinds.
_ALLERGEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "gluten": ("gluten",),
    "wheat": ("gluten",),
    "dairy": ("dairy",),
    "milk": ("dairy",),
    "lactose": ("dairy",),
    "cheese": ("dairy",),
    "egg": ("eggs",),
    "eggs": ("eggs",),
    "peanut": ("peanuts",),
    "peanuts": ("peanuts",),
    "tree_nuts": ("tree_nuts",),
    "tree nuts": ("tree_nuts",),
    "tree nut": ("tree_nuts",),
    "nut": ("peanuts", "tree_nuts"),
    "nuts": ("peanuts", "tree_nuts"),
    "soy": ("soy",),
    "soya": ("soy",),
    "fish": ("fish",),
    "shellfish": ("shellfish",),
    "seafood": ("fish", "shellfish"),
    "shrimp": ("shellfish",),
    "sesame": ("sesame",),
    "mustard": ("mustard",),
}


def normalize_allergens(values: Iterable[str]) -> tuple[str, ...]:
    """Map free-text allergen mentions onto the canonical vocabulary (unknown terms are dropped)."""
    out: list[str] = []
    for raw in values:
        key = raw.strip().lower().replace("-", " ").replace("_", " ")
        key = key.replace("free", "").strip()
        for candidate in (key, key.replace(" ", "_"), key.rstrip("s")):
            if candidate in _ALLERGEN_SYNONYMS:
                for a in _ALLERGEN_SYNONYMS[candidate]:
                    if a not in out:
                        out.append(a)
                break
    return tuple(out)


@dataclass(frozen=True)
class DishFilter:
    """Hard constraints extracted from a customer request (value object)."""

    max_price: float | None = None
    min_price: float | None = None
    max_calories: int | None = None
    min_calories: int | None = None
    min_protein: int | None = None
    exclude_allergens: tuple[str, ...] = ()
    exclude_ingredients: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    dietary: tuple[str, ...] = ()  # e.g. "vegetarian", "vegan"
    spicy_only: bool = False
    available_only: bool = True

    def is_empty(self) -> bool:
        """True when the filter imposes no constraint at all (not even availability)."""
        return self == DishFilter(available_only=False)

    def matches(self, dish: Dish) -> bool:
        return not self.violations(dish)

    def violations(self, dish: Dish) -> list[str]:
        """Human-readable reasons a dish fails the filter (empty list = allowed)."""
        reasons: list[str] = []
        if self.available_only and not dish.available:
            reasons.append("unavailable")
        price = float(dish.price)
        if self.max_price is not None and price > self.max_price:
            reasons.append(f"price {price:g} > {self.max_price:g}")
        if self.min_price is not None and price < self.min_price:
            reasons.append(f"price {price:g} < {self.min_price:g}")
        if self.max_calories is not None and dish.calories > self.max_calories:
            reasons.append(f"calories {dish.calories} > {self.max_calories}")
        if self.min_calories is not None and dish.calories < self.min_calories:
            reasons.append(f"calories {dish.calories} < {self.min_calories}")
        if self.min_protein is not None and dish.protein_grams < self.min_protein:
            reasons.append(f"protein {dish.protein_grams} < {self.min_protein}")
        hit = set(self.exclude_allergens) & set(dish.allergens)
        if hit:
            reasons.append("contains allergen: " + ", ".join(sorted(hit)))
        for ing in self.exclude_ingredients:
            needle = ing.lower()
            if any(needle in i.lower() for i in dish.ingredients):
                reasons.append(f"contains ingredient: {ing}")
        if self.categories and dish.category.lower() not in {c.lower() for c in self.categories}:
            reasons.append("category mismatch")
        for diet in self.dietary:
            if diet == "vegetarian" and not dish.is_vegetarian:
                reasons.append("not vegetarian")
            elif diet == "vegan" and "vegan" not in dish.tags:
                reasons.append("not vegan")
            elif diet == "gluten_free" and "gluten" in dish.allergens:
                reasons.append("contains gluten")
        if self.spicy_only and dish.spice_level <= 0:
            reasons.append("not spicy")
        return reasons

    @classmethod
    def from_dict(cls, d: dict) -> DishFilter:
        """Inverse of ``to_dict`` (which omits unset fields)."""
        return cls(
            max_price=d.get("max_price"), min_price=d.get("min_price"),
            max_calories=d.get("max_calories"), min_calories=d.get("min_calories"),
            min_protein=d.get("min_protein"),
            exclude_allergens=tuple(d.get("exclude_allergens", ())),
            exclude_ingredients=tuple(d.get("exclude_ingredients", ())),
            categories=tuple(d.get("categories", ())), dietary=tuple(d.get("dietary", ())),
            spicy_only=bool(d.get("spicy_only", False)),
            available_only=bool(d.get("available_only", False)),  # to_dict drops it when False
        )

    def to_dict(self) -> dict:
        d = {
            "max_price": self.max_price,
            "min_price": self.min_price,
            "max_calories": self.max_calories,
            "min_calories": self.min_calories,
            "min_protein": self.min_protein,
            "exclude_allergens": list(self.exclude_allergens),
            "exclude_ingredients": list(self.exclude_ingredients),
            "categories": list(self.categories),
            "dietary": list(self.dietary),
            "spicy_only": self.spicy_only,
            "available_only": self.available_only,
        }
        return {k: v for k, v in d.items() if v not in (None, [], False)}


@dataclass(frozen=True)
class FilterOutcome:
    allowed: list[Dish] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)  # dish_id -> reasons


def apply_filter(dishes: Iterable[Dish], flt: DishFilter) -> FilterOutcome:
    allowed: list[Dish] = []
    rejected: dict[str, list[str]] = {}
    for dish in dishes:
        why = flt.violations(dish)
        if why:
            rejected[dish.id] = why
        else:
            allowed.append(dish)
    return FilterOutcome(allowed=allowed, rejected=rejected)
