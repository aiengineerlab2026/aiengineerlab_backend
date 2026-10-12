from __future__ import annotations

from dataclasses import replace

import pytest

from app.application.chat.parser import SearchConstraints, to_dish_filter
from app.domain.restaurant.filters import DishFilter, apply_filter, normalize_allergens
from app.infrastructure.seed.menu_data import CATEGORIES, build_menu


def test_menu_fixture_is_deterministic_and_in_range():
    a, b = build_menu(), build_menu()
    assert a == b
    assert 50 <= len(a) <= 100
    assert len({d.id for d in a}) == len(a)
    assert {d.category for d in a} == {name for name, _ in CATEGORIES}
    assert any(not d.available for d in a), "fixture should include an unavailable dish"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["nuts"], ("peanuts", "tree_nuts")),
        (["Peanut"], ("peanuts",)),
        (["dairy-free"], ("dairy",)),
        (["seafood"], ("fish", "shellfish")),
        (["lactose", "milk"], ("dairy",)),
        (["unicorns"], ()),  # unknown terms are dropped, never guessed
    ],
)
def test_normalize_allergens(raw, expected):
    assert normalize_allergens(raw) == expected


def test_allergen_filter_is_exhaustive_over_the_menu(dishes):
    for allergen in ("dairy", "gluten", "peanuts", "shellfish"):
        outcome = apply_filter(dishes, DishFilter(exclude_allergens=(allergen,), available_only=False))
        assert all(allergen not in d.allergens for d in outcome.allowed)
        assert all(allergen in d.allergens for d in dishes if d.id in outcome.rejected)
        assert len(outcome.allowed) + len(outcome.rejected) == len(dishes)


def test_numeric_filters_are_inclusive(dishes):
    d = dishes[0]
    assert DishFilter(max_price=float(d.price)).matches(d)
    assert not DishFilter(max_price=float(d.price) - 1).matches(d)
    assert DishFilter(min_protein=d.protein_grams).matches(d)
    assert not DishFilter(min_protein=d.protein_grams + 1).matches(d)
    assert DishFilter(max_calories=d.calories).matches(d)


def test_unavailable_dishes_are_filtered_by_default(dishes):
    gone = replace(dishes[0], available=False)
    assert not DishFilter().matches(gone)
    assert DishFilter(available_only=False).matches(gone)


def test_dietary_and_ingredient_filters(dishes):
    veg = apply_filter(dishes, DishFilter(dietary=("vegetarian",))).allowed
    assert veg and all(d.is_vegetarian for d in veg)
    vegan = apply_filter(dishes, DishFilter(dietary=("vegan",))).allowed
    assert vegan and all("vegan" in d.tags for d in vegan)
    no_beef = apply_filter(dishes, DishFilter(exclude_ingredients=("beef",))).allowed
    assert no_beef and all(not any("beef" in i.lower() for i in d.ingredients) for d in no_beef)


def test_violations_explain_why(dishes):
    d = next(x for x in dishes if "peanuts" in x.allergens)
    reasons = DishFilter(exclude_allergens=("peanuts",)).violations(d)
    assert reasons == ["contains allergen: peanuts"]


def test_empty_filter():
    assert DishFilter(available_only=False).is_empty()
    assert not DishFilter().is_empty()  # availability is itself a constraint
    assert DishFilter(max_price=10).to_dict() == {"max_price": 10, "available_only": True}


def test_untrusted_constraints_are_sanitised():
    c = SearchConstraints(
        max_price=-5, exclude_allergens=["peanuts", "NUTS", "plutonium"], categories=["pizza", "Sandwiches"],
        dietary=["vegan", "keto"], exclude_ingredients=["  Mushrooms ", ""],
    )
    f = to_dish_filter(c, [name for name, _ in CATEGORIES])
    assert f.max_price is None
    assert set(f.exclude_allergens) == {"peanuts", "tree_nuts"}
    assert f.categories == ("Pizza",)
    assert f.dietary == ("vegan",)
    assert f.exclude_ingredients == ("mushrooms",)
