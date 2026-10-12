"""Versioned evaluation dataset (ground truth).

Ground-truth relevance is defined by *declarative constraints evaluated against the menu fixture*
(``matches_constraints``). It is therefore independent of every chunking/retrieval strategy and is
never computed from retrieval results. All threshold constraints are INCLUSIVE.

If you change a question, constraint or the menu fixture, bump ``EVAL_DATASET_VERSION``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.restaurant.entities import Dish
from app.infrastructure.seed.menu_data import MENU_DATASET_VERSION

EVAL_DATASET_NAME = "restaurant-eval"
EVAL_DATASET_VERSION = "v2"
EVAL_DATASET_DESCRIPTION = (
    "40 restaurant questions with constraint-derived dish-level relevance labels. Covers protein, "
    "budget, ingredient, allergen, calorie, multi-constraint, no-match, ambiguous and follow-up queries."
)

NO_MATCH_REFERENCE = (
    "No dish on the menu matches this request. The assistant should say so clearly and must not "
    "invent a dish."
)


def matches_constraints(dish: Dish, c: dict[str, Any]) -> bool:
    if not dish.available:
        return False
    price = float(dish.price)
    if "category" in c and dish.category not in c["category"]:
        return False
    if "exclude_categories" in c and dish.category in c["exclude_categories"]:
        return False
    if "max_price" in c and price > c["max_price"]:
        return False
    if "max_calories" in c and dish.calories > c["max_calories"]:
        return False
    if "min_calories" in c and dish.calories < c["min_calories"]:
        return False
    if "min_protein" in c and dish.protein_grams < c["min_protein"]:
        return False
    if "min_spice" in c and dish.spice_level < c["min_spice"]:
        return False
    if "exclude_allergens" in c and set(c["exclude_allergens"]) & set(dish.allergens):
        return False
    if "dietary" in c and not all(t in dish.tags or (t == "vegetarian" and "vegan" in dish.tags) for t in c["dietary"]):
        return False
    if "include_ingredients_any" in c:
        text = " | ".join(dish.ingredients).lower()
        if not any(i in text for i in c["include_ingredients_any"]):
            return False
    return True


MAINS = {"exclude_categories": ["Drinks", "Desserts"]}

# (category, question, constraints, expected_behavior, history)
_Q = list[tuple[str, str, dict[str, Any], str, list[dict[str, str]]]]

QUESTIONS: _Q = [
    # --- high protein ---
    ("high_protein", "Which dishes have at least 50 grams of protein?", {"min_protein": 50}, "recommend", []),
    ("high_protein", "What are the best high-protein options? I want 45 g of protein or more.", {"min_protein": 45}, "recommend", []),
    ("high_protein", "Find a chicken meal with at least 50 g of protein for up to $17.",
     {"include_ingredients_any": ["chicken"], "min_protein": 50, "max_price": 17}, "recommend", []),
    ("high_protein", "Which vegetarian dishes have at least 28 grams of protein?", {"dietary": ["vegetarian"], "min_protein": 28}, "recommend", []),
    # --- budget ---
    ("budget", "What desserts cost $6 or less?", {"category": ["Desserts"], "max_price": 6}, "recommend", []),
    ("budget", "Show me drinks for $4 or less.", {"category": ["Drinks"], "max_price": 4}, "recommend", []),
    ("budget", "Which pizzas cost up to $13?", {"category": ["Pizza"], "max_price": 13}, "recommend", []),
    ("budget", "Which salads cost $9 or less?", {"category": ["Salads"], "max_price": 9}, "recommend", []),
    # --- ingredient ---
    ("ingredient", "Which dishes contain shrimp?", {"include_ingredients_any": ["shrimp"]}, "recommend", []),
    ("ingredient", "I want something with mushrooms.", {"include_ingredients_any": ["mushroom"]}, "recommend", []),
    ("ingredient", "Which dishes have avocado?", {"include_ingredients_any": ["avocado"]}, "recommend", []),
    ("ingredient", "Is there anything made with salmon or tuna?", {"include_ingredients_any": ["salmon", "tuna"]}, "recommend", []),
    ("ingredient", "Which dishes include parmesan?", {"include_ingredients_any": ["parmesan"]}, "recommend", []),
    ("ingredient", "Which meals include chickpeas?", {"include_ingredients_any": ["chickpea"]}, "recommend", []),
    # --- allergen exclusions ---
    ("allergen", "Which chicken dishes are gluten free?", {"category": ["Chicken"], "exclude_allergens": ["gluten"]}, "recommend_safe_only", []),
    ("allergen", "I'm allergic to peanuts and tree nuts. Which salads can I eat?",
     {"category": ["Salads"], "exclude_allergens": ["peanuts", "tree_nuts"]}, "recommend_safe_only", []),
    ("allergen", "What desserts are dairy free?", {"category": ["Desserts"], "exclude_allergens": ["dairy"]}, "recommend_safe_only", []),
    ("allergen", "Which vegan main dishes have no soy and no sesame?",
     {**MAINS, "dietary": ["vegan"], "exclude_allergens": ["soy", "sesame"]}, "recommend_safe_only", []),
    ("allergen", "Which vegetarian pizzas do you have?", {"category": ["Pizza"], "dietary": ["vegetarian"]}, "recommend", []),
    # --- calories ---
    ("calories", "Show me meals with at most 350 calories, not drinks or desserts.", {**MAINS, "max_calories": 350}, "recommend", []),
    ("calories", "Which drinks have 100 calories or fewer?", {"category": ["Drinks"], "max_calories": 100}, "recommend", []),
    ("calories", "Recommend a low-calorie dinner, up to 450 kcal, with at least 30 g protein.",
     {**MAINS, "max_calories": 450, "min_protein": 30}, "recommend", []),
    ("calories", "I want something really indulgent, at least 960 calories.", {"min_calories": 960}, "recommend", []),
    # --- multiple constraints ---
    ("multi_constraint", "I want something really spicy without peanuts.", {"min_spice": 3, "exclude_allergens": ["peanuts"]}, "recommend_safe_only", []),
    ("multi_constraint", "Vegan meals up to $11 with at least 15 g protein.",
     {**MAINS, "dietary": ["vegan"], "max_price": 11, "min_protein": 15}, "recommend", []),
    ("multi_constraint", "A gluten-free dish with at least 40 g protein for up to $20.",
     {"exclude_allergens": ["gluten"], "min_protein": 40, "max_price": 20, **MAINS}, "recommend_safe_only", []),
    ("multi_constraint", "Dairy-free dinner up to $17 and under 500 calories.",
     {**MAINS, "exclude_allergens": ["dairy"], "max_price": 17, "max_calories": 499}, "recommend_safe_only", []),
    ("multi_constraint", "Burgers up to $13 with at least 30 g protein.",
     {"category": ["Burgers"], "max_price": 13, "min_protein": 30}, "recommend", []),
    # --- no valid match ---
    ("no_match", "Do you have sushi?", {"impossible": True, "include_ingredients_any": ["sushi rice", "nori"]}, "decline_no_match", []),
    ("no_match", "Is there a vegan dish with more than 60 grams of protein?", {"dietary": ["vegan"], "min_protein": 60}, "decline_no_match", []),
    ("no_match", "I want a gluten-free pizza.", {"category": ["Pizza"], "exclude_allergens": ["gluten"]}, "decline_no_match", []),
    ("no_match", "Do you have a dessert under 100 calories?", {"category": ["Desserts"], "max_calories": 99}, "decline_no_match", []),
    ("no_match", "Do you serve lamb dishes?", {"include_ingredients_any": ["lamb"]}, "decline_no_match", []),
    # --- ambiguous (interpretation made explicit in constraints) ---
    ("ambiguous", "Something light.", {**MAINS, "max_calories": 350}, "interpret_light_as_le_350_kcal", []),
    ("ambiguous", "I'm craving something sweet.", {"category": ["Desserts"]}, "interpret_sweet_as_desserts", []),
    ("ambiguous", "I want something healthy.", {"category": ["Healthy Meals"]}, "interpret_healthy_as_healthy_meals", []),
    # --- follow-ups (retrieval query = previous user turn + question) ---
    ("follow_up", "Which of those are vegan?", {"category": ["Desserts"], "dietary": ["vegan"]}, "use_history",
     [{"role": "user", "content": "Show me your desserts."}, {"role": "assistant", "content": "We have lava cake, cheesecake, baklava, brownie, fruit bowl, chia pudding and basbousa."}]),
    ("follow_up", "Anything cheaper than $13?", {"category": ["Chicken"], "max_price": 12.99}, "use_history",
     [{"role": "user", "content": "I want a chicken dish."}, {"role": "assistant", "content": "Here are some chicken dishes from our menu."}]),
    ("follow_up", "Which of them are spicy?", {"category": ["Pasta"], "min_spice": 1}, "use_history",
     [{"role": "user", "content": "What pasta do you have?"}, {"role": "assistant", "content": "We have eight pasta dishes."}]),
    ("follow_up", "I'm allergic to nuts, which are safe?", {"category": ["Salads"], "exclude_allergens": ["peanuts", "tree_nuts"]}, "use_history",
     [{"role": "user", "content": "I'm looking for salads."}, {"role": "assistant", "content": "We have eight salads."}]),
]


def build_reference_answer(relevant: list[Dish]) -> str:
    if not relevant:
        return NO_MATCH_REFERENCE
    parts = [
        f"{d.name} (${float(d.price):.2f}, {d.calories} kcal, {d.protein_grams} g protein)" for d in relevant
    ]
    return "Matching dishes: " + "; ".join(parts) + "."


def build_dataset(menu: list[Dish]) -> list[dict[str, Any]]:
    by_id = sorted(menu, key=lambda d: d.id)
    out: list[dict[str, Any]] = []
    for i, (category, question, constraints, behavior, history) in enumerate(QUESTIONS, start=1):
        c = {k: v for k, v in constraints.items() if k != "impossible"}
        relevant = [d for d in by_id if matches_constraints(d, c)]
        if behavior == "decline_no_match" and relevant:
            raise ValueError(f"Question {i} is declared no-match but the menu has matches: {question}")
        out.append(
            {
                "key": f"q_{i:03d}",
                "position": i,
                "category": category,
                "question": question,
                "history": history,
                "required_constraints": c,
                "expected_behavior": behavior,
                "relevant_dish_ids": [d.id for d in relevant],
                "reference_answer": build_reference_answer(relevant),
            }
        )
    return out


def dataset_content_hash(rows: list[dict[str, Any]]) -> str:
    blob = json.dumps(
        {"menu": MENU_DATASET_VERSION, "version": EVAL_DATASET_VERSION, "rows": rows},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()
