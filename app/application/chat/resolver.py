"""Deterministic mapping of free-text dish references ("the burger", "2 grilled chicken bowls") to dishes."""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain.restaurant.entities import Dish

_STOP = {"the", "a", "an", "some", "of", "please", "one", "my", "your", "with"}


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


@dataclass
class Resolution:
    status: str  # resolved | ambiguous | not_found
    dish: Dish | None = None
    candidates: list[Dish] = field(default_factory=list)


def resolve_dish(ref: str, pool: Sequence[Dish], prefer_ids: Sequence[str] = ()) -> Resolution:
    ref_tokens = set(tokens(ref))
    if not ref_tokens:
        return Resolution("not_found")
    exact: list[Dish] = []
    full: list[Dish] = []
    for d in pool:
        name_tokens = set(tokens(d.name))
        if ref_tokens == name_tokens:
            exact.append(d)
        elif ref_tokens <= name_tokens:
            full.append(d)
    matches = exact or full
    if len(matches) == 1:
        return Resolution("resolved", matches[0], matches)
    if len(matches) > 1:
        preferred = [d for d in matches if d.id in set(prefer_ids)]
        if len(preferred) == 1:
            return Resolution("resolved", preferred[0], matches)
        return Resolution("ambiguous", None, matches)
    names = {d.name.lower(): d for d in pool}
    close = difflib.get_close_matches(ref.lower(), list(names), n=3, cutoff=0.6)
    return Resolution("not_found", None, [names[c] for c in close])


_PRONOUNS = {"it", "that", "this", "that one", "this one", "the first one", "first one", "the first", "them"}
_ORDINALS = {"second": 1, "the second one": 1, "second one": 1, "the second": 1, "third": 2, "the third one": 2}


def resolve_reference(ref: str, pool: Sequence[Dish], last_dish_ids: Sequence[str], prefer_ids: Sequence[str] = ()) -> Resolution:
    """Like ``resolve_dish`` but also understands "it / that one / the second one" using the last shown dishes."""
    key = ref.strip().lower()
    by_id = {d.id: d for d in pool}
    if key in _PRONOUNS or key in _ORDINALS:
        idx = _ORDINALS.get(key, 0)
        if idx < len(last_dish_ids) and last_dish_ids[idx] in by_id:
            return Resolution("resolved", by_id[last_dish_ids[idx]])
        return Resolution("not_found")
    return resolve_dish(ref, pool, [*prefer_ids, *last_dish_ids])
