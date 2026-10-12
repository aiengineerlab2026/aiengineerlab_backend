"""Deterministic dish-level retrieval metrics.

Inputs are *dish* ids (already deduplicated and attributed - see ``rank_dishes``), so different
chunking strategies are compared on equal footing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

K_VALUES: tuple[int, ...] = (5, 10)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """relevant dishes in the top-k retrieved / number of dishes in the top-k retrieved.

    Returns ``None`` when the question has no relevant dishes (metric undefined, e.g. "no match"
    questions). Returns 0.0 when relevant dishes exist but nothing was retrieved.
    """
    rel = set(relevant)
    if not rel:
        return None
    top = list(dict.fromkeys(retrieved))[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in rel) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """relevant dishes in the top-k retrieved / total relevant dishes in the ground truth."""
    rel = set(relevant)
    if not rel:
        return None
    top = set(list(dict.fromkeys(retrieved))[:k])
    return len(top & rel) / len(rel)


def retrieval_metrics(retrieved: Sequence[str], relevant: Iterable[str]) -> dict[str, float | None]:
    rel = list(relevant)
    out: dict[str, float | None] = {}
    for k in K_VALUES:
        out[f"precision_at_{k}"] = precision_at_k(retrieved, rel, k)
        out[f"recall_at_{k}"] = recall_at_k(retrieved, rel, k)
    return out


def mean_ignoring_none(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
