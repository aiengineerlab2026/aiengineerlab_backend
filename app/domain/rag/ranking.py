"""Pure ranking algorithms (no I/O): RRF fusion, multi-query merge, dish-level attribution."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    score(d) = sum over rankings r containing d of 1 / (k + rank_r(d)), with 1-based ranks.
    Raw retriever scores are intentionally ignored: only positions matter.
    Ties are broken by first appearance (stable) so results are deterministic.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen: set[str] = set()
        for pos, item in enumerate(ranking, start=1):
            if item in seen:  # a single list can only vote once per item
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def merge_best_score(
    results_per_query: Sequence[Sequence[tuple[str, float]]],
) -> list[tuple[str, float, int]]:
    """Deduplicate candidates across multi-query result lists.

    Each input list is ``(chunk_id, similarity)`` (higher = better, same metric across lists, so the
    values are comparable). Output: ``(chunk_id, best_similarity, hits)`` sorted by best similarity,
    then by the number of query variants that retrieved the chunk, then by first appearance.
    """
    best: dict[str, float] = {}
    hits: dict[str, int] = {}
    first: dict[str, int] = {}
    for results in results_per_query:
        seen: set[str] = set()
        for cid, sim in results:
            if cid in seen:
                continue
            seen.add(cid)
            hits[cid] = hits.get(cid, 0) + 1
            if cid not in best or sim > best[cid]:
                best[cid] = sim
            first.setdefault(cid, len(first))
    ordered = sorted(best, key=lambda c: (-best[c], -hits[c], first[c]))
    return [(c, best[c], hits[c]) for c in ordered]


def rank_dishes(chunk_dish_ids: Sequence[Sequence[str]], limit: int | None = None) -> list[str]:
    """Dish-level ranking used by Precision@K / Recall@K.

    ATTRIBUTION POLICY: walk chunks in rank order; every dish a chunk references is credited to that
    chunk (a multi-dish chunk credits all of its dishes, in document order). A dish is counted only
    at its first (best) position - duplicates are dropped. The result is truncated to ``limit`` dishes.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ids in chunk_dish_ids:
        for d in ids:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out[:limit] if limit is not None else out
