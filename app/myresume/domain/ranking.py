"""Pure ranking maths (no I/O): Reciprocal Rank Fusion and match percentages."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ``score(d) = sum over rankings of 1 / (k + rank)`` with 1-based ranks.

    Only positions count, the retrievers' raw scores are ignored (BM25 and cosine are not comparable).
    Ties keep the order in which a job first appeared, so the result is deterministic.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for pos, item in enumerate(dict.fromkeys(ranking), start=1):  # one vote per list and item
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + pos)
            first_seen.setdefault(item, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def match_percent(method: str, score: float, top_score: float, low_score: float = 0.0) -> int:
    """A 0-100 reading of a ranking score.

    Vector: the cosine similarity. Hybrid: the fused RRF scores are min-max normalised over the returned jobs, ``(score - low)
    / (top - low)``, so the best is 100 and the last is 0 (RRF values sit too close together to read as they are; a single
    job, or all with the same score, is 100). BM25 has no fixed scale, so it is relative to the best-scoring job in the list.
    """
    if method == "vector":
        ratio = score
    elif method == "hybrid":
        span = top_score - low_score
        ratio = (score - low_score) / span if span > 0 else 1.0
    else:
        ratio = score / top_score if top_score > 0 else 0.0
    return max(0, min(100, round(ratio * 100)))
