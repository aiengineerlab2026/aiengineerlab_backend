"""Turning the K nearest labelled images into a country: two voting rules and a plain-language reason.

The "support score" is the share of the neighbour evidence that backs the winning country. It is a measure of
agreement among retrieved images, not a calibrated probability of being right.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from app.country_matcher.domain.models import Aggregation, CountryScore, Neighbor


def aggregate(neighbors: Sequence[Neighbor], aggregation: Aggregation) -> list[CountryScore]:
    """Countries ranked by their support among ``neighbors`` (best first). Empty input gives an empty ranking."""
    if not neighbors:
        return []
    groups: dict[str, list[Neighbor]] = defaultdict(list)
    for n in neighbors:
        groups[n.image.country_code].append(n)

    def weight(ns: list[Neighbor]) -> float:
        return sum(max(n.similarity, 0.0) for n in ns)  # a negative cosine carries no evidence

    total_votes = len(neighbors)
    total_weight = sum(weight(ns) for ns in groups.values())
    scored = []
    for code, ns in groups.items():
        w = weight(ns)
        if aggregation is Aggregation.WEIGHTED and total_weight > 0:
            support = w / total_weight
        else:  # majority, or every similarity <= 0 (then counting is the only meaningful evidence)
            support = len(ns) / total_votes
        scored.append(
            CountryScore(
                country_code=code,
                country_name=ns[0].image.country_name,
                votes=len(ns),
                weight=w,
                support=support,
                best_similarity=max(n.similarity for n in ns),
                image_ids=tuple(n.image.id for n in ns),
            )
        )
    # Primary evidence first, the other measure and then the single closest image break ties, the name makes it stable.
    if aggregation is Aggregation.WEIGHTED:
        scored.sort(key=lambda s: (-s.support, -s.votes, -s.best_similarity, s.country_name))
    else:
        scored.sort(key=lambda s: (-s.votes, -s.weight, -s.best_similarity, s.country_name))
    return scored


def explain(ranking: Sequence[CountryScore], aggregation: Aggregation) -> str:
    """Why the top country won, using only the retrieved neighbours (nothing about the picture itself is claimed)."""
    if not ranking:
        return "No labelled images were retrieved, so no country can be suggested."
    top = ranking[0]
    n = sum(c.votes for c in ranking)
    represented = ", ".join(f"{c.country_name} ({c.votes})" for c in ranking)
    pct = round(top.support * 100)
    if aggregation is Aggregation.WEIGHTED:
        reason = (
            f"{top.country_name} was selected because its {_images(top.votes)} among the {n} closest "
            f"carr{'ies' if top.votes == 1 else 'y'} {pct}% of the total similarity weight "
            f"(each image votes with its cosine similarity)."
        )
    else:
        reason = f"{top.country_name} was selected because {top.votes} of the {n} closest images are labeled {top.country_name}."
    if len(ranking) == 1:
        return f"{reason} All retrieved images agree. Countries represented: {represented}."
    runner = ranking[1]
    margin = top.support - runner.support
    caveat = " The margin over the runner-up is narrow, so treat this with caution." if margin < 0.2 else ""
    return (
        f"{reason} The runner-up is {runner.country_name} with {_images(runner.votes)}."
        f"{caveat} Countries represented: {represented}."
    )


def _images(count: int) -> str:
    return f"{count} image{'' if count == 1 else 's'}"
