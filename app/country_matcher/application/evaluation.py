"""Measuring CLIP and ResNet50 on a held-out test set (offline: ``cli evaluate``). The API only displays stored results.

Fairness rules, enforced here:

* the queries are the ``test`` split, which is never searched: queries retrieve from the ``index`` split only;
* both models are scored on the identical queries, against the identical indexed images, with the identical relevance;
* every query goes through the same ``PredictionService`` the API uses, so latencies describe the real code path
  (the stored image is fetched and decoded beforehand and is not timed).

Relevance is a proxy: an indexed image is relevant to a query when it carries the same country label. That is not a
judgement that the two photos look alike or show the same kind of place, so Precision@K, Recall@K, NDCG@K and MRR measure
"how many of the retrieved images are labelled with the query's country".
"""

from __future__ import annotations

import logging
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.country_matcher.application.ports import CountryStore, ImageEmbedder, ImageStorage
from app.country_matcher.application.prediction import PredictionService
from app.country_matcher.domain.models import Aggregation, ImageRecord, ModelName, Prediction, Split
from app.country_matcher.infrastructure.imaging import decode_image, hamming
from app.errors import ValidationFailed

log = logging.getLogger("country_matcher.evaluation")

StoreFactory = Callable[[], AbstractAsyncContextManager[CountryStore]]
EmbedderLookup = Callable[[ModelName], ImageEmbedder]

RELEVANCE_NOTE = (
    "Relevance is a proxy: an indexed image counts as relevant to a query when it has the same country label. "
    "Two photos of the same country can look completely different, and this does not measure visual similarity."
)
NEAR_DUPLICATE_DISTANCE = 6  # the same threshold ingestion uses


class EvaluationService:
    def __init__(
        self, open_store: StoreFactory, storage: ImageStorage, predictor: PredictionService, embedders: EmbedderLookup
    ):
        self.open_store, self.storage, self.predictor, self.embedders = open_store, storage, predictor, embedders

    async def run(self, *, k: int = 5, aggregation: Aggregation = Aggregation.WEIGHTED, save: bool = True) -> dict[str, Any]:
        async with self.open_store() as store:
            index_images = await store.list_images(split=Split.INDEX)
            test_images = await store.list_images(split=Split.TEST)
        if not index_images or not test_images:
            raise ValidationFailed("The dataset needs both indexed and held-out test images. Run the ingest command first.")

        queries = [(record, decode_image(await self.storage.get(record.storage_path), min_side=1)) for record in test_images]
        models: dict[str, Any] = {}
        for model in ModelName:
            await self._require_complete_index(model)
            models[model.value] = await self._evaluate_model(model, queries, index_images, k, aggregation)

        results = {
            "generated_at": datetime.now(UTC).isoformat(),
            "k": k,
            "aggregation": aggregation.value,
            "relevance_note": RELEVANCE_NOTE,
            "dataset": _dataset_summary(index_images, test_images),
            "environment": {"python": platform.python_version(), **_torch_environment()},
            "models": models,
        }
        if save:
            async with self.open_store() as store:
                await store.save_evaluation(results)
        return results

    async def _require_complete_index(self, model: ModelName) -> None:
        info = self.embedders(model).info
        async with self.open_store() as store:
            missing = await store.images_missing_embedding(info)
        if missing:
            raise ValidationFailed(f"{len(missing)} indexed images have no {model.value} embedding. Run the index command first.")

    async def _evaluate_model(
        self, model: ModelName, queries: list[tuple[ImageRecord, Any]], index_images: list[ImageRecord], k: int,
        aggregation: Aggregation,
    ) -> dict[str, Any]:
        await self.predictor.predict(queries[0][1], model, k=k, aggregation=aggregation)  # warm-up: not measured
        predictions: list[Prediction] = []
        for _, image in queries:
            predictions.append(await self.predictor.predict(image, model, k=k, aggregation=aggregation))
        embedder = self.embedders(model)
        info = embedder.info
        async with self.open_store() as store:
            stats = await store.embedding_stats(model)
        truth = [record.country_code for record, _ in queries]
        names = {r.country_code: r.country_name for r in index_images} | {r.country_code: r.country_name for r, _ in queries}
        return {
            "model_name": info.model_name,
            "model_version": info.model_version,
            "dimension": info.dimension,
            "retrieval": retrieval_metrics(queries, predictions, index_images, k),
            "classification": classification_metrics(truth, predictions, names),
            "performance": {
                "embed_ms": latency_summary([p.timings.embed_ms for p in predictions]),
                "search_ms": latency_summary([p.timings.search_ms for p in predictions]),
                "total_ms": latency_summary([p.timings.total_ms for p in predictions]),
                "dimension": info.dimension,
                "indexed_vectors": stats["count"],
                "vector_bytes_per_image": round(stats["vector_bytes"] / max(1, stats["count"]), 1),
                "vector_bytes_total": stats["vector_bytes"],
                "table_and_index_bytes": stats["total_bytes"],
                "device": getattr(embedder, "device", "unknown"),
            },
        }


def latency_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 2), "p50": round(float(np.percentile(arr, 50)), 2), "p95": round(float(np.percentile(arr, 95)), 2)}


def retrieval_metrics(queries, predictions: list[Prediction], index_images: list[ImageRecord], k: int) -> dict[str, Any]:
    """Precision@K, Recall@K, NDCG@K and MRR@K with ranx. A query whose country has no indexed image cannot retrieve
    anything relevant and is left out of this block (it still counts in the classification metrics)."""
    # ranx compiles with numba per string width, and its calls must not overlap: fixed-width ids, one process, one call.
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba-cache"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from ranx import Qrels, Run, evaluate

    doc_id = {r.id: f"d{n:06d}" for n, r in enumerate(index_images)}
    by_country: dict[str, list[str]] = {}
    for r in index_images:
        by_country.setdefault(r.country_code, []).append(doc_id[r.id])
    qrels_dict, run_dict = {}, {}
    for n, ((record, _), prediction) in enumerate(zip(queries, predictions, strict=True)):
        relevant = by_country.get(record.country_code)
        if not relevant:
            continue
        qid = f"q{n:06d}"
        qrels_dict[qid] = {d: 1 for d in relevant}
        run_dict[qid] = {doc_id[nb.image.id]: float(nb.similarity) for nb in prediction.neighbors}
    names = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}"]
    if not qrels_dict:
        return {"queries": 0, **{n: None for n in names}}
    scores = evaluate(Qrels(qrels_dict), Run(run_dict), names)
    return {"queries": len(qrels_dict), **{n: round(float(scores[n]), 4) for n in names}}


def classification_metrics(truth: list[str], predictions: list[Prediction], names: dict[str, str]) -> dict[str, Any]:
    """Top-1 (primary), Top-3, Macro F1, a per-country table and the confusion matrix (rows = true, columns = predicted)."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    predicted = [p.country.country_code for p in predictions]
    labels = sorted(set(truth) | set(predicted))
    top3 = [t in [c.country_code for c in p.ranking[:3]] for t, p in zip(truth, predictions, strict=True)]
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    return {
        "queries": len(truth),
        "top1_accuracy": round(sum(t == p for t, p in zip(truth, predicted, strict=True)) / len(truth), 4),
        "top3_accuracy": round(sum(top3) / len(top3), 4),
        "macro_f1": round(float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)), 4),
        "per_country": [
            {"code": c, "name": names.get(c, c), "precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
             "f1": round(float(f1[i]), 4), "support": int(support[i])}
            for i, c in enumerate(labels)
        ],
        "confusion_matrix": {
            "labels": labels,
            "names": [names.get(c, c) for c in labels],
            "matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        },
    }


def _dataset_summary(index_images: list[ImageRecord], test_images: list[ImageRecord]) -> dict[str, Any]:
    idx, tst = Counter(r.country_code for r in index_images), Counter(r.country_code for r in test_images)
    names = {r.country_code: r.country_name for r in [*index_images, *test_images]}
    shared_groups = {r.scene_group for r in index_images} & {r.scene_group for r in test_images}
    near_pairs = sum(
        1 for t in test_images for i in index_images if hamming(t.perceptual_hash, i.perceptual_hash) <= NEAR_DUPLICATE_DISTANCE
    )
    return {
        "index_images": len(index_images),
        "test_images": len(test_images),
        "countries": len(names),
        "per_country": [{"code": c, "name": names[c], "index": idx[c], "test": tst[c]} for c in sorted(names)],
        # Leakage audit between the two splits. All three must be zero for the numbers to be trustworthy.
        "leakage_audit": {
            "shared_image_hashes": len({r.image_hash for r in index_images} & {r.image_hash for r in test_images}),
            "shared_scene_groups": len(shared_groups),
            "near_duplicate_pairs": near_pairs,
        },
    }


def _torch_environment() -> dict[str, str]:
    try:
        import torch

        return {"torch": torch.__version__}
    except ImportError:  # pragma: no cover
        return {}
