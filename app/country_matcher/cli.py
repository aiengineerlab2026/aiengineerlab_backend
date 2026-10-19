"""Admin CLI of Country Matcher. These commands write to the database and to object storage with the server's own
credentials; none of them is reachable over HTTP.

    python -m app.country_matcher.cli check-sources                  # do the Commons categories exist? how many files?
    python -m app.country_matcher.cli ingest --per-country 40        # download, validate, de-duplicate, split, upload
    python -m app.country_matcher.cli index                          # embed the indexed images with CLIP and ResNet50
    python -m app.country_matcher.cli evaluate --k 5                 # held-out evaluation, stored for the dashboard
    python -m app.country_matcher.cli stats                          # what is in the dataset

Run them in this order. Each is idempotent and safe to repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.country_matcher.application.evaluation import EvaluationService
from app.country_matcher.application.indexing import IndexingService
from app.country_matcher.application.ingestion import IngestionConfig, IngestionService
from app.country_matcher.application.prediction import PredictionService
from app.country_matcher.container import sql_store_factory
from app.country_matcher.domain.models import Aggregation, ModelName, Split
from app.country_matcher.infrastructure.embedders import EmbedderRegistry
from app.country_matcher.infrastructure.safe_http import SafeImageDownloader
from app.country_matcher.infrastructure.seed_config import DEFAULT_COUNTRIES, CountrySeed
from app.country_matcher.infrastructure.storage import build_storage
from app.country_matcher.infrastructure.wikimedia import WikimediaCommonsSource
from app.db import create_engine, create_session_factory
from app.logging_config import configure_logging

log = logging.getLogger("country_matcher.cli")


def _select(codes: list[str] | None) -> list[CountrySeed]:
    if not codes:
        return list(DEFAULT_COUNTRIES)
    wanted = {c.upper() for c in codes}
    chosen = [c for c in DEFAULT_COUNTRIES if c.code in wanted]
    unknown = wanted - {c.code for c in chosen}
    if unknown:
        raise SystemExit(f"Unknown country codes: {', '.join(sorted(unknown))}. Known: {', '.join(c.code for c in DEFAULT_COUNTRIES)}")
    return chosen


async def check_sources(args) -> None:
    source = WikimediaCommonsSource(get_settings().country_matcher_user_agent)
    try:
        for country in _select(args.countries):
            info = await source.category_info([c for c, _ in country.places])
            for category, _ in country.places:
                found = info.get(category)
                print(f"{country.code}  {category:38} " + ("MISSING" if found is None else f"{found['files']} files, {found['subcategories']} subcategories"))
    finally:
        await source.aclose()


async def ingest(args) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    source = WikimediaCommonsSource(settings.country_matcher_user_agent)
    downloader = SafeImageDownloader(settings.country_matcher_allowed_hosts, settings.country_matcher_user_agent)
    try:
        service = IngestionService(sql_store_factory(create_session_factory(engine)), build_storage(settings), source, downloader)
        config = IngestionConfig(
            per_country=args.per_country, test_fraction=args.test_fraction, seed=args.seed, max_per_scene=args.max_per_scene,
            since_year=args.since_year,
        )
        reports = await service.ingest(_select(args.countries), config)
    finally:
        await downloader.aclose()
        await source.aclose()
        await engine.dispose()
    for code, r in reports.items():
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(r.skipped.items())) or "none"
        print(f"{code}: added {r.added}, already present {r.already_present}, skipped [{skipped}], failed {len(r.failed)}")
        for line in r.failed[:5]:
            print(f"    failed: {line}")


async def index(args) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    try:
        service = IndexingService(
            sql_store_factory(create_session_factory(engine)), build_storage(settings), EmbedderRegistry(settings).get
        )
        models = [ModelName(m) for m in args.models] if args.models else list(ModelName)
        reports = await service.index(models)
    finally:
        await engine.dispose()
    for model, r in reports.items():
        print(f"{model.value} ({r.model_name} @ {r.model_version}): embedded {r.embedded}, already indexed {r.already_indexed}, failed {len(r.failed)}")


async def evaluate(args) -> None:
    settings = get_settings()
    engine = create_engine(settings)
    try:
        open_store, storage, embedders = sql_store_factory(create_session_factory(engine)), build_storage(settings), EmbedderRegistry(settings)
        predictor = PredictionService(open_store, embedders.get, max_concurrent=1)
        results = await EvaluationService(open_store, storage, predictor, embedders.get).run(
            k=args.k, aggregation=Aggregation(args.aggregation), save=not args.dry_run
        )
    finally:
        await engine.dispose()
    d = results["dataset"]
    print(f"{d['index_images']} indexed images, {d['test_images']} held-out test images, {d['countries']} countries, k={results['k']}")
    print(f"leakage audit (all must be 0): {d['leakage_audit']}")
    k = results["k"]
    for name, r in results["models"].items():
        c, rt, p = r["classification"], r["retrieval"], r["performance"]
        print(
            f"{name:9} top1 {c['top1_accuracy']:.3f}  top3 {c['top3_accuracy']:.3f}  macroF1 {c['macro_f1']:.3f} | "
            f"P@{k} {rt[f'precision@{k}']}  R@{k} {rt[f'recall@{k}']}  NDCG@{k} {rt[f'ndcg@{k}']}  MRR@{k} {rt[f'mrr@{k}']} | "
            f"total {p['total_ms']['mean']} ms ({p['device']})"
        )
    print("Dry run: nothing stored." if args.dry_run else "Stored for the dashboard.")


async def stats(_args) -> None:
    engine = create_engine(get_settings())
    try:
        async with sql_store_factory(create_session_factory(engine))() as store:
            images = await store.list_images()
            counts: dict[str, list[int]] = {}
            for r in images:
                counts.setdefault(f"{r.country_code} {r.country_name}", [0, 0])[0 if r.split is Split.INDEX else 1] += 1
            print(f"{len(images)} images in {len(counts)} countries")
            for name, (idx, tst) in sorted(counts.items()):
                print(f"  {name:22} index {idx:3}  test {tst:3}")
            for m in ModelName:
                s = await store.embedding_stats(m)
                print(f"{m.value}: {s['count']} embeddings, versions {s['versions']}, {s['vector_bytes']} vector bytes")
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.country_matcher.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("check-sources", "Check the Commons categories"), ("ingest", "Build the dataset")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--countries", nargs="*", help="ISO codes to include (default: all configured)")
        if name == "ingest":
            p.add_argument("--per-country", type=int, default=40, help="images per country, test split included (default 40)")
            p.add_argument("--test-fraction", type=float, default=0.2, help="share held out for evaluation (default 0.2)")
            p.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
            p.add_argument("--since-year", type=int, default=None, help="only photos captured in this year or later (skips undated ones)")
            p.add_argument("--max-per-scene", type=int, default=2, help="max images per place and photographer (default 2)")
    p = sub.add_parser("index", help="Embed the indexed images with both models")
    p.add_argument("--models", nargs="*", choices=[m.value for m in ModelName])
    p = sub.add_parser("evaluate", help="Run the held-out evaluation and store the results")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--aggregation", choices=[a.value for a in Aggregation], default=Aggregation.WEIGHTED.value)
    p.add_argument("--dry-run", action="store_true", help="compute and print, do not store")
    sub.add_parser("stats", help="Dataset and embedding counts")
    args = parser.parse_args()
    configure_logging(get_settings().log_level, json_logs=False)
    asyncio.run({"check-sources": check_sources, "ingest": ingest, "index": index, "evaluate": evaluate, "stats": stats}[args.cmd](args))


if __name__ == "__main__":
    main()
