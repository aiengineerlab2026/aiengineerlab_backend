"""Building the dataset: pick licensed images per country, download them safely, validate, de-duplicate, split into the
searchable index and a held-out test set, store the bytes in object storage and the metadata in Postgres.

Admin-only: it runs from ``python -m app.country_matcher.cli ingest`` and is not exposed over HTTP.

Rules that keep the evaluation honest:

* sampling is reproducible (a seed) and balanced (the same target per country, spread over several cities);
* exact duplicates are caught by the content hash, near-duplicates (resized or re-compressed copies, burst shots) by a
  perceptual hash, across **all** countries and both splits;
* at most ``max_per_scene`` images share a scene group (same place and same photographer);
* the index/test split is made per scene group, so no scene appears on both sides.
It is idempotent: running it again tops each country up to the target and never re-adds an image.
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Protocol

from app.country_matcher.application.ports import CountryStore, ImageStorage
from app.country_matcher.domain.models import Candidate, NewImage, Split
from app.country_matcher.infrastructure.imaging import (
    decode_image,
    dhash,
    hamming,
    normalise_for_storage,
    sha256_hex,
)
from app.country_matcher.infrastructure.seed_config import CountrySeed
from app.errors import DomainError

log = logging.getLogger("country_matcher.ingest")

StoreFactory = Callable[[], AbstractAsyncContextManager[CountryStore]]


class ImageSource(Protocol):
    async def candidates(
        self, *, category: str, country_code: str, country_name: str, city: str | None, limit: int, since_year: int | None = None
    ) -> list[Candidate]: ...


class Downloader(Protocol):
    async def download(self, url: str) -> tuple[bytes, str]: ...


@dataclass(frozen=True)
class IngestionConfig:
    per_country: int = 40
    test_fraction: float = 0.2
    seed: int = 42
    max_per_scene: int = 2
    pool_per_category: int = 60  # how many acceptable files to look at in each category before sampling
    since_year: int | None = None  # only photos captured in this year or later (needs a recorded capture date)
    near_duplicate_distance: int = 6  # max Hamming distance (of 64 bits) between perceptual hashes to call two images the same


@dataclass
class CountryReport:
    added: int = 0
    already_present: int = 0
    skipped: Counter = field(default_factory=Counter)  # reason -> count
    failed: list[str] = field(default_factory=list)  # "url: reason"


def scene_group(candidate: Candidate) -> str:
    author = " ".join(candidate.author.lower().split())[:80]
    return f"{candidate.country_code}|{candidate.category}|{author}"[:300]


def storage_path(country_code: str, image_hash: str) -> str:
    return f"images/{country_code}/{image_hash[:2]}/{image_hash}.jpg"


@dataclass
class _Prepared:
    candidate: Candidate
    group: str
    data: bytes
    width: int
    height: int
    image_hash: str
    perceptual_hash: str


class IngestionService:
    def __init__(self, open_store: StoreFactory, storage: ImageStorage, source: ImageSource, downloader: Downloader):
        self.open_store, self.storage, self.source, self.downloader = open_store, storage, source, downloader

    async def ingest(self, countries: list[CountrySeed], config: IngestionConfig) -> dict[str, CountryReport]:
        reports: dict[str, CountryReport] = {}
        for seed_country in countries:
            reports[seed_country.code] = await self._ingest_country(seed_country, config)
        return reports

    async def _ingest_country(self, country: CountrySeed, config: IngestionConfig) -> CountryReport:
        report = CountryReport()
        async with self.open_store() as store:
            known = await store.list_images()
        mine = [r for r in known if r.country_code == country.code]
        report.already_present = len(mine)
        needed = config.per_country - len(mine)
        if needed <= 0:
            log.info("country already complete", extra={"country": country.code, "images": len(mine)})
            return report

        group_counts = Counter(r.scene_group for r in mine)
        group_split = {r.scene_group: r.split for r in known}
        phashes = [r.perceptual_hash for r in known]
        pool = await self._pool(country, config, report)
        rng = random.Random(f"{config.seed}:{country.code}")

        accepted: list[_Prepared] = []
        for candidate in _interleave(pool, rng):
            if len(accepted) >= needed:
                break
            group = scene_group(candidate)
            if group_counts[group] >= config.max_per_scene:
                report.skipped["scene_limit"] += 1
                continue
            try:
                prepared = await self._prepare(candidate, group, phashes, config, report)
            except DomainError as exc:
                report.failed.append(f"{candidate.download_url}: {exc.message}")
                log.warning("candidate rejected", extra={"url": candidate.download_url, "reason": exc.message})
                continue
            if prepared is None:
                continue
            accepted.append(prepared)
            phashes.append(prepared.perceptual_hash)
            group_counts[group] += 1

        splits = _assign_splits([p.group for p in accepted], group_split, len(mine) + len(accepted), mine, config, rng)
        for prepared in accepted:
            try:
                await self._store(prepared, splits[prepared.group])
                report.added += 1
            except Exception as exc:  # noqa: BLE001 - one bad upload must not stop the run; rerun resumes
                report.failed.append(f"{prepared.candidate.download_url}: store failed ({type(exc).__name__})")
                log.exception("could not store image", extra={"url": prepared.candidate.download_url})
        log.info("country ingested", extra={"country": country.code, "added": report.added, "needed": needed})
        return report

    async def _pool(self, country: CountrySeed, config: IngestionConfig, report: CountryReport) -> list[Candidate]:
        pool: list[Candidate] = []
        for category, city in country.places:
            try:
                found = await self.source.candidates(
                    category=category, country_code=country.code, country_name=country.name, city=city, limit=config.pool_per_category,
                    since_year=config.since_year,
                )
            except DomainError as exc:
                report.failed.append(f"category {category}: {exc.message}")
                continue
            pool.extend(found)
        seen, unique = set(), []
        for c in pool:
            if c.original_url not in seen:
                seen.add(c.original_url)
                unique.append(c)
        return unique

    async def _prepare(
        self, candidate: Candidate, group: str, phashes: list[str], config: IngestionConfig, report: CountryReport
    ) -> _Prepared | None:
        async with self.open_store() as store:
            if await store.has_original_url(candidate.original_url):
                report.skipped["already_stored"] += 1
                return None
        raw, _ = await self.downloader.download(candidate.download_url)
        image = decode_image(raw)
        data, width, height = normalise_for_storage(image)
        digest = sha256_hex(data)
        async with self.open_store() as store:
            if await store.find_by_hash(digest):
                report.skipped["exact_duplicate"] += 1
                return None
        phash = dhash(image)
        if any(hamming(phash, other) <= config.near_duplicate_distance for other in phashes):
            report.skipped["near_duplicate"] += 1
            return None
        return _Prepared(candidate, group, data, width, height, digest, phash)

    async def _store(self, p: _Prepared, split: Split) -> None:
        c = p.candidate
        path = storage_path(c.country_code, p.image_hash)
        await self.storage.put(path, p.data, "image/jpeg")  # same bytes -> same path, so a retry is harmless
        async with self.open_store() as store:
            await store.add_image(
                NewImage(
                    country_code=c.country_code, country_name=c.country_name, city=c.city, original_url=c.original_url,
                    source_page_url=c.page_url, storage_path=path, source_license=c.license[:100],
                    source_attribution=c.attribution, image_hash=p.image_hash, perceptual_hash=p.perceptual_hash,
                    scene_group=p.group, width=p.width, height=p.height, split=split,
                )
            )


def _interleave(pool: list[Candidate], rng: random.Random) -> list[Candidate]:
    """Shuffle within each category, then take turns between categories, so cities are evenly represented."""
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        by_category[c.category].append(c)
    queues = [by_category[k] for k in sorted(by_category)]
    for q in queues:
        rng.shuffle(q)
    out: list[Candidate] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop())
    return out


def _assign_splits(groups, existing_split, total, existing_of_country, config, rng) -> dict[str, Split]:
    """Whole scene groups go to either side. Groups already stored keep their split."""
    result: dict[str, Split] = {g: existing_split[g] for g in set(groups) if g in existing_split}
    fresh = sorted({g for g in groups if g not in result})
    size = Counter(g for g in groups if g in fresh)
    rng.shuffle(fresh)
    want_test = round(total * config.test_fraction)
    if config.test_fraction > 0 and total >= 2:
        want_test = max(1, want_test)
    have_test = sum(1 for r in existing_of_country if r.split is Split.TEST)
    have_test += sum(1 for g in groups if result.get(g) is Split.TEST)
    have_index = sum(1 for r in existing_of_country if r.split is Split.INDEX)
    have_index += sum(1 for g in groups if result.get(g) is Split.INDEX)
    for position, g in enumerate(fresh):
        last_and_no_index = position == len(fresh) - 1 and have_index == 0  # never leave a country without searchable images
        if have_test < want_test and not last_and_no_index:
            result[g], have_test = Split.TEST, have_test + size[g]
        else:
            result[g], have_index = Split.INDEX, have_index + size[g]
    return result
