"""Country Matcher: voting rules, SSRF-safe downloading, licensing, ingestion, indexing, prediction, evaluation and the
HTTP API, all against in-memory adapters and fake encoders (no database, network or neural network)."""

from __future__ import annotations

import io
import socket
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from cm_fakes import COLORS, Embedders, MemoryStorage, MemoryStore, jpeg, make_image, seed_dataset
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.country_matcher import presentation
from app.country_matcher.application.evaluation import EvaluationService, classification_metrics
from app.country_matcher.application.indexing import IndexingService
from app.country_matcher.application.ingestion import IngestionConfig, IngestionService, scene_group
from app.country_matcher.application.prediction import IndexNotReady, PredictionService
from app.country_matcher.domain.aggregation import aggregate, explain
from app.country_matcher.domain.models import (
    Aggregation,
    Candidate,
    ImageRecord,
    ModelName,
    Neighbor,
    Prediction,
    Split,
)
from app.country_matcher.infrastructure import safe_http, wikimedia
from app.country_matcher.infrastructure.imaging import decode_image, dhash, hamming, normalise_for_storage
from app.country_matcher.infrastructure.safe_http import (
    SafeImageDownloader,
    UnsafeUrl,
    check_url,
    is_public_ip,
)
from app.country_matcher.infrastructure.seed_config import CountrySeed
from app.country_matcher.infrastructure.storage import (
    LocalImageStorage,
    S3ImageStorage,
    UnconfiguredStorage,
    build_storage,
)
from app.error_handlers import register_error_handlers
from app.errors import NotFoundError, ServiceNotConfigured, ValidationFailed
from app.rate_limit import SlidingWindowLimiter

NOW = datetime.now(UTC)


def record(code: str, name: str | None = None, n: int = 0) -> ImageRecord:
    return ImageRecord(
        id=uuid.uuid4(), country_code=code, country_name=name or f"Country {code}", city=None, original_url=f"https://x/{code}/{n}",
        source_page_url=None, storage_path=f"p/{code}/{n}.jpg", source_license="CC BY 4.0", source_attribution="a",
        image_hash=f"h{code}{n}", perceptual_hash="0" * 16, scene_group=f"{code}|g{n}", width=10, height=10, split=Split.INDEX, created_at=NOW,
    )


def neighbors(*pairs: tuple[str, float]) -> list[Neighbor]:
    return [Neighbor(record(code, n=i), sim) for i, (code, sim) in enumerate(pairs)]


# ----------------------------------------------------------------------------------------- voting rules


class TestAggregation:
    def test_majority_counts_one_vote_per_neighbour(self):
        ranking = aggregate(neighbors(("DE", 0.9), ("FR", 0.8), ("FR", 0.7), ("FR", 0.6), ("DE", 0.5)), Aggregation.MAJORITY)
        assert [(c.country_code, c.votes) for c in ranking] == [("FR", 3), ("DE", 2)]
        assert ranking[0].support == pytest.approx(0.6)

    def test_weighted_lets_a_few_very_close_images_outvote_many_distant_ones(self):
        ns = neighbors(("DE", 0.99), ("DE", 0.98), ("FR", 0.30), ("FR", 0.29), ("FR", 0.28))
        assert aggregate(ns, Aggregation.MAJORITY)[0].country_code == "FR"
        weighted = aggregate(ns, Aggregation.WEIGHTED)
        assert weighted[0].country_code == "DE"
        assert weighted[0].support == pytest.approx((0.99 + 0.98) / (0.99 + 0.98 + 0.30 + 0.29 + 0.28))

    def test_support_shares_sum_to_one(self):
        ranking = aggregate(neighbors(("DE", 0.9), ("FR", 0.8), ("IT", 0.7)), Aggregation.WEIGHTED)
        assert sum(c.support for c in ranking) == pytest.approx(1.0)

    def test_negative_similarity_carries_no_weight_and_all_negative_falls_back_to_counting(self):
        ranking = aggregate(neighbors(("DE", -0.2), ("DE", -0.1), ("FR", -0.3)), Aggregation.WEIGHTED)
        assert ranking[0].country_code == "DE" and ranking[0].support == pytest.approx(2 / 3)

    def test_ties_are_broken_deterministically(self):
        ns = neighbors(("DE", 0.8), ("FR", 0.8))
        assert [c.country_code for c in aggregate(ns, Aggregation.MAJORITY)] == [c.country_code for c in aggregate(ns[::-1], Aggregation.MAJORITY)]

    def test_empty(self):
        assert aggregate([], Aggregation.WEIGHTED) == []
        assert "No labelled images" in explain([], Aggregation.WEIGHTED)

    def test_explanation_states_the_evidence_and_never_calls_it_a_probability(self):
        ns = neighbors(("DE", 0.9), ("DE", 0.85), ("DE", 0.8), ("DE", 0.75), ("FR", 0.7))
        majority = explain(aggregate(ns, Aggregation.MAJORITY), Aggregation.MAJORITY)
        assert "4 of the 5 closest images are labeled Country DE" in majority and "Country FR (1)" in majority
        weighted = explain(aggregate(ns, Aggregation.WEIGHTED), Aggregation.WEIGHTED)
        assert "similarity weight" in weighted and "probab" not in weighted.lower()

    def test_explanation_flags_a_narrow_margin_and_unanimity(self):
        close = explain(aggregate(neighbors(("DE", 0.8), ("DE", 0.7), ("FR", 0.78), ("FR", 0.72)), Aggregation.MAJORITY), Aggregation.MAJORITY)
        assert "narrow" in close
        assert "All retrieved images agree" in explain(aggregate(neighbors(("DE", 0.8), ("DE", 0.7)), Aggregation.MAJORITY), Aggregation.MAJORITY)


# ----------------------------------------------------------------------------------------- images


class TestImaging:
    def test_decode_accepts_the_three_formats_and_rejects_everything_else(self):
        img = make_image((10, 20, 30), 1)
        for fmt in ("JPEG", "PNG", "WEBP"):
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            assert decode_image(buf.getvalue()).mode == "RGB"
        buf = io.BytesIO()
        img.save(buf, format="GIF")
        with pytest.raises(ValidationFailed, match="Unsupported"):
            decode_image(buf.getvalue())
        for junk in (b"", b"hello world", b"\x89PNG\r\n\x1a\nnot really"):
            with pytest.raises(ValidationFailed):
                decode_image(junk)

    def test_too_small_and_too_many_pixels(self):
        small = io.BytesIO()
        Image.new("RGB", (30, 30)).save(small, format="PNG")
        with pytest.raises(ValidationFailed, match="too small"):
            decode_image(small.getvalue())
        bomb = io.BytesIO()
        Image.new("L", (7000, 7000)).save(bomb, format="PNG")  # 49M pixels, a few KB on disk
        with pytest.raises(ValidationFailed, match="too large"):
            decode_image(bomb.getvalue())

    def test_transparency_is_flattened_onto_white(self):
        buf = io.BytesIO()
        Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(buf, format="PNG")
        assert decode_image(buf.getvalue()).getpixel((5, 5)) == (255, 255, 255)

    def test_stored_copy_is_downscaled_never_upscaled(self):
        big, _, _ = normalise_for_storage(Image.new("RGB", (3000, 2000), "red"))
        assert max(Image.open(io.BytesIO(big)).size) == 1024
        data, w, h = normalise_for_storage(Image.new("RGB", (300, 200), "red"))
        assert (w, h) == (300, 200) and Image.open(io.BytesIO(data)).size == (300, 200)

    def test_perceptual_hash_matches_copies_and_separates_different_pictures(self):
        a = make_image((120, 60, 200), 1, size=(400, 300))
        resized = a.resize((200, 150))
        recompressed = Image.open(io.BytesIO(jpeg(a)))
        other = make_image((120, 60, 200), 2, size=(400, 300))
        assert hamming(dhash(a), dhash(resized)) <= 6
        assert hamming(dhash(a), dhash(recompressed)) <= 6
        assert hamming(dhash(a), dhash(other)) > 10


# ----------------------------------------------------------------------------------------- SSRF protection


def resolves_to(monkeypatch, *addresses: str):
    async def getaddrinfo(self, host, port, **kwargs):  # noqa: ARG001
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    monkeypatch.setattr("asyncio.base_events.BaseEventLoop.getaddrinfo", getaddrinfo)


ALLOWED = ["upload.wikimedia.org"]


class TestSafeUrls:
    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.9", "169.254.169.254", "0.0.0.0", "::1", "fe80::1", "::ffff:127.0.0.1", "100.64.0.1"])
    def test_non_public_addresses_are_rejected(self, address):
        assert not is_public_ip(address)

    def test_public_addresses_are_accepted(self):
        assert is_public_ip("8.8.8.8") and is_public_ip("2606:4700:4700::1111")

    @pytest.mark.parametrize(
        "url, message",
        [
            ("http://upload.wikimedia.org/a.jpg", "https"),
            ("ftp://upload.wikimedia.org/a.jpg", "https"),
            ("file:///etc/passwd", "https"),
            ("https://user:pw@upload.wikimedia.org/a.jpg", "Credentials"),
            ("https://upload.wikimedia.org:8443/a.jpg", "port"),
            ("https://evil.example.com/a.jpg", "not allowed"),
            ("https://upload.wikimedia.org.evil.com/a.jpg", "not allowed"),
            ("https://notupload.wikimedia.org/a.jpg", "not allowed"),
            ("https:///a.jpg", "no host"),
        ],
    )
    async def test_urls_that_must_not_be_fetched(self, url, message, monkeypatch):
        resolves_to(monkeypatch, "8.8.8.8")
        with pytest.raises(UnsafeUrl, match=message):
            await check_url(url, ALLOWED)

    async def test_an_allowed_host_resolving_to_a_private_address_is_rejected(self, monkeypatch):
        resolves_to(monkeypatch, "8.8.8.8", "169.254.169.254")  # one bad address among good ones is enough
        with pytest.raises(UnsafeUrl, match="public address"):
            await check_url("https://upload.wikimedia.org/a.jpg", ALLOWED)

    async def test_allowed_host_and_its_subdomains(self, monkeypatch):
        resolves_to(monkeypatch, "8.8.8.8")
        assert await check_url("https://upload.wikimedia.org/a.jpg", ALLOWED) == "upload.wikimedia.org"
        assert await check_url("https://cdn.upload.wikimedia.org/a.jpg", ALLOWED) == "cdn.upload.wikimedia.org"


def downloader(handler, monkeypatch, **kwargs) -> SafeImageDownloader:
    resolves_to(monkeypatch, "8.8.8.8")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
    return SafeImageDownloader(ALLOWED, "test-agent", client=client, min_interval_seconds=0, **kwargs)


class TestDownloader:
    async def test_downloads_an_image(self, monkeypatch):
        body = jpeg(make_image((1, 2, 3), 1))
        d = downloader(lambda r: httpx.Response(200, content=body, headers={"content-type": "image/jpeg"}), monkeypatch)
        assert await d.download("https://upload.wikimedia.org/a.jpg") == (body, "image/jpeg")

    async def test_a_redirect_to_a_host_that_is_not_allowed_is_refused(self, monkeypatch):
        hits = []

        def handler(request):
            hits.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://169.254.169.254/latest/meta-data"})

        with pytest.raises(UnsafeUrl):
            await downloader(handler, monkeypatch).download("https://upload.wikimedia.org/a.jpg")
        assert hits == ["https://upload.wikimedia.org/a.jpg"]  # the redirect target was never contacted

    async def test_redirects_within_allowed_hosts_are_followed_but_not_forever(self, monkeypatch):
        body = jpeg(make_image((1, 2, 3), 1))

        def handler(request):
            if request.url.path == "/a.jpg":
                return httpx.Response(301, headers={"location": "/b.jpg"})
            return httpx.Response(200, content=body, headers={"content-type": "image/jpeg"})

        assert (await downloader(handler, monkeypatch).download("https://upload.wikimedia.org/a.jpg"))[0] == body
        loop = downloader(lambda r: httpx.Response(302, headers={"location": "/again"}), monkeypatch)
        with pytest.raises(UnsafeUrl, match="Too many redirects"):
            await loop.download("https://upload.wikimedia.org/a.jpg")

    async def test_size_and_type_limits(self, monkeypatch):
        d = downloader(lambda r: httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/jpeg"}), monkeypatch, max_bytes=1000)
        with pytest.raises(ValidationFailed, match="larger"):
            await d.download("https://upload.wikimedia.org/a.jpg")
        d = downloader(lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"}), monkeypatch)
        with pytest.raises(ValidationFailed, match="Not an accepted image type"):
            await d.download("https://upload.wikimedia.org/a.jpg")
        d = downloader(lambda r: httpx.Response(404), monkeypatch)
        with pytest.raises(ValidationFailed, match="404"):
            await d.download("https://upload.wikimedia.org/a.jpg")

    async def test_throttling_and_server_errors_are_retried_then_given_up(self, monkeypatch):
        slept = []

        async def no_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(safe_http.asyncio, "sleep", no_sleep)
        body = jpeg(make_image((1, 2, 3), 1))
        calls = []

        def flaky(request):
            calls.append(1)
            if len(calls) < 3:
                return httpx.Response(429, headers={"retry-after": "7"}) if len(calls) == 1 else httpx.Response(503)
            return httpx.Response(200, content=body, headers={"content-type": "image/jpeg"})

        assert (await downloader(flaky, monkeypatch).download("https://upload.wikimedia.org/a.jpg"))[0] == body
        assert len(calls) == 3 and 7 in slept  # Retry-After is honoured

        from app.errors import ExternalServiceError

        with pytest.raises(ExternalServiceError):
            await downloader(lambda r: httpx.Response(503), monkeypatch, attempts=2).download("https://upload.wikimedia.org/a.jpg")


# ----------------------------------------------------------------------------------------- Wikimedia


def page(title="File:Street_in_Tokyo.jpg", *, license="CC BY-SA 4.0", mime="image/jpeg", width=2000, height=1500, **meta):
    extmetadata = {"LicenseShortName": {"value": license}, "Artist": {"value": '<a href="//x">Jane <b>Doe</b></a>'}, **meta}
    return {
        "title": title,
        "imageinfo": [{
            "url": f"https://upload.wikimedia.org/orig/{title}", "thumburl": f"https://upload.wikimedia.org/thumb/{title}",
            "descriptionurl": f"https://commons.wikimedia.org/wiki/{title}", "mime": mime, "width": width, "height": height,
            "extmetadata": extmetadata,
        }],
    }


class TestWikimedia:
    @pytest.mark.parametrize("name", ["CC BY 4.0", "CC BY-SA 3.0", "CC BY-SA 3.0 de", "CC BY 2.0", "CC0", "Public domain", "PD-old", "cc-by-sa-4.0"])
    def test_reuse_with_attribution_is_accepted(self, name):
        assert wikimedia.license_allowed(name), name

    @pytest.mark.parametrize("name", ["CC BY-NC 4.0", "CC BY-NC-SA 3.0", "CC BY-ND 2.0", "All rights reserved", "", "Fair use", "GFDL"])
    def test_everything_else_is_rejected(self, name):
        assert not wikimedia.license_allowed(name), name

    def test_candidate_carries_license_author_and_page(self):
        c = wikimedia._candidate(page(), "Streets in Tokyo", "JP", "Japan", "Tokyo")
        assert c and c.license == "CC BY-SA 4.0" and c.author == "Jane Doe" and c.attribution == "Jane Doe via Wikimedia Commons"
        assert c.page_url.endswith("Street_in_Tokyo.jpg") and c.download_url.startswith("https://upload.wikimedia.org/thumb/")

    @pytest.mark.parametrize(
        ("taken", "kept"),
        [("2021-06-01 10:00:00", True), ("2018:05:04", True), ("2012-01-01", False), ("", False), (None, False)],
    )
    def test_since_year_keeps_only_recently_captured_photos(self, taken, kept):
        meta = {} if taken is None else {"DateTimeOriginal": {"value": taken}}
        found = wikimedia._candidate(page(**meta), "c", "JP", "Japan", None, since_year=2018)
        assert (found is not None) is kept
        assert wikimedia._candidate(page(**meta), "c", "JP", "Japan", None) is not None  # no filter by default

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"license": "CC BY-NC 4.0"}, {"mime": "image/svg+xml"}, {"mime": "image/gif"}, {"width": 300}, {"height": 200},
            {"width": 5000, "height": 800}, {"title": "File:Tokyo_map.jpg"}, {"title": "File:Japan_flag.png"}, {"title": "File:Berlin_Postkarte_002.jpg"},
            {"Restrictions": {"value": "personality"}},
        ],
    )
    def test_unsuitable_files_are_filtered_out(self, kwargs):
        assert wikimedia._candidate(page(**kwargs), "c", "JP", "Japan", None) is None

    async def test_candidates_follow_pagination_and_stop_at_the_limit(self):
        pages = [[page(f"File:a{i}.jpg") for i in range(3)], [page(f"File:b{i}.jpg") for i in range(3)]]
        calls = []

        def handler(request):
            calls.append(dict(request.url.params))
            first = "gcmcontinue" not in request.url.params
            body = {"query": {"pages": pages[0 if first else 1]}}
            if first:
                body["continue"] = {"gcmcontinue": "x", "continue": "gcmcontinue||"}
            return httpx.Response(200, json=body)

        source = wikimedia.WikimediaCommonsSource("agent", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_interval_seconds=0)
        assert len(await source.candidates(category="Streets in X", country_code="JP", country_name="Japan", city="X", limit=5)) == 5
        assert len(calls) == 2 and calls[0]["iiurlwidth"] == "960"

    async def test_category_info_reports_missing_categories(self):
        def handler(request):
            return httpx.Response(200, json={"query": {"pages": [
                {"title": "Category:Streets in A", "categoryinfo": {"files": 12, "subcats": 3}},
                {"title": "Category:Streets in B", "missing": True},
            ]}})

        source = wikimedia.WikimediaCommonsSource("agent", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_interval_seconds=0)
        assert await source.category_info(["Streets in A", "Streets in B"]) == {"Streets in A": {"files": 12, "subcategories": 3}, "Streets in B": None}


# ----------------------------------------------------------------------------------------- ingestion


class FakeSource:
    def __init__(self, candidates: list[Candidate]):
        self.items = candidates

    async def candidates(self, *, category, country_code, country_name, city, limit, since_year=None):
        return [c for c in self.items if c.category == category and c.country_code == country_code][:limit]


class FakeDownloader:
    def __init__(self, bodies: dict[str, bytes], failing: tuple[str, ...] = ()):
        self.bodies, self.failing, self.calls = bodies, failing, []

    async def download(self, url):
        self.calls.append(url)
        if url in self.failing:
            raise UnsafeUrl("Host not allowed: evil.example.com")
        return self.bodies[url], "image/jpeg"


def candidate(code, category, n, author="a") -> Candidate:
    return Candidate(
        country_code=code, country_name=f"Country {code}", city=category, category=category, title=f"t{n}",
        download_url=f"https://upload.wikimedia.org/{code}/{category}/{n}.jpg", original_url=f"https://upload.wikimedia.org/o/{code}/{category}/{n}.jpg",
        page_url="https://commons.wikimedia.org/wiki/x", license="CC BY 4.0", attribution=f"{author} via Wikimedia Commons", author=author, width=2000, height=1500,
    )


def ingestion_world(per_category=8):
    """Two countries with two cities each, every picture different, one photographer per city plus a second one."""
    candidates, bodies, seed = [], {}, 100
    for code, color in (("AA", COLORS["AA"]), ("BB", COLORS["BB"])):
        for category in ("North", "South"):
            for n in range(per_category):
                seed += 1
                c = candidate(code, category, n, author="a" if n < 5 else "b")
                candidates.append(c)
                bodies[c.download_url] = jpeg(make_image(color, seed))
    store, storage = MemoryStore(), MemoryStorage()
    countries = [CountrySeed(code, f"Country {code}", (("North", "North"), ("South", "South"))) for code in ("AA", "BB")]
    return store, storage, FakeSource(candidates), FakeDownloader(bodies), countries


class TestIngestion:
    async def run(self, world, **config):
        store, storage, source, dl, countries = world
        service = IngestionService(store.factory(), storage, source, dl)
        return await service.ingest(countries, IngestionConfig(**{"per_country": 8, "seed": 1, **config}))

    async def test_balanced_sampling_with_storage_and_attribution(self):
        world = ingestion_world()
        reports = await self.run(world)
        store, storage = world[0], world[1]
        assert {c: r.added for c, r in reports.items()} == {"AA": 8, "BB": 8}
        images = await store.list_images()
        assert len(images) == 16 == len(storage.files)
        assert {i.city for i in images} == {"North", "South"}  # spread over both cities
        for i in images:
            assert i.source_license == "CC BY 4.0" and "Wikimedia Commons" in i.source_attribution and i.original_url
            assert i.storage_path == f"images/{i.country_code}/{i.image_hash[:2]}/{i.image_hash}.jpg" and i.storage_path in storage.files

    async def test_split_is_by_scene_group_so_no_scene_is_on_both_sides(self):
        world = ingestion_world()
        await self.run(world, test_fraction=0.25)
        images = await world[0].list_images()
        index_groups = {i.scene_group for i in images if i.split is Split.INDEX}
        test_groups = {i.scene_group for i in images if i.split is Split.TEST}
        assert test_groups and index_groups and not index_groups & test_groups
        for code in ("AA", "BB"):  # every country keeps searchable images and has held-out ones
            assert any(i.country_code == code and i.split is Split.TEST for i in images)
            assert any(i.country_code == code and i.split is Split.INDEX for i in images)

    async def test_scene_group_cap(self):
        world = ingestion_world()
        await self.run(world, max_per_scene=1)
        from collections import Counter

        assert max(Counter(i.scene_group for i in await world[0].list_images()).values()) == 1

    async def test_is_reproducible_for_a_seed_and_varies_with_it(self):
        async def picked(seed):
            world = ingestion_world()
            await self.run(world, seed=seed)
            return sorted(i.original_url for i in await world[0].list_images())

        assert await picked(5) == await picked(5)
        assert await picked(5) != await picked(6)

    async def test_rerun_is_idempotent_and_tops_up(self):
        world = ingestion_world()
        await self.run(world, per_country=4)
        before = {i.id: i.split for i in await world[0].list_images()}
        again = await self.run(world, per_country=4)
        assert all(r.added == 0 and r.already_present == 4 for r in again.values())
        topped = await self.run(world, per_country=8)
        assert all(r.added == 4 for r in topped.values())
        after = {i.id: i.split for i in await world[0].list_images()}
        assert len(after) == 16 and all(after[k] == v for k, v in before.items())  # earlier decisions are never changed

    async def test_duplicates_are_never_added(self):
        world = ingestion_world()
        store, storage, source, dl, countries = world
        original = source.items[0]
        copy = candidate("AA", "South", 99, author="zed")  # a different url and author serving the same picture, resized
        dl.bodies[copy.download_url] = jpeg(Image.open(io.BytesIO(dl.bodies[original.download_url])).resize((160, 120)))
        source.items.insert(0, copy)
        source.items.insert(0, candidate("AA", "North", 98, author="y"))
        dl.bodies[source.items[0].download_url] = dl.bodies[original.download_url]  # byte-identical
        reports = await self.run(world, per_country=8)
        images = await store.list_images()
        hashes = [i.perceptual_hash for i in images]
        assert all(hamming(a, b) > 6 for n, a in enumerate(hashes) for b in hashes[n + 1 :])
        assert sum(reports["AA"].skipped.values()) >= 1

    async def test_a_hostile_or_broken_candidate_is_skipped_without_stopping_the_run(self):
        world = ingestion_world()
        store, storage, source, dl, countries = world
        north = [c for c in source.items if c.country_code == "AA" and c.category == "North"]
        dl.failing = tuple(c.download_url for c in north[1:])  # hostile: the downloader refuses them
        dl.bodies[north[0].download_url] = b"definitely not an image"  # broken: downloads fine, does not decode
        reports = await self.run(world, max_per_scene=5)  # all of North is visited: the run alternates between cities
        assert reports["AA"].added == 8 and len(reports["AA"].failed) == 8
        assert sum("not allowed" in f for f in reports["AA"].failed) == 7 and sum("not a readable image" in f for f in reports["AA"].failed) == 1
        stored = {i.original_url for i in await store.list_images()}
        assert not stored & {c.original_url for c in north}

    async def test_scene_group_names_place_and_photographer(self):
        assert scene_group(candidate("AA", "North", 1, author=" Jane   DOE ")) == "AA|North|jane doe"


# ----------------------------------------------------------------------------------------- indexing, prediction


@pytest.fixture
async def world():
    store, storage, embedders = MemoryStore(), MemoryStorage(), Embedders()
    await seed_dataset(store, storage)
    indexing = IndexingService(store.factory(), storage, embedders.get, batch_size=4)
    predictor = PredictionService(store.factory(), embedders.get)
    return SimpleNamespace(store=store, storage=storage, embedders=embedders, indexing=indexing, predictor=predictor)


class TestIndexingAndPrediction:
    async def test_only_index_images_are_embedded_and_a_rerun_embeds_nothing(self, world):
        reports = await world.indexing.index(list(ModelName))
        assert all(r.embedded == 18 and r.failed == [] for r in reports.values())  # 3 countries x 6 index images, no test image
        assert set(world.store.vectors[ModelName.CLIP]) == {i.id for i in await world.store.list_images(split=Split.INDEX)}
        again = await world.indexing.index(list(ModelName))
        assert all(r.embedded == 0 and r.already_indexed == 18 for r in again.values())
        assert world.embedders.get(ModelName.CLIP).embedded == 18

    async def test_a_new_model_version_re_embeds_everything_and_old_vectors_are_not_searched(self, world):
        await world.indexing.index([ModelName.CLIP])
        clip = world.embedders.get(ModelName.CLIP)
        clip.version = "fake-2"
        report = (await world.indexing.index([ModelName.CLIP]))[ModelName.CLIP]
        assert report.embedded == 18 and report.model_version == "fake-2"
        stats = await world.store.embedding_stats(ModelName.CLIP)
        assert stats["versions"] == ["fake-2"]

    async def test_one_unreadable_stored_image_does_not_stop_indexing(self, world):
        broken = (await world.store.list_images(split=Split.INDEX))[0]
        world.storage.files[broken.storage_path] = b"corrupt"
        report = (await world.indexing.index([ModelName.CLIP]))[ModelName.CLIP]
        assert report.embedded == 17 and report.failed == [str(broken.id)]

    async def test_predict_returns_the_country_of_the_nearest_images(self, world):
        await world.indexing.index(list(ModelName))
        for code, color in COLORS.items():
            p = await world.predictor.predict(make_image(color, 999), ModelName.CLIP, k=5)
            assert p.country.country_code == code and p.country.votes == 5 and p.country.support == pytest.approx(1.0)
            assert len(p.neighbors) == 5 and all(n.image.split is Split.INDEX for n in p.neighbors)
            assert p.timings.total_ms >= p.timings.embed_ms >= 0 and p.embedding.model_version == "fake-1"

    async def test_switching_models_embeds_only_the_query(self, world):
        await world.indexing.index(list(ModelName))
        clip, resnet = world.embedders.get(ModelName.CLIP), world.embedders.get(ModelName.RESNET50)
        before = (clip.embedded, resnet.embedded)
        await world.predictor.predict(make_image(COLORS["AA"], 1), ModelName.RESNET50)
        await world.predictor.predict(make_image(COLORS["AA"], 1), ModelName.CLIP)
        assert (clip.embedded, resnet.embedded) == (before[0] + 1, before[1] + 1)

    async def test_compare_runs_both_models_on_the_same_image(self, world):
        await world.indexing.index(list(ModelName))
        results = await world.predictor.compare(make_image(COLORS["BB"], 5), k=3, aggregation=Aggregation.MAJORITY)
        assert set(results) == set(ModelName) and all(p.country.country_code == "BB" and p.k == 3 for p in results.values())

    async def test_the_held_out_images_are_never_retrieved(self, world):
        await world.indexing.index(list(ModelName))
        test_image = (await world.store.list_images(split=Split.TEST))[0]
        query = decode_image(await world.storage.get(test_image.storage_path), min_side=1)
        p = await world.predictor.predict(query, ModelName.CLIP, k=20)
        assert test_image.id not in {n.image.id for n in p.neighbors} and len(p.neighbors) == 18

    async def test_an_empty_index_is_reported_not_guessed(self, world):
        with pytest.raises(IndexNotReady, match="index is empty"):
            await world.predictor.predict(make_image(COLORS["AA"], 1), ModelName.CLIP)
        assert issubclass(IndexNotReady, ServiceNotConfigured)

    async def test_the_model_is_loaded_before_timing_starts(self, world):
        await world.indexing.index(list(ModelName))
        loads = world.embedders.get(ModelName.CLIP).loads
        await world.predictor.predict(make_image(COLORS["AA"], 1), ModelName.CLIP)
        assert world.embedders.get(ModelName.CLIP).loads == loads + 1  # load() is called (it is idempotent in the real class)


# ----------------------------------------------------------------------------------------- evaluation


class TestEvaluation:
    async def test_metrics_on_a_dataset_with_known_answers(self, world):
        await world.indexing.index(list(ModelName))
        service = EvaluationService(world.store.factory(), world.storage, world.predictor, world.embedders.get)
        results = await service.run(k=3)
        assert results["dataset"]["index_images"] == 18 and results["dataset"]["test_images"] == 6 and results["dataset"]["countries"] == 3
        assert results["dataset"]["leakage_audit"] == {"shared_image_hashes": 0, "shared_scene_groups": 0, "near_duplicate_pairs": 0}
        assert "proxy" in results["relevance_note"] and results["k"] == 3
        for name in ("clip", "resnet50"):
            m = results["models"][name]
            assert m["retrieval"] == {"queries": 6, "precision@3": 1.0, "recall@3": 0.5, "ndcg@3": 1.0, "mrr@3": 1.0}  # 6 relevant per country, 3 retrieved
            c = m["classification"]
            assert (c["top1_accuracy"], c["top3_accuracy"], c["macro_f1"]) == (1.0, 1.0, 1.0)
            assert c["confusion_matrix"]["matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]] and c["confusion_matrix"]["labels"] == ["AA", "BB", "CC"]
            p = m["performance"]
            assert p["total_ms"]["p95"] >= p["total_ms"]["p50"] > 0 and p["indexed_vectors"] == 18 and p["vector_bytes_per_image"] > 0
        assert (await world.store.latest_evaluation()).results == results  # stored for the dashboard

    async def test_identical_queries_for_both_models(self, world):
        await world.indexing.index(list(ModelName))
        service = EvaluationService(world.store.factory(), world.storage, world.predictor, world.embedders.get)
        results = await service.run(k=3, save=False)
        assert results["models"]["clip"]["classification"]["queries"] == results["models"]["resnet50"]["classification"]["queries"] == 6
        assert not world.store.evaluations  # --dry-run stores nothing

    async def test_refuses_to_run_on_an_incomplete_index_or_without_a_test_set(self, world):
        service = EvaluationService(world.store.factory(), world.storage, world.predictor, world.embedders.get)
        with pytest.raises(ValidationFailed, match="Run the index command"):
            await service.run()
        empty = EvaluationService(MemoryStore().factory(), world.storage, world.predictor, world.embedders.get)
        with pytest.raises(ValidationFailed, match="ingest"):
            await empty.run()

    def test_classification_metrics_with_mistakes(self):
        def pred(ranking):
            from app.country_matcher.domain.models import CountryScore

            scores = [CountryScore(c, c, 1, 1.0, 0.5, 0.9) for c in ranking]
            return Prediction(ModelName.CLIP, Aggregation.WEIGHTED, 5, scores[0], scores)

        truth = ["AA", "AA", "BB", "BB", "CC"]
        preds = [pred(["AA"]), pred(["BB", "AA"]), pred(["BB"]), pred(["CC", "AA", "BB"]), pred(["CC"])]
        m = classification_metrics(truth, preds, {"AA": "A", "BB": "B", "CC": "C"})
        assert m["top1_accuracy"] == 0.6 and m["top3_accuracy"] == 1.0
        assert m["confusion_matrix"]["matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 1]]
        by = {c["code"]: c for c in m["per_country"]}
        assert by["AA"]["precision"] == 1.0 and by["AA"]["recall"] == 0.5 and by["CC"]["precision"] == 0.5
        assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 2 / 3) / 3, abs=1e-3)


# ----------------------------------------------------------------------------------------- storage


class TestStorage:
    async def test_local_storage_roundtrip_and_no_path_escape(self, tmp_path):
        s = LocalImageStorage(tmp_path / "root")
        await s.put("a/b.jpg", b"data", "image/jpeg")
        assert await s.get("a/b.jpg") == b"data" and await s.exists("a/b.jpg") and not await s.exists("nope.jpg")
        (tmp_path / "secret.txt").write_text("secret")
        with pytest.raises(NotFoundError):
            await s.get("../secret.txt")
        with pytest.raises(NotFoundError):
            s.resolve("../../etc/passwd")
        with pytest.raises(NotFoundError):
            await s.get("missing.jpg")
        assert s.url("a/b.jpg").endswith("/local-images/a/b.jpg")

    def test_signed_urls_never_contain_the_secret_key(self):
        s3 = S3ImageStorage(
            endpoint="https://example.storage.supabase.co/storage/v1/s3", region="eu-central-1", bucket="country-matcher",
            access_key_id="AKIDEXAMPLE", secret_access_key="super-secret-value", signed_url_ttl_seconds=600,
        )
        url = s3.url("images/DE/ab/abc.jpg")
        assert "/country-matcher/images/DE/ab/abc.jpg" in url and "X-Amz-Signature" in url and "X-Amz-Expires=600" in url
        assert "super-secret-value" not in url

    def test_public_bucket_links_directly(self):
        s3 = S3ImageStorage(
            endpoint="https://e.example/s3", region="eu-central-1", bucket="b", access_key_id="k", secret_access_key="s",
            public_base_url="https://ref.supabase.co/storage/v1/object/public/country-matcher/",
        )
        assert s3.url("images/x.jpg") == "https://ref.supabase.co/storage/v1/object/public/country-matcher/images/x.jpg"

    def test_missing_credentials_do_not_break_startup_but_explain_themselves(self):
        storage = build_storage(Settings(country_matcher_storage="s3", country_matcher_s3_access_key_id=""))
        assert isinstance(storage, UnconfiguredStorage)
        with pytest.raises(ServiceNotConfigured, match="COUNTRY_MATCHER_S3"):
            storage.url("x")

    def test_settings(self):
        assert Settings(country_matcher_allowed_image_hosts=" Upload.Wikimedia.org , ,x.org").country_matcher_allowed_hosts == ["upload.wikimedia.org", "x.org"]
        assert isinstance(build_storage(Settings(country_matcher_storage="local", country_matcher_local_dir="/tmp/cm-test-local")), LocalImageStorage)


# ----------------------------------------------------------------------------------------- HTTP API


@pytest.fixture
async def api(world, tmp_path):
    await world.indexing.index(list(ModelName))
    settings = Settings(country_matcher_max_upload_mb=1, country_matcher_default_k=5, country_matcher_max_k=8)
    infra = SimpleNamespace(
        settings=settings, open_store=world.store.factory(), embedders=world.embedders, storage=world.storage,
        predictor=world.predictor, predict_limiter=SlidingWindowLimiter(4, 3600),
    )
    app = FastAPI()
    app.state.country_matcher = infra
    register_error_handlers(app)
    app.include_router(presentation.router, prefix="/api/v1")
    return SimpleNamespace(client=TestClient(app), infra=infra, world=world)


def upload(color_code="AA", seed=50):
    return {"file": ("street.jpg", jpeg(make_image(COLORS[color_code], seed)), "image/jpeg")}


class TestApi:
    def test_models_reports_index_and_readiness(self, api):
        body = api.client.get("/api/v1/country-matcher/models").json()
        assert body["ready"] is True and body["dataset"] == {"index_images": 18, "test_images": 6, "countries": 3}
        assert {m["id"]: m["index"]["images"] for m in body["models"]} == {"clip": 18, "resnet50": 18}
        assert body["default_k"] == 5 and body["aggregations"] == ["majority", "weighted"]

    async def test_models_not_ready_when_nothing_is_indexed(self, world):
        infra = SimpleNamespace(settings=Settings(), open_store=world.store.factory(), embedders=world.embedders, storage=world.storage)
        app = FastAPI()
        app.state.country_matcher = infra
        app.include_router(presentation.router)
        assert TestClient(app).get("/country-matcher/models").json()["ready"] is False

    def test_indexed_images_lists_only_the_index_with_both_models(self, api):
        body = api.client.get("/api/v1/country-matcher/indexed-images?limit=7").json()
        assert body["total"] == 18 and len(body["images"]) == 7 and body["limit"] == 7
        assert sum(c["images"] for c in body["countries"]) == 18 and len(body["countries"]) == 3
        held_out = {str(i.id) for i in api.world.store.images.values() if i.split is Split.TEST}
        assert not {i["id"] for i in body["images"]} & held_out  # the test photos never show up here
        assert all(i["models"] == ["clip", "resnet50"] and i["url"].startswith("https://storage.test/") for i in body["images"])
        rest = api.client.get("/api/v1/country-matcher/indexed-images?limit=60&offset=7").json()
        assert len(rest["images"]) == 11 and not {i["id"] for i in rest["images"]} & {i["id"] for i in body["images"]}

    def test_indexed_images_filter_by_country_and_validate_input(self, api):
        code = api.client.get("/api/v1/country-matcher/indexed-images").json()["countries"][0]
        body = api.client.get(f"/api/v1/country-matcher/indexed-images?country={code['code'].lower()}").json()
        assert body["total"] == code["images"] and {i["country_code"] for i in body["images"]} == {code["code"]}
        assert api.client.get("/api/v1/country-matcher/indexed-images?country=XYZ").status_code == 422
        assert api.client.get("/api/v1/country-matcher/indexed-images?limit=500").status_code == 422

    def test_examples_are_held_out_images_without_their_label(self, api):
        body = api.client.get("/api/v1/country-matcher/examples?limit=4").json()
        assert len(body["examples"]) == 4
        held_out = {str(i.id) for i in api.world.store.images.values() if i.split is Split.TEST}
        assert {e["id"] for e in body["examples"]} <= held_out
        assert all("country" not in k for e in body["examples"] for k in e)  # the answer is not given away
        assert all(e["url"].startswith("https://storage.test/") and e["license"] == "CC BY 4.0" for e in body["examples"])

    def test_predict_with_an_upload(self, api):
        r = api.client.post("/api/v1/country-matcher/predict", files=upload("BB"), data={"model": "resnet50", "k": "3", "aggregation": "majority"})
        assert r.status_code == 200
        p = r.json()["prediction"]
        assert p["model"] == "resnet50" and p["country"]["country_code"] == "BB" and p["k"] == 3 and len(p["neighbors"]) == 3
        assert p["neighbors"][0]["rank"] == 1 and p["neighbors"][0]["image"]["attribution"] and p["neighbors"][0]["image"]["license"]
        assert "not a calibrated probability" in p["support_note"] and p["timings"]["total_ms"] >= 0
        assert p["model_version"] == "fake-1" and "Country BB was selected" in p["explanation"]
        assert r.json()["query"] == {"example_id": None, "actual_country_code": None, "actual_country_name": None}

    def test_predict_defaults_to_clip_and_weighted_voting(self, api):
        p = api.client.post("/api/v1/country-matcher/predict", files=upload()).json()["prediction"]
        assert p["model"] == "clip" and p["aggregation"] == "weighted" and p["k"] == 5

    def test_predict_an_example_reveals_its_label_afterwards(self, api):
        example = api.client.get("/api/v1/country-matcher/examples?limit=1").json()["examples"][0]
        body = api.client.post("/api/v1/country-matcher/predict", data={"example_id": example["id"]}).json()
        assert body["query"]["example_id"] == example["id"] and body["query"]["actual_country_code"] in COLORS

    def test_an_indexed_image_is_not_a_valid_example(self, api):
        indexed = next(i for i in api.world.store.images.values() if i.split is Split.INDEX)
        r = api.client.post("/api/v1/country-matcher/predict", data={"example_id": str(indexed.id)})
        assert r.status_code == 404

    def test_compare_returns_both_models(self, api):
        r = api.client.post("/api/v1/country-matcher/compare", files=upload("CC"))
        assert r.status_code == 200 and set(r.json()["predictions"]) == {"clip", "resnet50"}
        assert all(p["country"]["country_code"] == "CC" for p in r.json()["predictions"].values())

    @pytest.mark.parametrize(
        "files, data, status, message",
        [
            ({"file": ("a.txt", b"hello", "text/plain")}, {}, 422, "not a readable image"),
            ({"file": ("a.gif", b"GIF89a" + b"0" * 30, "image/gif")}, {}, 422, "readable|Unsupported"),
            (None, {}, 422, "either an image file or an example_id"),
            (upload(), {"example_id": str(uuid.uuid4())}, 422, "either an image file or an example_id"),
            (None, {"example_id": str(uuid.uuid4())}, 404, "Unknown example"),
            (upload(), {"k": "0"}, 422, "k must be between 1 and 8"),
            (upload(), {"k": "9"}, 422, "k must be between 1 and 8"),
            (upload(), {"model": "vgg"}, 422, ""),
            (upload(), {"aggregation": "median"}, 422, ""),
        ],
    )
    def test_invalid_requests(self, api, files, data, status, message):
        r = api.client.post("/api/v1/country-matcher/predict", files=files, data=data)
        assert r.status_code == status
        import re

        assert re.search(message, r.json()["error"]["message"])

    def test_oversized_upload(self, api):
        big = {"file": ("big.jpg", b"\xff\xd8" + b"0" * (1024 * 1024 + 10), "image/jpeg")}
        r = api.client.post("/api/v1/country-matcher/predict", files=big)
        assert r.status_code == 422 and "larger than 1 MB" in r.json()["error"]["message"]

    def test_rate_limit_counts_compare_double_and_refunds_failures(self, api):
        url = "/api/v1/country-matcher"
        assert api.client.post(f"{url}/predict", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 422
        assert api.infra.predict_limiter.peek("testclient").used == 0  # nothing was predicted: nothing was spent
        assert api.client.post(f"{url}/compare", files=upload()).status_code == 200  # costs 2
        assert api.infra.predict_limiter.peek("testclient").used == 2
        assert api.client.post(f"{url}/predict", files=upload()).status_code == 200
        assert api.client.post(f"{url}/predict", files=upload()).status_code == 200  # 4 of 4 used
        blocked = api.client.post(f"{url}/predict", files=upload())
        assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "rate_limited" and "Retry-After" in blocked.headers
        over = api.client.post(f"{url}/compare", files=upload())  # needs 2, none left: nothing is half-spent
        assert over.status_code == 429 and api.infra.predict_limiter.peek("testclient").used == 4

    def test_evaluation_is_read_from_storage_never_recomputed(self, api):
        assert api.client.get("/api/v1/country-matcher/evaluation").json() == {"available": False, "generated_at": None, "results": None}
        asyncio_run(api.world.store.save_evaluation({"k": 5, "models": {"clip": {}, "resnet50": {}}}))
        body = api.client.get("/api/v1/country-matcher/evaluation").json()
        assert body["available"] and body["results"]["k"] == 5 and body["generated_at"]
        assert api.world.embedders.get(ModelName.CLIP).embedded == 18  # indexing only: the GET embedded nothing

    def test_local_image_route_is_absent_with_remote_storage(self, api):
        assert api.client.get("/api/v1/country-matcher/local-images/x.jpg").status_code == 404

    def test_local_image_route_serves_files_but_never_outside_the_root(self, tmp_path):
        storage = LocalImageStorage(tmp_path / "root")
        asyncio_run(storage.put("a/b.jpg", jpeg(make_image((1, 2, 3), 1)), "image/jpeg"))
        (tmp_path / "secret.txt").write_text("secret")
        app = FastAPI()
        app.state.country_matcher = SimpleNamespace(storage=storage)
        register_error_handlers(app)
        app.include_router(presentation.router)
        client = TestClient(app)
        assert client.get("/country-matcher/local-images/a/b.jpg").status_code == 200
        assert client.get("/country-matcher/local-images/..%2Fsecret.txt").status_code == 404
        assert client.get("/country-matcher/local-images/a/missing.jpg").status_code == 404

    def test_app_is_registered_in_the_catalogue(self):
        from app.api import APP_REGISTRY

        entry = next(a for a in APP_REGISTRY if a.id == "country-matcher")
        assert entry.route == "/country-matcher" and entry.name == "Country Matcher"


def asyncio_run(coro):
    import asyncio

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


async def test_missing_tables_become_a_clear_error():
    from sqlalchemy.exc import ProgrammingError

    from app.country_matcher import container
    from app.errors import ServiceNotConfigured

    @asynccontextmanager
    async def boom(_factory):
        raise ProgrammingError("SELECT 1", {}, Exception('relation "country_matcher_images" does not exist'))
        yield  # pragma: no cover

    original = container.transaction
    container.transaction = boom
    try:
        with pytest.raises(ServiceNotConfigured, match="alembic upgrade head"):
            async with container.sql_store_factory(None)():
                pass
    finally:
        container.transaction = original
