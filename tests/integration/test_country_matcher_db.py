"""Opt-in: Country Matcher's pgvector store against a real PostgreSQL (migration 0010 applied).

    docker compose up -d db && alembic upgrade head
    RUN_INTEGRATION=1 pytest tests/integration/test_country_matcher_db.py

Writes only rows of the fictional country code ``ZZ`` and removes them afterwards.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
from sqlalchemy import delete, text

pytestmark = pytest.mark.integration

if os.getenv("RUN_INTEGRATION") != "1":
    pytest.skip("set RUN_INTEGRATION=1 to run tests against the live database", allow_module_level=True)

from app.config import get_settings  # noqa: E402
from app.country_matcher.domain.models import EmbeddingInfo, ModelName, NewImage, Split  # noqa: E402
from app.country_matcher.infrastructure.models import CountryImageModel  # noqa: E402
from app.country_matcher.infrastructure.repository import SqlCountryStore  # noqa: E402
from app.db import create_engine, create_session_factory, transaction  # noqa: E402

DIMS = {ModelName.CLIP: 512, ModelName.RESNET50: 2048}


def unit(rng, dim, around=None):
    v = rng.normal(size=dim) if around is None else around + rng.normal(scale=0.002, size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
async def session_factory():
    engine = create_engine(get_settings())
    factory = create_session_factory(engine)
    yield factory
    async with transaction(factory) as session:
        await session.execute(delete(CountryImageModel).where(CountryImageModel.country_code == "ZZ"))  # embeddings cascade
        await session.execute(text("DELETE FROM country_matcher_evaluation_runs WHERE results ->> 'marker' = 'integration-test'"))
    await engine.dispose()


def new_image(tag: str, split: Split) -> NewImage:
    return NewImage(
        country_code="ZZ", country_name="Zedland", city=None, original_url=f"https://example.test/{tag}", source_page_url=None,
        storage_path=f"images/ZZ/{tag}.jpg", source_license="CC BY 4.0", source_attribution="tester", image_hash=tag.ljust(64, "0"),
        perceptual_hash="0" * 16, scene_group=f"ZZ|{tag}", width=10, height=10, split=split,
    )


@pytest.mark.parametrize("model", list(ModelName))
async def test_search_orders_by_cosine_similarity_and_only_sees_the_index(session_factory, model):
    rng = np.random.default_rng(7)
    dim = DIMS[model]
    anchor = unit(rng, dim)
    tag = uuid.uuid4().hex[:8]
    info = EmbeddingInfo(model, f"test/{model.value}", f"it-{tag}", dim)
    async with transaction(session_factory) as session:
        store = SqlCountryStore(session)
        near = await store.add_image(new_image(f"{tag}n", Split.INDEX))
        far = await store.add_image(new_image(f"{tag}f", Split.INDEX))
        held_out = await store.add_image(new_image(f"{tag}t", Split.TEST))
        await store.save_embeddings(info, [(near.id, unit(rng, dim, anchor)), (far.id, unit(rng, dim)), (held_out.id, anchor)])
    async with transaction(session_factory) as session:
        hits = await SqlCountryStore(session).search(info, anchor, 5)
    assert [h.image.id for h in hits] == [near.id, far.id]  # the identical held-out vector is invisible
    assert hits[0].similarity > 0.95 > hits[1].similarity and all(h.image.split is Split.INDEX for h in hits)
    async with transaction(session_factory) as session:  # another model version does not match
        other = EmbeddingInfo(model, info.model_name, f"other-{tag}", dim)
        assert await SqlCountryStore(session).search(other, anchor, 5) == []


async def test_storage_rules_missing_embeddings_and_evaluations(session_factory):
    tag = uuid.uuid4().hex[:8]
    rng = np.random.default_rng(1)
    info = EmbeddingInfo(ModelName.CLIP, "test/clip", f"it-{tag}", 512)
    async with transaction(session_factory) as session:
        store = SqlCountryStore(session)
        img = await store.add_image(new_image(tag, Split.INDEX))
        test_img = await store.add_image(new_image(f"{tag}x", Split.TEST))
        assert (await store.find_by_hash(img.image_hash)).id == img.id and await store.has_original_url(img.original_url)
        missing = {i.id for i in await store.images_missing_embedding(info)}
        assert img.id in missing and test_img.id not in missing  # held-out images never need an embedding
        await store.save_embeddings(info, [(img.id, unit(rng, 512))])
        await store.save_embeddings(info, [(img.id, unit(rng, 512))])  # saving again replaces, it does not duplicate
        assert img.id not in {i.id for i in await store.images_missing_embedding(info)}
        stats = await store.embedding_stats(ModelName.RESNET50)
        assert stats["dimension"] == 2048
        with pytest.raises(ValueError):
            await store.save_embeddings(info, [(img.id, unit(rng, 8))])  # a wrong-width vector is refused before the database
        run = await store.save_evaluation({"marker": "integration-test", "k": 5})
        assert (await store.latest_evaluation()).id == run.id
