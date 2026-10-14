from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api import api_router
from app.articles.container import AdminInfra, sql_store_factory
from app.config import get_settings
from app.db import warm_pool
from app.error_handlers import register_error_handlers
from app.logging_config import configure_logging
from app.myresume.application.evaluation import warm_up as warm_up_ranx
from app.myresume.container import MyResumeInfra
from app.myresume.infrastructure.bm25 import spacy_pipeline
from app.restaurant.container import Infra
from app.worker import JobWorker

log = logging.getLogger("app")
REQUEST_TIMEOUT_SECONDS = 180


async def _warm_up(infra: Infra) -> None:
    try:
        await infra.embedder.embed_query("warm up")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("embedding warm-up failed (will load lazily): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.app_env == "production")
    infra = Infra(settings)
    app.state.infra = infra
    app.state.myresume = MyResumeInfra(settings, infra.quota)  # stateless lab: no database, shares the provider quota tracker
    app.state.admin = AdminInfra(settings, sql_store_factory(infra.session_factory))
    worker = JobWorker(infra) if settings.run_worker else None
    try:
        await warm_pool(infra.engine, settings.db_warm_connections)  # fail fast & loudly if the database is unreachable
    except Exception as exc:
        log.error("database connection failed: %s", exc)
        raise
    try:  # the models picked in the Configuration tab survive restarts
        async with infra.uow() as svc:
            saved = await svc.config_repo.get()
        if saved and saved.models:
            infra.restore_models(saved.models)
        async with infra.uow() as svc:  # make sure every embedding model has its index queued/built
            cfg = await svc.config.get()
            await svc.config.update(cfg.chunking, cfg.retrieval)
    except Exception as exc:  # noqa: BLE001 - e.g. migration not applied yet: fall back to the .env models
        log.warning("could not restore the saved model selection: %s", exc)
    if worker:
        await worker.start()
    # Load the embedding model in the background so the first user query (and its measured latency)
    # doesn't pay the one-off model-load cost. Failures are non-fatal: it just loads lazily instead.
    warmup = asyncio.create_task(_warm_up(infra))
    asyncio.get_running_loop().run_in_executor(None, spacy_pipeline)  # load spaCy's model before the first BM25 request
    asyncio.get_running_loop().run_in_executor(None, warm_up_ranx)  # compile ranx before the first evaluation
    log.info("started", extra={"llm_configured": settings.llm_configured, "worker": bool(worker)})
    yield
    warmup.cancel()
    if worker:
        await worker.stop()
    await infra.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI RAG Lab API",
        version="0.1.0",
        description="Backend for AI RAG Lab - The Restaurant: menu, chat, cart, live-tunable RAG and RAGAS evaluation.",
        lifespan=lifespan,
    )
    @app.middleware("http")
    async def timeout_and_timing(request: Request, call_next):
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
        except TimeoutError:
            return JSONResponse(
                {"error": {"code": "timeout", "message": "The request timed out.", "details": {}}}, status_code=504
            )
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    register_error_handlers(app)
    app.include_router(api_router)

    # Added last so it is the outermost middleware: timeouts, 429s and error responses also get CORS headers.
    # No cookies/credentials are used, so "*" is valid.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After", "X-Process-Time-ms"],
    )

    @app.get("/health", tags=["meta"])
    async def health(request: Request) -> dict:
        infra: Infra = request.app.state.infra
        async with infra.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ok", "llm_configured": infra.llm.is_configured}

    return app


app = create_app()
