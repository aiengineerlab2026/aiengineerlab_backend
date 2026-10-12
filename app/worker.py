"""Durable background job worker (index builds, benchmark evaluation runs, live evaluations).

Jobs are rows in PostgreSQL (``rag_indexes.status`` / ``evaluation_runs.status``). The worker claims one
pending row at a time with ``FOR UPDATE SKIP LOCKED``, so it is safe to run it inside the API process
(default) or standalone (``python -m app.worker``) and even to run several workers. Jobs interrupted by a
crash/restart are re-queued on startup, so they are not lost. Live evaluations (one per answered customer question)
run in their own loop so they are never stuck behind a long benchmark run. Limitation: a job restarts from the
beginning (no mid-run checkpointing) and execution is one-job-at-a-time per worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from app.config import get_settings
from app.logging_config import configure_logging
from app.restaurant.application.rag.jobs import run_index_build
from app.restaurant.container import Infra

log = logging.getLogger("worker")


class JobWorker:
    def __init__(self, infra: Infra):
        self._infra = infra
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        async with self._infra.uow() as s:
            n_idx = await s.indexing.requeue_interrupted()
            n_eval = await s.eval_repo.requeue_interrupted()
            n_live = await s.eval_repo.requeue_interrupted_live()
        if n_idx or n_eval or n_live:
            log.info(
                "re-queued interrupted jobs", extra={"indexes": n_idx, "evaluations": n_eval, "live": n_live}
            )
        self._tasks = [
            asyncio.create_task(self._loop(self.tick), name="job-worker"),
            asyncio.create_task(self._loop(self.tick_live), name="live-eval-worker"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _loop(self, tick) -> None:
        poll = self._infra.settings.worker_poll_seconds
        while not self._stop.is_set():
            try:
                worked = await tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker tick failed")
                worked = False
            if not worked:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=poll)

    async def tick(self) -> bool:
        """Process at most one job. Returns True if a job was processed."""
        async with self._infra.uow() as s:
            index = await s.indexing.claim_pending()  # committed -> status visible as 'building'
        if index is not None:
            log.info("building index", extra={"index_id": index.id})
            try:
                await run_index_build(self._infra.uow, index)
            except Exception:
                log.exception("index build failed", extra={"index_id": index.id})
            return True
        async with self._infra.uow() as s:
            run = await s.eval_repo.claim_pending_run()
        if run is not None:
            log.info("running evaluation", extra={"run_id": run.id})
            async with self._infra.uow() as s:
                runner = s.evaluation_runner()
            await runner.execute(run)
            return True
        return False

    async def tick_live(self) -> bool:
        """Score at most one answered customer question. Returns True if one was processed."""
        async with self._infra.uow() as s:
            item = await s.eval_repo.claim_pending_live()  # committed -> visible as 'running'
        if item is None:
            return False
        log.info("live evaluation", extra={"id": item.id})
        async with self._infra.uow() as s:
            evaluator = s.live_evaluator()
        await evaluator.process(item)  # never raises; failures are stored on the item
        return True


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    infra = Infra(settings)
    worker = JobWorker(infra)
    await worker.start()
    try:
        await asyncio.Event().wait()
    finally:
        await worker.stop()
        await infra.dispose()


if __name__ == "__main__":
    asyncio.run(_main())
