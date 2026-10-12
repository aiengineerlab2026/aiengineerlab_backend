"""Shared helper to run an index build outside of any request, with committed progress updates."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from app.restaurant.domain.rag.entities import RagIndex

UnitOfWork = Callable[[], AbstractAsyncContextManager[Any]]  # yields a per-transaction Services object


async def run_index_build(uow: UnitOfWork, index: RagIndex) -> None:
    async def report(progress: dict) -> None:
        async with uow() as s:  # separate, short transaction so pollers see progress immediately
            await s.indexing.set_progress(index.id, progress)

    try:
        async with uow() as s:
            await s.indexing.build(index, report)
    except Exception as exc:
        async with uow() as s:
            await s.indexing.mark_failed(index.id, f"{type(exc).__name__}: {exc}")
        raise
