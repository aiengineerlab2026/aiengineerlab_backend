"""Process-level memory and CPU settings for the PyTorch models (built for small Cloud Run instances)."""

from __future__ import annotations

import ctypes
import gc
import logging
import sys

log = logging.getLogger("country_matcher.runtime")

_libc: ctypes.CDLL | None = None


def configure_torch(threads: int = 1) -> None:
    """Cap PyTorch's intra-op and inter-op thread pools. Call once at start-up, before any model runs.

    One thread per pool keeps the CPU from being oversubscribed on a small instance (extra threads add per-thread
    buffers and contention, not speed) at the price of slower single requests.
    """
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:  # only allowed before any parallel work has started; the intra-op cap above still applies
        log.warning("inter-op threads were already initialised; left unchanged")
    log.info("torch threads set", extra={"threads": threads})


def release_memory() -> None:
    """Collect cyclic garbage, then hand freed heap pages back to the operating system.

    ``gc.collect()`` frees Python objects, but glibc keeps the freed pages in its arenas, so the container's memory
    stays high. ``malloc_trim`` (Linux/glibc only) returns them. Blocking: run it in a worker thread.
    """
    global _libc
    gc.collect()
    if sys.platform != "linux":
        return
    try:
        _libc = _libc or ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except (OSError, AttributeError):  # musl (Alpine) has no malloc_trim; nothing to do
        pass
