"""Developer CLI.

    python -m app.cli seed               # menu + evaluation dataset + live configuration (idempotent)
    python -m app.cli seed --build-index # ...and build the live configuration's vector index now
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.logging_config import configure_logging
from app.restaurant.application.rag.config import DEFAULT_CHUNKING, DEFAULT_RETRIEVAL
from app.restaurant.container import Infra
from app.restaurant.domain.rag.entities import IndexStatus
from app.restaurant.infrastructure.seed.seeder import seed_evaluation_dataset, seed_menu

log = logging.getLogger("cli")


async def seed(build_index: bool) -> None:
    infra = Infra(get_settings())
    try:
        async with infra.uow() as s:
            n = await seed_menu(s.session)
            ds_id, q = await seed_evaluation_dataset(s.session)
            if await s.config_repo.get() is None:
                _, index = await s.config.update(DEFAULT_CHUNKING, DEFAULT_RETRIEVAL)
                print(f"Created the default configuration (index {index.status.value})")
        print(f"Seeded {n} dishes and evaluation dataset {ds_id} ({q} questions).")
        if build_index:
            from app.restaurant.application.rag.jobs import run_index_build

            async with infra.uow() as s:
                index = await s.config.ensure_index()
            if index.status != IndexStatus.READY:
                async with infra.uow() as s:
                    await s.index_repo.update_status(index.id, IndexStatus.BUILDING)
                print("Building index (first run downloads the embedding model)...")
                await run_index_build(infra.uow, index)
            print("Index ready.")
    finally:
        await infra.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed", help="Seed menu, evaluation dataset and the live configuration")
    p.add_argument("--build-index", action="store_true", help="Also build the live configuration's index")
    args = parser.parse_args()
    configure_logging(get_settings().log_level, json_logs=False)
    if args.cmd == "seed":
        asyncio.run(seed(args.build_index))


if __name__ == "__main__":
    main()
