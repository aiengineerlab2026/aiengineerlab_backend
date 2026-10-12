"""Chunking strategies (Strategy pattern) over the canonical menu corpus.

fixed / recursive / semantic split the single canonical document with LangChain splitters and then
attribute every chunk to the dish(es) it overlaps (``attribute_chunks``). ``entity`` emits one chunk
per dish. All strategies emit identical metadata keys so downstream code is strategy-agnostic.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings

from app.errors import ExternalServiceError, ValidationFailed
from app.restaurant.domain.canonical import CanonicalCorpus, DishSpan, normalize_ws, render_dish
from app.restaurant.domain.rag.entities import ChunkDraft
from app.restaurant.domain.rag.ports import Chunker
from app.restaurant.domain.rag.value_objects import ChunkingConfig, ChunkingStrategy, LengthUnit

_TOKEN_ENCODING = "cl100k_base"


def _chunk_metadata(strategy: str, spans: list[DishSpan], start: int, end: int) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "dish_names": [s.dish_name for s in spans],
        "categories": sorted({s.category for s in spans}),
        "multi_dish": len(spans) > 1,
        "char_start": start,
        "char_end": end,
        "char_length": end - start,
    }


def attribute_chunks(corpus: CanonicalCorpus, strategy: str, texts: list[str]) -> list[ChunkDraft]:
    """Map chunk texts back to dishes via offsets in the whitespace-normalised canonical text.

    Chunks are produced in document order, so each chunk is searched for from just after the previous
    chunk's start (overlap-safe). If a chunk cannot be located the build fails loudly rather than
    silently producing chunks without provenance.
    """
    haystack = corpus.normalized_text
    prev_start = -1
    drafts: list[ChunkDraft] = []
    for raw in texts:
        needle = normalize_ws(raw)
        if not needle:
            continue
        pos = haystack.find(needle, prev_start + 1)
        if pos < 0:
            pos = haystack.find(needle)  # tolerate splitters that reorder (should not happen)
        if pos < 0:
            raise ExternalServiceError(
                f"Chunk provenance failed: could not locate chunk in canonical text: {needle[:80]!r}"
            )
        end = pos + len(needle)
        spans = corpus.dishes_in_range(pos, end)
        if not spans:
            continue
        prev_start = pos
        drafts.append(
            ChunkDraft(
                content=raw.strip(),
                dish_ids=[s.dish_id for s in spans],
                metadata=_chunk_metadata(strategy, spans, pos, end),
            )
        )
    return drafts


class EntityChunker(Chunker):
    async def chunk(self, corpus: CanonicalCorpus, config: ChunkingConfig) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for dish in corpus.dishes:
            content = render_dish(dish)
            drafts.append(
                ChunkDraft(
                    content=content,
                    dish_ids=[dish.id],
                    metadata={
                        "strategy": "entity",
                        "dish_names": [dish.name],
                        "categories": [dish.category],
                        "multi_dish": False,
                        "char_length": len(content),
                        "price": float(dish.price),
                        "calories": dish.calories,
                        "protein_grams": dish.protein_grams,
                    },
                )
            )
        return drafts


class _SplitterChunker(Chunker):
    strategy_name: str

    @abstractmethod
    def _make_splitter(self, config: ChunkingConfig) -> Any: ...

    async def chunk(self, corpus: CanonicalCorpus, config: ChunkingConfig) -> list[ChunkDraft]:
        def _run() -> list[ChunkDraft]:
            try:
                splitter = self._make_splitter(config)
                texts = [d.page_content for d in splitter.create_documents([corpus.text])]
            except ImportError as exc:
                raise ValidationFailed(f"Chunking dependency missing: {exc}") from exc
            except ValueError as exc:
                raise ValidationFailed(f"Invalid chunking configuration: {exc}") from exc
            return attribute_chunks(corpus, self.strategy_name, texts)

        return await asyncio.to_thread(_run)


class FixedSizeChunker(_SplitterChunker):
    """Pure length-based windows (no semantic separators): chunks may cut through dishes."""

    strategy_name = "fixed"

    def _make_splitter(self, config: ChunkingConfig) -> Any:
        from langchain_text_splitters import CharacterTextSplitter

        kwargs = dict(separator="", chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap, keep_separator=False)
        if config.length_unit == LengthUnit.TOKENS:
            return CharacterTextSplitter.from_tiktoken_encoder(encoding_name=_TOKEN_ENCODING, **kwargs)
        return CharacterTextSplitter(**kwargs)


class RecursiveChunker(_SplitterChunker):
    """Splits on the first separator that keeps pieces under chunk_size (paragraph -> line -> sentence ...)."""

    strategy_name = "recursive"

    def _make_splitter(self, config: ChunkingConfig) -> Any:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        kwargs = dict(
            separators=list(config.separators), chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        if config.length_unit == LengthUnit.TOKENS:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name=_TOKEN_ENCODING, **kwargs)
        return RecursiveCharacterTextSplitter(**kwargs)


class SemanticChunker(_SplitterChunker):
    """Embedding-based breakpoints (LangChain experimental SemanticChunker)."""

    strategy_name = "semantic"

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings

    def _make_splitter(self, config: ChunkingConfig) -> Any:
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

        return LCSemanticChunker(
            self._embeddings,
            breakpoint_threshold_type=config.breakpoint_threshold_type,
            breakpoint_threshold_amount=config.breakpoint_threshold_amount,
        )


class ChunkerFactory:
    """Resolves a ChunkingStrategy to its Chunker implementation."""

    def __init__(self, embeddings_provider):
        self._embeddings_provider = embeddings_provider  # () -> langchain Embeddings (lazy)

    def create(self, strategy: ChunkingStrategy) -> Chunker:
        if strategy == ChunkingStrategy.ENTITY:
            return EntityChunker()
        if strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker()
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker()
        if strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(self._embeddings_provider())
        raise ValidationFailed(f"Unknown chunking strategy '{strategy}'")
