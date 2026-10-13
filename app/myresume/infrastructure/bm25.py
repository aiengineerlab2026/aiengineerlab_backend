"""BM25 keyword search: spaCy preprocessing + the Okapi BM25 ranking function (``rank_bm25``)."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from app.myresume.domain.models import Job
from app.myresume.domain.retrieval import Bm25Settings

log = logging.getLogger(__name__)
SPACY_MODEL = "en_core_web_sm"


class SpacyPipeline:
    """spaCy tokeniser (+ lemmatiser when the English model is installed). Loaded once per process."""

    def __init__(self) -> None:
        import spacy

        try:
            # The parser and NER are not needed; the tagger and attribute ruler feed the lemmatiser.
            self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
            self.has_lemmatizer = True
        except OSError:
            log.warning("spaCy model %s is not installed: lemmatisation is unavailable (run: python -m spacy download %s)", SPACY_MODEL, SPACY_MODEL)
            self.nlp = spacy.blank("en")  # tokeniser and stop-word list only
            self.has_lemmatizer = False

    def tokens(self, texts: Sequence[str], s: Bm25Settings) -> list[list[str]]:
        lemmas = s.lemmatize and self.has_lemmatizer
        out: list[list[str]] = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            toks: list[str] = []
            for t in doc:
                if t.is_space:
                    continue
                if s.remove_punctuation and not any(c.isalnum() for c in t.text):
                    continue
                if s.remove_stopwords and t.is_stop:
                    continue
                tok = t.lemma_ if lemmas else t.text
                if s.lowercase:
                    tok = tok.lower()
                if len(tok) >= s.min_token_length:
                    toks.append(tok)
            out.append(toks)
        return out


_pipeline: SpacyPipeline | None = None
_pipeline_lock = threading.Lock()


def spacy_pipeline() -> SpacyPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = SpacyPipeline()
        return _pipeline


class Bm25Index:
    """BM25 over the job documents, built for one preprocessing/parameter setting."""

    def __init__(self, jobs: Sequence[Job], settings: Bm25Settings):
        self.settings = settings
        self._pipe = spacy_pipeline()
        self.job_ids = [j.id for j in jobs]
        corpus = self._pipe.tokens([j.text for j in jobs], settings)
        self._bm25 = BM25Okapi(corpus, k1=settings.k1, b=settings.b)

    @property
    def lemmatizer_missing(self) -> bool:
        return self.settings.lemmatize and not self._pipe.has_lemmatizer

    def score(self, resume_text: str) -> tuple[np.ndarray, int]:
        """BM25 score of every job for the resume, plus the number of query tokens that survived preprocessing."""
        query = self._pipe.tokens([resume_text], self.settings)[0]
        if not query:
            return np.zeros(len(self.job_ids)), 0
        return np.asarray(self._bm25.get_scores(query), dtype=float), len(query)


class Bm25Registry:
    """Builds an index per distinct setting (the corpus is tiny, so this is milliseconds) and keeps the recent ones."""

    def __init__(self, jobs: Sequence[Job], max_size: int = 32):
        self._jobs, self._max = list(jobs), max_size
        self._indexes: OrderedDict[Bm25Settings, Bm25Index] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, settings: Bm25Settings) -> Bm25Index:
        with self._lock:
            idx = self._indexes.get(settings)
            if idx is None:
                idx = self._indexes[settings] = Bm25Index(self._jobs, settings)
                while len(self._indexes) > self._max:
                    self._indexes.popitem(last=False)
            else:
                self._indexes.move_to_end(settings)
            return idx


@lru_cache
def lemmatizer_available() -> bool:
    return spacy_pipeline().has_lemmatizer
