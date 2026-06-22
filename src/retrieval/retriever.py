"""Two-stage retrieval with a cross-encoder reranker and a confidence gate.

- **Stage 1 — recall (bi-encoder):** the bge embedding model pulls ``top_k``
  candidate chunks from Chroma. Fast, but only approximately ranked.
- **Stage 2 — precision (cross-encoder):** a cross-encoder re-scores each
  ``(query, chunk)`` pair *jointly*, which judges true relevance far more
  accurately than comparing independent embeddings. We keep the best
  ``rerank_top_n``.
- **Hallucination guard:** scores are sigmoid-normalised to 0-1; if even the
  best reranked chunk falls below ``min_relevance_score``, we declare the
  question unanswerable from our knowledge base (``has_answer == False``) so the
  generator can refuse instead of inventing an answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import settings
from src.indexing.embedder import get_embeddings
from src.indexing.vector_store import load_vector_store


@dataclass
class ScoredDocument:
    """A retrieved chunk paired with its cross-encoder relevance score (0-1)."""

    document: Document
    score: float


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call, including the guard decision."""

    query: str
    documents: list[ScoredDocument]
    top_score: float

    @property
    def has_answer(self) -> bool:
        """True if at least one chunk cleared the relevance threshold."""
        return len(self.documents) > 0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@lru_cache(maxsize=1)
def _load_reranker() -> CrossEncoder:
    """Load the cross-encoder once (downloaded & cached on first use)."""
    return CrossEncoder(settings.retrieval.rerank_model)


class Retriever:
    """Recall → rerank → confidence-gate retrieval pipeline."""

    def __init__(
        self,
        vector_store=None,
        reranker: CrossEncoder | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        min_relevance_score: float | None = None,
    ) -> None:
        cfg = settings.retrieval
        self.vector_store = vector_store or load_vector_store(get_embeddings())
        self.reranker = reranker or _load_reranker()
        self.top_k = top_k or cfg.top_k
        self.rerank_top_n = rerank_top_n or cfg.rerank_top_n
        self.min_relevance_score = (
            cfg.min_relevance_score if min_relevance_score is None
            else min_relevance_score
        )

    def retrieve(self, query: str) -> RetrievalResult:
        # Stage 1: recall candidates with the bi-encoder.
        candidates: list[Document] = self.vector_store.similarity_search(
            query, k=self.top_k
        )
        if not candidates:
            return RetrievalResult(query, documents=[], top_score=0.0)

        # Stage 2: rerank candidates with the cross-encoder.
        pairs = [(query, doc.page_content) for doc in candidates]
        logits = self.reranker.predict(pairs)
        scored = [
            ScoredDocument(document=doc, score=_sigmoid(float(logit)))
            for doc, logit in zip(candidates, logits)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        top_score = scored[0].score

        # Confidence gate + keep the strongest few that pass it.
        kept = [s for s in scored if s.score >= self.min_relevance_score]
        kept = kept[: self.rerank_top_n]
        return RetrievalResult(query, documents=kept, top_score=top_score)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Shared Retriever singleton (loads the embedding + reranker models once)."""
    return Retriever()
