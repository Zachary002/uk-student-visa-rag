"""Unit tests for the rerank + confidence-gate logic.

Fakes stand in for the vector store and cross-encoder, so these run instantly
with no model downloads — they isolate the *logic* (sigmoid, sort, threshold,
top-n) from the ML models.
"""
from langchain_core.documents import Document

from src.retrieval.retriever import Retriever, _sigmoid


class FakeVectorStore:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._docs[:k]


class FakeReranker:
    """Returns pre-set logits in the order the candidate pairs are given."""

    def __init__(self, logits: list[float]) -> None:
        self._logits = logits

    def predict(self, pairs):
        return self._logits[: len(pairs)]


def _docs(n: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {i}", metadata={"title": f"T{i}", "source_url": "u"})
        for i in range(n)
    ]


def test_sigmoid_maps_logits_to_unit_range() -> None:
    assert _sigmoid(0.0) == 0.5
    assert _sigmoid(10.0) > 0.99
    assert _sigmoid(-10.0) < 0.01


def test_rerank_sorts_by_score_and_applies_top_n() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore(_docs(4)),
        reranker=FakeReranker([-5.0, 2.0, 0.0, 5.0]),  # T0 low … T3 highest
        top_k=4,
        rerank_top_n=2,
        min_relevance_score=0.5,
    )
    result = retriever.retrieve("q")

    assert result.has_answer
    # Sorted by relevance, capped at top_n=2: T3 (5.0) then T1 (2.0).
    assert [s.document.metadata["title"] for s in result.documents] == ["T3", "T1"]
    assert result.top_score == _sigmoid(5.0)


def test_refuses_when_all_candidates_below_threshold() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore(_docs(3)),
        reranker=FakeReranker([-5.0, -4.0, -3.0]),  # all sigmoids < 0.5
        top_k=3,
        rerank_top_n=3,
        min_relevance_score=0.5,
    )
    result = retriever.retrieve("q")

    assert not result.has_answer
    assert result.documents == []
    assert result.top_score == _sigmoid(-3.0)  # best (least-bad) candidate


def test_empty_vector_store_returns_no_answer() -> None:
    retriever = Retriever(
        vector_store=FakeVectorStore([]),
        reranker=FakeReranker([]),
        top_k=8,
    )
    result = retriever.retrieve("q")
    assert not result.has_answer
    assert result.top_score == 0.0
