"""Unit tests for the RAG pipeline's two-layer hallucination guard.

A fake retriever and a fake LLM let us assert the control flow without an API
key, network, or models:
- Layer 1: no relevant context -> refuse, and the LLM is never called.
- Happy path: context -> grounded answer with numbered sources.
- Layer 2: LLM emits the refusal sentence -> sources are cleared.
"""
from types import SimpleNamespace

from langchain_core.documents import Document

from src.generation.rag_pipeline import RagPipeline
from src.prompts import REFUSAL_MESSAGE
from src.retrieval.retriever import RetrievalResult, ScoredDocument


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self._result = result

    def retrieve(self, query: str) -> RetrievalResult:
        return self._result


class RecordingLLM:
    """Fake chat model that records whether it was invoked."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.content)


def _result(*, has_answer: bool, n: int = 2, top: float = 0.9) -> RetrievalResult:
    docs = [
        ScoredDocument(
            Document(page_content=f"c{i}", metadata={"title": f"T{i}", "source_url": "u"}),
            0.9,
        )
        for i in range(n)
    ] if has_answer else []
    return RetrievalResult(query="q", documents=docs, top_score=top)


def test_layer1_refusal_skips_the_llm() -> None:
    llm = RecordingLLM("this should never be returned")
    pipe = RagPipeline(retriever=FakeRetriever(_result(has_answer=False, top=0.2)), llm=llm)

    response = pipe.answer("best pizza in London?")

    assert not response.has_answer
    assert response.answer == REFUSAL_MESSAGE
    assert response.sources == []
    assert llm.calls == 0  # cost guard: no API call when nothing is relevant


def test_grounded_answer_maps_numbered_sources() -> None:
    llm = RecordingLLM("You can work up to 20 hours per week [1].")
    pipe = RagPipeline(retriever=FakeRetriever(_result(has_answer=True, n=2)), llm=llm)

    response = pipe.answer("how many hours can I work?")

    assert response.has_answer
    assert response.answer == "You can work up to 20 hours per week [1]."
    assert llm.calls == 1
    assert [s.index for s in response.sources] == [1, 2]
    assert response.sources[0].title == "T0"


def test_layer2_llm_refusal_clears_sources() -> None:
    # Retrieval passed the threshold, but the LLM judged the context irrelevant.
    llm = RecordingLLM(REFUSAL_MESSAGE)
    pipe = RagPipeline(retriever=FakeRetriever(_result(has_answer=True, n=3, top=0.9)), llm=llm)

    response = pipe.answer("how do I apply for a driving licence?")

    assert llm.calls == 1
    assert not response.has_answer
    assert response.sources == []
