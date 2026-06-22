"""End-to-end RAG pipeline: retrieve -> guard -> grounded generation.

This ties P3 (retrieval + rerank + threshold) to P4 (LLM generation) and
implements the two-layer anti-hallucination guard:

- **Layer 1 (retrieval):** if nothing clears the relevance threshold, we return
  the refusal message *without* calling the LLM — fast, cheap, and certain.
- **Layer 2 (generation):** the LLM is instructed (see prompts.py) to emit the
  exact refusal sentence when the retrieved context does not actually answer the
  question. This catches "related-but-wrong" matches that slip past Layer 1.

The answer is returned with structured ``Source`` objects so the UI can render
citations ([1], [2], ...) back to the originating documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.llm import get_llm
from src.prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.retrieval.retriever import Retriever, ScoredDocument, get_retriever


@dataclass
class Source:
    """A citation, mapped to the [n] markers used in the answer."""

    index: int
    title: str
    source_url: str
    score: float
    snippet: str


@dataclass
class RagResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    has_answer: bool = True
    top_score: float = 0.0


def _format_context(documents: list[ScoredDocument]) -> str:
    """Number each chunk so the model can cite it as [n], tagged with its title."""
    blocks = []
    for index, scored in enumerate(documents, start=1):
        title = scored.document.metadata.get("title", "Source")
        blocks.append(f"[{index}] (Source: {title})\n{scored.document.page_content}")
    return "\n\n".join(blocks)


def _build_sources(documents: list[ScoredDocument]) -> list[Source]:
    sources = []
    for index, scored in enumerate(documents, start=1):
        meta = scored.document.metadata
        snippet = " ".join(scored.document.page_content.split())[:200]
        sources.append(
            Source(
                index=index,
                title=meta.get("title", "Source"),
                source_url=meta.get("source_url", ""),
                score=scored.score,
                snippet=snippet,
            )
        )
    return sources


def _is_refusal(answer: str) -> bool:
    """Detect when the LLM (Layer 2) chose to refuse rather than answer."""
    return "could not find a reliable answer" in answer.lower()


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.retriever = retriever or get_retriever()
        self.llm = llm or get_llm()

    def answer(self, query: str) -> RagResponse:
        result = self.retriever.retrieve(query)

        # Layer 1: nothing relevant retrieved -> refuse without spending an LLM call.
        if not result.has_answer:
            return RagResponse(
                answer=REFUSAL_MESSAGE, sources=[], has_answer=False,
                top_score=result.top_score,
            )

        context = _format_context(result.documents)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT_TEMPLATE.format(context=context, question=query)
            ),
        ]
        response = self.llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        answer = answer.strip()

        # Layer 2: the LLM judged the context insufficient and refused.
        if _is_refusal(answer):
            return RagResponse(
                answer=REFUSAL_MESSAGE, sources=[], has_answer=False,
                top_score=result.top_score,
            )

        return RagResponse(
            answer=answer,
            sources=_build_sources(result.documents),
            has_answer=True,
            top_score=result.top_score,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Shared pipeline singleton (loads retriever models + LLM client once)."""
    return RagPipeline()
