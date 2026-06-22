"""Central configuration for the UK Student Visa RAG system.

Every tunable parameter — filesystem paths, model names, chunking sizes,
retrieval thresholds and LLM settings — lives here, so the rest of the
codebase imports a single ``settings`` object instead of hard-coding values.

Secrets (API keys) are read from the environment via a local ``.env`` file in
development and from real environment variables in production (e.g. Hugging
Face Spaces "Secrets"). They are never committed to version control.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present. This is a no-op in
# production where the variables are already set in the real environment.
load_dotenv()

# Project root = the directory that contains this ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All paths are absolute and derived from PROJECT_ROOT."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"           # scraped HTML
    data_processed: Path = PROJECT_ROOT / "data" / "processed"  # clean Markdown docs
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"      # persisted vector store
    eval_dir: Path = PROJECT_ROOT / "eval"


@dataclass(frozen=True)
class ChunkingConfig:
    """Text-splitting parameters.

    chunk_size / chunk_overlap are measured in characters (not tokens) because
    we use ``RecursiveCharacterTextSplitter``. The rationale for these specific
    numbers is documented in the README ("Why these chunking parameters?").
    """

    chunk_size: int = 800        # ~150-200 tokens: big enough for one coherent idea
    chunk_overlap: int = 120     # 15% overlap so answers spanning a boundary survive


@dataclass(frozen=True)
class EmbeddingConfig:
    """Open-source embedding model — runs locally / on free CPU, no API cost."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    # BGE models retrieve best when the *query* (not the documents) is prefixed
    # with a short instruction. Documents are embedded as-is.
    query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )
    normalize_embeddings: bool = True  # enables cosine similarity via dot product


@dataclass(frozen=True)
class VectorStoreConfig:
    """Chroma collection settings (shared by the indexer and the retriever)."""

    collection_name: str = "uk_student_visa"


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval + cross-encoder reranking + hallucination-guard thresholds."""

    top_k: int = 8               # candidate chunks pulled from the vector store
    rerank_top_n: int = 4        # chunks kept after cross-encoder reranking
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Cross-encoder relevance score (sigmoid-normalised to 0-1). If the BEST
    # candidate scores below this, we refuse to answer rather than hallucinate.
    # This is layer 1 of a two-layer guard (layer 2 = the LLM refusal contract
    # in prompts.py, which catches "related-but-wrong" matches that slip past).
    # 0.5 is a neutral midpoint default; it is tuned against the eval set in P7.
    min_relevance_score: float = 0.50


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings. Designed to be switchable (Claude by default)."""

    provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    temperature: float = 0.0     # deterministic, factual answers for a QA bot
    max_tokens: int = 1024
    # API keys (read from environment; may be None until the user provides them).
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


@dataclass(frozen=True)
class Settings:
    """Top-level settings object aggregating all config sections."""

    paths: Paths = field(default_factory=Paths)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not yet exist."""
        for path in (
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.chroma_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Singleton imported across the codebase: ``from src.config import settings``.
settings = Settings()
