"""Chroma vector store helpers: build (persist) and load.

Chroma is a local, file-backed vector database — free, zero-ops, and easy to
ship inside a Hugging Face Space. The index is persisted under
``settings.paths.chroma_dir`` so it is built once and reused at query time.
"""
from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings


def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> Chroma:
    """Embed chunks and persist them to a fresh Chroma collection.

    The existing index directory is removed first so rebuilds are deterministic
    (no duplicate vectors accumulating across runs).
    """
    if settings.paths.chroma_dir.exists():
        shutil.rmtree(settings.paths.chroma_dir)
    settings.paths.chroma_dir.mkdir(parents=True, exist_ok=True)

    # Chroma persists automatically when persist_directory is set.
    # hnsw:space=cosine makes relevance scores true cosine similarities (0-1),
    # which is what the P3 confidence threshold is calibrated against.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.vector_store.collection_name,
        persist_directory=str(settings.paths.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_vector_store(embeddings: Embeddings) -> Chroma:
    """Open the previously built, persisted Chroma collection for querying."""
    return Chroma(
        collection_name=settings.vector_store.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.paths.chroma_dir),
    )


def index_exists() -> bool:
    """True if a persisted Chroma index appears to be present on disk."""
    chroma_dir = settings.paths.chroma_dir
    return chroma_dir.exists() and any(chroma_dir.iterdir())


def ensure_index() -> None:
    """Build the index from data/processed if it isn't there yet.

    This makes the app self-bootstrapping on a fresh deploy (e.g. a new Hugging
    Face Space): the committed Markdown corpus is embedded into Chroma on first
    launch, so no scraping or manual build step is needed on the server.
    """
    if index_exists():
        return
    # Lazy imports: skip the heavy import chain when the index already exists.
    from src.chunking.splitter import load_documents, split_documents
    from src.indexing.embedder import get_embeddings

    documents = load_documents()
    if not documents:
        raise RuntimeError(
            "No documents in data/processed/. Commit the corpus or run "
            "scripts/1_scrape.py before building the index."
        )
    build_vector_store(split_documents(documents), get_embeddings())
