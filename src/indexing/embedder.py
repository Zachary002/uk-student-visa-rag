"""Embedding model factory.

Uses the open-source ``BAAI/bge-small-en-v1.5`` model via sentence-transformers,
which runs locally on CPU (no API cost). BGE models are trained *asymmetrically*:
documents are embedded as-is, but a short instruction should be prepended to
queries for best retrieval. We capture that with a thin subclass so the rest of
the code just calls ``embed_query`` / ``embed_documents`` normally.
"""
from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


class BGEEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings that prepends the BGE retrieval instruction to queries."""

    query_instruction: str = ""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{self.query_instruction}{text}")


def get_embeddings() -> BGEEmbeddings:
    """Construct the configured embedding model (downloaded & cached on first use)."""
    cfg = settings.embedding
    return BGEEmbeddings(
        model_name=cfg.model_name,
        # Normalising enables cosine similarity via a simple dot product.
        encode_kwargs={"normalize_embeddings": cfg.normalize_embeddings},
        query_instruction=cfg.query_instruction,
    )
