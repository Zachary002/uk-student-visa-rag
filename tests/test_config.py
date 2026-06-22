"""Smoke tests for the configuration layer.

These are intentionally tiny — they verify the config object loads, paths are
sane, and the documented defaults are in place. They also prove the test
harness (pytest + pythonpath) is wired up correctly from P0 onwards.
"""
from src.config import settings


def test_paths_are_absolute() -> None:
    assert settings.paths.root.is_absolute()
    assert settings.paths.data_processed.name == "processed"
    assert settings.paths.chroma_dir.parent == settings.paths.root / "data"


def test_default_models() -> None:
    assert settings.embedding.model_name == "BAAI/bge-small-en-v1.5"
    assert settings.retrieval.rerank_model.startswith("cross-encoder/")
    # Default LLM is Claude; remains overridable via the LLM_MODEL env var.
    assert settings.llm.model.startswith("claude")


def test_chunk_overlap_is_smaller_than_chunk_size() -> None:
    assert 0 <= settings.chunking.chunk_overlap < settings.chunking.chunk_size


def test_relevance_threshold_in_unit_range() -> None:
    # Cross-encoder scores are sigmoid-normalised to 0-1, so the guard
    # threshold must live in that range.
    assert 0.0 <= settings.retrieval.min_relevance_score <= 1.0
