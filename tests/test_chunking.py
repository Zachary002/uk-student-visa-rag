"""Unit tests for document loading and chunking.

Fast and dependency-light: no embedding model, no network. They exercise the
front-matter parser and the splitter's two guarantees — tiny fragments are
dropped, and survivors get sequential chunk_ids while keeping their metadata.
"""
from langchain_core.documents import Document

from src.chunking.splitter import (
    MIN_CHUNK_CHARS,
    load_documents,
    parse_frontmatter,
    split_documents,
)
from src.config import settings


def test_parse_frontmatter_valid() -> None:
    text = (
        '---\n'
        'title: "Student visa: Money"\n'
        'source_url: "https://www.gov.uk/student-visa/money"\n'
        'retrieved_at: 2026-06-22\n'
        '---\n\n'
        '# Student visa: Money\n\nBody text.'
    )
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "Student visa: Money"
    assert meta["source_url"] == "https://www.gov.uk/student-visa/money"
    assert body.startswith("# Student visa: Money")


def test_parse_frontmatter_missing_block() -> None:
    meta, body = parse_frontmatter("Just some text, no front matter.")
    assert meta == {}
    assert body == "Just some text, no front matter."


def test_split_drops_documents_below_minimum() -> None:
    tiny = Document(page_content="# Hi", metadata={"title": "T", "source_url": "u"})
    assert split_documents([tiny]) == []


def test_split_assigns_sequential_ids_and_keeps_metadata() -> None:
    long_text = "## Section\n\n" + ("word " * 600)  # well over one chunk_size
    doc = Document(page_content=long_text, metadata={"title": "Guide", "source_url": "u"})

    chunks = split_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content.strip()) >= MIN_CHUNK_CHARS for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["title"] == "Guide" for c in chunks)
    # No chunk should wildly exceed the configured size.
    assert max(len(c.page_content) for c in chunks) <= settings.chunking.chunk_size + 50


def test_load_documents_carry_citation_metadata() -> None:
    """Integration check against the built corpus (no models involved)."""
    docs = load_documents()
    assert len(docs) >= 15  # we curated ~19
    for doc in docs:
        assert doc.metadata["source_url"].startswith("http")
        assert doc.metadata["title"]
        assert doc.page_content.strip()
