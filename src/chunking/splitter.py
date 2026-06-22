"""Load the cleaned Markdown corpus and split it into retrieval chunks.

Why these chunking parameters? (see also the README)
- ``chunk_size = 800`` characters ≈ 150-200 English tokens. That is small enough
  to stay well inside bge-small-en-v1.5's 512-token window (no truncation) and
  to keep each chunk focused on a *single* idea, which sharpens its embedding
  and improves retrieval precision. Larger chunks blur multiple topics together
  and hurt ranking; much smaller chunks fragment answers across many results.
- ``chunk_overlap = 120`` characters (~15%) carries a little context across
  boundaries, so a fact that straddles a split is still fully present in at
  least one chunk.
- Separators are markdown-aware (headings → paragraphs → sentences → words), so
  splits fall on natural boundaries instead of mid-sentence wherever possible.
"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

# Matches a leading YAML front-matter block: ---\n ... \n--- \n body
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

# Try these separators in order; earlier ones keep larger semantic units intact.
_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]

# Chunks shorter than this are almost always isolated headings/fragments with
# no answer value, so we drop them to keep the index clean.
MIN_CHUNK_CHARS = 80


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into (metadata_dict, body). Robust to a missing block."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    meta_block, body = match.group(1), match.group(2)
    metadata: dict[str, str] = {}
    for line in meta_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip().strip('"')
    return metadata, body.strip()


def load_documents(directory: Path | None = None) -> list[Document]:
    """Read every processed Markdown file into a LangChain Document.

    The front-matter (title, source_url, retrieved_at) becomes document
    metadata so it can travel with each chunk and power citations later.
    """
    directory = directory or settings.paths.data_processed
    documents: list[Document] = []
    for path in sorted(Path(directory).glob("*.md")):
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "source_file": path.name,
                    "source_url": metadata.get("source_url", ""),
                    "title": metadata.get("title", path.stem),
                    "retrieved_at": metadata.get("retrieved_at", ""),
                },
            )
        )
    return documents


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
        separators=_SEPARATORS,
        add_start_index=True,  # records each chunk's char offset in its source
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks, dropping tiny fragments and
    tagging each survivor with a sequential chunk_id."""
    raw_chunks = make_splitter().split_documents(documents)
    chunks = [c for c in raw_chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
