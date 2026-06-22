"""P2 entrypoint — build the Chroma vector index from the processed corpus.

Steps: load Markdown docs -> split into chunks -> embed with bge-small-en-v1.5
-> persist to Chroma. Ends with a sanity-check similarity query.

Usage:
    python scripts/2_build_index.py
"""
from __future__ import annotations

import logging
import os
import statistics
import sys
from pathlib import Path

# Quiet third-party telemetry/tokenizer noise before importing the ML stack.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking.splitter import load_documents, split_documents  # noqa: E402
from src.config import settings                                    # noqa: E402
from src.indexing.embedder import get_embeddings                   # noqa: E402
from src.indexing.vector_store import build_vector_store           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3",
               "sentence_transformers", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("build_index")

SANITY_QUERY = "How many hours can I work per week on a Student visa?"


def main() -> int:
    settings.ensure_dirs()

    documents = load_documents()
    if not documents:
        logger.error("No documents found in %s. Run scripts/1_scrape.py first.",
                     settings.paths.data_processed)
        return 1
    logger.info("Loaded %d documents.", len(documents))

    chunks = split_documents(documents)
    lengths = [len(c.page_content) for c in chunks]
    logger.info(
        "Split into %d chunks (chars: min=%d, median=%d, max=%d).",
        len(chunks), min(lengths), int(statistics.median(lengths)), max(lengths),
    )

    logger.info("Loading embedding model '%s' (first run downloads it)...",
                settings.embedding.model_name)
    embeddings = get_embeddings()

    logger.info("Embedding %d chunks and persisting to Chroma...", len(chunks))
    store = build_vector_store(chunks, embeddings)
    logger.info("Index persisted to %s", settings.paths.chroma_dir)

    # --- Sanity check: the index should retrieve relevant chunks ---
    logger.info("Sanity query: %r", SANITY_QUERY)
    hits = store.similarity_search_with_relevance_scores(SANITY_QUERY, k=3)
    for rank, (doc, score) in enumerate(hits, start=1):
        preview = " ".join(doc.page_content.split())[:140]
        logger.info("  #%d [%.3f] %s -> %s", rank, score,
                    doc.metadata.get("title", "?"), preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
