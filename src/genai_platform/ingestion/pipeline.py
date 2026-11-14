"""Ingestion pipeline: load, chunk, embed, upsert to vector store, and persist to jsonl."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from genai_platform.config import ARTIFACTS
from genai_platform.embeddings.embedder import Embedder, get_embedder
from genai_platform.ingestion.chunking import chunk_documents, chunk_stats
from genai_platform.ingestion.loaders import load_path
from genai_platform.ingestion.schemas import Chunk, Document
from genai_platform.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_FILE = ARTIFACTS / "chunks.jsonl"


def build_chunks(
    docs: Sequence[Document],
    strategy: str = "recursive",
    chunk_size: int = 800,
    overlap: int = 100,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Helper to convert documents to chunks."""
    return chunk_documents(
        docs=docs,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        embedder=embedder,
    )


def save_chunks(chunks: Sequence[Chunk], path: Path | str | None = None) -> Path:
    """Save chunks to a JSON lines file."""
    dest = Path(path) if path else DEFAULT_CHUNKS_FILE
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk)) + "\n")
    return dest


def load_chunks(path: Path | str | None = None) -> list[Chunk]:
    """Load chunks from a JSON lines file."""
    src = Path(path) if path else DEFAULT_CHUNKS_FILE
    if not src.exists():
        return []
    chunks: list[Chunk] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                chunks.append(Chunk(**data))
    return chunks


def ingest_path(
    path: Path | str,
    store: QdrantStore,
    embedder: Embedder | None = None,
    strategy: str = "recursive",
    chunk_size: int = 800,
    overlap: int = 100,
) -> dict:
    """
    Ingest all documents from a path, chunk them, embed, upsert to vector store,
    and persist to artifacts/chunks.jsonl.
    """
    active_embedder = embedder or get_embedder()
    docs = load_path(path)
    if not docs:
        logger.warning(f"No documents extracted from {path}")
        return {"documents": 0, "chunks": 0, "sources": []}

    chunks = build_chunks(
        docs,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        embedder=active_embedder if strategy == "semantic" else None,
    )

    if chunks:
        texts = [c.text for c in chunks]
        vectors = active_embedder.embed_documents(texts)
        store.upsert(chunks, vectors)
        save_chunks(chunks)

    stats = chunk_stats(chunks)
    stats["documents"] = len(docs)
    return stats
