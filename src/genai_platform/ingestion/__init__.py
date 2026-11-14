"""Ingestion module exports."""

from genai_platform.ingestion.chunking import (
    chunk_documents,
    chunk_stats,
    recursive_split,
    semantic_split,
)
from genai_platform.ingestion.loaders import (
    clean_text,
    load_file,
    load_html,
    load_path,
    load_pdf,
    load_text,
)
from genai_platform.ingestion.schemas import Chunk, Document

__all__ = [
    "Chunk",
    "Document",
    "clean_text",
    "load_file",
    "load_html",
    "load_path",
    "load_pdf",
    "load_text",
    "chunk_documents",
    "chunk_stats",
    "recursive_split",
    "semantic_split",
]


def __getattr__(name: str):
    if name in {"build_chunks", "ingest_path", "load_chunks", "save_chunks"}:
        from genai_platform.ingestion import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
