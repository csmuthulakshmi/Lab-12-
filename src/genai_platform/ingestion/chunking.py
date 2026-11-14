"""Chunking strategies: Recursive character split and Semantic split."""

from __future__ import annotations

import logging
import re
from typing import Sequence

import numpy as np

from genai_platform.embeddings.embedder import Embedder, get_embedder
from genai_platform.embeddings.math_ops import cosine_loop
from genai_platform.ingestion.schemas import Chunk, Document

logger = logging.getLogger(__name__)


def recursive_split(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text recursively by hierarchy: paragraph -> line -> sentence -> semicolon -> comma -> space -> char.
    Greedy merge pieces with overlap starting on word boundaries.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", r"(?<=[.!?])\s+", "; ", ", ", " ", ""]

    def _split_into_atoms(curr_text: str, sep_idx: int) -> list[str]:
        if len(curr_text) <= chunk_size or sep_idx >= len(separators):
            return [curr_text] if curr_text else []

        sep = separators[sep_idx]
        if sep == r"(?<=[.!?])\s+":
            pieces = [p for p in re.split(sep, curr_text) if p]
        elif sep == "":
            pieces = list(curr_text)
        else:
            pieces = curr_text.split(sep)

        result: list[str] = []
        for p in pieces:
            p_strip = p.strip()
            if not p_strip:
                continue
            if len(p_strip) > chunk_size and sep_idx + 1 < len(separators):
                result.extend(_split_into_atoms(p_strip, sep_idx + 1))
            else:
                result.append(p_strip)
        return result

    atoms = _split_into_atoms(text, 0)
    if not atoms:
        return []

    chunks: list[str] = []
    current_chunk = atoms[0]

    for next_atom in atoms[1:]:
        if len(current_chunk) + 1 + len(next_atom) <= chunk_size:
            current_chunk = f"{current_chunk} {next_atom}"
        else:
            chunks.append(current_chunk)
            # Find overlap on word boundary from end of current_chunk
            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                # Snap to next word boundary if possible
                word_break = overlap_text.find(" ")
                if word_break != -1 and word_break < len(overlap_text) - 1:
                    overlap_seed = overlap_text[word_break + 1 :]
                else:
                    overlap_seed = overlap_text
                current_chunk = f"{overlap_seed} {next_atom}".strip()
            else:
                current_chunk = next_atom

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def semantic_split(
    text: str,
    embedder: Embedder | None = None,
    chunk_size: int = 800,
    max_chars: int | None = None,
) -> list[str]:
    """
    Split text into sentences, embed them, compute neighbour cosine distance,
    and cut at distances above the 85th percentile.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if max_chars is None:
        max_chars = 2 * chunk_size

    # Sentence boundary split
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 2:
        return recursive_split(text, chunk_size=chunk_size, overlap=100)

    active_embedder = embedder or get_embedder()
    try:
        embeddings = active_embedder.embed_documents(sentences)
    except Exception as e:
        logger.warning(f"Semantic split embedding failed, falling back to recursive: {e}")
        return recursive_split(text, chunk_size=chunk_size, overlap=100)

    # Compute adjacent cosine distances (1 - cos)
    distances: list[float] = []
    for i in range(len(embeddings) - 1):
        cos_sim = cosine_loop(embeddings[i], embeddings[i + 1])
        dist = max(0.0, 1.0 - cos_sim)
        distances.append(dist)

    threshold = float(np.percentile(distances, 85)) if distances else 0.5

    chunks: list[str] = []
    curr_group: list[str] = [sentences[0]]

    for i in range(len(distances)):
        next_sent = sentences[i + 1]
        dist = distances[i]
        curr_chars = sum(len(s) + 1 for s in curr_group)

        # Cut condition: high semantic distance and min 2 sentences, or max_chars reached
        should_cut = (
            (dist >= threshold and len(curr_group) >= 2)
            or (curr_chars + len(next_sent) > max_chars)
        )

        if should_cut:
            chunks.append(" ".join(curr_group))
            curr_group = [next_sent]
        else:
            curr_group.append(next_sent)

    if curr_group:
        chunks.append(" ".join(curr_group))

    return chunks


def chunk_documents(
    docs: Sequence[Document],
    strategy: str = "recursive",
    chunk_size: int = 800,
    overlap: int = 100,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Process a list of documents into indexed Chunk instances."""
    chunks: list[Chunk] = []

    for doc in docs:
        if not doc.text.strip():
            continue

        if strategy == "semantic":
            text_pieces = semantic_split(
                doc.text,
                embedder=embedder,
                chunk_size=chunk_size,
            )
        else:
            text_pieces = recursive_split(
                doc.text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

        for idx, piece in enumerate(text_pieces):
            chunk = Chunk(
                text=piece,
                source=doc.source,
                doc_type=doc.doc_type,
                page=doc.page,
                index=idx,
                strategy=strategy,
            )
            chunks.append(chunk)

    return chunks


def chunk_stats(chunks: Sequence[Chunk]) -> dict:
    """Calculate summary statistics for a collection of Chunks."""
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
            "sources": [],
            "strategies": {},
        }

    lengths = [len(c.text) for c in chunks]
    sources = sorted(list({c.source for c in chunks}))
    strategies: dict[str, int] = {}
    for c in chunks:
        strategies[c.strategy] = strategies.get(c.strategy, 0) + 1

    return {
        "total_chunks": len(chunks),
        "avg_chars": float(np.mean(lengths)),
        "min_chars": int(np.min(lengths)),
        "max_chars": int(np.max(lengths)),
        "sources": sources,
        "strategies": strategies,
    }
