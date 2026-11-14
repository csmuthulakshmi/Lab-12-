"""Qdrant Vector Store wrapper with idempotent UUIDs, HNSW indexing, and source stats."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from genai_platform.config import Settings, get_settings
from genai_platform.embeddings.embedder import get_embedder
from genai_platform.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

NAMESPACE_CHUNKS = uuid.UUID("a3b8e192-56c4-4d8e-90f1-1a2b3c4d5e6f")


@dataclass
class Hit:
    id: str
    score: float
    text: str
    source: str
    page: int
    doc_type: str
    chunk_id: str
    retrieval_score: float | None = None
    rerank_score: float | None = None

    @property
    def cite(self) -> str:
        name = Path(self.source).name
        if self.page > 0:
            return f"{name}, p.{self.page}"
        return name


class QdrantStore:
    """Wrapper around Qdrant vector database with idempotent point IDs and metadata indexing."""

    def __init__(self, url: str, collection: str, dim: int, recreate: bool = False):
        self.url = url
        self.collection = collection
        self.dim = dim

        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=url)

        self._init_collection(recreate=recreate)

    def _init_collection(self, recreate: bool = False) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if recreate and self.collection in collections:
            self.client.delete_collection(self.collection)
            collections.remove(self.collection)

        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=rest_models.VectorParams(
                    size=self.dim,
                    distance=rest_models.Distance.COSINE,
                ),
                hnsw_config=rest_models.HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                ),
            )
            # Create keyword payload index on source for fast filtered search
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """Upsert chunks and their vectors in batches of 128 using idempotent UUIDs."""
        if not chunks or not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and vectors ({len(vectors)})")

        batch_size = 128
        total = len(chunks)

        for i in range(0, total, batch_size):
            chunk_batch = chunks[i : i + batch_size]
            vector_batch = vectors[i : i + batch_size]

            points = []
            for chunk, vec in zip(chunk_batch, vector_batch):
                # Deterministic UUID5 based on chunk id
                point_id = str(uuid.uuid5(NAMESPACE_CHUNKS, chunk.id))
                payload = {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "doc_type": chunk.doc_type,
                    "chunk_id": chunk.id,
                    "index": chunk.index,
                    "strategy": chunk.strategy,
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=list(vec),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )

    def search(
        self,
        vector: Sequence[float],
        k: int = 4,
        source: str | None = None,
        ef: int = 128,
    ) -> list[Hit]:
        query_filter: rest_models.Filter | None = None
        if source:
            query_filter = rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="source",
                        match=rest_models.MatchValue(value=source),
                    )
                ]
            )

        search_params = rest_models.SearchParams(
            hnsw_ef=ef,
            exact=False,
        )

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection,
                query_vector=list(vector),
                limit=k,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
            )
        else:
            results = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                limit=k,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
            ).points

        hits: list[Hit] = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                Hit(
                    id=str(r.id),
                    score=float(r.score),
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    page=int(payload.get("page", 0)),
                    doc_type=payload.get("doc_type", "txt"),
                    chunk_id=payload.get("chunk_id", ""),
                    retrieval_score=float(r.score),
                )
            )
        return hits

    def timed_search(
        self,
        vector: Sequence[float],
        k: int = 4,
        source: str | None = None,
        ef: int = 128,
    ) -> tuple[list[Hit], float]:
        start = time.perf_counter()
        hits = self.search(vector, k=k, source=source, ef=ef)
        latency = time.perf_counter() - start
        return hits, latency

    def all_chunks(self) -> list[Chunk]:
        """Scroll all chunks from the collection."""
        chunks: list[Chunk] = []
        offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                p = r.payload or {}
                chunks.append(
                    Chunk(
                        text=p.get("text", ""),
                        source=p.get("source", ""),
                        doc_type=p.get("doc_type", "txt"),
                        page=int(p.get("page", 0)),
                        index=int(p.get("index", 0)),
                        strategy=p.get("strategy", "recursive"),
                        id=p.get("chunk_id", ""),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset

        return chunks

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def sources(self) -> list[str]:
        chunks = self.all_chunks()
        return sorted(list({c.source for c in chunks}))

    def delete_source(self, source: str) -> int:
        """Delete points matching the source filter and return count of deleted points."""
        before = self.count()
        query_filter = rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="source",
                    match=rest_models.MatchValue(value=source),
                )
            ]
        )
        self.client.delete(
            collection_name=self.collection,
            points_selector=rest_models.FilterSelector(filter=query_filter),
            wait=True,
        )
        after = self.count()
        return max(0, before - after)

    def source_stats(self) -> list[dict[str, Any]]:
        """Return aggregated stats per source: source, doc_type, chunks, pages."""
        chunks = self.all_chunks()
        stats_map: dict[str, dict[str, Any]] = {}

        for c in chunks:
            if c.source not in stats_map:
                stats_map[c.source] = {
                    "source": c.source,
                    "doc_type": c.doc_type,
                    "chunks": 0,
                    "pages": set(),
                }
            stats_map[c.source]["chunks"] += 1
            if c.page > 0:
                stats_map[c.source]["pages"].add(c.page)

        result = []
        for src, item in sorted(stats_map.items()):
            result.append({
                "source": item["source"],
                "doc_type": item["doc_type"],
                "chunks": item["chunks"],
                "pages": len(item["pages"]) if item["pages"] else 1,
            })
        return result


def get_store(recreate: bool = False) -> QdrantStore:
    settings = get_settings()
    embedder = get_embedder()
    # Name the collection with dim suffix to prevent cross-embedder corruption
    coll_name = f"{settings.QDRANT_COLLECTION}_{embedder.dim}"
    return QdrantStore(
        url=settings.QDRANT_URL,
        collection=coll_name,
        dim=embedder.dim,
        recreate=recreate,
    )
