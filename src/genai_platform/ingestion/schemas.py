"""Ingestion data schemas: Document and Chunk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    text: str
    source: str
    doc_type: str
    page: int = 1
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str
    doc_type: str
    page: int
    index: int
    strategy: str
    id: str = ""

    def __post_init__(self):
        if not self.id:
            # First 16 hex characters of sha1(source|page|index|text[:64])
            raw = f"{self.source}|{self.page}|{self.index}|{self.text[:64]}"
            self.id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def cite(self) -> str:
        name = Path(self.source).name
        if self.page > 0:
            return f"{name}, p.{self.page}"
        return name
