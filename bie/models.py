"""
Core data models for BIE.

These mirror the schemas defined in the BIE PRD (Section 7), but kept
intentionally lean for the OSS / single-process edition.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Document(BaseModel):
    """A crawled and cleaned document, ready for indexing."""

    doc_id: str = Field(default_factory=_new_id)
    url: str
    title: str = ""
    text: str = ""
    publish_date: Optional[str] = None
    site: str = ""
    lang: str = "en"
    content_type: str = "article"
    trust_score: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)
    crawled_at: float = Field(default_factory=time.time)

    model_config = {"extra": "allow"}


class Chunk(BaseModel):
    """A paragraph/section-level chunk of a Document, used for retrieval."""

    chunk_id: str = Field(default_factory=_new_id)
    doc_id: str
    text: str
    start_offset: int = 0
    end_offset: int = 0
    embedding: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single ranked search result returned to the caller."""

    title: str
    url: str
    snippet: str
    source: str
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    trust_score: float = 0.5
    publish_date: Optional[str] = None
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.score:.3f}] {self.title} — {self.url}"


class SearchResponse(BaseModel):
    """Full response payload for /search and the Python API."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    took_ms: float = 0.0
    total_indexed_documents: int = 0
