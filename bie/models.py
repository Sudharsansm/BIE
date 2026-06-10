"""
BIE data models — shared across all modules.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


# ── Raw document after Bitscrape crawl ──────────────────────────────────────

class DocumentRecord(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    title: str = ""
    publish_date: str = ""
    authors: list[str] = []
    text: str = ""
    chunk_ids: list[str] = []
    metadata: dict[str, Any] = {}
    crawled_at: float = Field(default_factory=time.time)

    @property
    def domain(self) -> str:
        from urllib.parse import urlparse
        return urlparse(self.url).netloc


# ── Chunk (paragraph / section granularity) ──────────────────────────────────

class ChunkRecord(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    start_offset: int = 0
    end_offset: int = 0
    tokens: int = 0
    embeddings: list[float] = []
    metadata: dict[str, Any] = {}
    trust_score: float = 0.5


# ── Search request / response ─────────────────────────────────────────────────

class SearchFilters(BaseModel):
    lang: str | None = None
    domain: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    min_trust: float | None = None
    content_type: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    use_reranker: bool = True
    stream: bool = False


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    url: str
    snippet: str
    source: str
    publish_date: str = ""
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    trust_score: float = 0.5
    contradiction_flags: list[str] = []


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_found: int
    latency_ms: float
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── Agent / LLM response ──────────────────────────────────────────────────────

class Citation(BaseModel):
    index: int
    url: str
    title: str
    snippet: str
    trust_score: float = 0.5


class AgentResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    contradiction_flags: list[str] = []
    latency_ms: float
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""


# ── Crawl request ─────────────────────────────────────────────────────────────

class CrawlRequest(BaseModel):
    url: str
    priority: int = Field(5, ge=1, le=10)
    force_recrawl: bool = False


class CrawlResponse(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    status: str = "queued"
    message: str = ""


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    index_size: int = 0
    uptime_seconds: float = 0.0
