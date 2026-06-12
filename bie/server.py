"""
BIE REST API server — implements the core endpoints from PRD Section 6.

Run with::

    bie serve --port 8000

or::

    uvicorn bie.server:app --reload
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from bie import __version__
from bie.config import BIESettings
from bie.engine import BIE
from bie.models import SearchResponse

logger = logging.getLogger("bie.server")

settings = BIESettings()
engine = BIE(settings)

app = FastAPI(
    title="BitSearch Intelligence Engine (BIE)",
    description=(
        "AI-native, real-time web search & extraction API. "
        "Built on the Bitscrape crawling framework. "
        "https://github.com/Sudharsansm/BIE"
    ),
    version=__version__,
)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def require_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(10, ge=1, le=100)


class SearchLiveRequest(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(10, ge=1, le=100)
    discovery_results: int = Field(8, ge=1, le=20)
    deep: bool = Field(
        True,
        description="If true (default), crawl discovered URLs with Bitscrape and "
        "rank via BIE's hybrid index. If false, return raw discovery order "
        "without snippets.",
    )


class CrawlRequest(BaseModel):
    urls: list[str] = Field(..., description="Seed URLs to crawl and index")
    allowed_domains: list[str] | None = Field(
        default=None, description="Restrict link-following to these domains"
    )


class CrawlResponse(BaseModel):
    documents_indexed: int
    total_indexed_documents: int


class IndexTextRequest(BaseModel):
    url: str
    title: str = ""
    text: str
    trust_score: float = 0.5


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    indexed_documents: int
    vector_search: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(
        indexed_documents=len(engine),
        vector_search=engine.index.vector_enabled,
    )


@app.post("/search", response_model=SearchResponse, tags=["search"], dependencies=[Depends(require_api_key)])
async def search(req: SearchRequest) -> SearchResponse:
    """Hybrid (BM25 + vector) search over the in-memory index."""
    return engine.search_full(req.query, top_k=req.top_k)


@app.post("/search/live", tags=["search"], dependencies=[Depends(require_api_key)])
async def search_live(req: SearchLiveRequest) -> dict:
    """Search the live internet for `query` — no seed URLs, no API key,
    no subscription required.

    Discovers relevant URLs via free public search endpoints (DuckDuckGo,
    with a Bing fallback), crawls them with Bitscrape, and ranks the
    extracted content via BIE's hybrid index.
    """
    import bie

    results = bie.websearch(
        req.query,
        top_k=req.top_k,
        discovery_results=req.discovery_results,
        deep=req.deep,
        use_embeddings=False,
    )

    return {"query": req.query, "results": [r.model_dump() for r in results]}


@app.post("/crawl/url", response_model=CrawlResponse, tags=["ingest"], dependencies=[Depends(require_api_key)])
async def crawl(req: CrawlRequest) -> CrawlResponse:
    """Trigger an on-demand crawl of one or more URLs and add to the index."""
    n = await _async_crawl(req.urls, req.allowed_domains)
    return CrawlResponse(documents_indexed=n, total_indexed_documents=len(engine))


@app.post("/indices/update", tags=["ingest"], dependencies=[Depends(require_api_key)])
async def index_text(req: IndexTextRequest) -> dict:
    """Push a pre-extracted document directly into the index."""
    engine.add_text(req.url, req.text, title=req.title, trust_score=req.trust_score)
    return {"status": "indexed", "total_indexed_documents": len(engine)}


@app.get("/metrics", tags=["meta"], dependencies=[Depends(require_api_key)])
async def metrics() -> dict:
    return {
        "indexed_documents": len(engine),
        "indexed_chunks": len(engine.index.chunks),
        "vector_search_enabled": engine.index.vector_enabled,
        "settings": settings.model_dump(exclude={"api_key"}),
    }


async def _async_crawl(urls: list[str], allowed_domains: list[str] | None) -> int:
    documents = await engine.crawler.acrawl(urls, allowed_domains=allowed_domains)
    for doc in documents:
        engine.add_document(doc)
    return len(documents)
