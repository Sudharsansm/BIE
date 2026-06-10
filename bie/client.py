"""
BIE Python SDK — High-level client
====================================
Use this in your own AI applications to search BIE programmatically.

Example::

    import asyncio
    from bie import BIEClient

    async def main():
        async with BIEClient(base_url="http://localhost:8000", api_key="my-key") as client:
            # Simple hybrid search
            resp = await client.search("latest AI research 2026", top_k=5)
            for r in resp.results:
                print(r.rank, r.title, r.url, r.trust_score)

            # RAG: grounded LLM answer with citations
            answer = await client.agent_query("What happened in TSMC Q2 2026?")
            print(answer.answer)
            for c in answer.citations:
                print(f"  [{c.index}] {c.url}")

            # On-demand crawl
            await client.crawl_url("https://example.com/new-article")

    asyncio.run(main())

Sync wrapper::

    from bie.client import BIEClientSync

    client = BIEClientSync(base_url="http://localhost:8000", api_key="my-key")
    resp = client.search("semiconductor supply chain")
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx

from bie.models import (
    AgentResponse,
    CrawlRequest,
    CrawlResponse,
    HealthResponse,
    SearchFilters,
    SearchRequest,
    SearchResponse,
)


class BIEClient:
    """
    Async HTTP client for the BIE REST API.
    Use as an async context manager or call `.close()` manually.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "dev-key",
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    async def __aenter__(self) -> "BIEClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        use_reranker: bool = True,
    ) -> SearchResponse:
        """Hybrid BM25 + vector search. Returns ranked results."""
        payload = SearchRequest(
            query=query,
            top_k=top_k,
            filters=filters or SearchFilters(),
            use_reranker=use_reranker,
        )
        resp = await self._client.post("/search", content=payload.model_dump_json())
        resp.raise_for_status()
        return SearchResponse.model_validate(resp.json())

    async def search_stream(
        self, query: str, top_k: int = 10
    ) -> AsyncIterator[str]:
        """Stream search results as SSE events."""
        async with self._client.stream(
            "GET", "/search/stream", params={"query": query, "top_k": top_k}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

    # ── Agent / RAG ───────────────────────────────────────────────────────────

    async def agent_query(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> AgentResponse:
        """Full RAG: retrieve → build context → LLM answer with citations."""
        payload = SearchRequest(
            query=query,
            top_k=top_k,
            filters=filters or SearchFilters(),
        )
        resp = await self._client.post("/agent/query", content=payload.model_dump_json())
        resp.raise_for_status()
        return AgentResponse.model_validate(resp.json())

    async def agent_stream(
        self, query: str, top_k: int = 10
    ) -> AsyncIterator[str]:
        """Stream LLM tokens via SSE."""
        async with self._client.stream(
            "GET", "/agent/stream", params={"query": query, "top_k": top_k}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield data

    # ── Crawler ───────────────────────────────────────────────────────────────

    async def crawl_url(self, url: str, priority: int = 5) -> CrawlResponse:
        """Trigger an on-demand crawl + index of a single URL."""
        payload = CrawlRequest(url=url, priority=priority)
        resp = await self._client.post("/crawl/url", content=payload.model_dump_json())
        resp.raise_for_status()
        return CrawlResponse.model_validate(resp.json())

    async def crawl_batch(self, urls: list[str]) -> dict:
        """Batch crawl up to 50 URLs."""
        resp = await self._client.post("/crawl/batch", json=urls)
        resp.raise_for_status()
        return resp.json()

    # ── Feedback ──────────────────────────────────────────────────────────────

    async def feedback(self, url: str, positive: bool) -> None:
        """Send thumbs-up / thumbs-down to improve trust scoring."""
        resp = await self._client.post(
            "/feedback", params={"url": url, "positive": str(positive).lower()}
        )
        resp.raise_for_status()

    # ── Ops ───────────────────────────────────────────────────────────────────

    async def health(self) -> HealthResponse:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return HealthResponse.model_validate(resp.json())

    async def metrics(self) -> dict:
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        return resp.json()


# ── Sync wrapper ───────────────────────────────────────────────────────────────

class BIEClientSync:
    """
    Synchronous wrapper around BIEClient.
    Useful in scripts, Jupyter notebooks, or non-async frameworks.
    """

    def __init__(self, **kwargs):
        self._async_client = BIEClient(**kwargs)
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def search(self, query: str, **kwargs) -> SearchResponse:
        return self._run(self._async_client.search(query, **kwargs))

    def agent_query(self, query: str, **kwargs) -> AgentResponse:
        return self._run(self._async_client.agent_query(query, **kwargs))

    def crawl_url(self, url: str, **kwargs) -> CrawlResponse:
        return self._run(self._async_client.crawl_url(url, **kwargs))

    def health(self) -> HealthResponse:
        return self._run(self._async_client.health())

    def close(self):
        self._run(self._async_client.close())
        self._loop.close()
