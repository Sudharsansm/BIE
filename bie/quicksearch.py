"""
``bie.search()`` and ``bie.websearch()`` — the simplest entry points.

.. code-block:: python

    import bie

    # Search specific sites you already know about
    results = bie.search("AI regulation news 2026", urls=["https://example.com/news"])

    # Search the live internet — no URLs, no API key, no subscription
    results = bie.websearch("latest AI regulation news 2026")
    for r in results:
        print(r.title, r.url)
        print(r.snippet)
"""

from __future__ import annotations

import re

from bie.config import BIESettings
from bie.discovery import discover_urls
from bie.engine import BIE
from bie.models import SearchResult


def search(
    query: str,
    urls: list[str],
    top_k: int = 10,
    **settings_kwargs,
) -> list[SearchResult]:
    """Crawl ``urls`` and return the top-``top_k`` results for ``query``.

    This spins up a fresh, in-memory :class:`bie.BIE` instance — convenient
    for scripts and one-off queries. For repeated queries against the same
    sources, create a :class:`bie.BIE` instance and reuse it instead.

    Args:
        query: The search query.
        urls: Seed URLs to crawl.
        top_k: Number of results to return.
        **settings_kwargs: Forwarded to :class:`bie.config.BIESettings`
            (e.g. ``max_pages=10``, ``use_embeddings=False``).
    """
    engine = BIE(BIESettings(**settings_kwargs))
    return engine.search_web(query, urls, top_k=top_k)


def websearch(
    query: str,
    top_k: int = 10,
    discovery_results: int = 8,
    deep: bool = True,
    **settings_kwargs,
) -> list[SearchResult]:
    """Search the **live internet** for ``query`` — no seed URLs, no API
    key, no subscription required.

    This is BIE's primary entry point — a "type a question, get a
    real-time answer" experience:

      1. **Discovery** — :func:`bie.discovery.discover_urls` finds
         candidate URLs for ``query`` using free, public, no-key search
         endpoints (DuckDuckGo, with a Bing fallback).
      2. **Crawl + rank** — the discovered URLs are crawled with
         Bitscrape, their text extracted and chunked, and ranked against
         ``query`` using BIE's hybrid BM25 + vector index.

    Args:
        query: The natural-language search query.
        top_k: Number of results to return.
        discovery_results: How many candidate URLs to discover before
            crawling. Higher values improve result quality at the cost of
            crawl time.
        deep: If True (default), crawl discovered URLs with Bitscrape and
            rank the extracted page text via BIE's hybrid index — gives
            full-page snippets and proper relevance scoring. If False,
            skip crawling and return the raw discovery order with empty
            snippets (fast, but low quality — mainly useful for debugging
            discovery itself).
        **settings_kwargs: Forwarded to :class:`bie.config.BIESettings`
            (e.g. ``max_pages=1``, ``use_embeddings=False``,
            ``request_timeout=10``).

    Example::

        import bie
        results = bie.websearch("who won the latest F1 race")
        for r in results:
            print(r.title, "-", r.url)
            print(r.snippet)
    """
    urls = discover_urls(query, max_results=discovery_results)

    if not urls:
        return []

    if not deep:
        return [
            SearchResult(
                title=url,
                url=url,
                snippet="",
                source=_domain(url),
                score=1.0 / (i + 1),
            )
            for i, url in enumerate(urls[:top_k])
        ]

    settings_kwargs.setdefault("max_pages", 1)
    settings_kwargs.setdefault("max_depth", 0)
    engine = BIE(BIESettings(**settings_kwargs))
    results = engine.search_web(query, urls, top_k=top_k)

    if results:
        return results

    # Fallback: crawling produced nothing usable (e.g. all JS-rendered
    # pages, or every page failed/blocked) — return discovered URLs
    # without snippets rather than an empty list, so the caller still
    # gets *something* to work with.
    return [
        SearchResult(
            title=url,
            url=url,
            snippet="",
            source=_domain(url),
            score=1.0 / (i + 1),
        )
        for i, url in enumerate(urls[:top_k])
    ]


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1) if m else url
