"""
BIE as a Model Context Protocol (MCP) server.

Exposes two tools to any MCP-compatible client (Claude Desktop, Claude
Code, etc.):

  - ``bie_search``       — crawl seed URLs (or the existing in-memory
                            index) and return ranked, cited results.
  - ``bie_crawl``        — crawl & index URLs without searching.
  - ``bie_index_search`` — search the in-memory index built so far,
                            without crawling anything new.

Run with::

    bie mcp

Requires the ``mcp`` package: ``pip install 'bits-bie[mcp]'``
"""

from __future__ import annotations

import json
import logging

from bie.config import BIESettings
from bie.engine import BIE

logger = logging.getLogger("bie.mcp")

_engine: BIE | None = None


def _get_engine() -> BIE:
    global _engine
    if _engine is None:
        _engine = BIE(BIESettings())
    return _engine


def run_mcp_server() -> None:
    """Start the BIE MCP server over stdio."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'mcp' package is required for MCP support. "
            "Install with: pip install 'bits-bie[mcp]'"
        ) from exc

    mcp = FastMCP("bie")

    @mcp.tool()
    def bie_search(query: str, urls: list[str], top_k: int = 5, max_pages: int = 10) -> str:
        """Search the live web for `query` by crawling `urls` (and a few
        linked pages) with the Bitscrape-powered crawler, then return
        ranked, citation-ready results (title, url, snippet, score).

        Args:
            query: The natural-language search query.
            urls: One or more seed URLs to crawl (e.g. news sites, docs).
            top_k: Max number of results to return.
            max_pages: Max pages to crawl per seed URL.
        """
        settings = BIESettings(max_pages=max_pages)
        engine = BIE(settings)
        engine.crawl(urls)
        response = engine.search_full(query, top_k=top_k)
        return response.model_dump_json(indent=2)

    @mcp.tool()
    def bie_crawl(urls: list[str], max_pages: int = 20) -> str:
        """Crawl and index `urls` into BIE's persistent in-memory index for
        this session, without searching yet. Use `bie_index_search`
        afterwards to query the indexed content repeatedly.

        Args:
            urls: Seed URLs to crawl.
            max_pages: Max pages to crawl per seed URL.
        """
        engine = _get_engine()
        engine.settings.max_pages = max_pages
        n = engine.crawl(urls)
        return json.dumps(
            {"documents_indexed": n, "total_indexed_documents": len(engine)}
        )

    @mcp.tool()
    def bie_index_search(query: str, top_k: int = 5) -> str:
        """Search documents already crawled into BIE's session index via
        `bie_crawl`. Returns ranked, citation-ready results.

        Args:
            query: The natural-language search query.
            top_k: Max number of results to return.
        """
        engine = _get_engine()
        response = engine.search_full(query, top_k=top_k)
        return response.model_dump_json(indent=2)

    logger.info("Starting BIE MCP server (stdio)...")
    mcp.run()
