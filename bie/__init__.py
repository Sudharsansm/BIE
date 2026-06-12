"""
BIE — BitSearch Intelligence Engine
=====================================

A real-time web search engine for AI applications — built on top of
**Bitscrape** (https://pypi.org/project/bitscrape/). Give it a query,
get ranked, cited results from the live internet. No API keys, no
subscriptions, no third-party search services.

Quick start
-----------

.. code-block:: python

    import bie

    # Search the live internet — no URLs, no API key, no subscription
    results = bie.websearch("who won the latest F1 race")
    for r in results:
        print(r.title, r.url)
        print(r.snippet)

You can also search specific sites you already know about::

    results = bie.search("latest semiconductor export rules 2026", urls=[
        "https://www.reuters.com/technology/",
        "https://www.bloomberg.com/technology",
    ])

Or build a persistent index you can query repeatedly::

    engine = bie.BIE()
    engine.crawl(["https://example.com"])
    hits = engine.search("example query", top_k=5)

Run as a server::

    bie serve --port 8000

Run as an MCP tool (for Claude Desktop, Claude Code, etc.)::

    bie mcp
"""

from __future__ import annotations

from bie.config import BIESettings
from bie.engine import BIE
from bie.models import Document, SearchResult
from bie.quicksearch import search, websearch

__version__ = "0.4.0"

__all__ = [
    "BIE",
    "BIESettings",
    "Document",
    "SearchResult",
    "search",
    "websearch",
    "__version__",
]
