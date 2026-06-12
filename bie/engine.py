"""
The ``BIE`` class — the main entry point of the BitSearch Intelligence
Engine Python API.

.. code-block:: python

    import bie

    engine = bie.BIE()
    engine.crawl(["https://example.com/news"])
    results = engine.search("what happened today")
    for r in results:
        print(r)
"""

from __future__ import annotations

import logging
import time

from bie.chunker import chunk_document
from bie.config import BIESettings
from bie.crawler import Crawler
from bie.index import HybridIndex
from bie.models import Document, SearchResponse, SearchResult

logger = logging.getLogger("bie")


class BIE:
    """The BitSearch Intelligence Engine — crawl, index, and search the web.

    Args:
        settings: Optional :class:`bie.config.BIESettings`. If omitted,
            settings are loaded from environment variables / ``.env``.

    Example::

        engine = bie.BIE()
        engine.crawl(["https://www.bbc.com/news"])
        for hit in engine.search("global markets"):
            print(hit.title, hit.url, hit.score)
    """

    def __init__(self, settings: BIESettings | None = None) -> None:
        self.settings = settings or BIESettings()
        self.index = HybridIndex(self.settings)
        self.crawler = Crawler(self.settings)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def crawl(self, urls: list[str], allowed_domains: list[str] | None = None) -> int:
        """Crawl ``urls`` (and linked pages, bounded by settings) and add
        the extracted documents to the index.

        Returns the number of documents added.
        """
        documents = self.crawler.crawl(urls, allowed_domains=allowed_domains)
        for doc in documents:
            self.add_document(doc)
        return len(documents)

    def add_document(self, doc: Document) -> None:
        """Add a single pre-fetched :class:`~bie.models.Document` to the index."""
        chunks = chunk_document(
            doc, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        if not chunks:
            return
        self.index.add_document(doc, chunks)

    def add_text(
        self,
        url: str,
        text: str,
        title: str = "",
        trust_score: float = 0.5,
        **metadata,
    ) -> None:
        """Add raw text directly (no crawling) — useful for indexing local
        documents, PDFs you've already extracted, API responses, etc."""
        doc = Document(
            url=url,
            title=title or url,
            text=text,
            trust_score=trust_score,
            metadata=metadata,
        )
        self.add_document(doc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Run a hybrid (BM25 + vector) search over the indexed documents."""
        return self.index.search(query, top_k=top_k)

    def search_full(self, query: str, top_k: int = 10) -> SearchResponse:
        """Like :meth:`search`, but returns a full :class:`SearchResponse`
        with timing and index-size metadata (matches the ``/search`` API)."""
        start = time.perf_counter()
        results = self.search(query, top_k=top_k)
        took_ms = (time.perf_counter() - start) * 1000
        return SearchResponse(
            query=query,
            results=results,
            took_ms=round(took_ms, 2),
            total_indexed_documents=len(self.index),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def search_web(self, query: str, urls: list[str], top_k: int = 10) -> list[SearchResult]:
        """One-shot: crawl ``urls``, then immediately search the freshly
        indexed content. Equivalent to ``engine.crawl(urls)`` followed by
        ``engine.search(query)``."""
        self.crawl(urls)
        return self.search(query, top_k=top_k)

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BIE documents={len(self.index)} "
            f"vector_search={'on' if self.index.vector_enabled else 'off'}>"
        )
