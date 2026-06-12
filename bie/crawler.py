"""
BIE Crawler — thin orchestration layer over Bitscrape's Engine.

Implements PRD Module 1 (Crawler) for the OSS edition: runs the
:class:`bie.spiders.generic.BIESpider` against one or more seed URLs,
collects extracted pages in-memory as :class:`bie.models.Document`
objects, ready for chunking + indexing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import bitscrape
from bitscrape.pipeline.pipelines import BasePipeline

from bie.config import BIESettings
from bie.models import Document
from bie.spiders.generic import BIESpider

logger = logging.getLogger("bie.crawler")


class _CollectorPipeline(BasePipeline):
    """Collects every scraped item into an in-memory list."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    async def process_item(self, item: Any, spider: Any) -> Any:
        self.items.append(item)
        return item


class Crawler:
    """Crawls a list of seed URLs using Bitscrape and returns Documents."""

    def __init__(self, settings: BIESettings | None = None) -> None:
        self.settings = settings or BIESettings()

    def crawl(self, urls: list[str], allowed_domains: list[str] | None = None) -> list[Document]:
        """Synchronous convenience wrapper around :meth:`acrawl`."""
        return asyncio.run(self.acrawl(urls, allowed_domains))

    async def acrawl(
        self, urls: list[str], allowed_domains: list[str] | None = None
    ) -> list[Document]:
        if not urls:
            return []

        if allowed_domains is None:
            allowed_domains = sorted({urlparse(u).netloc for u in urls if urlparse(u).netloc})

        bs_settings = bitscrape.Settings(
            concurrent_requests=self.settings.concurrent_requests,
            download_delay=self.settings.download_delay,
            user_agent=self.settings.user_agent,
            robotstxt_obey=self.settings.robotstxt_obey,
            download_timeout=self.settings.request_timeout,
            max_depth=self.settings.max_depth,
        )

        spider = BIESpider(settings=bs_settings)
        spider.start_urls = list(urls)
        spider.allowed_domains = allowed_domains
        spider.max_pages = self.settings.max_pages
        spider.max_depth = self.settings.max_depth

        collector = _CollectorPipeline()

        middlewares = [
            bitscrape.UserAgentMiddleware(),
            bitscrape.CookieMiddleware(),
        ]
        if bs_settings.robotstxt_obey:
            middlewares.insert(0, bitscrape.RobotsMiddleware())

        engine = bitscrape.Engine(
            spider=spider,
            settings=bs_settings,
            pipelines=[collector],
            middlewares=middlewares,
        )

        stats = await engine.run()
        logger.info(
            "Crawled %d page(s) from %d seed URL(s) — %d failed",
            stats.items_scraped,
            len(urls),
            stats.requests_failed,
        )

        documents: list[Document] = []
        for item in collector.items:
            if not item.get("text"):
                continue
            documents.append(
                Document(
                    url=item["url"],
                    title=item.get("title", item["url"]),
                    text=item["text"],
                    site=urlparse(item["url"]).netloc,
                    metadata={"depth": item.get("depth", 0)},
                )
            )
        return documents
