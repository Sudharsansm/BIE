"""
M01 — BIE Crawler
=================
Powered by the Bitscrape framework.  Handles URL frontier management,
robots.txt compliance, multi-format extraction, and chunk production.

Usage::

    from bie.crawler import BIECrawler

    crawler = BIECrawler()
    docs = await crawler.crawl_urls(["https://example.com/article"])
    # or trigger a single on-demand crawl
    doc = await crawler.crawl_single("https://example.com/article")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from typing import AsyncIterator
from urllib.parse import urlparse

import bitscrape
from bitscrape import Item, Request, Response, Settings, Spider

from bie.config import BIESettings, settings
from bie.models import ChunkRecord, DocumentRecord

logger = logging.getLogger(__name__)


# ── Bitscrape Item ─────────────────────────────────────────────────────────────

class PageItem(Item):
    url: str = ""
    title: str = ""
    text: str = ""
    publish_date: str = ""
    authors: list[str] = []
    lang: str = "en"
    content_type: str = "article"


# ── BIE Web Spider (Bitscrape Spider subclass) ────────────────────────────────

class BIESpider(Spider):
    """
    General-purpose BIE spider.  Extracts title + body text from any URL.
    Subclass for domain-specific extraction (news, docs, APIs, etc.).
    """

    name = "bie_spider"

    def __init__(self, urls: list[str], cfg: BIESettings = settings):
        super().__init__()
        self.start_urls = urls
        self._cfg = cfg

    async def parse(self, response: Response) -> AsyncIterator:  # type: ignore[override]
        url = response.url
        html = response.text or ""

        title = self._extract_title(response)
        text = self._extract_text(response, html)
        publish_date = self._extract_date(response, html)

        if not text.strip():
            logger.debug("Empty body — skipping %s", url)
            return

        yield PageItem(
            source_url=url,
            url=url,
            title=title,
            text=text,
            publish_date=publish_date,
            authors=[],
            lang=self._detect_lang(text),
            content_type="article",
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract_title(self, response: Response) -> str:
        title = response.css("title::text").get("")
        if not title:
            title = response.css("h1::text").get("")
        return title.strip()[:300]

    def _extract_text(self, response: Response, html: str) -> str:
        # Remove script / style / nav / footer noise
        for tag in ["script", "style", "nav", "footer", "header", "aside"]:
            for node in response.css(tag):
                pass  # Bitscrape CSS selectors — just select paragraphs instead

        # Select paragraphs, headings, list items
        parts: list[str] = []
        for sel in ["p", "h1", "h2", "h3", "h4", "li", "td", "th", "pre", "blockquote"]:
            parts.extend(response.css(f"{sel}::text").getall())

        text = " ".join(t.strip() for t in parts if t.strip())
        # Collapse whitespace
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()

    def _extract_date(self, response: Response, html: str) -> str:
        # Try <time> / meta tags
        date = response.css("time::attr(datetime)").get("")
        if not date:
            date = response.css('meta[property="article:published_time"]::attr(content)').get("")
        if not date:
            # ISO-8601 pattern fallback
            m = re.search(r"\d{4}-\d{2}-\d{2}", html)
            date = m.group(0) if m else ""
        return date[:20]

    def _detect_lang(self, text: str) -> str:
        # Lightweight: count ASCII ratio
        ascii_ratio = sum(1 for c in text[:500] if ord(c) < 128) / max(len(text[:500]), 1)
        return "en" if ascii_ratio > 0.8 else "xx"


# ── Chunker ────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits document text into paragraph-level chunks suitable for
    embedding and BM25 indexing.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: DocumentRecord) -> list[ChunkRecord]:
        text = doc.text.strip()
        if not text:
            return []

        # Split on double-newline / paragraph boundaries first
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        chunks: list[ChunkRecord] = []
        buffer: list[str] = []
        buf_tokens = 0
        offset = 0

        for para in paragraphs:
            para_tokens = len(para.split())
            if buf_tokens + para_tokens > self.chunk_size and buffer:
                chunk_text = " ".join(buffer)
                chunks.append(
                    ChunkRecord(
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_offset=offset,
                        end_offset=offset + len(chunk_text),
                        tokens=buf_tokens,
                        metadata={"section_title": doc.title},
                        trust_score=doc.metadata.get("trust_score", 0.5),
                    )
                )
                offset += len(chunk_text)
                # Keep last sentence for overlap
                overlap_text = " ".join(buffer[-2:]) if len(buffer) >= 2 else ""
                buffer = [overlap_text] if overlap_text else []
                buf_tokens = len(overlap_text.split())

            buffer.append(para)
            buf_tokens += para_tokens

        # Flush remainder
        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(
                ChunkRecord(
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_offset=offset,
                    end_offset=offset + len(chunk_text),
                    tokens=buf_tokens,
                    metadata={"section_title": doc.title},
                    trust_score=doc.metadata.get("trust_score", 0.5),
                )
            )

        # Wire chunk_ids back into the doc
        doc.chunk_ids = [c.chunk_id for c in chunks]
        return chunks


# ── Fingerprinter (MinHash-like dedup) ────────────────────────────────────────

class ContentFingerprinter:
    def __init__(self):
        self._seen: set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        fp = hashlib.md5(text[:500].encode(), usedforsecurity=False).hexdigest()
        if fp in self._seen:
            return True
        self._seen.add(fp)
        return False


# ── Main Crawler facade ────────────────────────────────────────────────────────

class BIECrawler:
    """
    High-level crawler facade used by the rest of BIE.

    Internally runs Bitscrape spiders and converts scraped pages
    into (DocumentRecord, [ChunkRecord]) tuples ready for indexing.
    """

    def __init__(self, cfg: BIESettings = settings):
        self._cfg = cfg
        self._chunker = TextChunker(chunk_size=cfg.chunk_size)
        self._dedup = ContentFingerprinter()
        self._crawl_queue: asyncio.Queue[str] = asyncio.Queue()
        self._bitscrape_settings = Settings(
            concurrent_requests=cfg.crawl_concurrent_requests,
            download_delay=cfg.crawl_download_delay,
            download_timeout=cfg.crawl_timeout,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def crawl_urls(
        self, urls: list[str]
    ) -> list[tuple[DocumentRecord, list[ChunkRecord]]]:
        """Crawl a batch of URLs and return (doc, chunks) pairs."""
        if not urls:
            return []

        collected: list[PageItem] = []
        spider = BIESpider(urls=urls, cfg=self._cfg)

        # Collect via bitscrape pipeline
        collected = await self._run_spider(spider)
        results = []
        for item in collected:
            doc, chunks = self._item_to_doc(item)
            if not self._dedup.is_duplicate(doc.text):
                results.append((doc, chunks))
        return results

    async def crawl_single(self, url: str) -> tuple[DocumentRecord, list[ChunkRecord]] | None:
        """On-demand single-URL crawl (POST /crawl/url handler)."""
        results = await self.crawl_urls([url])
        return results[0] if results else None

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_spider(self, spider: BIESpider) -> list[PageItem]:
        """
        Run a Bitscrape spider and collect all yielded PageItems.
        Uses bitscrape.Engine directly for programmatic access.
        """
        items: list[PageItem] = []

        class CollectPipeline(bitscrape.BasePipeline):
            async def process_item(self, item, spider_):
                if isinstance(item, PageItem):
                    items.append(item)
                return item

        engine = bitscrape.Engine(
            spider=spider,
            settings=self._bitscrape_settings,
            pipelines=[CollectPipeline()],
        )
        try:
            await engine.run()
        except Exception as exc:
            logger.warning("Crawler error: %s", exc)

        return items

    def _item_to_doc(
        self, item: PageItem
    ) -> tuple[DocumentRecord, list[ChunkRecord]]:
        domain = urlparse(item.url).netloc
        trust = _estimate_trust(domain)
        doc = DocumentRecord(
            url=item.url,
            title=item.title,
            text=item.text,
            publish_date=item.publish_date,
            authors=item.authors,
            metadata={
                "site": domain,
                "lang": item.lang,
                "content_type": item.content_type,
                "trust_score": trust,
            },
        )
        chunks = self._chunker.chunk(doc)
        for c in chunks:
            c.trust_score = trust
        return doc, chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

# Lightweight trust heuristic — replace with a full domain DB in production
_HIGH_TRUST_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "nature.com", "science.org",
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "gov", "edu",
    "github.com", "docs.python.org",
}

def _estimate_trust(domain: str) -> float:
    domain = domain.lower().lstrip("www.")
    for td in _HIGH_TRUST_DOMAINS:
        if domain == td or domain.endswith(f".{td}"):
            return 0.9
    # Generic heuristic: known TLDs
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95
    return 0.6  # default middle trust
