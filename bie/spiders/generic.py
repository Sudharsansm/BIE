"""
BIE's default spider — built directly on the Bitscrape Spider class.

This implements PRD Module 1 (Crawler) at OSS scope: generic article/page
extraction with link-following, using Bitscrape's selector API and
RobotsMiddleware for politeness.

Note: Bitscrape's Engine calls ``parse(self, parsed)`` where ``parsed`` is
already a :class:`bitscrape.ParsedResponse` (a CSS/XPath-selectable wrapper
around the raw Response) — see ``bitscrape/engine.py::_parse_response``.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator
from urllib.parse import urljoin, urlparse

import bitscrape

_WS_RE = re.compile(r"\s+")
_SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|svg|css|js|ico|pdf|zip|mp4|mp3|woff2?)$", re.I)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


class BIESpider(bitscrape.Spider):
    """Generic article-extraction spider with bounded link-following.

    Configured dynamically per-crawl via instance attributes
    ``start_urls``, ``allowed_domains``, ``max_pages``, and ``max_depth``
    (set by :class:`bie.crawler.Crawler`).
    """

    name = "bie_spider"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.allowed_domains: list[str] = []
        self.max_pages: int = 40
        self.max_depth: int = 2
        self._seen_pages: int = 0

    async def parse(self, parsed: "bitscrape.ParsedResponse") -> AsyncGenerator:
        if parsed.status >= 400:
            return

        self._seen_pages += 1

        title_sel = parsed.css("title::text").get() or parsed.css("h1::text").get() or ""
        title = _clean(title_sel)

        candidate_selectors = [
            "article ::text",
            "main ::text",
            "[role=main] ::text",
            ".content ::text",
            ".post ::text",
            "p::text",
            "h1::text",
            "h2::text",
            "h3::text",
        ]
        candidates: list[str] = []
        for sel in candidate_selectors:
            try:
                candidates.extend(parsed.css(sel).getall())
            except Exception:
                continue

        if not candidates:
            try:
                candidates = parsed.css("body ::text").getall()
            except Exception:
                candidates = []

        blocks = [_clean(c) for c in candidates]
        blocks = [b for b in blocks if len(b) > 1]
        text = "\n\n".join(_dedupe_consecutive(blocks))

        depth = parsed.request.depth
        meta = parsed.request.meta

        yield {
            "url": parsed.url,
            "title": title or parsed.url,
            "text": text,
            "depth": depth,
        }

        # Link-following (bounded by depth + page budget)
        if depth < self.max_depth and self._seen_pages < self.max_pages:
            hrefs = parsed.css("a::attr(href)").getall()
            for href in hrefs[:50]:
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue
                next_url = urljoin(parsed.url, href)
                if _SKIP_EXT.search(next_url):
                    continue
                if self.allowed_domains and not _same_site(next_url, self.allowed_domains):
                    continue
                yield self.follow(next_url, meta={**meta, "depth": depth + 1})


def _dedupe_consecutive(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if not out or out[-1] != item:
            out.append(item)
    return out


def _same_site(url: str, allowed_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith(f".{d}") for d in allowed_domains)
