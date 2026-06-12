"""
Free, no-API-key, real-time web search discovery for BIE.

This module answers the question: *"given a query, what URLs on the
internet are relevant right now?"* — without requiring any paid API,
subscription, or API key.

It tries multiple lightweight, no-JS public search endpoints in order,
falling back automatically if one is blocked, rate-limited, or returns
no results:

  1. DuckDuckGo HTML  (``https://html.duckduckgo.com/html/``)
  2. DuckDuckGo Lite  (``https://lite.duckduckgo.com/lite/``)
  3. Bing HTML        (``https://www.bing.com/search``)

This is the **discovery** step. BIE then crawls the discovered URLs with
Bitscrape and ranks the extracted content with its hybrid BM25+vector
index — giving a genuine "type a query, get a real-time answer from the
internet" experience with zero configuration and zero cost.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

logger = logging.getLogger("bie.discovery")

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded-url>&...
_DDG_REDIRECT_RE = re.compile(r"^(?:https?:)?//duckduckgo\.com/l/\?")

# Tracking/redirector domains we never want to treat as a "real" result
_BLOCKED_HOST_FRAGMENTS = (
    "duckduckgo.com",
    "bing.com/search",
    "go.microsoft.com",
    "r.search.yahoo.com",
)


def discover_urls(query: str, max_results: int = 5, timeout: float = 15.0) -> list[str]:
    """Return up to ``max_results`` candidate URLs for ``query`` from the
    live web — no API key required.

    Tries DuckDuckGo (HTML, then Lite) and falls back to Bing HTML search
    if both fail or return nothing.

    Args:
        query: The natural-language search query.
        max_results: Maximum number of URLs to return.
        timeout: HTTP request timeout in seconds (per attempt).

    Returns:
        A list of absolute, deduplicated URLs in result order. Returns an
        empty list only if every backend fails — callers should treat this
        as "try again" rather than "no results exist".
    """
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }

    attempts = [
        ("ddg_html", _fetch_ddg_html, query),
        ("ddg_lite", _fetch_ddg_lite, query),
        ("bing_html", _fetch_bing_html, query),
    ]

    for name, fetch, q in attempts:
        try:
            with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
                html = fetch(client, q)
        except httpx.HTTPError as exc:
            logger.warning("Discovery backend %s failed: %s", name, exc)
            continue

        if not html:
            continue

        urls = _parse_result_urls(html, max_results)
        if urls:
            logger.debug("Discovery backend %s returned %d url(s)", name, len(urls))
            return urls

    logger.warning("All discovery backends failed or returned no results for query=%r", query)
    return []


def _fetch_ddg_html(client: httpx.Client, query: str) -> str:
    resp = client.post("https://html.duckduckgo.com/html/", data={"q": query})
    resp.raise_for_status()
    return resp.text


def _fetch_ddg_lite(client: httpx.Client, query: str) -> str:
    resp = client.post("https://lite.duckduckgo.com/lite/", data={"q": query})
    resp.raise_for_status()
    return resp.text


def _fetch_bing_html(client: httpx.Client, query: str) -> str:
    resp = client.get("https://www.bing.com/search", params={"q": query, "form": "QBLH"})
    resp.raise_for_status()
    return resp.text


def _parse_result_urls(html: str, max_results: int) -> list[str]:
    """Extract organic result URLs from a search results page.

    Handles:
      - DuckDuckGo HTML:  ``<a class="result__a" href="...">``
      - DuckDuckGo Lite:  ``<a class="result-link" href="...">``
      - Bing:             ``<li class="b_algo">...<a href="...">``
    """
    hrefs = re.findall(r'class="result__a"[^>]*href="([^"]+)"', html)
    if not hrefs:
        hrefs = re.findall(r'class="result-link"[^>]*href="([^"]+)"', html)
    if not hrefs:
        hrefs = _extract_bing_hrefs(html)

    urls: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        url = _resolve_redirect(href)
        if not url or not url.startswith("http"):
            continue
        if _is_blocked_host(url):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max_results:
            break

    return urls


def _extract_bing_hrefs(html: str) -> list[str]:
    """Extract result links from Bing's organic result blocks
    (``<li class="b_algo">``)."""
    hrefs: list[str] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, flags=re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"', block)
        if m:
            hrefs.append(m.group(1))
    return hrefs


def _resolve_redirect(href: str) -> str | None:
    """Unwrap DuckDuckGo's ``//duckduckgo.com/l/?uddg=<url-encoded-target>``
    redirect links to get the real target URL. Other links pass through
    unchanged."""
    href = href.strip().replace("&amp;", "&")

    if _DDG_REDIRECT_RE.match(href):
        parsed = urlparse(href if href.startswith("http") else f"https:{href}")
        qs = parse_qs(parsed.query)
        target = qs.get("uddg", [None])[0]
        return unquote(target) if target else None

    return href


def _is_blocked_host(url: str) -> bool:
    return any(fragment in url for fragment in _BLOCKED_HOST_FRAGMENTS)
