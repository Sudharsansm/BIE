"""
M05 — Trust Engine
==================
Scores source reliability and adjusts retrieval ranking.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


# Known high-quality domains → base trust score
_DOMAIN_TRUST: dict[str, float] = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "bbc.com": 0.92,
    "bbc.co.uk": 0.92,
    "nature.com": 0.97,
    "science.org": 0.97,
    "arxiv.org": 0.88,
    "pubmed.ncbi.nlm.nih.gov": 0.96,
    "github.com": 0.80,
    "stackoverflow.com": 0.78,
    "docs.python.org": 0.93,
    "en.wikipedia.org": 0.75,
    "nytimes.com": 0.88,
    "wsj.com": 0.87,
    "ft.com": 0.88,
}

# Domains to block outright
_BLOCKED: set[str] = {
    "spam-example.com",
    "malware-example.net",
}

# TLD trust priors
_TLD_TRUST: dict[str, float] = {
    ".gov": 0.93,
    ".edu": 0.90,
    ".ac.uk": 0.88,
    ".org": 0.70,
    ".com": 0.60,
    ".net": 0.55,
    ".io": 0.58,
}


class TrustEngine:
    """
    Computes a [0, 1] trust score for a given URL / domain.
    Integrates user feedback via a simple exponential moving average.
    """

    def __init__(self):
        self._feedback: dict[str, float] = {}  # domain → EMA feedback score

    def score(self, url: str) -> float:
        domain = _extract_domain(url)
        if domain in _BLOCKED:
            return 0.0

        base = self._domain_base(domain)
        feedback = self._feedback.get(domain, 0.5)

        # Blend: 80% domain signal, 20% user feedback
        return round(0.8 * base + 0.2 * feedback, 3)

    def register_feedback(self, url: str, positive: bool) -> None:
        """Thumbs up/down feedback — updates EMA."""
        domain = _extract_domain(url)
        current = self._feedback.get(domain, 0.5)
        signal = 1.0 if positive else 0.0
        # EMA alpha=0.1
        self._feedback[domain] = 0.9 * current + 0.1 * signal

    def is_blocked(self, url: str) -> bool:
        return _extract_domain(url) in _BLOCKED

    # ── private ───────────────────────────────────────────────────────────────

    def _domain_base(self, domain: str) -> float:
        if domain in _DOMAIN_TRUST:
            return _DOMAIN_TRUST[domain]
        bare = domain.lstrip("www.")
        if bare in _DOMAIN_TRUST:
            return _DOMAIN_TRUST[bare]
        for tld, score in _TLD_TRUST.items():
            if domain.endswith(tld):
                return score
        return 0.55  # unknown default


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""
