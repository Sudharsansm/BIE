"""
M08 — Context Builder
=====================
Assembles top-K chunks into a token-budgeted, citation-tagged context
string ready for injection into an LLM system prompt.
"""

from __future__ import annotations

import re
from typing import Iterator

from bie.config import BIESettings, settings
from bie.models import Citation, SearchResult


class ContextBuilder:
    """
    Builds an LLM-ready context block from ranked search results.

    Output format::

        [1] Title — domain.com (trust: 0.91)
        "Snippet text here..."

        [2] Another Title — other.com (trust: 0.78)
        "Another snippet..."

    Each result gets a numeric citation tag [N] that the LLM is
    instructed to echo in its answer.
    """

    def __init__(self, cfg: BIESettings = settings):
        self._cfg = cfg

    def build(
        self,
        results: list[SearchResult],
        query: str,
        max_tokens: int | None = None,
    ) -> tuple[str, list[Citation]]:
        """
        Returns (context_string, citations_list).
        context_string is injected into the LLM system prompt.
        """
        budget = max_tokens or self._cfg.max_context_tokens
        lines: list[str] = [
            f'Answer the question using ONLY the sources below. '
            f'Cite each fact with its [N] tag.\n\nQuestion: {query}\n\nSources:\n'
        ]
        citations: list[Citation] = []
        used_tokens = _count_tokens(lines[0])

        for i, result in enumerate(results, start=1):
            snippet = _clean_snippet(result.snippet)
            entry = (
                f"[{i}] {result.title} — {result.source} (trust: {result.trust_score})\n"
                f'"{snippet}"\n'
            )
            entry_tokens = _count_tokens(entry)
            if used_tokens + entry_tokens > budget:
                break

            lines.append(entry)
            used_tokens += entry_tokens
            citations.append(
                Citation(
                    index=i,
                    url=result.url,
                    title=result.title,
                    snippet=snippet,
                    trust_score=result.trust_score,
                )
            )

        context = "\n".join(lines)
        return context, citations


def _count_tokens(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 chars."""
    return max(1, len(text) // 4)


def _clean_snippet(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]
