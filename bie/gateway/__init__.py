"""
M10 — LLM Gateway
==================
Unified interface to any OpenAI-compatible LLM endpoint.
Default: sudharsansm/bie_qwen_2.5_3b via Ollama or vLLM.

Falls back to a deterministic extractive answer if LLM is unreachable —
so BIE stays useful even without a running model server.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

import httpx

from bie.config import BIESettings, settings
from bie.models import AgentResponse, Citation, SearchResult

logger = logging.getLogger(__name__)


class LLMGateway:
    """
    Sends context-injected prompts to the configured LLM and returns
    grounded, citation-tagged answers.
    """

    def __init__(self, cfg: BIESettings = settings):
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.llm_base_url,
            headers={"Authorization": f"Bearer {cfg.llm_api_key}"},
            timeout=60.0,
        )

    async def generate(
        self,
        query: str,
        context: str,
        citations: list[Citation],
        results: list[SearchResult],
    ) -> AgentResponse:
        t0 = time.perf_counter()

        answer = await self._call_llm(context)

        elapsed = (time.perf_counter() - t0) * 1000
        return AgentResponse(
            query=query,
            answer=answer,
            citations=citations,
            contradiction_flags=[],
            latency_ms=round(elapsed, 1),
            model=self._cfg.llm_model,
        )

    async def generate_stream(
        self,
        context: str,
    ) -> AsyncIterator[str]:
        """Yield token chunks via SSE-compatible async generator."""
        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self._cfg.llm_model,
                    "messages": [{"role": "user", "content": context}],
                    "temperature": self._cfg.llm_temperature,
                    "max_tokens": self._cfg.llm_max_tokens,
                    "stream": True,
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        import json
                        try:
                            obj = json.loads(data)
                            token = obj["choices"][0]["delta"].get("content", "")
                            if token:
                                yield token
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Stream failed: %s", exc)
            yield "[LLM unavailable — showing extractive fallback]\n\n"

    # ── internals ─────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str:
        try:
            resp = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._cfg.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self._cfg.llm_temperature,
                    "max_tokens": self._cfg.llm_max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("LLM call failed (%s). Using extractive fallback.", exc)
            return _extractive_fallback(prompt)

    async def close(self):
        await self._client.aclose()


def _extractive_fallback(context: str) -> str:
    """
    Returns the first substantive sentence from each source block.
    Used when the LLM server is not reachable.
    """
    lines = [l.strip() for l in context.splitlines() if l.strip().startswith('"')]
    snippets = [l.strip('"').split(".")[0] for l in lines[:3]]
    if snippets:
        return (
            "Based on retrieved sources: "
            + " | ".join(snippets)
            + ". (LLM server unavailable — extractive answer)"
        )
    return "No relevant information found in the index for this query."
