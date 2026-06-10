"""
M07 — Multi-Agent Orchestrator
================================
Lead agent decomposes a query into sub-tasks (web search, KG lookup,
summarization, fact verification), runs sub-agents in parallel
(async fan-out) or sequentially (linear chain), and merges results
via a shared in-memory (or Redis-backed) memory store.

Usage::

    from bie.agents import AgentOrchestrator

    orch = AgentOrchestrator(retriever, kg, llm, fact_verifier)
    result = await orch.run("Compare TSMC and Samsung's 2026 capex plans")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from bie.config import BIESettings, settings
from bie.context import ContextBuilder
from bie.models import AgentResponse, Citation, SearchFilters, SearchResult

logger = logging.getLogger(__name__)


# ── Shared memory store ────────────────────────────────────────────────────────

class SharedMemory:
    """
    Persists intermediate sub-agent findings across turns.
    Default: in-memory dict. Set `redis_client` for Redis-backed
    cross-process sharing (per PRD M07).
    """

    def __init__(self, redis_client: Any = None, ttl_seconds: int = 3600):
        self._store: dict[str, dict[str, Any]] = {}
        self._redis = redis_client
        self._ttl = ttl_seconds

    async def set(self, session_id: str, key: str, value: Any) -> None:
        if self._redis is not None:
            await self._redis.hset(f"bie:session:{session_id}", key, json.dumps(value))
            await self._redis.expire(f"bie:session:{session_id}", self._ttl)
            return
        self._store.setdefault(session_id, {})[key] = value

    async def get(self, session_id: str, key: str) -> Any:
        if self._redis is not None:
            raw = await self._redis.hget(f"bie:session:{session_id}", key)
            return json.loads(raw) if raw else None
        return self._store.get(session_id, {}).get(key)

    async def get_all(self, session_id: str) -> dict[str, Any]:
        if self._redis is not None:
            raw = await self._redis.hgetall(f"bie:session:{session_id}")
            return {k: json.loads(v) for k, v in raw.items()}
        return dict(self._store.get(session_id, {}))


# ── Token budget tracker ───────────────────────────────────────────────────────

class TokenBudget:
    """Per-agent / per-session token budget enforcement."""

    def __init__(self, max_tokens: int):
        self._max = max_tokens
        self._used = 0

    def consume(self, tokens: int) -> bool:
        """Returns False if consuming would exceed budget."""
        if self._used + tokens > self._max:
            return False
        self._used += tokens
        return True

    @property
    def remaining(self) -> int:
        return max(0, self._max - self._used)

    @property
    def used(self) -> int:
        return self._used


# ── Sub-task definitions ────────────────────────────────────────────────────────

class TaskType(str, Enum):
    SEARCH_WEB = "search_web"
    SEARCH_KG = "search_kg"
    SUMMARIZE = "summarize"
    VERIFY_FACT = "verify_fact"


@dataclass
class SubTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: TaskType = TaskType.SEARCH_WEB
    query: str = ""
    depends_on: list[str] = field(default_factory=list)


@dataclass
class SubTaskResult:
    task_id: str
    type: TaskType
    output: Any
    elapsed_ms: float


# ── Query decomposition ────────────────────────────────────────────────────────

class QueryDecomposer:
    """
    Splits a complex query into sub-tasks.
    Heuristic decomposition: detects "compare", "and", multi-entity
    queries → fan-out search_web tasks per entity, plus a KG lookup
    and a final summarize task. Production can swap this for an
    LLM-based planner.
    """

    _COMPARISON_WORDS = {"compare", "vs", "versus", "difference between"}

    def decompose(self, query: str) -> list[SubTask]:
        tasks: list[SubTask] = []
        q_lower = query.lower()

        # Always include a primary web search
        primary = SubTask(type=TaskType.SEARCH_WEB, query=query)
        tasks.append(primary)

        # KG lookup for named-entity-like capitalized terms
        import re
        entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?\b", query)
        if entities:
            tasks.append(SubTask(type=TaskType.SEARCH_KG, query=" ".join(entities[:3])))

        # Comparison → split into sub-searches per entity
        if any(w in q_lower for w in self._COMPARISON_WORDS) and len(entities) >= 2:
            for ent in entities[:2]:
                tasks.append(SubTask(type=TaskType.SEARCH_WEB, query=f"{ent} {query}"))

        # Final synthesis depends on all prior tasks
        summarize = SubTask(
            type=TaskType.SUMMARIZE,
            query=query,
            depends_on=[t.task_id for t in tasks],
        )
        tasks.append(summarize)

        return tasks


# ── Orchestrator ────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Executes a multi-agent plan: decompose → fan-out sub-agents
    (async) → merge → synthesize via LLM with fact verification.
    """

    def __init__(
        self,
        retriever,           # HybridRetriever
        kg=None,              # KnowledgeGraph
        llm=None,             # LLMGateway
        fact_verifier=None,   # FactVerifier
        cfg: BIESettings = settings,
        memory: SharedMemory | None = None,
    ):
        self._retriever = retriever
        self._kg = kg
        self._llm = llm
        self._fact_verifier = fact_verifier
        self._cfg = cfg
        self._decomposer = QueryDecomposer()
        self._context_builder = ContextBuilder(cfg)
        self._memory = memory or SharedMemory(ttl_seconds=cfg.redis_ttl_seconds)

    async def run(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
        mode: str = "async",  # "async" (fan-out) | "sync" (linear chain)
        token_budget: int = 4000,
    ) -> dict:
        """
        Returns a dict with: answer, citations, sub_results, contradiction_flags,
        latency_ms, mode, session_id.
        """
        session_id = session_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        budget = TokenBudget(token_budget)

        tasks = self._decomposer.decompose(query)
        logger.debug("Decomposed '%s' into %d sub-tasks", query, len(tasks))

        # Separate the synthesis task (always last, depends on others)
        sub_tasks = [t for t in tasks if t.type != TaskType.SUMMARIZE]
        synth_task = next((t for t in tasks if t.type == TaskType.SUMMARIZE), None)

        if mode == "async":
            sub_results = await self._run_parallel(sub_tasks, top_k, budget, session_id)
        else:
            sub_results = await self._run_sequential(sub_tasks, top_k, budget, session_id)

        # Merge all search results for context building
        all_search_results: list[SearchResult] = []
        kg_results: list[dict] = []
        for sr in sub_results:
            if sr.type == TaskType.SEARCH_WEB:
                all_search_results.extend(sr.output)
            elif sr.type == TaskType.SEARCH_KG:
                kg_results.extend(sr.output)

        # Dedup by chunk_id, keep highest rrf_score
        merged: dict[str, SearchResult] = {}
        for r in all_search_results:
            if r.chunk_id not in merged or r.rrf_score > merged[r.chunk_id].rrf_score:
                merged[r.chunk_id] = r
        ranked = sorted(merged.values(), key=lambda r: r.rrf_score, reverse=True)[:top_k]
        for i, r in enumerate(ranked, start=1):
            r.rank = i

        # Synthesize final answer
        context, citations = self._context_builder.build(ranked, query, max_tokens=budget.remaining * 4)
        if kg_results:
            context += "\n\nKnowledge Graph facts:\n" + json.dumps(kg_results[:5], indent=2)

        if self._llm is not None and ranked:
            agent_resp = await self._llm.generate(query, context, citations, ranked)
            answer = agent_resp.answer
        elif ranked:
            answer = "Based on retrieved sources: " + " ".join(
                r.snippet[:150] for r in ranked[:2]
            )
        else:
            answer = "No relevant information found across sub-agent searches."

        # Fact verification pass
        contradiction_flags: list[str] = []
        if self._fact_verifier is not None and ranked:
            verification = await self._fact_verifier.verify(answer, ranked)
            contradiction_flags = [v["claim"] for v in verification if not v["verified"]]

        await self._memory.set(session_id, "last_query", query)
        await self._memory.set(session_id, "last_answer", answer)

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "query": query,
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "sub_results": [
                {"task_id": sr.task_id, "type": sr.type.value, "elapsed_ms": round(sr.elapsed_ms, 1)}
                for sr in sub_results
            ],
            "kg_facts": kg_results[:5],
            "contradiction_flags": contradiction_flags,
            "tokens_used": budget.used,
            "session_id": session_id,
            "mode": mode,
            "latency_ms": round(elapsed, 1),
        }

    # ── Execution strategies ───────────────────────────────────────────────────

    async def _run_parallel(
        self, tasks: list[SubTask], top_k: int, budget: TokenBudget, session_id: str
    ) -> list[SubTaskResult]:
        coros = [self._execute_task(t, top_k, budget, session_id) for t in tasks]
        return await asyncio.gather(*coros)

    async def _run_sequential(
        self, tasks: list[SubTask], top_k: int, budget: TokenBudget, session_id: str
    ) -> list[SubTaskResult]:
        results = []
        for t in tasks:
            results.append(await self._execute_task(t, top_k, budget, session_id))
        return results

    async def _execute_task(
        self, task: SubTask, top_k: int, budget: TokenBudget, session_id: str
    ) -> SubTaskResult:
        t0 = time.perf_counter()

        if task.type == TaskType.SEARCH_WEB:
            results = await self._retriever.search(task.query, top_k=top_k)
            output: Any = results

        elif task.type == TaskType.SEARCH_KG:
            if self._kg is not None:
                output = self._kg.search_entities(task.query, limit=5)
            else:
                output = []

        elif task.type == TaskType.VERIFY_FACT:
            output = []  # handled post-hoc by FactVerifier

        else:  # SUMMARIZE — handled by caller
            output = None

        elapsed_ms = (time.perf_counter() - t0) * 1000
        await self._memory.set(session_id, f"task:{task.task_id}", {"type": task.type.value, "elapsed_ms": elapsed_ms})

        return SubTaskResult(task_id=task.task_id, type=task.type, output=output, elapsed_ms=elapsed_ms)
