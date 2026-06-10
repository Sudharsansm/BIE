"""
M11 — BIE v1.0 Agent API
==========================
Full v1.0 REST API: all v0.1 endpoints plus Knowledge Graph,
Multi-Agent Orchestrator, Contradiction Detector, Fact Verifier,
SSO/Enterprise auth, compliance endpoints, and multi-region status.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from bie.agents import AgentOrchestrator, SharedMemory
from bie.auth import APIKeyStore, JWTManager, RBAC, Role
from bie.compliance import (
    AuditLogger, AuditEvent, AuditEventType,
    ComplianceChecker, DataRetentionPolicy, PIIDetector,
)
from bie.config import BIESettings, settings
from bie.context import ContextBuilder
from bie.contradiction import ContradictionDetector
from bie.crawler import BIECrawler
from bie.gateway import LLMGateway
from bie.indexer import HybridIndex, HybridRetriever
from bie.kg import KnowledgeGraph
from bie.models import (
    AgentResponse, CrawlRequest, CrawlResponse,
    HealthResponse, SearchRequest, SearchResponse, SearchResult,
)
from bie.regions import RegionRegistry, GeoRouter, ReplicationManager, ShardRouter
from bie.trust import TrustEngine
from bie.verifier import FactVerifier

logger = logging.getLogger(__name__)

# ── App-wide singletons ────────────────────────────────────────────────────────
_index         = HybridIndex()
_retriever     = HybridRetriever(_index)
_crawler       = BIECrawler()
_trust         = TrustEngine()
_context_builder = ContextBuilder()
_llm           = LLMGateway()
_kg            = KnowledgeGraph()
_contradiction = ContradictionDetector()
_fact_verifier = FactVerifier(kg=_kg)
_memory        = SharedMemory()
_orchestrator  = AgentOrchestrator(
    retriever=_retriever, kg=_kg, llm=_llm,
    fact_verifier=_fact_verifier, memory=_memory,
)

# Auth + compliance singletons
_key_store     = APIKeyStore()
_jwt_manager   = JWTManager()
_audit         = AuditLogger()
_pii           = PIIDetector()
_retention     = DataRetentionPolicy()
_compliance    = ComplianceChecker(settings)

# Multi-region singletons
_region_registry = RegionRegistry()
_geo_router      = GeoRouter(_region_registry)
_shard_router    = ShardRouter(_region_registry)
_replication     = ReplicationManager(_region_registry)

_start_time = time.time()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BIE v1.0 starting — region=%s", settings.region)
    _region_registry.get(settings.region)  # validate local region
    yield
    await _llm.close()
    logger.info("BIE v1.0 shut down.")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app(cfg: BIESettings = settings) -> FastAPI:
    app = FastAPI(
        title="BitSearch Intelligence Engine v1.0",
        description=(
            "AI-native real-time retrieval — Bitscrape-powered.\n"
            "Multi-region · 10B-doc index · SOC 2 · Enterprise Auth\n"
            "Knowledge Graph · Contradiction Detection · Multi-Agent Orchestration"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth dependency ────────────────────────────────────────────────────────

    async def require_api_key(
        request: Request,
        x_api_key: str = Header(..., alias="X-API-Key"),
    ) -> tuple:
        result = _key_store.validate(x_api_key)
        if result is None:
            _audit.log_auth_failure(
                ip=request.client.host if request.client else "",
                endpoint=str(request.url.path),
                reason="invalid_api_key",
            )
            raise HTTPException(status_code=401, detail="Invalid API key")
        key_rec, tenant = result
        if not _key_store.record_usage(x_api_key):
            raise HTTPException(status_code=429, detail="Monthly quota exceeded")
        if cfg.audit_log_enabled:
            _audit.log_request(
                api_key=x_api_key,
                endpoint=str(request.url.path),
                tenant_id=tenant.tenant_id,
                ip=request.client.host if request.client else "",
            )
        return key_rec, tenant

    def require_role(required: Role):
        async def dep(auth=Depends(require_api_key)):
            key_rec, tenant = auth
            if not RBAC.has_permission(key_rec.role, _endpoint_permission(required)):
                raise HTTPException(status_code=403, detail=f"Role '{key_rec.role}' lacks required permission.")
            return key_rec, tenant
        return dep

    def _endpoint_permission(role: Role) -> str:
        return {
            Role.VIEWER: "search:read",
            Role.DEVELOPER: "agent:read",
            Role.ADMIN: "indices:write",
            Role.OWNER: "tenant:manage",
        }.get(role, "search:read")

    # ══════════════════════════════════════════════════════════════════════════
    # Search endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/search", response_model=SearchResponse, tags=["Search"])
    async def search(req: SearchRequest, auth=Depends(require_api_key)) -> SearchResponse:
        """Hybrid BM25 + vector search with trust reweighting and contradiction flags."""
        t0 = time.perf_counter()
        results = await _retriever.search(req.query, top_k=req.top_k, filters=req.filters)

        # Contradiction detection
        flags = _contradiction.detect(results)
        flag_map: dict[str, list[str]] = {}
        for f in flags:
            for cid in (f.chunk_id_a, f.chunk_id_b):
                flag_map.setdefault(cid, []).append(f.explanation)
        for r in results:
            r.contradiction_flags = flag_map.get(r.chunk_id, [])

        return SearchResponse(
            query=req.query,
            results=results,
            total_found=len(results),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    @app.get("/search/stream", tags=["Search"])
    async def search_stream(query: str, top_k: int = 10, auth=Depends(require_api_key)):
        """SSE streaming search — emits each result as it scores."""
        async def gen() -> AsyncIterator[dict]:
            results = await _retriever.search(query=query, top_k=top_k)
            for r in results:
                yield {"event": "result", "data": r.model_dump_json()}
                await asyncio.sleep(0)
            yield {"event": "done", "data": "[DONE]"}
        return EventSourceResponse(gen())

    # ══════════════════════════════════════════════════════════════════════════
    # Agent / RAG endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/agent/query", response_model=AgentResponse, tags=["Agent"])
    async def agent_query(req: SearchRequest, auth=Depends(require_api_key)) -> AgentResponse:
        """Full RAG: retrieve → context → LLM → grounded answer + citations + fact check."""
        results = await _retriever.search(req.query, top_k=req.top_k, filters=req.filters)
        if not results:
            return AgentResponse(
                query=req.query,
                answer="No relevant information found.",
                citations=[], latency_ms=0.0,
            )
        context, citations = _context_builder.build(results, req.query)
        resp = await _llm.generate(req.query, context, citations, results)

        # Post-generation fact verification
        verifications = await _fact_verifier.verify(resp.answer, results)
        unverified = [v["claim"] for v in verifications if not v["verified"]]
        if unverified:
            resp.answer = _fact_verifier.annotate_answer(resp.answer, verifications)
            resp.contradiction_flags = unverified

        # Contradiction check vs retrieved evidence
        c_flags = _contradiction.verify_answer(resp.answer, results)
        if c_flags:
            resp.contradiction_flags.extend([f.explanation for f in c_flags])

        return resp

    @app.post("/agent/orchestrate", tags=["Agent"])
    async def agent_orchestrate(
        req: SearchRequest,
        session_id: str | None = None,
        mode: str = "async",
        auth=Depends(require_api_key),
    ) -> dict:
        """
        Multi-Agent Orchestrator (M07): decomposes query → parallel sub-agents
        (web search, KG lookup) → merges → synthesizes grounded answer.
        """
        key_rec, tenant = auth
        return await _orchestrator.run(
            query=req.query,
            session_id=session_id,
            top_k=req.top_k,
            mode=mode,
            token_budget=settings.agent_token_budget,
        )

    @app.get("/agent/stream", tags=["Agent"])
    async def agent_stream(query: str, top_k: int = 10, auth=Depends(require_api_key)):
        """Streaming LLM token output via SSE."""
        results = await _retriever.search(query=query, top_k=top_k)
        context, _ = _context_builder.build(results, query)
        async def token_gen():
            async for token in _llm.generate_stream(context):
                yield {"event": "token", "data": token}
            yield {"event": "done", "data": "[DONE]"}
        return EventSourceResponse(token_gen())

    # ══════════════════════════════════════════════════════════════════════════
    # Crawler endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/crawl/url", response_model=CrawlResponse, tags=["Crawler"])
    async def crawl_url(req: CrawlRequest, auth=Depends(require_api_key)) -> CrawlResponse:
        """On-demand single-URL crawl with PII detection, trust scoring, and indexing."""
        if _trust.is_blocked(req.url):
            raise HTTPException(status_code=400, detail="URL domain is blocked.")

        result = await _crawler.crawl_single(req.url)
        if result is None:
            return CrawlResponse(url=req.url, status="failed", message="Could not extract content.")

        doc, chunks = result
        trust_score = _trust.score(req.url)
        doc.metadata["trust_score"] = trust_score

        # PII scan and redact each chunk before indexing
        if settings.pii_detection_enabled:
            for chunk in chunks:
                redacted, findings = _pii.redact(chunk.text)
                if findings:
                    chunk.text = redacted
                    _audit.log(AuditEvent(
                        event_type=AuditEventType.PII_DETECTED,
                        resource_id=chunk.chunk_id,
                        details={"findings": len(findings), "url": req.url},
                    ))
            chunk.trust_score = trust_score

        count = await _index.add_documents([(doc, chunks)])
        _kg.ingest_document(doc, chunks)
        _retention.register(doc.doc_id, doc.url, doc.crawled_at)

        # Shard routing info
        region_id, shard = _shard_router.route(doc.doc_id)

        return CrawlResponse(
            url=req.url, status="indexed",
            message=f"Indexed {count} chunks → region={region_id}, shard={shard}",
        )

    @app.post("/crawl/batch", tags=["Crawler"])
    async def crawl_batch(urls: list[str], auth=Depends(require_api_key)) -> dict:
        """Batch crawl up to 50 URLs."""
        if len(urls) > 50:
            raise HTTPException(status_code=400, detail="Max 50 URLs per batch call.")
        docs = await _crawler.crawl_urls(urls)
        enriched = []
        for doc, chunks in docs:
            ts = _trust.score(doc.url)
            doc.metadata["trust_score"] = ts
            if settings.pii_detection_enabled:
                for chunk in chunks:
                    redacted, _ = _pii.redact(chunk.text)
                    chunk.text = redacted
                    chunk.trust_score = ts
            enriched.append((doc, chunks))
        total = await _index.add_documents(enriched)
        for doc, chunks in enriched:
            _kg.ingest_document(doc, chunks)
            _retention.register(doc.doc_id, doc.url, doc.crawled_at)
        return {"status": "ok", "urls_attempted": len(urls), "chunks_indexed": total}

    # ══════════════════════════════════════════════════════════════════════════
    # Knowledge Graph endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/kg/search", tags=["Knowledge Graph"])
    async def kg_search(q: str, limit: int = 10, auth=Depends(require_api_key)) -> dict:
        """Entity search in the Knowledge Graph."""
        entities = _kg.search_entities(q, limit=limit)
        return {"query": q, "entities": entities, "total": len(entities)}

    @app.post("/kg/query", tags=["Knowledge Graph"])
    async def kg_query(
        source_type: str | None = None,
        relation: str | None = None,
        target_type: str | None = None,
        limit: int = 50,
        auth=Depends(require_api_key),
    ) -> dict:
        """Graph pattern query (SPARQL-compatible filter on node types and relation types)."""
        results = _kg.query_pattern(source_type, relation, target_type, limit)
        return {"results": results, "total": len(results)}

    @app.get("/kg/entity/{entity_id}", tags=["Knowledge Graph"])
    async def kg_entity(entity_id: str, auth=Depends(require_api_key)) -> dict:
        """Get full entity graph: node + all neighbors."""
        graph = _kg.get_entity_graph(entity_id)
        if graph is None:
            raise HTTPException(status_code=404, detail="Entity not found")
        return graph

    @app.get("/kg/stats", tags=["Knowledge Graph"])
    async def kg_stats(auth=Depends(require_api_key)) -> dict:
        return {"nodes": _kg.node_count, "edges": _kg.edge_count}

    # ══════════════════════════════════════════════════════════════════════════
    # Trust & feedback
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/feedback", tags=["Trust"])
    async def feedback(url: str, positive: bool, auth=Depends(require_api_key)) -> dict:
        _trust.register_feedback(url, positive)
        return {"status": "ok", "url": url, "positive": positive}

    # ══════════════════════════════════════════════════════════════════════════
    # Compliance endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/compliance/deletion", tags=["Compliance"])
    async def request_deletion(identifier: str, reason: str = "gdpr_erasure", auth=Depends(require_api_key)) -> dict:
        """GDPR Art. 17 — Right to erasure. Returns a deletion ticket with 24-hour SLA."""
        return _retention.request_deletion(identifier, reason)

    @app.get("/compliance/audit", tags=["Compliance"])
    async def get_audit_log(limit: int = 100, auth=Depends(require_api_key)) -> dict:
        """SOC 2 CC7.2 — returns recent audit events for this tenant."""
        key_rec, tenant = auth
        events = _audit.query(tenant_id=tenant.tenant_id, limit=limit)
        return {"events": events, "total": len(events)}

    @app.get("/compliance/checklist", tags=["Compliance"])
    async def compliance_checklist(auth=Depends(require_api_key)) -> dict:
        """SOC 2 + GDPR readiness checklist for the current configuration."""
        return _compliance.run()

    @app.get("/compliance/retention", tags=["Compliance"])
    async def retention_status(auth=Depends(require_api_key)) -> dict:
        """Data retention tier distribution."""
        return _retention.docs_by_tier()

    # ══════════════════════════════════════════════════════════════════════════
    # Multi-region endpoints
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/regions", tags=["Multi-Region"])
    async def list_regions(auth=Depends(require_api_key)) -> dict:
        return _replication.status()

    @app.get("/regions/route", tags=["Multi-Region"])
    async def route_region(lat: float | None = None, lon: float | None = None, auth=Depends(require_api_key)) -> dict:
        region = _geo_router.route(lat, lon)
        return {
            "routed_to": region.region_id,
            "name": region.name,
            "endpoint": region.endpoint,
            "avg_latency_ms": region.avg_latency_ms,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Indices management (enterprise)
    # ══════════════════════════════════════════════════════════════════════════

    @app.post("/indices/update", tags=["Indices"])
    async def indices_update(
        doc: dict,
        auth=Depends(require_role(Role.ADMIN)),
    ) -> dict:
        """Push a document directly into BIE indexes (enterprise token required)."""
        from bie.models import DocumentRecord
        from bie.crawler import TextChunker
        d = DocumentRecord(**doc)
        chunks = TextChunker(chunk_size=settings.chunk_size).chunk(d)
        count = await _index.add_documents([(d, chunks)])
        _kg.ingest_document(d, chunks)
        return {"status": "ok", "chunks_indexed": count, "doc_id": d.doc_id}

    # ══════════════════════════════════════════════════════════════════════════
    # Operations
    # ══════════════════════════════════════════════════════════════════════════

    @app.get("/metrics", tags=["Operations"])
    async def metrics(auth=Depends(require_api_key)) -> dict:
        key_rec, tenant = auth
        quota = _key_store.quota_status(key_rec.api_key)
        return {
            "tenant_id": tenant.tenant_id,
            "tier": tenant.tier.value,
            "api_requests_used": key_rec.requests_this_month,
            "quota": quota,
            "index_docs": _index.doc_count,
            "index_chunks": _index.chunk_count,
            "kg_nodes": _kg.node_count,
            "kg_edges": _kg.edge_count,
            "audit_events": _audit.count,
            "region": settings.region,
            "uptime_seconds": round(time.time() - _start_time, 1),
        }

    @app.get("/health", response_model=HealthResponse, tags=["Operations"])
    async def health() -> HealthResponse:
        """Service health check — no auth required."""
        return HealthResponse(
            status="ok",
            index_size=_index.doc_count,
            uptime_seconds=round(time.time() - _start_time, 1),
        )

    return app


app = create_app()
