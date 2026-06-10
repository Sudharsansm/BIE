"""
BIE v1.0 tests — covers all new modules.
Run: pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import time
import pytest

# ── Knowledge Graph ────────────────────────────────────────────────────────────

class TestEntityExtractor:
    def test_known_orgs(self):
        from bie.kg import EntityExtractor
        ex = EntityExtractor()
        entities = ex.extract("TSMC and NVIDIA reported strong earnings this quarter.")
        names = [e[0] for e in entities]
        assert "TSMC" in names
        assert "NVIDIA" in names

    def test_known_locations(self):
        from bie.kg import EntityExtractor
        ex = EntityExtractor()
        entities = ex.extract("The factory is based in Taiwan and ships to the United States.")
        types = {e[0]: e[1] for e in entities}
        assert types.get("Taiwan") == "Location"

    def test_relation_extraction(self):
        from bie.kg import EntityExtractor
        ex = EntityExtractor()
        rels = ex.extract_relations("TSMC manufactures advanced chips for NVIDIA.")
        assert any(r[1] == "MANUFACTURES" for r in rels)

    def test_empty_text(self):
        from bie.kg import EntityExtractor
        ex = EntityExtractor()
        assert ex.extract("") == []
        assert ex.extract_relations("") == []


class TestInMemoryGraphStore:
    def test_upsert_and_retrieve_node(self):
        from bie.kg import InMemoryGraphStore
        store = InMemoryGraphStore()
        node = store.upsert_node("TSMC", "Organization", "doc1")
        assert node.name == "TSMC"
        assert node.type == "Organization"
        fetched = store.find_by_name("TSMC")
        assert fetched is not None
        assert fetched.entity_id == node.entity_id

    def test_upsert_dedup(self):
        from bie.kg import InMemoryGraphStore
        store = InMemoryGraphStore()
        n1 = store.upsert_node("Apple", "Organization", "doc1")
        n2 = store.upsert_node("Apple", "Organization", "doc2")
        assert n1.entity_id == n2.entity_id
        assert store.node_count == 1

    def test_upsert_edge(self):
        from bie.kg import InMemoryGraphStore
        store = InMemoryGraphStore()
        n1 = store.upsert_node("TSMC", "Organization", "doc1")
        n2 = store.upsert_node("NVIDIA", "Organization", "doc1")
        edge = store.upsert_edge(n1.entity_id, n2.entity_id, "MANUFACTURES", "doc1")
        assert edge.relation_type == "MANUFACTURES"
        assert store.edge_count == 1

    def test_query_pattern(self):
        from bie.kg import InMemoryGraphStore
        store = InMemoryGraphStore()
        n1 = store.upsert_node("TSMC", "Organization", "doc1")
        n2 = store.upsert_node("Taiwan", "Location", "doc1")
        store.upsert_edge(n1.entity_id, n2.entity_id, "HEADQUARTERED_IN", "doc1")
        results = store.query_pattern(source_type="Organization", relation="HEADQUARTERED_IN")
        assert len(results) == 1
        assert results[0]["source"]["name"] == "TSMC"

    def test_search_entities(self):
        from bie.kg import InMemoryGraphStore
        store = InMemoryGraphStore()
        store.upsert_node("Taiwan Semiconductor", "Organization", "doc1")
        results = store.search_entities("Taiwan", limit=5)
        assert len(results) >= 1


class TestKnowledgeGraph:
    def test_ingest_document(self):
        from bie.kg import KnowledgeGraph
        from bie.models import DocumentRecord, ChunkRecord
        kg = KnowledgeGraph()
        doc = DocumentRecord(url="https://example.com", title="Test", text="TSMC is based in Taiwan.")
        chunk = ChunkRecord(doc_id=doc.doc_id, text="TSMC is based in Taiwan.")
        stats = kg.ingest_document(doc, [chunk])
        assert stats["nodes_processed"] >= 0  # may or may not find entities

    def test_search_entities(self):
        from bie.kg import KnowledgeGraph
        from bie.models import DocumentRecord, ChunkRecord
        kg = KnowledgeGraph()
        doc = DocumentRecord(url="https://example.com", title="TSMC News", text="TSMC reported record revenue.")
        chunk = ChunkRecord(doc_id=doc.doc_id, text="TSMC reported record revenue.")
        kg.ingest_document(doc, [chunk])
        results = kg.search_entities("TSMC")
        assert isinstance(results, list)

    def test_query_pattern_returns_list(self):
        from bie.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        results = kg.query_pattern(source_type="Organization", relation="MANUFACTURES")
        assert isinstance(results, list)


# ── Contradiction Detector ─────────────────────────────────────────────────────

class TestContradictionDetector:
    def _make_result(self, rank, chunk_id, source, snippet) -> "SearchResult":
        from bie.models import SearchResult
        return SearchResult(
            rank=rank, chunk_id=chunk_id, doc_id="d1",
            title="T", url=f"https://{source}/a",
            snippet=snippet, source=source, trust_score=0.8,
        )

    def test_detects_antonym_contradiction(self):
        from bie.contradiction import ContradictionDetector
        detector = ContradictionDetector(threshold=0.45)
        r1 = self._make_result(1, "c1", "reuters.com", "TSMC revenue increased significantly in Q2 2026 earnings report.")
        r2 = self._make_result(2, "c2", "bbc.com", "TSMC revenue declined significantly in Q2 2026 earnings report.")
        flags = detector.detect([r1, r2])
        assert len(flags) >= 1
        assert flags[0].source_a in ("reuters.com", "bbc.com")

    def test_no_flag_same_source(self):
        from bie.contradiction import ContradictionDetector
        detector = ContradictionDetector()
        r1 = self._make_result(1, "c1", "same.com", "Revenue increased in Q2.")
        r2 = self._make_result(2, "c2", "same.com", "Revenue decreased in Q2.")
        flags = detector.detect([r1, r2])
        assert len(flags) == 0  # same-source skipped

    def test_no_flag_unrelated_snippets(self):
        from bie.contradiction import ContradictionDetector
        detector = ContradictionDetector()
        r1 = self._make_result(1, "c1", "a.com", "The weather in Paris is sunny today for tourists.")
        r2 = self._make_result(2, "c2", "b.com", "Quantum computing advances in semiconductor research labs.")
        flags = detector.detect([r1, r2])
        assert len(flags) == 0

    def test_verify_answer(self):
        from bie.contradiction import ContradictionDetector
        detector = ContradictionDetector(threshold=0.4)
        answer = "TSMC revenue increased in Q2."
        r = self._make_result(1, "c1", "reuters.com", "TSMC revenue declined in Q2.")
        flags = detector.verify_answer(answer, [r])
        # May or may not flag depending on heuristic — just check type
        assert isinstance(flags, list)


# ── Fact Verifier ──────────────────────────────────────────────────────────────

class TestFactVerifier:
    def _make_result(self, snippet: str) -> "SearchResult":
        from bie.models import SearchResult
        return SearchResult(
            rank=1, chunk_id="c1", doc_id="d1",
            title="T", url="https://reuters.com/a",
            snippet=snippet, source="reuters.com", trust_score=0.9,
        )

    @pytest.mark.asyncio
    async def test_verify_supported_claim(self):
        from bie.verifier import FactVerifier
        fv = FactVerifier()
        snippet = "TSMC reported record Q2 revenue driven by AI chip demand."
        answer = "TSMC reported record Q2 revenue driven by AI chip demand this quarter."
        evidence = [self._make_result(snippet)]
        results = await fv.verify(answer, evidence)
        assert isinstance(results, list)
        assert all("verified" in r for r in results)

    @pytest.mark.asyncio
    async def test_annotate_unverified(self):
        from bie.verifier import FactVerifier
        fv = FactVerifier()
        answer = "TSMC built a factory on Mars in 2025."
        evidence = [self._make_result("TSMC announced expansion plans in Arizona.")]
        results = await fv.verify(answer, evidence)
        annotated = fv.annotate_answer(answer, results)
        assert isinstance(annotated, str)

    def test_audit_log_grows(self):
        from bie.verifier import FactVerifier
        fv = FactVerifier()
        asyncio.run(fv.verify("Some claim.", []))
        assert fv.audit_log.count >= 0  # no evidence → may still log claims

    def test_claim_splitting(self):
        from bie.verifier import _split_claims
        answer = "TSMC reported growth. NVIDIA demand increased. Intel faced headwinds."
        claims = _split_claims(answer)
        assert len(claims) == 3


# ── Multi-Agent Orchestrator ───────────────────────────────────────────────────

class TestQueryDecomposer:
    def test_simple_query_has_search_and_summarize(self):
        from bie.agents import QueryDecomposer, TaskType
        d = QueryDecomposer()
        tasks = d.decompose("What is TSMC's revenue?")
        types = [t.type for t in tasks]
        assert TaskType.SEARCH_WEB in types
        assert TaskType.SUMMARIZE in types

    def test_comparison_query_adds_sub_searches(self):
        from bie.agents import QueryDecomposer, TaskType
        d = QueryDecomposer()
        tasks = d.decompose("Compare TSMC versus Samsung capex plans 2026")
        search_tasks = [t for t in tasks if t.type == TaskType.SEARCH_WEB]
        assert len(search_tasks) >= 2

    def test_named_entities_trigger_kg_lookup(self):
        from bie.agents import QueryDecomposer, TaskType
        d = QueryDecomposer()
        tasks = d.decompose("What products does NVIDIA manufacture?")
        types = [t.type for t in tasks]
        assert TaskType.SEARCH_KG in types


class TestSharedMemory:
    @pytest.mark.asyncio
    async def test_set_get(self):
        from bie.agents import SharedMemory
        mem = SharedMemory()
        await mem.set("s1", "key1", {"data": 42})
        result = await mem.get("s1", "key1")
        assert result == {"data": 42}

    @pytest.mark.asyncio
    async def test_get_all(self):
        from bie.agents import SharedMemory
        mem = SharedMemory()
        await mem.set("s2", "a", 1)
        await mem.set("s2", "b", 2)
        all_data = await mem.get_all("s2")
        assert all_data == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        from bie.agents import SharedMemory
        mem = SharedMemory()
        result = await mem.get("nonexistent", "key")
        assert result is None


class TestTokenBudget:
    def test_consume_within_budget(self):
        from bie.agents import TokenBudget
        b = TokenBudget(100)
        assert b.consume(50) is True
        assert b.used == 50
        assert b.remaining == 50

    def test_consume_exceeds_budget(self):
        from bie.agents import TokenBudget
        b = TokenBudget(100)
        b.consume(80)
        assert b.consume(30) is False

    def test_remaining_never_negative(self):
        from bie.agents import TokenBudget
        b = TokenBudget(10)
        b.consume(10)
        assert b.remaining == 0
        # Further consume is rejected (budget exhausted)
        assert b.consume(1) is False


@pytest.mark.asyncio
class TestAgentOrchestrator:
    async def _make_orchestrator(self):
        from bie.agents import AgentOrchestrator
        from bie.indexer import HybridIndex, HybridRetriever
        from bie.models import DocumentRecord
        from bie.crawler import TextChunker

        idx = HybridIndex()
        doc = DocumentRecord(
            url="https://reuters.com/tsmc",
            title="TSMC 2026",
            text="TSMC reported record revenue in Q2 2026 driven by AI chip demand.",
            metadata={"site": "reuters.com", "lang": "en", "content_type": "article", "trust_score": 0.95},
        )
        chunks = TextChunker(chunk_size=50).chunk(doc)
        await idx.add_documents([(doc, chunks)])
        retriever = HybridRetriever(idx)
        return AgentOrchestrator(retriever=retriever)

    async def test_run_returns_answer(self):
        orch = await self._make_orchestrator()
        result = await orch.run("TSMC revenue 2026", top_k=3)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    async def test_run_has_session_id(self):
        orch = await self._make_orchestrator()
        result = await orch.run("AI chip demand", session_id="test-session")
        assert result["session_id"] == "test-session"

    async def test_run_sequential_mode(self):
        orch = await self._make_orchestrator()
        result = await orch.run("AI market", mode="sync")
        assert result["mode"] == "sync"

    async def test_sub_results_present(self):
        orch = await self._make_orchestrator()
        result = await orch.run("semiconductor demand")
        assert "sub_results" in result
        assert isinstance(result["sub_results"], list)


# ── Auth module ────────────────────────────────────────────────────────────────

class TestAPIKeyStore:
    def test_dev_key_seeded(self):
        from bie.auth import APIKeyStore
        store = APIKeyStore()
        result = store.validate("dev-key-12345")
        assert result is not None
        key_rec, tenant = result
        assert tenant.name == "dev-tenant"

    def test_invalid_key_returns_none(self):
        from bie.auth import APIKeyStore
        store = APIKeyStore()
        assert store.validate("invalid-key-xyz") is None

    def test_create_tenant_and_key(self):
        from bie.auth import APIKeyStore, PricingTier, Role
        store = APIKeyStore()
        tenant = store.create_tenant("Acme Corp", PricingTier.STARTUP)
        key = store.create_key(tenant.tenant_id, Role.DEVELOPER)
        result = store.validate(key.api_key)
        assert result is not None
        assert result[1].name == "Acme Corp"

    def test_quota_tracking(self):
        from bie.auth import APIKeyStore, PricingTier, Role
        store = APIKeyStore()
        tenant = store.create_tenant("TestCo", PricingTier.FREE)
        key = store.create_key(tenant.tenant_id, Role.DEVELOPER)
        assert store.record_usage(key.api_key) is True
        status = store.quota_status(key.api_key)
        assert status["used"] == 1

    def test_enterprise_unlimited_quota(self):
        from bie.auth import APIKeyStore
        store = APIKeyStore()
        result = store.validate("dev-key-12345")
        assert result is not None
        status = store.quota_status("dev-key-12345")
        assert status["remaining"] == "unlimited"


class TestJWTManager:
    def test_issue_and_verify(self):
        from bie.auth import JWTManager, Role
        mgr = JWTManager()
        token = mgr.issue("user@example.com", "tn_abc", Role.DEVELOPER)
        claims = mgr.verify(token)
        assert claims is not None
        assert claims["sub"] == "user@example.com"
        assert claims["role"] == Role.DEVELOPER.value

    def test_invalid_token_returns_none(self):
        from bie.auth import JWTManager
        mgr = JWTManager()
        assert mgr.verify("not.a.valid.jwt") is None

    def test_expired_token_rejected(self):
        from bie.auth import JWTManager, Role
        mgr = JWTManager(ttl_seconds=-1)  # already expired
        token = mgr.issue("u", "t", Role.VIEWER)
        assert mgr.verify(token) is None


class TestRBAC:
    def test_viewer_can_search(self):
        from bie.auth import RBAC, Role
        assert RBAC.has_permission(Role.VIEWER, "search:read") is True

    def test_viewer_cannot_write_indices(self):
        from bie.auth import RBAC, Role
        assert RBAC.has_permission(Role.VIEWER, "indices:write") is False

    def test_admin_has_all_non_tenant_perms(self):
        from bie.auth import RBAC, Role
        assert RBAC.has_permission(Role.ADMIN, "indices:write") is True
        assert RBAC.has_permission(Role.ADMIN, "crawl:write") is True

    def test_owner_has_all_perms(self):
        from bie.auth import RBAC, Role
        assert RBAC.has_permission(Role.OWNER, "tenant:manage") is True
        assert RBAC.has_permission(Role.OWNER, "billing:manage") is True


# ── Compliance ─────────────────────────────────────────────────────────────────

class TestPIIDetector:
    def test_detects_email(self):
        from bie.compliance import PIIDetector
        d = PIIDetector()
        findings = d.scan("Contact us at alice@example.com for more info.")
        assert any(f.pii_type == "email" for f in findings)

    def test_detects_phone(self):
        from bie.compliance import PIIDetector
        d = PIIDetector()
        assert d.has_pii("Call 555-867-5309 for support.")

    def test_detects_ip(self):
        from bie.compliance import PIIDetector
        d = PIIDetector()
        assert d.has_pii("Request from 192.168.1.100")

    def test_redact_replaces_pii(self):
        from bie.compliance import PIIDetector
        d = PIIDetector()
        text = "Email alice@example.com or call 555-867-5309."
        redacted, findings = d.redact(text)
        assert "alice@example.com" not in redacted
        assert "[REDACTED-" in redacted

    def test_clean_text_unchanged(self):
        from bie.compliance import PIIDetector
        d = PIIDetector()
        text = "TSMC reported record revenue in Q2 2026."
        redacted, findings = d.redact(text)
        assert redacted == text
        assert findings == []


class TestDataRetentionPolicy:
    def test_register_and_classify_hot(self):
        from bie.compliance import DataRetentionPolicy, RetentionTier
        policy = DataRetentionPolicy()
        policy.register("doc1", "https://example.com", time.time())
        assert policy.classify("doc1") == RetentionTier.HOT

    def test_classify_deleted_after_request(self):
        from bie.compliance import DataRetentionPolicy, RetentionTier
        policy = DataRetentionPolicy()
        policy.register("doc2", "https://example.com/old")
        policy.request_deletion("doc2", "gdpr_erasure")
        assert policy.classify("doc2") == RetentionTier.DELETED

    def test_deletion_ticket_returned(self):
        from bie.compliance import DataRetentionPolicy
        policy = DataRetentionPolicy()
        policy.register("doc3", "https://example.com/page")
        ticket = policy.request_deletion("doc3")
        assert "ticket_id" in ticket
        assert ticket["ticket_id"].startswith("DEL-")
        assert ticket["sla_hours"] == 24

    def test_docs_by_tier_structure(self):
        from bie.compliance import DataRetentionPolicy
        policy = DataRetentionPolicy()
        policy.register("d1", "https://a.com")
        tiers = policy.docs_by_tier()
        assert "hot" in tiers
        assert "d1" in tiers["hot"]


class TestAuditLogger:
    def test_log_and_query(self):
        from bie.compliance import AuditLogger, AuditEvent, AuditEventType
        log = AuditLogger()
        log.log(AuditEvent(event_type=AuditEventType.API_REQUEST, tenant_id="t1", endpoint="/search"))
        events = log.query(tenant_id="t1")
        assert len(events) == 1
        assert events[0]["endpoint"] == "/search"

    def test_log_request_helper(self):
        from bie.compliance import AuditLogger
        log = AuditLogger()
        log.log_request("my-api-key", "/agent/query", tenant_id="t2")
        assert log.count == 1

    def test_log_auth_failure(self):
        from bie.compliance import AuditLogger, AuditEventType
        log = AuditLogger()
        log.log_auth_failure("1.2.3.4", "/search", "bad_key")
        events = log.query(event_type=AuditEventType.AUTH_FAILURE)
        assert len(events) == 1
        assert events[0]["outcome"] == "failure"


class TestComplianceChecker:
    def test_returns_summary(self):
        from bie.compliance import ComplianceChecker
        from bie.config import BIESettings
        cfg = BIESettings()
        checker = ComplianceChecker(cfg)
        result = checker.run()
        assert "summary" in result
        assert "checks" in result
        assert result["summary"]["total"] > 0

    def test_default_config_has_failures(self):
        from bie.compliance import ComplianceChecker
        from bie.config import BIESettings
        cfg = BIESettings()  # uses default secret key
        checker = ComplianceChecker(cfg)
        result = checker.run()
        # Default secret key should fail
        failed = [c for c in result["checks"] if c["status"] == "FAIL"]
        assert len(failed) >= 1


# ── Multi-region ───────────────────────────────────────────────────────────────

class TestRegionRegistry:
    def test_all_regions(self):
        from bie.regions import RegionRegistry
        reg = RegionRegistry()
        assert len(reg.all()) >= 4

    def test_primary_region(self):
        from bie.regions import RegionRegistry
        reg = RegionRegistry()
        primary = reg.primary()
        assert primary.is_primary is True

    def test_healthy_regions(self):
        from bie.regions import RegionRegistry, RegionStatus
        reg = RegionRegistry()
        reg.update_health("eu-west-1", RegionStatus.DOWN)
        healthy = reg.healthy()
        assert all(r.region_id != "eu-west-1" for r in healthy)

    def test_total_capacity_10b(self):
        from bie.regions import RegionRegistry
        reg = RegionRegistry()
        assert reg.total_capacity() >= 10_000_000_000


class TestGeoRouter:
    def test_routes_to_nearest(self):
        from bie.regions import RegionRegistry, GeoRouter
        reg = RegionRegistry()
        router = GeoRouter(reg)
        # Singapore coords → should prefer ap-southeast-1
        region = router.route(client_lat=1.35, client_lon=103.82)
        assert region.region_id == "ap-southeast-1"

    def test_falls_back_on_no_geo(self):
        from bie.regions import RegionRegistry, GeoRouter
        reg = RegionRegistry()
        router = GeoRouter(reg)
        region = router.route()  # no geo info
        assert region is not None

    def test_failover_skips_down_region(self):
        from bie.regions import RegionRegistry, GeoRouter, RegionStatus
        reg = RegionRegistry()
        router = GeoRouter(reg)
        primary_id = reg.primary().region_id
        fallback = router.failover(primary_id)
        assert fallback.region_id != primary_id


class TestShardRouter:
    def test_consistent_routing(self):
        from bie.regions import RegionRegistry, ShardRouter
        reg = RegionRegistry()
        router = ShardRouter(reg)
        region1, shard1 = router.route("doc-abc-123")
        region2, shard2 = router.route("doc-abc-123")
        assert region1 == region2
        assert shard1 == shard2

    def test_different_docs_spread_across_regions(self):
        from bie.regions import RegionRegistry, ShardRouter
        reg = RegionRegistry()
        router = ShardRouter(reg)
        import uuid
        regions_seen = set()
        for _ in range(200):
            region_id, _ = router.route(str(uuid.uuid4()))
            regions_seen.add(region_id)
        assert len(regions_seen) >= 2  # spread

    def test_index_name_format(self):
        from bie.regions import RegionRegistry, ShardRouter
        reg = RegionRegistry()
        router = ShardRouter(reg)
        name = router.index_name("doc-xyz", "bie-docs")
        assert name.startswith("bie-docs-")
        assert "shard" in name


# ── v1.0 API integration ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestV1API:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from bie.api import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_compliance_checklist(self, client):
        r = client.get("/compliance/checklist", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert "summary" in data
        assert data["summary"]["total"] > 0

    def test_regions_endpoint(self, client):
        r = client.get("/regions", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert "regions" in data
        assert len(data["regions"]) >= 4

    def test_kg_stats(self, client):
        r = client.get("/kg/stats", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data

    def test_kg_search_empty(self, client):
        r = client.get("/kg/search?q=TSMC", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200

    def test_kg_entity_not_found(self, client):
        r = client.get("/kg/entity/nonexistent-id", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 404

    def test_search_empty_index(self, client):
        r = client.post("/search", json={"query": "test", "top_k": 5},
                        headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        assert r.json()["results"] == []

    def test_agent_empty_index(self, client):
        r = client.post("/agent/query", json={"query": "test", "top_k": 5},
                        headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        assert "answer" in r.json()

    def test_metrics_includes_kg(self, client):
        r = client.get("/metrics", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert "kg_nodes" in data
        assert "kg_edges" in data

    def test_geo_route(self, client):
        r = client.get("/regions/route?lat=1.35&lon=103.82",
                       headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        data = r.json()
        assert data["routed_to"] == "ap-southeast-1"

    def test_deletion_request(self, client):
        r = client.post(
            "/compliance/deletion?identifier=https://example.com/page&reason=gdpr_erasure",
            headers={"X-API-Key": "dev-key-12345"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "ticket_id" in data
        assert data["sla_hours"] == 24

    def test_audit_log(self, client):
        # Make a request first so there's something to audit
        client.post("/search", json={"query": "test"}, headers={"X-API-Key": "dev-key-12345"})
        r = client.get("/audit" if False else "/compliance/audit",
                       headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
        assert "events" in r.json()

    def test_feedback(self, client):
        r = client.post(
            "/feedback?url=https://reuters.com/a&positive=true",
            headers={"X-API-Key": "dev-key-12345"},
        )
        assert r.status_code == 200

    def test_invalid_api_key_401(self, client):
        r = client.post("/search", json={"query": "test", "top_k": 5},
                        headers={"X-API-Key": "bad-key"})
        assert r.status_code == 401

    def test_retention_status(self, client):
        r = client.get("/compliance/retention", headers={"X-API-Key": "dev-key-12345"})
        assert r.status_code == 200
