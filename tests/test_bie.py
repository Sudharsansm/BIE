"""
BIE unit tests — fast, no network, no LLM required.
Run: pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bie.config import BIESettings
from bie.models import (
    ChunkRecord,
    DocumentRecord,
    SearchFilters,
    SearchRequest,
    SearchResponse,
)
from bie.indexer import HybridIndex, HybridRetriever, TextIndex, VectorIndex, _rrf_fuse
from bie.crawler import TextChunker, ContentFingerprinter, _estimate_trust
from bie.trust import TrustEngine
from bie.context import ContextBuilder, _count_tokens


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_doc() -> DocumentRecord:
    return DocumentRecord(
        url="https://reuters.com/tech/ai-news-2026",
        title="AI News 2026",
        text=(
            "Artificial intelligence continues to advance rapidly in 2026. "
            "Major language models are now deployed in production environments worldwide. "
            "Semiconductor companies report record demand for AI chips. "
            "TSMC announced record Q2 wafer shipments driven by AI demand. "
            "The semiconductor supply chain faces both opportunities and challenges."
        ),
        publish_date="2026-06-01",
        metadata={"site": "reuters.com", "lang": "en", "content_type": "article", "trust_score": 0.95},
    )


@pytest.fixture
def sample_chunks(sample_doc) -> list[ChunkRecord]:
    chunker = TextChunker(chunk_size=50)
    return chunker.chunk(sample_doc)


@pytest.fixture
def cfg() -> BIESettings:
    return BIESettings(
        embedding_model="BAAI/bge-m3",
        embedding_device="cpu",
        llm_base_url="http://localhost:11434/v1",
    )


# ── TextChunker ────────────────────────────────────────────────────────────────

class TestTextChunker:
    def test_produces_chunks(self, sample_doc):
        chunker = TextChunker(chunk_size=30)
        chunks = chunker.chunk(sample_doc)
        assert len(chunks) >= 1

    def test_chunk_ids_written_to_doc(self, sample_doc):
        chunker = TextChunker(chunk_size=30)
        chunks = chunker.chunk(sample_doc)
        assert sample_doc.chunk_ids == [c.chunk_id for c in chunks]

    def test_empty_doc_returns_empty(self):
        doc = DocumentRecord(url="https://example.com", title="", text="")
        chunker = TextChunker()
        assert chunker.chunk(doc) == []

    def test_chunk_has_doc_id(self, sample_doc):
        chunker = TextChunker(chunk_size=30)
        chunks = chunker.chunk(sample_doc)
        for c in chunks:
            assert c.doc_id == sample_doc.doc_id

    def test_chunk_text_non_empty(self, sample_doc):
        chunker = TextChunker(chunk_size=30)
        chunks = chunker.chunk(sample_doc)
        for c in chunks:
            assert c.text.strip()


# ── ContentFingerprinter ──────────────────────────────────────────────────────

class TestContentFingerprinter:
    def test_first_occurrence_not_duplicate(self):
        fp = ContentFingerprinter()
        assert not fp.is_duplicate("Hello world unique content here.")

    def test_second_occurrence_is_duplicate(self):
        fp = ContentFingerprinter()
        text = "Duplicate content test string."
        fp.is_duplicate(text)
        assert fp.is_duplicate(text)

    def test_different_texts_not_duplicate(self):
        fp = ContentFingerprinter()
        assert not fp.is_duplicate("First piece of content.")
        assert not fp.is_duplicate("Second piece of content.")


# ── Trust Engine ──────────────────────────────────────────────────────────────

class TestTrustEngine:
    def test_high_trust_domain(self):
        te = TrustEngine()
        assert te.score("https://reuters.com/news/article") >= 0.85

    def test_unknown_domain_gets_default(self):
        te = TrustEngine()
        score = te.score("https://random-unknown-blog-xyz123.com/post")
        assert 0.4 <= score <= 0.8

    def test_gov_domain_high_trust(self):
        te = TrustEngine()
        assert te.score("https://whitehouse.gov/press") >= 0.80

    def test_feedback_positive_increases_score(self):
        te = TrustEngine()
        url = "https://newsite.io/article"
        before = te.score(url)
        for _ in range(10):
            te.register_feedback(url, positive=True)
        after = te.score(url)
        assert after >= before

    def test_blocked_domain_returns_zero(self):
        te = TrustEngine()
        te._blocked = {"blocked-domain.com"}  # type: ignore[attr-defined]
        from bie.trust import _BLOCKED
        _BLOCKED.add("blocked-test-domain.com")
        assert te.score("https://blocked-test-domain.com/page") == 0.0
        _BLOCKED.discard("blocked-test-domain.com")


# ── TextIndex (BM25) ──────────────────────────────────────────────────────────

class TestTextIndex:
    def _make_chunk(self, text: str, doc_id: str = "doc1") -> ChunkRecord:
        return ChunkRecord(doc_id=doc_id, text=text, trust_score=0.8)

    def test_returns_results(self):
        idx = TextIndex()
        idx.add(self._make_chunk("Python programming language tutorial 2026"))
        idx.add(self._make_chunk("Java enterprise applications development"))
        results = idx.search("Python programming", top_k=2)
        assert len(results) >= 1
        # Both docs indexed; Python doc must appear in results
        texts = [r[0].text for r in results]
        assert any("Python" in t for t in texts)

    def test_empty_index_returns_empty(self):
        idx = TextIndex()
        assert idx.search("anything") == []

    def test_size_property(self):
        idx = TextIndex()
        for i in range(5):
            idx.add(self._make_chunk(f"Document number {i} with unique content here"))
        assert idx.size == 5


# ── VectorIndex ───────────────────────────────────────────────────────────────

class TestVectorIndex:
    def test_add_and_search(self):
        idx = VectorIndex(dim=4)
        c1 = ChunkRecord(doc_id="d1", text="AI research")
        c2 = ChunkRecord(doc_id="d2", text="cooking recipes")
        idx.add(c1, [0.9, 0.1, 0.0, 0.0])
        idx.add(c2, [0.0, 0.0, 0.9, 0.1])
        results = idx.search([0.9, 0.1, 0.0, 0.0], top_k=2)
        assert results[0][0].doc_id == "d1"

    def test_empty_returns_empty(self):
        idx = VectorIndex(dim=4)
        assert idx.search([0.1, 0.2, 0.3, 0.4], top_k=5) == []


# ── RRF Fusion ────────────────────────────────────────────────────────────────

class TestRRFFusion:
    def _chunk(self, cid: str) -> ChunkRecord:
        c = ChunkRecord(doc_id="doc", text="text")
        c.chunk_id = cid
        return c

    def test_fusion_ranks_by_combined_score(self):
        c_a = self._chunk("a")
        c_b = self._chunk("b")
        c_c = self._chunk("c")

        bm25_hits = [(c_a, 0.9), (c_b, 0.5), (c_c, 0.1)]
        vec_hits  = [(c_b, 0.95), (c_a, 0.4), (c_c, 0.2)]

        fused = _rrf_fuse(bm25_hits, vec_hits)
        ids = [f[0].chunk_id for f in fused]
        # Both a and b score highly; order may vary but both top-2
        assert set(ids[:2]) == {"a", "b"}

    def test_fusion_includes_all_chunks(self):
        c1, c2, c3 = [self._chunk(str(i)) for i in range(3)]
        bm25 = [(c1, 0.8), (c2, 0.5)]
        vec  = [(c2, 0.9), (c3, 0.7)]
        fused = _rrf_fuse(bm25, vec)
        assert len(fused) == 3


# ── HybridIndex (integration) ─────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHybridIndex:
    async def test_add_and_search(self, sample_doc, sample_chunks):
        idx = HybridIndex()
        count = await idx.add_documents([(sample_doc, sample_chunks)])
        assert count == len(sample_chunks)
        assert idx.doc_count == 1

    async def test_bm25_search_returns_results(self, sample_doc, sample_chunks):
        idx = HybridIndex()
        await idx.add_documents([(sample_doc, sample_chunks)])
        results = idx.bm25_search("TSMC semiconductor")
        assert len(results) >= 1

    async def test_get_doc(self, sample_doc, sample_chunks):
        idx = HybridIndex()
        await idx.add_documents([(sample_doc, sample_chunks)])
        doc = idx.get_doc(sample_doc.doc_id)
        assert doc is not None
        assert doc.url == sample_doc.url


# ── HybridRetriever ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHybridRetriever:
    async def _populated_retriever(self, sample_doc, sample_chunks):
        idx = HybridIndex()
        await idx.add_documents([(sample_doc, sample_chunks)])
        return HybridRetriever(idx)

    async def test_search_returns_results(self, sample_doc, sample_chunks):
        retriever = await self._populated_retriever(sample_doc, sample_chunks)
        results = await retriever.search("AI chips semiconductor")
        assert len(results) >= 1

    async def test_results_have_required_fields(self, sample_doc, sample_chunks):
        retriever = await self._populated_retriever(sample_doc, sample_chunks)
        results = await retriever.search("language models")
        for r in results:
            assert r.url
            assert r.trust_score >= 0
            assert r.rrf_score >= 0

    async def test_top_k_respected(self, sample_doc, sample_chunks):
        retriever = await self._populated_retriever(sample_doc, sample_chunks)
        results = await retriever.search("AI", top_k=1)
        assert len(results) <= 1

    async def test_empty_index_returns_empty(self):
        idx = HybridIndex()
        retriever = HybridRetriever(idx)
        results = await retriever.search("anything")
        assert results == []

    async def test_domain_filter(self, sample_doc, sample_chunks):
        idx = HybridIndex()
        await idx.add_documents([(sample_doc, sample_chunks)])
        retriever = HybridRetriever(idx)

        # Filter to matching domain
        f_match = SearchFilters(domain="reuters.com")
        results_match = await retriever.search("AI", filters=f_match)

        # Filter to non-matching domain
        f_no_match = SearchFilters(domain="notexist.com")
        results_no = await retriever.search("AI", filters=f_no_match)

        assert len(results_match) >= len(results_no)


# ── ContextBuilder ────────────────────────────────────────────────────────────

class TestContextBuilder:
    def _make_result(self, i: int) -> "SearchResult":
        from bie.models import SearchResult
        return SearchResult(
            rank=i,
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            title=f"Article {i}",
            url=f"https://example{i}.com/article",
            snippet=f"This is snippet number {i} about AI research and technology.",
            source=f"example{i}.com",
            trust_score=0.8,
        )

    def test_build_returns_context_and_citations(self):
        cb = ContextBuilder()
        results = [self._make_result(i) for i in range(3)]
        context, citations = cb.build(results, "AI research")
        assert "[1]" in context
        assert len(citations) == 3

    def test_token_budget_respected(self):
        cb = ContextBuilder()
        results = [self._make_result(i) for i in range(20)]
        context, citations = cb.build(results, "test query", max_tokens=200)
        # Should truncate at budget
        assert len(citations) < 20

    def test_token_counter(self):
        assert _count_tokens("hello world") >= 1
        assert _count_tokens("a" * 100) > _count_tokens("a" * 10)


# ── API integration ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from bie.api import app
        return TestClient(app)

    def test_health_no_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_search_requires_auth(self, client):
        resp = client.post("/search", json={"query": "test", "top_k": 5})
        assert resp.status_code == 422  # missing header

    def test_search_empty_index(self, client):
        resp = client.post(
            "/search",
            json={"query": "anything", "top_k": 5},
            headers={"X-API-Key": "dev-key-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["query"] == "anything"

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics", headers={"X-API-Key": "dev-key-12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert "index_docs" in data

    def test_crawl_blocked_url(self, client):
        from bie.trust import _BLOCKED
        _BLOCKED.add("blocked-api-test.com")
        resp = client.post(
            "/crawl/url",
            json={"url": "https://blocked-api-test.com/page"},
            headers={"X-API-Key": "dev-key-12345"},
        )
        assert resp.status_code == 400
        _BLOCKED.discard("blocked-api-test.com")
