"""
Hybrid retrieval index — BM25 (lexical) + optional dense vector (semantic),
fused with Reciprocal Rank Fusion (RRF).

This is the OSS, single-process implementation of the BIE PRD's
Module 2 (Indexes) + Module 3 (Hybrid Retriever). It's designed to be
fast to start (`pip install bits-bie`, no external services required) while
being a drop-in interface that a future Elasticsearch/Milvus-backed
implementation could replace.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Optional

from bie.config import BIESettings
from bie.models import Chunk, Document, SearchResult

logger = logging.getLogger("bie.index")

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """A minimal, dependency-free BM25 index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.df: Counter = Counter()
        self.chunk_ids: list[str] = []
        self.avgdl: float = 0.0

    def add(self, chunk_id: str, text: str) -> None:
        tokens = _tokenize(text)
        freqs = Counter(tokens)
        self.doc_freqs.append(freqs)
        self.doc_lens.append(len(tokens))
        self.chunk_ids.append(chunk_id)
        for term in freqs:
            self.df[term] += 1
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        q_tokens = _tokenize(query)
        n = len(self.chunk_ids)
        scores = [0.0] * n
        for term in q_tokens:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom

        ranked = sorted(
            ((self.chunk_ids[i], s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


class VectorIndex:
    """Optional dense-vector index using sentence-transformers, if installed."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self.chunk_ids: list[str] = []
        self.vectors: list[list[float]] = []
        self.available = self._try_load()

    def _try_load(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info(
                "Vector search disabled (sentence-transformers not available: %s). "
                "Falling back to BM25-only. Install with: pip install 'bits-bie[embeddings]'",
                exc,
            )
            return False

    def add(self, chunk_id: str, text: str) -> None:
        if not self.available:
            return
        vec = self._model.encode(text, normalize_embeddings=True).tolist()
        self.chunk_ids.append(chunk_id)
        self.vectors.append(vec)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if not self.available or not self.vectors:
            return []
        qvec = self._model.encode(query, normalize_embeddings=True)
        scores = []
        for cid, vec in zip(self.chunk_ids, self.vectors):
            sim = sum(a * b for a, b in zip(qvec, vec))
            scores.append((cid, float(sim)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridIndex:
    """Combines BM25 + vector search with weighted RRF fusion."""

    RRF_K = 60

    def __init__(self, settings: Optional[BIESettings] = None) -> None:
        self.settings = settings or BIESettings()
        self.bm25 = BM25Index()
        self.vector = VectorIndex(self.settings.embedding_model) if self.settings.use_embeddings else None
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    @property
    def vector_enabled(self) -> bool:
        return bool(self.vector and self.vector.available)

    def add_document(self, doc: Document, chunks: list[Chunk]) -> None:
        self.documents[doc.doc_id] = doc
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            self.bm25.add(chunk.chunk_id, chunk.text)
            if self.vector:
                self.vector.add(chunk.chunk_id, chunk.text)

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        bm25_hits = dict(self.bm25.search(query, top_k=max(50, top_k * 5)))
        vector_hits: dict[str, float] = {}
        if self.vector_enabled:
            vector_hits = dict(self.vector.search(query, top_k=max(50, top_k * 5)))

        # Reciprocal Rank Fusion across both ranked lists
        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, (cid, _) in enumerate(
            sorted(bm25_hits.items(), key=lambda x: x[1], reverse=True)
        ):
            rrf_scores[cid] += self.settings.bm25_weight / (self.RRF_K + rank + 1)

        if vector_hits:
            for rank, (cid, _) in enumerate(
                sorted(vector_hits.items(), key=lambda x: x[1], reverse=True)
            ):
                rrf_scores[cid] += self.settings.vector_weight / (self.RRF_K + rank + 1)

        if not rrf_scores:
            return []

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for cid, fused_score in ranked:
            chunk = self.chunks.get(cid)
            if not chunk:
                continue
            doc = self.documents.get(chunk.doc_id)
            if not doc:
                continue
            # Trust-score multiplier (PRD Module 5: Trust Engine)
            final_score = fused_score * (0.5 + 0.5 * doc.trust_score)
            results.append(
                SearchResult(
                    title=doc.title or doc.url,
                    url=doc.url,
                    snippet=_snippet(chunk.text, query),
                    source=doc.site or _extract_domain(doc.url),
                    score=round(final_score, 6),
                    bm25_score=round(bm25_hits.get(cid, 0.0), 4),
                    vector_score=round(vector_hits.get(cid, 0.0), 4),
                    trust_score=doc.trust_score,
                    publish_date=doc.publish_date,
                    chunk_id=cid,
                    doc_id=doc.doc_id,
                )
            )
        return results


def _snippet(text: str, query: str, length: int = 240) -> str:
    """Return a snippet centered on the first matching query term, if any."""
    q_terms = [t for t in _tokenize(query) if len(t) > 2]
    lower = text.lower()
    idx = -1
    for term in q_terms:
        idx = lower.find(term)
        if idx != -1:
            break
    if idx == -1:
        snippet = text[:length]
    else:
        start = max(0, idx - length // 3)
        snippet = text[start : start + length]
    snippet = snippet.strip().replace("\n", " ")
    if len(snippet) >= length:
        snippet += "..."
    return snippet


def _extract_domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)/?", url)
    return m.group(1) if m else url
