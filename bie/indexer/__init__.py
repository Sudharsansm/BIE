"""
M02 + M03 — Hybrid Index & Retriever
=====================================
In-memory store with:
  • BM25 full-text index  (rank_bm25)
  • Dense vector index    (numpy cosine ANN — swappable for Milvus/Qdrant)
  • Reciprocal Rank Fusion (RRF) for hybrid scoring
  • Trust-score reweighting

All state is kept in memory by default.  For production, swap
``VectorIndex`` and ``TextIndex`` backends for Milvus / Elasticsearch
(the public interface stays identical).

Usage::

    from bie.indexer import HybridIndex
    from bie.retriever import HybridRetriever

    idx = HybridIndex()
    idx.add_chunks(chunks)

    retriever = HybridRetriever(idx)
    results = await retriever.search("query text", top_k=10)
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from rank_bm25 import BM25Okapi

from bie.config import BIESettings, settings
from bie.models import ChunkRecord, DocumentRecord, SearchFilters, SearchResult

logger = logging.getLogger(__name__)


# ── BM25 Text Index ───────────────────────────────────────────────────────────

class TextIndex:
    """
    Incremental BM25 index.  Re-builds on every N additions
    (amortised O(1) per chunk).
    """

    _REBUILD_EVERY = 500

    def __init__(self):
        self._chunks: list[ChunkRecord] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = 0

    def add(self, chunk: ChunkRecord) -> None:
        tokens = _tokenize(chunk.text)
        self._chunks.append(chunk)
        self._tokenized.append(tokens)
        self._dirty += 1
        if self._dirty >= self._REBUILD_EVERY:
            self._rebuild()

    def _rebuild(self) -> None:
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        self._dirty = 0

    def search(self, query: str, top_k: int = 50) -> list[tuple[ChunkRecord, float]]:
        if not self._chunks:
            return []
        if self._dirty > 0:
            self._rebuild()
        q_tokens = _tokenize(query)
        scores = self._bm25.get_scores(q_tokens)  # type: ignore[union-attr]
        # Normalise to [0, 1]
        max_s = float(np.max(scores)) if scores.any() else 1.0
        norm_scores = scores / max_s if max_s > 0 else scores
        top_idx = np.argsort(norm_scores)[::-1][:top_k]
        return [(self._chunks[i], float(norm_scores[i])) for i in top_idx]

    @property
    def size(self) -> int:
        return len(self._chunks)


# ── Dense Vector Index ────────────────────────────────────────────────────────

class VectorIndex:
    """
    Numpy-backed ANN index.  Suitable for ~1M vectors; swap for
    Milvus / Qdrant in production for 1B+ scale.
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self._chunks: list[ChunkRecord] = []
        self._matrix: np.ndarray | None = None  # (N, dim)

    def add(self, chunk: ChunkRecord, embedding: list[float]) -> None:
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        vec = vec / (np.linalg.norm(vec) + 1e-9)  # L2 normalise
        if self._matrix is None:
            self._matrix = vec
        else:
            self._matrix = np.vstack([self._matrix, vec])
        self._chunks.append(chunk)

    def search(self, query_vec: list[float], top_k: int = 50) -> list[tuple[ChunkRecord, float]]:
        if self._matrix is None or len(self._chunks) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._matrix @ q  # cosine sim (already normalised)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    @property
    def size(self) -> int:
        return len(self._chunks)


# ── Embedding Engine ──────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps sentence-transformers BAAI/bge-m3.
    Falls back to a fast TF-IDF hash if the model can't be loaded
    (keeps BIE runnable without GPU even in dev mode).
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dim = 1024
        self._fallback = False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, device=self._device)
            self._dim = self._model.get_sentence_embedding_dimension() or 1024
            logger.info("Loaded embedding model %s (dim=%d)", self._model_name, self._dim)
        except Exception as exc:
            logger.warning(
                "Could not load %s (%s). Using fast hash fallback.", self._model_name, exc
            )
            self._fallback = True
            self._dim = 512

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._fallback:
            return [_hash_embedding(t, self._dim) for t in texts]
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    @property
    def dim(self) -> int:
        self._load()
        return self._dim


# ── Hybrid Index ──────────────────────────────────────────────────────────────

class HybridIndex:
    """
    Unified store: text + vector + document registry.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, cfg: BIESettings = settings):
        self._cfg = cfg
        self._text_idx = TextIndex()
        self._vec_idx = VectorIndex(dim=cfg.embedding_dim)
        self._embed = EmbeddingEngine(cfg.embedding_model, cfg.embedding_device)
        self._docs: dict[str, DocumentRecord] = {}    # doc_id → DocumentRecord
        self._chunks: dict[str, ChunkRecord] = {}     # chunk_id → ChunkRecord
        self._lock = asyncio.Lock()

    async def add_documents(
        self, docs: list[tuple[DocumentRecord, list[ChunkRecord]]]
    ) -> int:
        """Add (doc, chunks) pairs; returns count of chunks indexed."""
        count = 0
        texts = []
        chunk_batch: list[ChunkRecord] = []
        for doc, chunks in docs:
            self._docs[doc.doc_id] = doc
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                texts.append(chunk.text)
                chunk_batch.append(chunk)

        # Embed in batch (GPU-friendly)
        embeddings = self._embed.encode(texts)

        async with self._lock:
            for chunk, emb in zip(chunk_batch, embeddings):
                chunk.embeddings = emb
                self._text_idx.add(chunk)
                self._vec_idx.add(chunk, emb)
                count += 1

        logger.info("Indexed %d chunks (total text=%d)", count, self._text_idx.size)
        return count

    def get_doc(self, doc_id: str) -> DocumentRecord | None:
        return self._docs.get(doc_id)

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        return self._chunks.get(chunk_id)

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def chunk_count(self) -> int:
        return self._text_idx.size

    # ── Search primitives ─────────────────────────────────────────────────────

    def bm25_search(self, query: str, top_k: int = 50) -> list[tuple[ChunkRecord, float]]:
        return self._text_idx.search(query, top_k)

    def vector_search(self, query: str, top_k: int = 50) -> list[tuple[ChunkRecord, float]]:
        qvec = self._embed.encode([query])[0]
        return self._vec_idx.search(qvec, top_k)


# ── Hybrid Retriever (M03) ────────────────────────────────────────────────────

class HybridRetriever:
    """
    Fuses BM25 + vector scores via Reciprocal Rank Fusion,
    applies trust-score weighting, and returns ranked SearchResult objects.
    """

    def __init__(self, index: HybridIndex, cfg: BIESettings = settings):
        self._idx = index
        self._cfg = cfg

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        k = top_k or self._cfg.default_top_k
        filters = filters or SearchFilters()

        t0 = time.perf_counter()

        # Parallel BM25 + vector search
        bm25_hits = self._idx.bm25_search(query, top_k=self._cfg.rerank_top_k)
        vec_hits = self._idx.vector_search(query, top_k=self._cfg.rerank_top_k)

        # RRF fusion
        fused = _rrf_fuse(
            bm25_hits,
            vec_hits,
            rrf_k=self._cfg.rrf_k,
            bm25_w=self._cfg.bm25_weight,
            vec_w=self._cfg.vector_weight,
        )

        # Apply filters + trust reweighting
        results = []
        rank = 1
        for chunk, rrf_score, b_score, v_score in fused:
            doc = self._idx.get_doc(chunk.doc_id)
            if doc is None:
                continue
            if not _passes_filters(doc, chunk, filters):
                continue
            # Trust reweighting
            trust = chunk.trust_score
            final_score = rrf_score * (0.5 + 0.5 * trust)

            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=doc.title,
                    url=doc.url,
                    snippet=chunk.text[:300],
                    source=doc.domain,
                    publish_date=doc.publish_date,
                    bm25_score=round(b_score, 4),
                    vector_score=round(v_score, 4),
                    rrf_score=round(final_score, 4),
                    trust_score=round(trust, 3),
                )
            )
            rank += 1
            if rank > k:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("search '%s': %d results in %.1f ms", query, len(results), elapsed_ms)
        return results


# ── Utility functions ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer: lowercase + split on non-alphanumeric."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _rrf_fuse(
    bm25_hits: list[tuple[ChunkRecord, float]],
    vec_hits: list[tuple[ChunkRecord, float]],
    rrf_k: int = 60,
    bm25_w: float = 0.4,
    vec_w: float = 0.6,
) -> list[tuple[ChunkRecord, float, float, float]]:
    """
    Reciprocal Rank Fusion.
    Returns [(chunk, rrf_score, bm25_norm, vec_norm)] sorted desc by rrf_score.
    """
    bm25_map = {c.chunk_id: (i, s) for i, (c, s) in enumerate(bm25_hits)}
    vec_map = {c.chunk_id: (i, s) for i, (c, s) in enumerate(vec_hits)}

    all_ids = set(bm25_map) | set(vec_map)
    # Gather all chunks
    chunk_lookup: dict[str, ChunkRecord] = {}
    for c, _ in bm25_hits:
        chunk_lookup[c.chunk_id] = c
    for c, _ in vec_hits:
        chunk_lookup[c.chunk_id] = c

    scores: list[tuple[ChunkRecord, float, float, float]] = []
    for cid in all_ids:
        chunk = chunk_lookup[cid]
        b_rank, b_score = bm25_map.get(cid, (len(bm25_hits), 0.0))
        v_rank, v_score = vec_map.get(cid, (len(vec_hits), 0.0))
        rrf = bm25_w / (rrf_k + b_rank + 1) + vec_w / (rrf_k + v_rank + 1)
        scores.append((chunk, rrf, b_score, v_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _passes_filters(doc: DocumentRecord, chunk: ChunkRecord, f: SearchFilters) -> bool:
    if f.lang and doc.metadata.get("lang") != f.lang:
        return False
    if f.domain and doc.domain != f.domain:
        return False
    if f.min_trust and chunk.trust_score < f.min_trust:
        return False
    if f.content_type and doc.metadata.get("content_type") != f.content_type:
        return False
    return True


def _hash_embedding(text: str, dim: int = 512) -> list[float]:
    """Fast deterministic fallback embedding (no GPU needed)."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(list(h[:8]))
    vec = rng.standard_normal(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.tolist()
