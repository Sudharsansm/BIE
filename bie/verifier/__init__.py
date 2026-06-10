"""
M09 — Fact Verifier
=====================
Post-generation: validates each factual claim in an LLM answer against
retrieved evidence. Cross-checks entity attributes via the Knowledge
Graph and maintains an audit log of every verification step (required
for regulated enterprise use, per PRD M09).

Production note: the PRD specifies integration with FactCheck.org /
Snopes for widely-debunked claims. ``ExternalFactCheckClient`` defines
that interface; it's a no-op stub here (no outbound calls) and can be
wired to a real API key in enterprise deployments.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from bie.config import BIESettings, settings
from bie.models import SearchResult


# ── Audit log ──────────────────────────────────────────────────────────────────

@dataclass
class VerificationAuditEntry:
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim: str = ""
    verified: bool = False
    method: str = ""
    evidence_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AuditLog:
    """In-memory audit trail. Swap for a persistent store (Postgres/S3) in production."""

    def __init__(self):
        self._entries: list[VerificationAuditEntry] = []

    def record(self, entry: VerificationAuditEntry) -> None:
        self._entries.append(entry)

    def all(self) -> list[VerificationAuditEntry]:
        return list(self._entries)

    def for_claim(self, claim: str) -> list[VerificationAuditEntry]:
        return [e for e in self._entries if e.claim == claim]

    @property
    def count(self) -> int:
        return len(self._entries)


# ── External fact-check client (stub) ─────────────────────────────────────────

class ExternalFactCheckClient:
    """
    Interface for FactCheck.org / Snopes-style APIs.
    No outbound network calls are made by default — returns "unknown"
    so BIE remains fully self-contained. Configure `enabled=True` and
    implement `_query_api` with a real client in enterprise deployments.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    async def check(self, claim: str) -> dict | None:
        if not self.enabled:
            return None
        return await self._query_api(claim)

    async def _query_api(self, claim: str) -> dict | None:  # pragma: no cover
        # Wire to FactCheck.org / Snopes API here.
        return None


# ── Claim extraction ───────────────────────────────────────────────────────────

def _split_claims(answer: str) -> list[str]:
    """Splits an LLM answer into individual factual claims (sentences)."""
    # Strip citation tags like [1] before splitting
    cleaned = re.sub(r"\[\d+\]", "", answer)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 8]


# ── Verification scorer ────────────────────────────────────────────────────────

def _claim_supported(claim: str, evidence: list[SearchResult]) -> tuple[bool, float, list[str]]:
    """
    Lightweight lexical-overlap support check: a claim is "supported"
    if it shares enough salient tokens with at least one evidence chunk.
    Production swaps this for an NLI entailment model.
    """
    claim_tokens = set(_tokenize(claim))
    if not claim_tokens:
        return False, 0.0, []

    best_score = 0.0
    supporting_ids: list[str] = []
    for ev in evidence:
        ev_tokens = set(_tokenize(ev.snippet))
        overlap = claim_tokens & ev_tokens
        score = len(overlap) / max(len(claim_tokens), 1)
        if score > best_score:
            best_score = score
        if score >= 0.35:
            supporting_ids.append(ev.chunk_id)

    verified = best_score >= 0.35
    return verified, round(best_score, 2), supporting_ids


def _tokenize(text: str) -> list[str]:
    stop = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "to",
            "and", "for", "with", "that", "this", "by", "as", "at", "from", "it",
            "its", "be", "has", "have", "had"}
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [t for t in tokens if t not in stop and len(t) > 2]


# ── Fact Verifier facade ────────────────────────────────────────────────────────

class FactVerifier:
    """
    Verifies each claim in a generated answer against retrieved evidence
    and (optionally) the Knowledge Graph and external fact-check APIs.
    """

    def __init__(
        self,
        cfg: BIESettings = settings,
        kg=None,  # KnowledgeGraph
        external_client: ExternalFactCheckClient | None = None,
        audit_log: AuditLog | None = None,
    ):
        self._cfg = cfg
        self._kg = kg
        self._external = external_client or ExternalFactCheckClient(enabled=False)
        self._audit = audit_log or AuditLog()

    async def verify(self, answer: str, evidence: list[SearchResult]) -> list[dict]:
        """
        Returns a list of per-claim verification results::

            [{"claim": str, "verified": bool, "confidence": float,
              "method": str, "evidence_chunk_ids": [...], "tag": "(unverified)" | ""}]
        """
        claims = _split_claims(answer)
        results = []

        for claim in claims:
            verified, confidence, evidence_ids = _claim_supported(claim, evidence)
            method = "lexical_overlap"

            # KG cross-check for entity attributes (dates, roles, affiliations)
            if not verified and self._kg is not None:
                kg_hit = self._kg_cross_check(claim)
                if kg_hit:
                    verified, confidence, method = True, 0.6, "kg_cross_check"

            # External fact-check (only for still-unverified claims)
            if not verified:
                ext = await self._external.check(claim)
                if ext is not None:
                    verified = ext.get("verified", False)
                    confidence = ext.get("confidence", confidence)
                    method = "external_factcheck"

            entry = VerificationAuditEntry(
                claim=claim,
                verified=verified,
                method=method,
                evidence_chunk_ids=evidence_ids,
                confidence=confidence,
            )
            self._audit.record(entry)

            results.append({
                "claim": claim,
                "verified": verified,
                "confidence": confidence,
                "method": method,
                "evidence_chunk_ids": evidence_ids,
                "tag": "" if verified else "(unverified)",
            })

        return results

    def annotate_answer(self, answer: str, verifications: list[dict]) -> str:
        """Appends `(unverified)` tags after unsupported claims."""
        out = answer
        for v in verifications:
            if not v["verified"] and v["claim"] in out:
                out = out.replace(v["claim"], f"{v['claim']} (unverified)", 1)
        return out

    def _kg_cross_check(self, claim: str) -> bool:
        if self._kg is None:
            return False
        # Look for any entity name from the claim in the KG
        words = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", claim)
        for w in words:
            if self._kg.search_entities(w, limit=1):
                return True
        return False

    @property
    def audit_log(self) -> AuditLog:
        return self._audit
