"""
M06 — Contradiction Detector
==============================
NLI-based cross-source conflict detection. Compares top-K result pairs
and flags statements that entail / contradict one another.

Production swaps `_NLIScorer` for a fine-tuned DeBERTa-v3-MNLI model
(per PRD M06). The lightweight scorer here uses negation-aware lexical
overlap heuristics so BIE runs without a GPU/transformers dependency.

Usage::

    from bie.contradiction import ContradictionDetector

    detector = ContradictionDetector()
    flags = detector.detect(results)  # list[ContradictionFlag]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from bie.config import BIESettings, settings
from bie.models import SearchResult


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ContradictionFlag:
    chunk_id_a: str
    chunk_id_b: str
    source_a: str
    source_b: str
    statement_a: str
    statement_b: str
    confidence: float
    explanation: str
    conflicting_terms: list[str] = field(default_factory=list)


# ── Negation / numeric conflict heuristics ─────────────────────────────────────

_NEGATIONS = {"not", "no", "never", "n't", "without", "lacks", "fails to"}

_ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"increase", "increased", "rose", "rising", "grew", "growth", "up", "higher", "surge", "surged"},
     {"decrease", "decreased", "fell", "falling", "declined", "decline", "down", "lower", "drop", "dropped"}),
    ({"approved", "approves", "confirmed", "confirms"},
     {"rejected", "rejects", "denied", "denies"}),
    ({"safe", "secure", "reliable"}, {"unsafe", "insecure", "risky", "vulnerable"}),
    ({"profit", "profitable", "profits"}, {"loss", "losses", "unprofitable"}),
    ({"banned", "prohibited", "illegal"}, {"legal", "permitted", "allowed"}),
]

_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\s*(?:%|percent|million|billion|thousand|k|m|b)?\b", re.I)


class _NLIScorer:
    """
    Lightweight NLI heuristic scorer.
    Returns a contradiction probability in [0, 1] and the conflicting
    term pairs found, given two statements about (likely) the same topic.
    """

    def score(self, text_a: str, text_b: str) -> tuple[float, list[str]]:
        a_tokens = set(_tokenize(text_a))
        b_tokens = set(_tokenize(text_b))

        # Require meaningful topical overlap before flagging
        overlap = a_tokens & b_tokens
        overlap_ratio = len(overlap) / max(min(len(a_tokens), len(b_tokens)), 1)
        if overlap_ratio < 0.15:
            return 0.0, []

        conflicts: list[str] = []
        score = 0.0

        # 1) Antonym pair detection
        for set_x, set_y in _ANTONYM_PAIRS:
            x_in_a, y_in_a = bool(a_tokens & set_x), bool(a_tokens & set_y)
            x_in_b, y_in_b = bool(b_tokens & set_x), bool(b_tokens & set_y)
            if (x_in_a and y_in_b) or (y_in_a and x_in_b):
                score = max(score, 0.75)
                conflicts.extend(sorted((a_tokens | b_tokens) & (set_x | set_y)))

        # 2) Negation asymmetry on shared subject terms
        a_neg = bool(a_tokens & _NEGATIONS)
        b_neg = bool(b_tokens & _NEGATIONS)
        if a_neg != b_neg and overlap_ratio > 0.3:
            score = max(score, 0.55)
            conflicts.append("negation-mismatch")

        # 3) Conflicting numeric values for the same topic
        nums_a = _NUMBER_RE.findall(text_a)
        nums_b = _NUMBER_RE.findall(text_b)
        if nums_a and nums_b and overlap_ratio > 0.25:
            norm_a = {_norm_num(n) for n in nums_a}
            norm_b = {_norm_num(n) for n in nums_b}
            if norm_a and norm_b and not (norm_a & norm_b):
                # Different numbers cited for an overlapping claim
                score = max(score, 0.45)
                conflicts.append("numeric-mismatch")

        return round(score, 2), conflicts


# ── Main detector ──────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Compares all pairs in the top-K result set and emits
    `ContradictionFlag`s for pairs whose contradiction score
    exceeds the configured threshold.
    """

    def __init__(self, cfg: BIESettings = settings, threshold: float = 0.45):
        self._cfg = cfg
        self._threshold = threshold
        self._scorer = _NLIScorer()

    def detect(self, results: list[SearchResult], max_pairs: int = 50) -> list[ContradictionFlag]:
        flags: list[ContradictionFlag] = []
        pairs = list(combinations(results, 2))[:max_pairs]

        for a, b in pairs:
            # Skip same-source comparisons (internal consistency assumed)
            if a.source == b.source:
                continue

            score, conflicts = self._scorer.score(a.snippet, b.snippet)
            if score >= self._threshold:
                flags.append(
                    ContradictionFlag(
                        chunk_id_a=a.chunk_id,
                        chunk_id_b=b.chunk_id,
                        source_a=a.source,
                        source_b=b.source,
                        statement_a=a.snippet[:200],
                        statement_b=b.snippet[:200],
                        confidence=score,
                        explanation=_build_explanation(conflicts, a.source, b.source),
                        conflicting_terms=conflicts,
                    )
                )
        return flags

    def verify_answer(self, answer: str, results: list[SearchResult]) -> list[ContradictionFlag]:
        """
        Post-generation check: does the generated answer contradict
        any retrieved evidence?
        """
        flags: list[ContradictionFlag] = []
        for r in results:
            score, conflicts = self._scorer.score(answer, r.snippet)
            if score >= self._threshold:
                flags.append(
                    ContradictionFlag(
                        chunk_id_a="generated_answer",
                        chunk_id_b=r.chunk_id,
                        source_a="LLM Answer",
                        source_b=r.source,
                        statement_a=answer[:200],
                        statement_b=r.snippet[:200],
                        confidence=score,
                        explanation=_build_explanation(conflicts, "generated answer", r.source),
                        conflicting_terms=conflicts,
                    )
                )
        return flags


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _norm_num(raw: str) -> str:
    """Normalise '23.4 billion' / '23,400 million' to a comparable bucket."""
    s = raw.lower().replace(",", "").strip()
    m = re.match(r"([\d.]+)\s*(billion|million|thousand|k|m|b|%|percent)?", s)
    if not m:
        return s
    num = float(m.group(1))
    unit = m.group(2) or ""
    multiplier = {
        "billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3, "%": 1, "percent": 1,
    }.get(unit, 1)
    return f"{num * multiplier:.0f}"


def _build_explanation(conflicts: list[str], source_a: str, source_b: str) -> str:
    if not conflicts:
        return f"{source_a} and {source_b} appear to disagree on this topic."
    if "negation-mismatch" in conflicts:
        return f"{source_a} and {source_b} differ on whether this claim holds (negation mismatch)."
    if "numeric-mismatch" in conflicts:
        return f"{source_a} and {source_b} cite different figures for the same metric."
    terms = ", ".join(t for t in conflicts if t not in ("negation-mismatch", "numeric-mismatch"))
    return f"{source_a} and {source_b} use conflicting terms: {terms}."
