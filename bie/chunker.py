"""
Lightweight text chunker — splits cleaned document text into
paragraph/section-sized chunks for indexing (PRD Module 8: Context Builder).

No heavy NLP deps; sentence/paragraph aware, with overlap.
"""

from __future__ import annotations

import re

from bie.models import Chunk, Document

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_document(doc: Document, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split a document's text into overlapping chunks.

    Strategy:
      1. Split on paragraph boundaries.
      2. Greedily pack paragraphs into chunks up to ``chunk_size`` chars.
      3. If a single paragraph exceeds ``chunk_size``, split it by sentence.
      4. Apply a small character-overlap between consecutive chunks so
         retrieval doesn't lose context at boundaries.
    """
    text = (doc.text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sentences = _SENT_SPLIT.split(para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= chunk_size:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf:
                        units.append(buf)
                    buf = sent
            if buf:
                units.append(buf)

    chunks: list[Chunk] = []
    buf = ""
    offset = 0
    for unit in units:
        candidate = f"{buf}\n\n{unit}".strip() if buf else unit
        if len(candidate) <= chunk_size:
            buf = candidate
            continue

        if buf:
            chunks.append(_make_chunk(doc, buf, offset))
            offset += max(len(buf) - overlap, 0)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            buf = unit

    if buf:
        chunks.append(_make_chunk(doc, buf, offset))

    return chunks


def _make_chunk(doc: Document, text: str, start_offset: int) -> Chunk:
    return Chunk(
        doc_id=doc.doc_id,
        text=text,
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        metadata={"site": doc.site, "title": doc.title},
    )
