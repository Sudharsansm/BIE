"""
BitSearch Intelligence Engine (BIE) v1.0
==========================================
AI-native real-time retrieval — Bitscrape-powered.
Multi-region, 10B-doc index, SOC 2 compliant.
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Sudharsansm"

from bie.client import BIEClient
from bie.config import BIESettings
from bie.models import (
    ChunkRecord, DocumentRecord,
    SearchRequest, SearchResponse, SearchResult,
    AgentResponse, Citation,
)

__all__ = [
    "BIEClient", "BIESettings",
    "DocumentRecord", "ChunkRecord",
    "SearchRequest", "SearchResponse", "SearchResult",
    "AgentResponse", "Citation",
    "__version__",
]
