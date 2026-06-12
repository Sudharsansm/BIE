"""
BIE configuration.

All settings can be overridden via environment variables prefixed with
``BIE_`` (e.g. ``BIE_MAX_PAGES=200``) or passed directly to
``BIESettings(...)``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BIESettings(BaseSettings):
    # --- Crawl behaviour (delegated to Bitscrape) -----------------------
    max_pages: int = Field(40, ge=1, description="Max pages to crawl per source URL")
    max_depth: int = Field(2, ge=0, description="Max link-follow depth")
    concurrent_requests: int = Field(16, ge=1, le=256)
    download_delay: float = Field(0.0, ge=0.0)
    user_agent: str = "BIE/0.1 (+https://github.com/Sudharsansm/BIE) bitscrape"
    robotstxt_obey: bool = True
    request_timeout: float = Field(20.0, ge=1.0)
    use_playwright: bool = False

    # --- Indexing / retrieval --------------------------------------------
    chunk_size: int = Field(800, ge=100, description="Approx characters per chunk")
    chunk_overlap: int = Field(100, ge=0)
    use_embeddings: bool = Field(
        True,
        description="Enable semantic (vector) search via sentence-transformers. "
        "Falls back to BM25-only if the model can't be loaded.",
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_weight: float = Field(0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(0.5, ge=0.0, le=1.0)

    # --- Storage -----------------------------------------------------------
    index_dir: str = Field(".bie_index", description="Directory for persisted index")
    persist: bool = Field(False, description="Persist index to disk between runs")

    # --- Server --------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = Field(
        default=None,
        description="If set, all /search and /crawl endpoints require "
        "an `Authorization: Bearer <key>` header.",
    )

    model_config = SettingsConfigDict(
        env_prefix="BIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
