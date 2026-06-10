"""
BIE v1.0 Configuration
========================
All settings driven by env-vars (prefix BIE_) or a .env file.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BIESettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIE_", env_file=".env", extra="ignore")

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    secret_key: str = "change-me-in-production"
    region: str = "us-east-1"
    environment: str = "production"   # production | staging | development

    # ── Crawler (Bitscrape) ───────────────────────────────────────────────────
    crawl_concurrent_requests: int = 64
    crawl_download_delay: float = 0.5
    crawl_timeout: float = 30.0
    crawl_max_retries: int = 3
    crawl_freshness_hours: int = 1

    # ── Index ─────────────────────────────────────────────────────────────────
    max_index_size: int = Field(10_000_000_000, description="10B-doc target for v1.0")
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = 10
    rerank_top_k: int = 50
    rrf_k: int = 60
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # ── Trust Engine ──────────────────────────────────────────────────────────
    trust_high_threshold: float = 0.8
    trust_low_threshold: float = 0.3

    # ── Context Builder ───────────────────────────────────────────────────────
    max_context_tokens: int = 16_000
    chunk_size: int = 512

    # ── LLM Gateway ───────────────────────────────────────────────────────────
    llm_model: str = "sudharsansm/bie_qwen_2.5_3b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600

    # ── Multi-Agent ───────────────────────────────────────────────────────────
    agent_token_budget: int = 8000
    agent_max_parallel_tasks: int = 8

    # ── Multi-Region ─────────────────────────────────────────────────────────
    shards_per_region: int = 64
    replication_enabled: bool = False   # enable when multiple regions deployed

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_free: int = 1_667        # ~50K/month ÷ 30
    rate_limit_startup: int = 33_333

    # ── Compliance ────────────────────────────────────────────────────────────
    pii_detection_enabled: bool = True
    audit_log_enabled: bool = True
    data_retention_hot_days: int = 90
    data_retention_warm_days: int = 365
    data_retention_cold_days: int = 730

    # ── Auth / SSO ────────────────────────────────────────────────────────────
    jwt_ttl_seconds: int = 3600
    sso_enabled: bool = False
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_jwks_uri: str = ""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = True   # structured JSON in production


# Singleton
settings = BIESettings()
