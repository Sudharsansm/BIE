"""
Multi-Region Support
=====================
Region registry, geo-routing, and cross-region index replication
hooks for the v1.0 "Multi-region, 10B doc index" target.

This module doesn't run actual cross-datacenter infrastructure (that's
Kubernetes/Istio's job per the PRD tech stack) — it provides the
application-level primitives:

  - ``RegionRegistry``     — known regions + their endpoints/capacity
  - ``GeoRouter``          — picks the nearest/healthiest region for a request
  - ``ReplicationManager``  — async fan-out of index writes to replicas
  - ``ShardRouter``        — consistent-hash routing for the 10B-doc index
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from bie.config import BIESettings, settings

logger = logging.getLogger(__name__)


# ── Region definitions ──────────────────────────────────────────────────────────

class RegionStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class Region:
    region_id: str            # e.g. "us-east-1"
    name: str                 # e.g. "US East (N. Virginia)"
    endpoint: str             # base URL of the regional API
    is_primary: bool = False
    status: RegionStatus = RegionStatus.HEALTHY
    last_health_check: float = field(default_factory=time.time)
    avg_latency_ms: float = 50.0
    shard_capacity: int = 2_500_000_000  # docs per region (10B / 4 regions default)
    shard_count: int = 0


_DEFAULT_REGIONS = [
    Region("us-east-1", "US East (N. Virginia)", "https://us-east-1.bie.example.com", is_primary=True),
    Region("eu-west-1", "EU West (Ireland)", "https://eu-west-1.bie.example.com"),
    Region("ap-southeast-1", "Asia Pacific (Singapore)", "https://ap-southeast-1.bie.example.com"),
    Region("ap-south-1", "Asia Pacific (Mumbai)", "https://ap-south-1.bie.example.com"),
]


class RegionRegistry:
    """Registry of all deployment regions and their health/capacity."""

    def __init__(self, regions: list[Region] | None = None):
        self._regions: dict[str, Region] = {
            r.region_id: r for r in (regions or _DEFAULT_REGIONS)
        }

    def all(self) -> list[Region]:
        return list(self._regions.values())

    def get(self, region_id: str) -> Region | None:
        return self._regions.get(region_id)

    def healthy(self) -> list[Region]:
        return [r for r in self._regions.values() if r.status == RegionStatus.HEALTHY]

    def primary(self) -> Region:
        return next((r for r in self._regions.values() if r.is_primary), self.all()[0])

    def update_health(self, region_id: str, status: RegionStatus, latency_ms: float | None = None) -> None:
        region = self._regions.get(region_id)
        if region is None:
            return
        region.status = status
        region.last_health_check = time.time()
        if latency_ms is not None:
            region.avg_latency_ms = 0.7 * region.avg_latency_ms + 0.3 * latency_ms

    def total_capacity(self) -> int:
        return sum(r.shard_capacity for r in self._regions.values())

    def total_docs(self) -> int:
        return sum(r.shard_count for r in self._regions.values())

    def utilization(self) -> dict[str, float]:
        return {
            r.region_id: round(r.shard_count / r.shard_capacity, 4) if r.shard_capacity else 0.0
            for r in self._regions.values()
        }


# ── Geo-routing ────────────────────────────────────────────────────────────────

# Approximate region geo-coordinates for nearest-region routing
_REGION_COORDS: dict[str, tuple[float, float]] = {
    "us-east-1": (38.13, -78.45),
    "eu-west-1": (53.41, -8.24),
    "ap-southeast-1": (1.35, 103.82),
    "ap-south-1": (19.08, 72.88),
}


class GeoRouter:
    """
    Routes a request to the nearest healthy region based on the
    client's approximate lat/lon (e.g. derived from IP geolocation
    at the edge / CloudFront).
    """

    def __init__(self, registry: RegionRegistry):
        self._registry = registry

    def route(self, client_lat: float | None = None, client_lon: float | None = None) -> Region:
        healthy = self._registry.healthy()
        if not healthy:
            return self._registry.primary()

        if client_lat is None or client_lon is None:
            # No geo info — pick lowest-latency healthy region
            return min(healthy, key=lambda r: r.avg_latency_ms)

        def distance(region: Region) -> float:
            lat, lon = _REGION_COORDS.get(region.region_id, (0.0, 0.0))
            return ((lat - client_lat) ** 2 + (lon - client_lon) ** 2) ** 0.5

        return min(healthy, key=distance)

    def failover(self, failed_region_id: str) -> Region:
        """Returns the next-best healthy region when `failed_region_id` is down."""
        self._registry.update_health(failed_region_id, RegionStatus.DOWN)
        healthy = self._registry.healthy()
        if not healthy:
            return self._registry.primary()
        return min(healthy, key=lambda r: r.avg_latency_ms)


# ── Shard routing (for 10B-doc scale) ───────────────────────────────────────────

class ShardRouter:
    """
    Consistent-hash routing of documents to (region, shard) pairs.
    At 10B documents, each region holds ~2.5B docs split across
    `shards_per_region` logical shards (mapped to Elasticsearch/Milvus
    index aliases in production).
    """

    def __init__(self, registry: RegionRegistry, shards_per_region: int = 64):
        self._registry = registry
        self._shards_per_region = shards_per_region

    def route(self, doc_id: str) -> tuple[str, int]:
        """Returns (region_id, shard_index) for a given doc_id."""
        h = int(hashlib.md5(doc_id.encode(), usedforsecurity=False).hexdigest(), 16)
        regions = self._registry.all()
        if not regions:
            raise RuntimeError("No regions configured")
        region = regions[h % len(regions)]
        shard = (h // len(regions)) % self._shards_per_region
        return region.region_id, shard

    def index_name(self, doc_id: str, base_name: str = "bie-docs") -> str:
        region_id, shard = self.route(doc_id)
        return f"{base_name}-{region_id}-shard{shard:03d}"


# ── Replication ───────────────────────────────────────────────────────────────

class ReplicationManager:
    """
    Fans out index writes to replica regions asynchronously.
    Primary write succeeds synchronously; replication to other
    regions happens in the background (eventual consistency),
    matching the v1.0 multi-region target.
    """

    def __init__(self, registry: RegionRegistry, cfg: BIESettings = settings):
        self._registry = registry
        self._cfg = cfg
        self._replication_lag_ms: dict[str, float] = {}

    async def write(self, doc_ids: list[str], primary_writer, replica_writers: dict[str, callable] | None = None) -> dict:
        """
        `primary_writer` — async callable performing the synchronous write.
        `replica_writers` — optional {region_id: async callable} for replicas.
        Returns status including which regions were replicated.
        """
        t0 = time.perf_counter()
        await primary_writer(doc_ids)
        primary_ms = (time.perf_counter() - t0) * 1000

        replicated_to: list[str] = []
        if replica_writers:
            async def _replicate(region_id: str, writer):
                t1 = time.perf_counter()
                try:
                    await writer(doc_ids)
                    self._replication_lag_ms[region_id] = (time.perf_counter() - t1) * 1000
                    replicated_to.append(region_id)
                except Exception as exc:
                    logger.warning("Replication to %s failed: %s", region_id, exc)

            await asyncio.gather(*[_replicate(rid, w) for rid, w in replica_writers.items()])

        return {
            "doc_count": len(doc_ids),
            "primary_write_ms": round(primary_ms, 1),
            "replicated_to": replicated_to,
            "replication_lag_ms": dict(self._replication_lag_ms),
        }

    def status(self) -> dict:
        return {
            "regions": [
                {
                    "region_id": r.region_id,
                    "status": r.status.value,
                    "avg_latency_ms": round(r.avg_latency_ms, 1),
                    "shard_count": r.shard_count,
                    "capacity": r.shard_capacity,
                }
                for r in self._registry.all()
            ],
            "total_docs": self._registry.total_docs(),
            "total_capacity": self._registry.total_capacity(),
            "replication_lag_ms": dict(self._replication_lag_ms),
        }
