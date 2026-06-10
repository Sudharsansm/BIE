"""
Enterprise Authentication & Authorization
===========================================
SSO (OAuth2/OIDC), JWT session tokens, API-key tenancy, and
role-based access control (RBAC) for the v1.0 Enterprise tier.

Components:
  - ``APIKeyStore``     — per-tenant API keys with tier + quota
  - ``JWTManager``      — issue/verify short-lived session JWTs (post-SSO)
  - ``OIDCConfig``      — OIDC provider configuration (Okta/Azure AD/Google)
  - ``RBAC``            — role → permission mapping
  - ``require_role``    — FastAPI dependency factory

This module has zero hard dependency on a real IdP — ``OIDCConfig``
holds connection details, and ``verify_oidc_token`` validates tokens
issued by any standards-compliant OIDC provider via JWKS.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from jose import jwt, JWTError
from pydantic import BaseModel, Field

from bie.config import BIESettings, settings


# ── Roles & permissions (RBAC) ─────────────────────────────────────────────────

class Role(str, Enum):
    VIEWER = "viewer"          # search only
    DEVELOPER = "developer"    # search + agent + crawl
    ADMIN = "admin"            # all + indices/update + webhooks
    OWNER = "owner"            # all + billing + tenant management


_ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: {"search:read"},
    Role.DEVELOPER: {"search:read", "agent:read", "crawl:write", "feedback:write"},
    Role.ADMIN: {
        "search:read", "agent:read", "crawl:write", "feedback:write",
        "indices:write", "webhooks:write", "metrics:read", "kg:read",
    },
    Role.OWNER: {
        "search:read", "agent:read", "crawl:write", "feedback:write",
        "indices:write", "webhooks:write", "metrics:read", "kg:read",
        "tenant:manage", "billing:manage",
    },
}


class RBAC:
    @staticmethod
    def has_permission(role: Role, permission: str) -> bool:
        return permission in _ROLE_PERMISSIONS.get(role, set())

    @staticmethod
    def permissions_for(role: Role) -> set[str]:
        return _ROLE_PERMISSIONS.get(role, set())


# ── Tenant / API key model ──────────────────────────────────────────────────────

class PricingTier(str, Enum):
    FREE = "free"
    STARTUP = "startup"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


_TIER_QUOTAS: dict[PricingTier, int] = {
    PricingTier.FREE: 50_000,        # queries / month
    PricingTier.STARTUP: 1_000_000,
    PricingTier.BUSINESS: 10_000_000,
    PricingTier.ENTERPRISE: -1,      # unlimited
}


@dataclass
class Tenant:
    tenant_id: str = field(default_factory=lambda: f"tn_{uuid.uuid4().hex[:12]}")
    name: str = ""
    tier: PricingTier = PricingTier.FREE
    region: str = "us-east-1"
    sso_enabled: bool = False
    oidc_issuer: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class APIKeyRecord:
    api_key: str
    tenant_id: str
    role: Role = Role.DEVELOPER
    monthly_quota: int = -1
    requests_this_month: int = 0
    period_start: float = field(default_factory=time.time)
    active: bool = True


class APIKeyStore:
    """
    In-memory multi-tenant API key store with quota tracking.
    Swap for a Postgres/DynamoDB-backed implementation in production —
    the interface (`validate`, `record_usage`, `create_key`) stays the same.
    """

    def __init__(self):
        self._tenants: dict[str, Tenant] = {}
        self._keys: dict[str, APIKeyRecord] = {}
        self._seed_dev_key()

    def _seed_dev_key(self) -> None:
        tenant = Tenant(name="dev-tenant", tier=PricingTier.ENTERPRISE)
        self._tenants[tenant.tenant_id] = tenant
        self._keys["dev-key-12345"] = APIKeyRecord(
            api_key="dev-key-12345",
            tenant_id=tenant.tenant_id,
            role=Role.OWNER,
            monthly_quota=-1,
        )

    def create_tenant(self, name: str, tier: PricingTier, region: str = "us-east-1") -> Tenant:
        tenant = Tenant(name=name, tier=tier, region=region)
        self._tenants[tenant.tenant_id] = tenant
        return tenant

    def create_key(self, tenant_id: str, role: Role = Role.DEVELOPER) -> APIKeyRecord:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise ValueError(f"Unknown tenant {tenant_id}")
        quota = _TIER_QUOTAS[tenant.tier]
        key = APIKeyRecord(
            api_key=f"bie_{uuid.uuid4().hex}",
            tenant_id=tenant_id,
            role=role,
            monthly_quota=quota,
        )
        self._keys[key.api_key] = key
        return key

    def validate(self, api_key: str) -> tuple[APIKeyRecord, Tenant] | None:
        record = self._keys.get(api_key)
        if record is None or not record.active:
            return None
        tenant = self._tenants.get(record.tenant_id)
        if tenant is None:
            return None
        self._maybe_reset_period(record)
        return record, tenant

    def record_usage(self, api_key: str) -> bool:
        """Returns False if quota exceeded."""
        record = self._keys.get(api_key)
        if record is None:
            return False
        self._maybe_reset_period(record)
        if record.monthly_quota >= 0 and record.requests_this_month >= record.monthly_quota:
            return False
        record.requests_this_month += 1
        return True

    def _maybe_reset_period(self, record: APIKeyRecord) -> None:
        elapsed = time.time() - record.period_start
        if elapsed > 30 * 86400:  # 30-day rolling period
            record.requests_this_month = 0
            record.period_start = time.time()

    def quota_status(self, api_key: str) -> dict:
        record = self._keys.get(api_key)
        if record is None:
            return {}
        return {
            "quota": record.monthly_quota,
            "used": record.requests_this_month,
            "remaining": (
                "unlimited" if record.monthly_quota < 0
                else max(0, record.monthly_quota - record.requests_this_month)
            ),
        }


# ── JWT session management (post-SSO) ───────────────────────────────────────────

class JWTManager:
    """
    Issues and verifies short-lived JWTs after a successful SSO/OIDC
    login. Used for browser-based dashboard sessions; API traffic uses
    API keys (`APIKeyStore`).
    """

    def __init__(self, cfg: BIESettings = settings, ttl_seconds: int = 3600):
        self._secret = cfg.secret_key
        self._ttl = ttl_seconds
        self._algorithm = "HS256"

    def issue(self, subject: str, tenant_id: str, role: Role) -> str:
        now = int(time.time())
        payload = {
            "sub": subject,
            "tenant_id": tenant_id,
            "role": role.value,
            "iat": now,
            "exp": now + self._ttl,
        }
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def verify(self, token: str) -> dict | None:
        try:
            return jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except JWTError:
            return None


# ── OIDC / SSO configuration ─────────────────────────────────────────────────────

class OIDCConfig(BaseModel):
    """
    Connection details for an enterprise SSO provider
    (Okta, Azure AD, Google Workspace, OneLogin, etc).
    Tokens issued by the provider are validated via JWKS against
    `jwks_uri` — no provider-specific code required.
    """

    issuer: str
    client_id: str
    jwks_uri: str
    audience: str
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])


async def verify_oidc_token(token: str, oidc: OIDCConfig, jwks_keys: list[dict]) -> dict | None:
    """
    Verifies an OIDC ID token against the provider's JWKS.
    `jwks_keys` should be fetched from `oidc.jwks_uri` and cached
    by the caller (e.g. refreshed hourly).
    """
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        key = next((k for k in jwks_keys if k.get("kid") == kid), None)
        if key is None:
            return None
        claims = jwt.decode(
            token, key, algorithms=oidc.algorithms,
            audience=oidc.audience, issuer=oidc.issuer,
        )
        return claims
    except JWTError:
        return None
