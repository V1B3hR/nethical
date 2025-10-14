from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Set
import fnmatch

from nethical.hooks.interfaces import CommsPolicy


class NoopCommsPolicy(CommsPolicy):
    """
    Permissive policy that always allows connections. Useful for development.
    """
    def connection_allowed(self, peer_id: str, context: Mapping[str, Any]) -> bool:
        return True

    def identities(self) -> Mapping[str, Any]:
        return {"impl": "noop"}


class MTLSCommsPolicy(CommsPolicy):
    """
    mTLS + SPIFFE-aware zero-trust policy with layered controls.

    Expected context keys (recommended by CommsPolicy protocol):
    - mtls_peer_san: str                 # mTLS SAN of peer certificate
    - spiffe_id: str                     # 'spiffe://<trust_domain>/...'
    - device_attested: bool
    - runtime_attested: bool
    - purpose: str                       # declared purpose-of-use
    - peer_roles: Iterable[str]          # caller roles
    - region: str                        # e.g., 'US', 'EU'
    - meta: Mapping[str, Any]            # extra metadata

    Enforcement order (first failure denies):
    1) Deny-list (patterns supported)
    2) Allow-list (if configured; patterns supported)
    3) SPIFFE trust domain check (if spiffe_id present)
    4) Attestation requirements
    5) Region allow-list (if configured)
    6) Purpose allow-list (if configured)
    7) Role requirements (if configured; all-of or any-of)

    Identity matching considers: peer_id, spiffe_id, and mtls_peer_san.
    Pattern examples: 'spiffe://prod.local/*', '*.svc.cluster.local', 'team:alpha/*'
    """

    def __init__(
        self,
        trust_domain: str,
        allowed_identities: Optional[Iterable[str]] = None,
        denied_identities: Optional[Iterable[str]] = None,
        *,
        allowed_regions: Optional[Iterable[str]] = None,
        allowed_purposes: Optional[Iterable[str]] = None,
        required_roles: Optional[Iterable[str]] = None,
        require_all_roles: bool = True,
        require_device_attested: bool = False,
        require_runtime_attested: bool = False,
    ) -> None:
        self.trust_domain: str = trust_domain.strip()
        self.allowed: Set[str] = _as_str_set(allowed_identities)
        self.denied: Set[str] = _as_str_set(denied_identities)

        self.allowed_regions: Set[str] = _as_str_set(allowed_regions)
        self.allowed_purposes: Set[str] = _as_str_set(allowed_purposes)
        self.required_roles: Set[str] = _as_str_set(required_roles)
        self.require_all_roles: bool = bool(require_all_roles)

        self.require_device_attested: bool = bool(require_device_attested)
        self.require_runtime_attested: bool = bool(require_runtime_attested)

    # -----------------------
    # CommsPolicy interface
    # -----------------------

    def connection_allowed(self, peer_id: str, context: Mapping[str, Any]) -> bool:
        # Candidate identities we will evaluate against allow/deny patterns
        spiffe_id = _as_opt_str(context.get("spiffe_id"))
        mtls_san = _as_opt_str(context.get("mtls_peer_san"))
        candidates = {c for c in (peer_id, spiffe_id, mtls_san) if c}

        # 1) Deny-list (patterns apply)
        if self.denied and _match_any(candidates, self.denied):
            return False

        # 2) Allow-list (if provided, at least one must match)
        if self.allowed and not _match_any(candidates, self.allowed):
            return False

        # 3) SPIFFE trust-domain check (if a spiffe_id is provided)
        if spiffe_id and not _spiffe_in_domain(spiffe_id, self.trust_domain):
            return False

        # 4) Attestation gates
        if self.require_device_attested and not bool(context.get("device_attested")):
            return False
        if self.require_runtime_attested and not bool(context.get("runtime_attested")):
            return False

        # 5) Region allow-list
        if self.allowed_regions:
            region = _as_opt_str(context.get("region"))
            if not region or region not in self.allowed_regions:
                return False

        # 6) Purpose allow-list
        if self.allowed_purposes:
            purpose = _as_opt_str(context.get("purpose"))
            if not purpose or purpose not in self.allowed_purposes:
                return False

        # 7) Role requirements
        if self.required_roles:
            roles = {str(r) for r in _as_iterable(context.get("peer_roles"))}
            if self.require_all_roles:
                if not self.required_roles.issubset(roles):
                    return False
            else:
                if not (self.required_roles & roles):
                    return False

        return True

    def identities(self) -> Mapping[str, Any]:
        return {
            "impl": "mtls",
            "trust_domain": self.trust_domain,
            "allowed_identities": sorted(self.allowed),
            "denied_identities": sorted(self.denied),
            "allowed_regions": sorted(self.allowed_regions),
            "allowed_purposes": sorted(self.allowed_purposes),
            "required_roles": sorted(self.required_roles),
            "require_all_roles": self.require_all_roles,
            "require_device_attested": self.require_device_attested,
            "require_runtime_attested": self.require_runtime_attested,
        }

    # -----------------------
    # Convenience constructor
    # -----------------------

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "MTLSCommsPolicy":
        """
        Build a policy from a dict-like config.

        Required:
        - trust_domain: str

        Optional:
        - allowed_identities: Iterable[str]
        - denied_identities: Iterable[str]
        - allowed_regions: Iterable[str]
        - allowed_purposes: Iterable[str]
        - required_roles: Iterable[str]
        - require_all_roles: bool (default True)
        - require_device_attested: bool
        - require_runtime_attested: bool
        """
        trust_domain = _as_opt_str(cfg.get("trust_domain")) or ""
        if not trust_domain:
            raise ValueError("trust_domain is required for MTLSCommsPolicy")

        return cls(
            trust_domain=trust_domain,
            allowed_identities=cfg.get("allowed_identities"),
            denied_identities=cfg.get("denied_identities"),
            allowed_regions=cfg.get("allowed_regions"),
            allowed_purposes=cfg.get("allowed_purposes"),
            required_roles=cfg.get("required_roles"),
            require_all_roles=bool(cfg.get("require_all_roles", True)),
            require_device_attested=bool(cfg.get("require_device_attested", False)),
            require_runtime_attested=bool(cfg.get("require_runtime_attested", False)),
        )


# ----------
# Utilities
# ----------

def _spiffe_in_domain(spiffe_id: str, trust_domain: str) -> bool:
    """
    Returns True if ID looks like 'spiffe://<trust_domain>/...' and matches given domain.
    """
    sid = spiffe_id.strip()
    dom = trust_domain.strip()
    if not sid.startswith("spiffe://"):
        return False
    prefix = f"spiffe://{dom}/"
    return sid.startswith(prefix)


def _match_any(candidates: Set[str], patterns: Set[str]) -> bool:
    """
    True if any candidate matches any pattern (glob-style).
    """
    for cand in candidates:
        for pat in patterns:
            if fnmatch.fnmatch(cand, pat):
                return True
    return False


def _as_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except Exception:
        return [value]


def _as_opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        return s if s else None
    except Exception:
        return None


def _as_str_set(values: Optional[Iterable[Any]]) -> Set[str]:
    if not values:
        return set()
    out: Set[str] = set()
    for v in values:
        s = _as_opt_str(v)
        if s:
            out.add(s)
    return out
