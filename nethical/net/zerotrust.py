from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Set, Tuple, List, Union
import fnmatch

from nethical.hooks.interfaces import CommsPolicy


# =======================================
# Simple, permissive policy (unchanged)
# =======================================


class NoopCommsPolicy(CommsPolicy):
    """
    Permissive policy that always allows connections. Useful for development.
    """

    __slots__ = ()

    def connection_allowed(self, peer_id: str, context: Mapping[str, Any]) -> bool:
        return True

    def identities(self) -> Mapping[str, Any]:
        return {"impl": "noop", "version": 1}


# =======================================
# Advanced mTLS + SPIFFE Zero-Trust Policy
# =======================================


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
      - region: str                        # e.g. 'US', 'EU'
      - meta: Mapping[str, Any]            # extra metadata

    Enforcement order (first failure denies):
      1) Deny-list (patterns supported)
      2) Allow-list (if configured; patterns supported)
      3) SPIFFE trust domain check (if spiffe_id present)
      4) Attestation requirements
      5) Region allow-list (if configured)
      6) Purpose allow-list (if configured)
      7) Role requirements (if configured; all-of or any-of)

    Pattern matching considers any of: peer_id, spiffe_id, mtls_peer_san.
    """

    __slots__ = (
        "trust_domain",
        "allowed",
        "denied",
        "allowed_regions",
        "allowed_purposes",
        "required_roles",
        "require_all_roles",
        "require_device_attested",
        "require_runtime_attested",
        "case_sensitive",
        "_normalized_allowed",
        "_normalized_denied",
    )

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
        case_sensitive: bool = True,
    ) -> None:
        # Normalize trust domain (strip scheme if accidentally included).
        normalized_domain = _normalize_trust_domain(trust_domain)
        if not normalized_domain:
            raise ValueError("trust_domain must be a non-empty string")
        self.trust_domain: str = normalized_domain

        # Original pattern sets (preserved for display).
        self.allowed: Set[str] = _as_str_set(allowed_identities)
        self.denied: Set[str] = _as_str_set(denied_identities)

        self.allowed_regions: Set[str] = _as_str_set(allowed_regions)
        self.allowed_purposes: Set[str] = _as_str_set(allowed_purposes)
        self.required_roles: Set[str] = _as_str_set(required_roles)
        self.require_all_roles: bool = bool(require_all_roles)

        self.require_device_attested: bool = bool(require_device_attested)
        self.require_runtime_attested: bool = bool(require_runtime_attested)

        self.case_sensitive: bool = bool(case_sensitive)

        # Pre-normalized pattern caches (lowercase if case-insensitive).
        if self.case_sensitive:
            self._normalized_allowed = self.allowed
            self._normalized_denied = self.denied
        else:
            self._normalized_allowed = {p.lower() for p in self.allowed}
            self._normalized_denied = {p.lower() for p in self.denied}

    # -----------------------
    # Public evaluation
    # -----------------------

    def evaluate(
        self, peer_id: str, context: Mapping[str, Any], *, explain: bool = False
    ) -> Union[bool, Tuple[bool, List[str]]]:
        """
        Perform the full policy evaluation. If explain=True, returns (allowed, reasons)
        with a stepwise trace; otherwise returns only a boolean.

        This is an extended interface; connection_allowed() still provides the legacy bool.
        """
        reasons: List[str] = [] if explain else None

        spiffe_id = _as_opt_str(context.get("spiffe_id"))
        mtls_san = _as_opt_str(context.get("mtls_peer_san"))
        candidates_raw = {c for c in (peer_id, spiffe_id, mtls_san) if c}

        if not self.case_sensitive:
            candidates = {c.lower() for c in candidates_raw}
        else:
            candidates = candidates_raw

        # 1) Deny patterns
        if self.denied:
            if _match_any_globs(
                candidates,
                self._normalized_denied,
                case_sensitive=self.case_sensitive,
            ):
                if explain:
                    reasons.append("DENY: Matched denied identity pattern.")
                    return False, reasons
                return False

        # 2) Allow patterns (if provided, require at least one match)
        if self.allowed:
            if not _match_any_globs(
                candidates,
                self._normalized_allowed,
                case_sensitive=self.case_sensitive,
            ):
                if explain:
                    reasons.append(
                        "DENY: No candidate matched any allowed identity pattern."
                    )
                    return False, reasons
                return False

        # 3) SPIFFE trust domain (if present)
        if spiffe_id:
            peer_domain = _spiffe_trust_domain(spiffe_id)
            if not peer_domain:
                if explain:
                    reasons.append("DENY: Invalid SPIFFE ID format.")
                    return False, reasons
                return False
            if peer_domain != self.trust_domain:
                if explain:
                    reasons.append(
                        f"DENY: SPIFFE trust domain mismatch (got '{peer_domain}', expected '{self.trust_domain}')."
                    )
                    return False, reasons
                return False

        # 4) Attestation requirements
        if self.require_device_attested and not bool(context.get("device_attested")):
            if explain:
                reasons.append("DENY: Device attestation required but not present.")
                return False, reasons
            return False
        if self.require_runtime_attested and not bool(context.get("runtime_attested")):
            if explain:
                reasons.append("DENY: Runtime attestation required but not present.")
                return False, reasons
            return False

        # 5) Region allow-list
        if self.allowed_regions:
            region = _as_opt_str(context.get("region"))
            if not region or region not in self.allowed_regions:
                if explain:
                    reasons.append(
                        f"DENY: Region '{region or '∅'}' not in allowed regions {sorted(self.allowed_regions)}."
                    )
                    return False, reasons
                return False

        # 6) Purpose allow-list
        if self.allowed_purposes:
            purpose = _as_opt_str(context.get("purpose"))
            if not purpose or purpose not in self.allowed_purposes:
                if explain:
                    reasons.append(
                        f"DENY: Purpose '{purpose or '∅'}' not in allowed purposes {sorted(self.allowed_purposes)}."
                    )
                    return False, reasons
                return False

        # 7) Role requirements
        if self.required_roles:
            roles = {str(r) for r in _as_iterable(context.get("peer_roles"))}
            if self.require_all_roles:
                if not self.required_roles.issubset(roles):
                    if explain:
                        missing = sorted(self.required_roles - roles)
                        reasons.append(
                            f"DENY: Missing required roles (all-of) {missing}."
                        )
                        return False, reasons
                    return False
            else:
                if not (self.required_roles & roles):
                    if explain:
                        reasons.append(
                            f"DENY: None of required roles (any-of) {sorted(self.required_roles)} present."
                        )
                        return False, reasons
                    return False

        if explain:
            reasons.append("ALLOW: All policy checks passed.")
            return True, reasons
        return True

    # Legacy interface
    def connection_allowed(self, peer_id: str, context: Mapping[str, Any]) -> bool:
        result = self.evaluate(peer_id, context, explain=False)
        assert isinstance(result, bool)
        return result

    def identities(self) -> Mapping[str, Any]:
        return {
            "impl": "mtls",
            "version": 2,
            "trust_domain": self.trust_domain,
            "allowed_identities": sorted(self.allowed),
            "denied_identities": sorted(self.denied),
            "allowed_regions": sorted(self.allowed_regions),
            "allowed_purposes": sorted(self.allowed_purposes),
            "required_roles": sorted(self.required_roles),
            "require_all_roles": self.require_all_roles,
            "require_device_attested": self.require_device_attested,
            "require_runtime_attested": self.require_runtime_attested,
            "case_sensitive": self.case_sensitive,
        }

    def __repr__(self) -> str:
        return (
            f"MTLSCommsPolicy(trust_domain={self.trust_domain!r}, "
            f"allowed={len(self.allowed)}, denied={len(self.denied)}, "
            f"regions={len(self.allowed_regions)}, purposes={len(self.allowed_purposes)}, "
            f"roles={len(self.required_roles)}, all_roles={self.require_all_roles}, "
            f"device_attest={self.require_device_attested}, runtime_attest={self.require_runtime_attested}, "
            f"case_sensitive={self.case_sensitive})"
        )

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
          - case_sensitive: bool (default True)
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
            case_sensitive=bool(cfg.get("case_sensitive", True)),
        )


# ----------
# Utilities
# ----------


def _normalize_trust_domain(raw: str) -> str:
    """
    Accepts forms like:
      'prod.example' -> 'prod.example'
      'spiffe://prod.example/' -> 'prod.example'
      'spiffe://prod.example/ns/team' -> 'prod.example'
    Returns lowercase trust domain or '' if invalid.
    """
    if not raw:
        return ""
    s = raw.strip()
    if s.startswith("spiffe://"):
        s = s[len("spiffe://") :]
        # cut at first slash
        if "/" in s:
            s = s.split("/", 1)[0]
    s = s.strip().lower()
    return s


def _spiffe_trust_domain(spiffe_id: str) -> Optional[str]:
    """
    Extract the trust domain from a SPIFFE ID of form 'spiffe://domain/path'.
    Returns None if malformed.
    """
    sid = spiffe_id.strip()
    if not sid.startswith("spiffe://"):
        return None
    remainder = sid[len("spiffe://") :]
    if not remainder:
        return None
    # domain ends at first '/'
    if "/" in remainder:
        dom = remainder.split("/", 1)[0]
    else:
        # SPIFFE spec requires a path after domain; treat missing slash as invalid.
        return None
    dom = dom.strip().lower()
    return dom or None


def _match_any_globs(
    candidates: Set[str],
    patterns: Set[str],
    *,
    case_sensitive: bool = True,
) -> bool:
    """
    Returns True if any candidate matches any pattern (glob-style).
    Performs case normalization if case_sensitive=False.
    Assumes candidates and patterns are already normalized if case_insensitive path chosen
    (i.e., callers pre-lowered them).
    """
    if not candidates or not patterns:
        return False

    if case_sensitive:
        for cand in candidates:
            for pat in patterns:
                if fnmatch.fnmatch(cand, pat):
                    return True
        return False
    else:
        # Both sets assumed lowercase; simplest approach: still use fnmatch (case-insensitive by data).
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
