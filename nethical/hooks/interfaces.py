from __future__ import annotations

"""
Nethical Hooks — Interfaces

This module defines narrow, security-focused extension points for attestation,
cryptographic role signaling, communications policy, geofencing, offline
persistence, and jurisdiction-aware authorization.

Design goals:
- Minimal, explicit, and composable interfaces
- Structural typing with runtime checkability for plugin validation
- Immutability and stable return shapes
- Compatibility with high-assurance and zero-trust environments
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    runtime_checkable,
    NewType,
)

# --------- Type aliases for security-critical identifiers ---------

PrincipalId = NewType("PrincipalId", str)
PeerId = NewType("PeerId", str)
RoleName = NewType("RoleName", str)
FeatureName = NewType("FeatureName", str)
TokenStr = NewType("TokenStr", str)
EventId = NewType("EventId", str)
ZoneName = NewType("ZoneName", str)

# --------- Enumerations ---------


class Region(str, Enum):
    """
    Jurisdictional regions that can impact allowed features, roles, and policy.
    Note: NATO is included for defensive/dual-use profiles; healthcare defaults
    should keep this inert unless explicitly enabled.
    """

    US = "US"
    UK = "UK"
    EU = "EU"
    NATO = "NATO"  # NATO profile hooks; healthcare defaults keep this inert


class Purpose(str, Enum):
    """
    Purpose-of-use categories commonly referenced in healthcare and regulated
    data handling contexts. Extend as needed.
    """

    TREATMENT = "treatment"
    PAYMENT = "payment"
    OPERATIONS = "operations"
    TESTING = "testing"
    RESEARCH = "research"
    OTHER = "other"


# --------- Structured payloads ---------


class Evidence(TypedDict, total=False):
    """
    Structured evidence accompanying attestation results.

    Fields are intentionally optional to allow different attesters to provide
    only the relevant subset.
    """

    runtime_quote: str  # e.g., TEE quote, VBS enclave report
    tcb_version: str  # Trusted Computing Base version
    cert_chain_pem: str  # PEM-encoded verifier/attester chain
    verifier: str  # Name/version of verifier used
    measurements: Dict[str, str]  # Component -> hash/measurement
    meta: Dict[str, Any]  # Free-form metadata (stable keys preferred)


class TokenMeta(TypedDict, total=False):
    """
    Metadata attached to issued or verified role tokens.
    """

    issuer: str
    subject: str
    audience: str
    issued_at: float  # epoch seconds
    not_before: float
    expires_at: float
    key_id: str
    algorithm: str
    meta: Dict[str, Any]


class OfflineEvent(TypedDict):
    """
    Append-only event record used during contested/offline operation.
    """

    type: str
    ts: float  # epoch seconds (UTC)
    payload: Dict[str, Any]
    meta: Dict[str, Any]


class OfflineSnapshot(TypedDict, total=False):
    """
    Snapshot of the offline store for auditing or reconciliation.
    """

    events: Iterable[OfflineEvent]
    last_event_id: str
    meta: Dict[str, Any]


# --------- Results ---------


@dataclass(frozen=True, slots=True)
class AttestationResult:
    """
    Result of an attestation operation.

    ok: True if attestation passes policy.
    evidence: Structured evidence documenting the decision.
    reason: Human-readable explanation when ok is False.
    code: Optional machine-readable error code for automated responses.
    created_at: UTC timestamp of when this result was produced.
    """

    ok: bool
    evidence: Evidence
    reason: Optional[str] = None
    code: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class SignalResult:
    """
    Result of issuing or verifying a cryptographic role signal.

    ok: True if issuance/verification succeeded and token is usable.
    token: The issued or validated token (opaque to callers).
    meta: Parsed or attached token metadata.
    reason: Human-readable explanation when ok is False.
    code: Optional machine-readable error code for automated responses.
    created_at: UTC timestamp of when this result was produced.
    """

    ok: bool
    token: Optional[TokenStr] = None
    meta: Optional[TokenMeta] = None
    reason: Optional[str] = None
    code: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# --------- Interfaces (Protocols) ---------


@runtime_checkable
class AttestationProvider(Protocol):
    """
    Provides runtime and hardware attestation signals suitable for zero-trust
    admission decisions and continuous verification.
    """

    def attest_runtime(self) -> AttestationResult:
        """
        Attest the executing runtime (e.g., container, VM, enclave).
        Must be side-effect free and fast enough for on-connection checks.
        """
        ...

    def attest_hardware(self) -> AttestationResult:
        """
        Attest the underlying hardware or TEE/SEV/TDX/SGX environment.
        Should include sufficient evidence for offline audit.
        """
        ...


@runtime_checkable
class CryptoSignalProvider(Protocol):
    """
    Defensive-friendly authenticated role signaling.
    Disabled by default unless explicitly configured.
    """

    def issue_medical_role_token(
        self, role: RoleName, ttl_seconds: int = 60
    ) -> SignalResult:
        """
        Issue a short-lived, signed token asserting a 'medical' role for
        purpose-of-use enforcement. Implementations should constrain TTL,
        audience, and binding to the current principal context.
        """
        ...

    def verify_peer_role_token(self, token: TokenStr) -> SignalResult:
        """
        Verify a peer's role token, returning parsed metadata when valid.
        Implementations should enforce audience, expiration, and binding.
        """
        ...


@runtime_checkable
class CommsPolicy(Protocol):
    """
    Zero-trust communications policy (mTLS, identity enforcement, allowlists).
    Implementations should make deterministic decisions based on peer identity
    and provided context.

    Expected context keys (recommendation; not enforced by type):
    - "mtls_peer_san": str
    - "spiffe_id": str
    - "device_attested": bool
    - "runtime_attested": bool
    - "purpose": str|Purpose
    - "peer_roles": Iterable[str]
    - "region": str|Region
    - "meta": Mapping[str, Any]
    """

    def connection_allowed(self, peer_id: PeerId, context: Mapping[str, Any]) -> bool:
        """
        Return True if the connection from peer_id is allowed under the
        provided context; False otherwise. Must be side-effect free.
        """
        ...

    def identities(self) -> Mapping[str, Any]:
        """
        Return identity material (e.g., cert/key references, trust roots,
        allowlist entries) used by this policy.
        """
        ...


@runtime_checkable
class GeoFenceProvider(Protocol):
    """
    Geofencing advisory for location-bound operations.
    """

    def allowed(self, lat: float, lon: float, purpose: str) -> bool:
        """
        Return True if the given coordinates are acceptable for the stated
        purpose. Implementations should account for hysteresis and accuracy.
        """
        ...

    def current_zone(self) -> Optional[ZoneName]:
        """
        Return the current geofence zone name if known (e.g., 'EU', 'US-CA').
        """
        ...


@runtime_checkable
class OfflineStore(Protocol):
    """
    Tamper-evident, append-only store for contested/offline operation.
    Implementations should ensure monotonicity and integrity, and be able
    to reconcile/flush when connectivity is restored.
    """

    def append_event(self, event: OfflineEvent) -> EventId:
        """
        Append an event and return a stable, tamper-evident identifier.
        """
        ...

    def snapshot(self) -> OfflineSnapshot:
        """
        Return a snapshot suitable for audit, backup, or remote reconciliation.
        """
        ...

    def flush_to_remote(self) -> Tuple[bool, Optional[str]]:
        """
        Attempt to flush pending events to a remote durable store.
        Returns (ok, reason) where reason is filled on failure.
        """
        ...


@runtime_checkable
class RoleAuthorityResolver(Protocol):
    """
    Maps identities (principals) to roles with jurisdiction-specific constraints.
    """

    def resolve(self, principal: PrincipalId, region: Region) -> Iterable[RoleName]:
        """
        Resolve the set of roles granted to the principal in the given region.
        """
        ...


@runtime_checkable
class ExportControlAdvisor(Protocol):
    """
    Advises on dual-use/export constraints. Disabled in healthcare profiles
    unless explicitly enabled.
    """

    def allowed_feature(self, feature: FeatureName, region: Region) -> bool:
        """
        Return True if the feature is allowed in the given region with respect
        to export/dual-use constraints.
        """
        ...


__all__ = [
    # Enums
    "Region",
    "Purpose",
    # Type aliases
    "PrincipalId",
    "PeerId",
    "RoleName",
    "FeatureName",
    "TokenStr",
    "EventId",
    "ZoneName",
    # TypedDicts
    "Evidence",
    "TokenMeta",
    "OfflineEvent",
    "OfflineSnapshot",
    # Results
    "AttestationResult",
    "SignalResult",
    # Protocols
    "AttestationProvider",
    "CryptoSignalProvider",
    "CommsPolicy",
    "GeoFenceProvider",
    "OfflineStore",
    "RoleAuthorityResolver",
    "ExportControlAdvisor",
]
