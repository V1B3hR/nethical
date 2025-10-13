from __future__ import annotations
from typing import Protocol, Optional, Dict, Any, Tuple, Iterable
from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    US = "US"
    UK = "UK"
    EU = "EU"
    NATO = "NATO"  # NATO profile hooks; healthcare defaults keep this inert


@dataclass
class AttestationResult:
    ok: bool
    evidence: Dict[str, Any]
    reason: Optional[str] = None


class AttestationProvider(Protocol):
    def attest_runtime(self) -> AttestationResult: ...
    def attest_hardware(self) -> AttestationResult: ...


@dataclass
class SignalResult:
    ok: bool
    token: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class CryptoSignalProvider(Protocol):
    # Defensive-friendly authenticated “medical” role signaling; disabled by default
    def issue_medical_role_token(
        self, role: str, ttl_seconds: int = 60
    ) -> SignalResult: ...
    def verify_peer_role_token(self, token: str) -> SignalResult: ...


class CommsPolicy(Protocol):
    # Zero-trust comms policy (mTLS, identity enforcement, allowlists)
    def connection_allowed(self, peer_id: str, context: Dict[str, Any]) -> bool: ...
    def identities(self) -> Dict[str, Any]: ...


class GeoFenceProvider(Protocol):
    def allowed(self, lat: float, lon: float, purpose: str) -> bool: ...
    def current_zone(self) -> Optional[str]: ...


class OfflineStore(Protocol):
    # Tamper-evident, append-only store for contested/offline operation
    def append_event(self, event: Dict[str, Any]) -> str: ...
    def snapshot(self) -> Dict[str, Any]: ...
    def flush_to_remote(self) -> Tuple[bool, Optional[str]]: ...


class RoleAuthorityResolver(Protocol):
    # Maps identities to roles with jurisdiction-specific constraints
    def resolve(self, principal: str, region: Region) -> Iterable[str]: ...


class ExportControlAdvisor(Protocol):
    # Advises on dual-use/export constraints, disabled in healthcare profiles
    def allowed_feature(self, feature: str, region: Region) -> bool: ...
