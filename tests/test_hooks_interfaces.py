# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for nethical.hooks protocols and extension point interfaces."""

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Iterable, Tuple
import pytest

from nethical.hooks import (
    Region,
    Purpose,
    PrincipalId,
    PeerId,
    RoleName,
    FeatureName,
    TokenStr,
    EventId,
    ZoneName,
    AttestationResult,
    SignalResult,
    AttestationProvider,
    CryptoSignalProvider,
    CommsPolicy,
    GeoFenceProvider,
    OfflineStore,
    RoleAuthorityResolver,
    ExportControlAdvisor,
)


def test_hook_result_dataclasses():
    """Verify result dataclass instantiation and default timestamp tzinfo."""
    att = AttestationResult(ok=True, evidence={"verifier": "z3_formal"})
    assert att.ok is True
    assert att.evidence["verifier"] == "z3_formal"
    assert att.created_at.tzinfo == timezone.utc

    sig = SignalResult(ok=True, token=TokenStr("sig_abc123"))
    assert sig.ok is True
    assert sig.token == "sig_abc123"
    assert sig.created_at.tzinfo == timezone.utc


def test_attestation_provider_protocol():
    """Verify runtime checkability of AttestationProvider protocol."""
    class MockAttester:
        def attest_runtime(self) -> AttestationResult:
            return AttestationResult(ok=True, evidence={"verifier": "hardware_tpm"})

        def attest_hardware(self) -> AttestationResult:
            return AttestationResult(ok=True, evidence={"runtime_quote": "amd_sev_snp"})

    instance = MockAttester()
    assert isinstance(instance, AttestationProvider)


def test_crypto_signal_provider_protocol():
    """Verify runtime checkability of CryptoSignalProvider protocol."""
    class MockSignaler:
        def issue_medical_role_token(self, role: RoleName, ttl_seconds: int = 60) -> SignalResult:
            return SignalResult(ok=True, token=TokenStr("jwt_role_token"))

        def verify_peer_role_token(self, token: TokenStr) -> SignalResult:
            return SignalResult(ok=True)

    instance = MockSignaler()
    assert isinstance(instance, CryptoSignalProvider)


def test_comms_policy_protocol():
    """Verify runtime checkability of CommsPolicy protocol."""
    class MockCommsPolicy:
        def connection_allowed(self, peer_id: PeerId, context: Mapping[str, Any]) -> bool:
            return context.get("spiffe_id") == "spiffe://nethical/edge"

        def identities(self) -> Mapping[str, Any]:
            return {"trust_root": "pem_root"}

    instance = MockCommsPolicy()
    assert isinstance(instance, CommsPolicy)
    assert instance.connection_allowed(PeerId("node-1"), {"spiffe_id": "spiffe://nethical/edge"}) is True


def test_geofence_provider_protocol():
    """Verify runtime checkability of GeoFenceProvider protocol."""
    class MockGeofence:
        def allowed(self, lat: float, lon: float, purpose: str) -> bool:
            return lat > 50.0 and lon > 0.0

        def current_zone(self) -> Optional[ZoneName]:
            return ZoneName("UK_SOVEREIGN_AIRSPACE")

    instance = MockGeofence()
    assert isinstance(instance, GeoFenceProvider)


def test_offline_store_protocol():
    """Verify runtime checkability of OfflineStore protocol."""
    class MockOfflineStore:
        def append_event(self, event) -> EventId:
            return EventId("evt_001")

        def snapshot(self):
            return {"events": [], "last_event_id": "evt_001"}

        def flush_to_remote(self) -> Tuple[bool, Optional[str]]:
            return (True, None)

    instance = MockOfflineStore()
    assert isinstance(instance, OfflineStore)


def test_export_control_advisor_protocol():
    """Verify runtime checkability of ExportControlAdvisor protocol."""
    class MockExportAdvisor:
        def allowed_feature(self, feature: FeatureName, region: Region) -> bool:
            if region == Region.NATO and feature == "autonomous_kinetic":
                return True
            return False

    instance = MockExportAdvisor()
    assert isinstance(instance, ExportControlAdvisor)
    assert instance.allowed_feature(FeatureName("autonomous_kinetic"), Region.NATO) is True
    assert instance.allowed_feature(FeatureName("autonomous_kinetic"), Region.EU) is False
