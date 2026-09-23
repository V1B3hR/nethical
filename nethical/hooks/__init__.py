# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Nethical Hooks Package (nethical.hooks).

Defines extension points, protocols, and interfaces for attestation,
cryptographic role signaling, communications policy, geofencing, offline
persistence, and jurisdiction-aware authorization.
"""

from __future__ import annotations

from nethical.hooks.interfaces import (
    Region,
    Purpose,
    PrincipalId,
    PeerId,
    RoleName,
    FeatureName,
    TokenStr,
    EventId,
    ZoneName,
    Evidence,
    TokenMeta,
    OfflineEvent,
    OfflineSnapshot,
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

__all__ = [
    "Region",
    "Purpose",
    "PrincipalId",
    "PeerId",
    "RoleName",
    "FeatureName",
    "TokenStr",
    "EventId",
    "ZoneName",
    "Evidence",
    "TokenMeta",
    "OfflineEvent",
    "OfflineSnapshot",
    "AttestationResult",
    "SignalResult",
    "AttestationProvider",
    "CryptoSignalProvider",
    "CommsPolicy",
    "GeoFenceProvider",
    "OfflineStore",
    "RoleAuthorityResolver",
    "ExportControlAdvisor",
]
