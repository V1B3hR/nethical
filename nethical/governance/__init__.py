# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Governance Module

Governance and ethics features including:
- Ethics benchmark system
- Threshold configuration versioning
- Policy grammar specification
"""

from .ethics_benchmark import (
    EthicsBenchmark,
    BenchmarkCase,
    DetectionResult,
    ViolationType,
    BenchmarkMetrics
)
from .threshold_config import (
    ThresholdVersionManager,
    Threshold,
    ThresholdType,
    ThresholdConfig,
    DEFAULT_THRESHOLDS
)

__all__ = [
    'EthicsBenchmark',
    'BenchmarkCase',
    'DetectionResult',
    'ViolationType',
    'BenchmarkMetrics',
    'ThresholdVersionManager',
    'Threshold',
    'ThresholdType',
    'ThresholdConfig',
    'DEFAULT_THRESHOLDS',
    # UK Gov Teal Book GovS 002 DoAM
    'DelegationOfAuthorityMatrix',
    'AuthorityLevel',
    'ReservedPowerCategory',
    'DOAMStatus',
    'DOAMEvaluationResult',
    # Global Jurisdictional & Institutional Intelligence
    'JurisdictionalTrustEngine',
    'JurisdictionProfile',
    'GovernanceDimension',
    'DataClassification',
    'TransferVerdict',
    'OECDRegulatoryImpactResult',
]

from .doam_matrix import (
    DelegationOfAuthorityMatrix,
    AuthorityLevel,
    ReservedPowerCategory,
    DOAMStatus,
    DOAMEvaluationResult,
)

from .jurisdictional_intel import (
    JurisdictionalTrustEngine,
    JurisdictionProfile,
    GovernanceDimension,
    DataClassification,
    TransferVerdict,
    OECDRegulatoryImpactResult,
)

