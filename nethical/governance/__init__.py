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
    # Institutional Governance & Wave 1
    'WorkflowEngine',
    'WorkflowInstance',
    'StageInstance',
    'ImpactAssessmentEngine',
    'ImpactAssessment',
    # NTSG Tactical Gate & ROE
    'NTSGCommandGate',
    'DeterministicIHLGate',
    'TwoManPQCAuthenticator',
    'IHLArticle',
    'TargetClassification',
    'EffectorCategory',
    'ROEGateVerdict',
    'CanonicalOperationToken',
    'OfficerSignature',
    'ROEDecisionReceipt',
    # Wave 3: Board Governance Dashboard & Executive Briefings
    'BoardGovernanceDashboard',
    'RiskAppetiteThresholds',
    'EthicalDebtItem',
    'RegulatoryPosture',
    'DepartmentalRiskSummary',
    'ExecutiveBoardPacket',
    'RAGStatus',
    'AppetiteBreachStatus',
    'EthicalDebtCategory',
    'ExecutiveBriefingGenerator',
    'ExecutiveBriefingDocument',
    'BriefingKPIs',
    'IncidentHighlight',
    'BriefingType',
    'ClassificationMarking',
    # Conflict of Laws & Jurisdictional Balancing
    'ConflictOfLawsEngine',
    'LegalObligation',
    'ConflictAdjudicationResult',
    'NormativeHierarchy',
    'MandateAction',
    'ResolutionPrinciple',
    # Precedent Database & Jurisprudential Stare Decisis
    'PrecedentDatabase',
    'PrecedentCase',
    'PrecedentRuling',
    'PrecedentBindingLevel',
    'PrecedentMatch',
    # Governance KPI Framework
    'AIGovernanceKPIEngine',
    'GovernanceKPISnapshot',
    'ReviewTicketMetric',
    'PairedReviewRating',
    'KPICategory',
    'KPIHealth',
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

from .workflow_engine import (
    WorkflowEngine,
    WorkflowInstance,
    StageInstance,
)

from .impact_assessment import (
    ImpactAssessmentEngine,
    ImpactAssessment,
)

from .roe_gate import (
    NTSGCommandGate,
    DeterministicIHLGate,
    TwoManPQCAuthenticator,
    IHLArticle,
    TargetClassification,
    EffectorCategory,
    ROEGateVerdict,
    CanonicalOperationToken,
    OfficerSignature,
    ROEDecisionReceipt,
)

from .board_dashboard import (
    BoardGovernanceDashboard,
    RiskAppetiteThresholds,
    EthicalDebtItem,
    RegulatoryPosture,
    DepartmentalRiskSummary,
    ExecutiveBoardPacket,
    RAGStatus,
    AppetiteBreachStatus,
    EthicalDebtCategory,
)

from .executive_briefing import (
    ExecutiveBriefingGenerator,
    ExecutiveBriefingDocument,
    BriefingKPIs,
    IncidentHighlight,
    BriefingType,
    ClassificationMarking,
)

from .conflict_of_laws import (
    ConflictOfLawsEngine,
    LegalObligation,
    ConflictAdjudicationResult,
    NormativeHierarchy,
    MandateAction,
    ResolutionPrinciple,
)

from .precedent_database import (
    PrecedentDatabase,
    PrecedentCase,
    PrecedentRuling,
    PrecedentBindingLevel,
    PrecedentMatch,
)

from .kpi_framework import (
    AIGovernanceKPIEngine,
    GovernanceKPISnapshot,
    ReviewTicketMetric,
    PairedReviewRating,
    KPICategory,
    KPIHealth,
)

