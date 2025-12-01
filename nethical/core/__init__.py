"""Core components of the Nethical safety governance system."""

from .risk_engine import RiskEngine, RiskTier, RiskProfile
from .correlation_engine import CorrelationEngine, CorrelationMatch
from .fairness_sampler import FairnessSampler, Sample, SamplingJob, SamplingStrategy
from .ethical_drift_reporter import EthicalDriftReporter, EthicalDriftReport, CohortProfile
from .performance_optimizer import PerformanceOptimizer, DetectorTier, DetectorMetrics
from .phase3_integration import Phase3IntegratedGovernance

# Fundamental Laws - Ethical Backbone
from .fundamental_laws import (
    LawCategory,
    FundamentalLaw,
    FundamentalLawsRegistry,
    FUNDAMENTAL_LAWS,
    get_fundamental_laws,
)

# Phase 4 components
from .audit_merkle import MerkleAnchor, AuditChunk, MerkleNode
from .policy_diff import PolicyDiffAuditor, PolicyDiffResult, PolicyChange, ChangeType, RiskLevel
from .quarantine import QuarantineManager, QuarantineReason, QuarantineStatus, QuarantinePolicy, HardwareIsolationLevel
from .ethical_taxonomy import EthicalTaxonomy, EthicalTag, ViolationTagging, EthicalDimension
from .sla_monitor import SLAMonitor, SLAStatus, SLATarget, SLABreach
from .phase4_integration import Phase4IntegratedGovernance

# Kill Switch Protocol - Emergency Override System
from .kill_switch import (
    ShutdownMode,
    CommandType,
    KeyType,
    ConnectionType,
    IsolationLevel,
    ActuatorState,
    KillSwitchConfig,
    AgentRecord,
    ActuatorRecord,
    SignedCommand,
    AuditLogEntry,
    KillSwitchResult,
    KillSwitchCallback,
    GlobalKillSwitch,
    ActuatorSevering,
    CryptoSignedCommands,
    HardwareIsolation,
    KillSwitchProtocol,
)

# Phase 5 components
from .ml_shadow import MLShadowClassifier, ShadowPrediction, ShadowMetrics, MLModelType

# Phase 6 components
from .ml_blended_risk import MLBlendedRiskEngine, BlendedDecision, BlendingMetrics, RiskZone

# Phase 7 components
from .anomaly_detector import (
    AnomalyDriftMonitor,
    SequenceAnomalyDetector,
    DistributionDriftDetector,
    AnomalyAlert,
    AnomalyType,
    DriftSeverity,
    DriftMetrics,
)

# Phase 5-7 Integration
from .phase567_integration import Phase567IntegratedGovernance

# Phase 8 components
from .human_feedback import (
    EscalationQueue,
    FeedbackTag,
    ReviewStatus,
    ReviewPriority,
    HumanFeedback,
    EscalationCase,
    SLAMetrics,
)

# Phase 9 components
from .optimization import (
    MultiObjectiveOptimizer,
    Configuration,
    PerformanceMetrics,
    OptimizationObjective,
    OptimizationTechnique,
    ConfigStatus,
    PromotionGate,
    AdaptiveThresholdTuner,
    ABTestingFramework,
    OutcomeRecord,
)

# Phase 8-9 Integration
from .phase89_integration import Phase89IntegratedGovernance

# Unified Integration (All Phases)
from .integrated_governance import IntegratedGovernance

# F2: Detector & Policy Extensibility
from .plugin_interface import (
    DetectorPlugin,
    PluginManager,
    PluginMetadata,
    PluginStatus,
    get_plugin_manager,
)
from .policy_dsl import (
    Policy,
    PolicyAction,
    PolicyEngine,
    PolicyParser,
    PolicyRule,
    RuleEvaluator,
    RuleSeverity,
    get_policy_engine,
)

# Phase 1: Security & Governance
from .rbac import (
    Role,
    Permission,
    RBACManager,
    AccessDeniedError,
    require_role,
    require_permission,
    get_rbac_manager,
    set_rbac_manager,
)

__all__ = [
    # Fundamental Laws - Ethical Backbone
    "LawCategory",
    "FundamentalLaw",
    "FundamentalLawsRegistry",
    "FUNDAMENTAL_LAWS",
    "get_fundamental_laws",
    # Phase 3
    "RiskEngine",
    "RiskTier",
    "RiskProfile",
    "CorrelationEngine",
    "CorrelationMatch",
    "FairnessSampler",
    "Sample",
    "SamplingJob",
    "SamplingStrategy",
    "EthicalDriftReporter",
    "EthicalDriftReport",
    "CohortProfile",
    "PerformanceOptimizer",
    "DetectorTier",
    "DetectorMetrics",
    "Phase3IntegratedGovernance",
    # Phase 4
    "MerkleAnchor",
    "AuditChunk",
    "MerkleNode",
    "PolicyDiffAuditor",
    "PolicyDiffResult",
    "PolicyChange",
    "ChangeType",
    "RiskLevel",
    "QuarantineManager",
    "QuarantineReason",
    "QuarantineStatus",
    "QuarantinePolicy",
    "HardwareIsolationLevel",
    "EthicalTaxonomy",
    "EthicalTag",
    "ViolationTagging",
    "EthicalDimension",
    "SLAMonitor",
    "SLAStatus",
    "SLATarget",
    "SLABreach",
    "Phase4IntegratedGovernance",
    # Kill Switch Protocol
    "ShutdownMode",
    "CommandType",
    "KeyType",
    "ConnectionType",
    "IsolationLevel",
    "ActuatorState",
    "KillSwitchConfig",
    "AgentRecord",
    "ActuatorRecord",
    "SignedCommand",
    "AuditLogEntry",
    "KillSwitchResult",
    "KillSwitchCallback",
    "GlobalKillSwitch",
    "ActuatorSevering",
    "CryptoSignedCommands",
    "HardwareIsolation",
    "KillSwitchProtocol",
    # Phase 5
    "MLShadowClassifier",
    "ShadowPrediction",
    "ShadowMetrics",
    "MLModelType",
    # Phase 6
    "MLBlendedRiskEngine",
    "BlendedDecision",
    "BlendingMetrics",
    "RiskZone",
    # Phase 7
    "AnomalyDriftMonitor",
    "SequenceAnomalyDetector",
    "DistributionDriftDetector",
    "AnomalyAlert",
    "AnomalyType",
    "DriftSeverity",
    "DriftMetrics",
    # Phase 5-7 Integration
    "Phase567IntegratedGovernance",
    # Phase 8
    "EscalationQueue",
    "FeedbackTag",
    "ReviewStatus",
    "ReviewPriority",
    "HumanFeedback",
    "EscalationCase",
    "SLAMetrics",
    # Phase 9
    "MultiObjectiveOptimizer",
    "Configuration",
    "PerformanceMetrics",
    "OptimizationObjective",
    "OptimizationTechnique",
    "ConfigStatus",
    "PromotionGate",
    "AdaptiveThresholdTuner",
    "ABTestingFramework",
    "OutcomeRecord",
    # Phase 8-9 Integration
    "Phase89IntegratedGovernance",
    # Unified Integration (All Phases)
    "IntegratedGovernance",
    # F2: Detector & Policy Extensibility
    "DetectorPlugin",
    "PluginManager",
    "PluginMetadata",
    "PluginStatus",
    "get_plugin_manager",
    "Policy",
    "PolicyAction",
    "PolicyEngine",
    "PolicyParser",
    "PolicyRule",
    "RuleEvaluator",
    "RuleSeverity",
    "get_policy_engine",
    # Phase 1: Security & Governance
    "Role",
    "Permission",
    "RBACManager",
    "AccessDeniedError",
    "require_role",
    "require_permission",
    "get_rbac_manager",
    "set_rbac_manager",
]
