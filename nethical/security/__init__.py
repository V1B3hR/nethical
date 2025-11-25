# Package init for security

from .attestation import (
    NoopAttestation,
    TrustedAttestation,
    select_attestation_provider,
    normalize_attestation_result,
    AttestationErrorCodes,
    register_attestation_provider,
    compute_measurements_digest,
)

from .auth import (
    AuthManager,
    TokenPayload,
    TokenType,
    APIKey,
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
    authenticate_request,
    get_auth_manager,
    set_auth_manager,
)

from .mfa import (
    MFAMethod,
    MFASetup,
    MFAManager,
    MFARequiredError,
    InvalidMFACodeError,
    get_mfa_manager,
    set_mfa_manager,
)

from .sso import (
    SSOProvider,
    SSOConfig,
    SAMLConfig,
    SSOManager,
    SSOError,
    get_sso_manager,
    set_sso_manager,
)

# Phase 1: Military-Grade Security Enhancements
from .authentication import (
    AuthCredentials,
    AuthResult,
    ClearanceLevel,
    PKICertificateValidator,
    MultiFactorAuthEngine,
    SecureSessionManager,
    LDAPConnector,
    MilitaryGradeAuthProvider,
)

from .encryption import (
    EncryptionAlgorithm,
    KeyRotationPolicy,
    HSMConfig,
    EncryptedData,
    MilitaryGradeEncryption,
    KeyManagementService,
)

from .input_validation import (
    ValidationResult,
    ThreatLevel,
    SemanticAnomalyDetector,
    ThreatIntelligenceDB,
    BehavioralAnalyzer,
    AdversarialInputDefense,
)

# Phase 2: Advanced Anomaly Detection
from .anomaly_detection import (
    AnomalyType,
    AnomalyDetectionResult,
    LSTMSequenceDetector,
    TransformerContextAnalyzer,
    GraphRelationshipAnalyzer,
    InsiderThreatDetector,
    APTBehavioralDetector,
    AdvancedAnomalyDetectionEngine,
)

# Phase 2: SOC Integration
from .soc_integration import (
    SIEMFormat,
    AlertSeverity,
    IncidentStatus,
    SIEMEvent,
    Incident,
    SIEMConnector,
    IncidentManager,
    ThreatHuntingEngine,
    AlertingEngine,
    ForensicCollector,
    SOCIntegrationHub,
)

# Phase 3: Compliance & Audit Framework
from .compliance import (
    ComplianceFramework,
    ComplianceStatus,
    ControlSeverity,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceReport,
    NIST80053ControlMapper,
    HIPAAComplianceValidator,
    FedRAMPMonitor,
    ComplianceReportGenerator,
    EvidenceCollector,
)

# Phase 3: Enhanced Audit Logging
from .audit_logging import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    BlockchainBlock,
    TimestampAuthority,
    DigitalSignature,
    AuditBlockchain,
    ForensicAnalyzer,
    ChainOfCustodyManager,
    EnhancedAuditLogger,
)

# Phase 4: Zero Trust Architecture
from .zero_trust import (
    TrustLevel,
    DeviceHealthStatus,
    NetworkSegment,
    ServiceMeshConfig,
    DeviceHealthCheck,
    PolicyEnforcer,
    ContinuousAuthEngine,
    ZeroTrustController,
)

# Phase 4: Secret Management
from .secret_management import (
    SecretType,
    SecretRotationPolicy,
    VaultConfig,
    Secret,
    SecretScanner,
    DynamicSecretGenerator,
    SecretRotationManager,
    VaultIntegration,
    SecretManagementSystem,
)

# Regulatory Compliance Framework (EU AI Act, UK Law, US Standards)
from .regulatory_compliance import (
    AIRiskLevel,
    RegulatoryFramework,
    ComplianceStatus as RegulatoryComplianceStatus,
    ControlCategory,
    RegulatoryRequirement,
    RegulatoryMapping,
    EUAIActCompliance,
    UKLawCompliance,
    USStandardsCompliance,
    RegulatoryMappingGenerator,
    generate_regulatory_mapping_table,
)

__all__ = [
    # Attestation
    "NoopAttestation",
    "TrustedAttestation",
    "select_attestation_provider",
    "normalize_attestation_result",
    "AttestationErrorCodes",
    "register_attestation_provider",
    "compute_measurements_digest",
    # Authentication
    "AuthManager",
    "TokenPayload",
    "TokenType",
    "APIKey",
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "authenticate_request",
    "get_auth_manager",
    "set_auth_manager",
    # Multi-Factor Authentication
    "MFAMethod",
    "MFASetup",
    "MFAManager",
    "MFARequiredError",
    "InvalidMFACodeError",
    "get_mfa_manager",
    "set_mfa_manager",
    # Single Sign-On
    "SSOProvider",
    "SSOConfig",
    "SAMLConfig",
    "SSOManager",
    "SSOError",
    "get_sso_manager",
    "set_sso_manager",
    # Phase 1: Military-Grade Authentication
    "AuthCredentials",
    "AuthResult",
    "ClearanceLevel",
    "PKICertificateValidator",
    "MultiFactorAuthEngine",
    "SecureSessionManager",
    "LDAPConnector",
    "MilitaryGradeAuthProvider",
    # Phase 1: Encryption
    "EncryptionAlgorithm",
    "KeyRotationPolicy",
    "HSMConfig",
    "EncryptedData",
    "MilitaryGradeEncryption",
    "KeyManagementService",
    # Phase 1: Input Validation
    "ValidationResult",
    "ThreatLevel",
    "SemanticAnomalyDetector",
    "ThreatIntelligenceDB",
    "BehavioralAnalyzer",
    "AdversarialInputDefense",
    # Phase 2: Advanced Anomaly Detection
    "AnomalyType",
    "AnomalyDetectionResult",
    "LSTMSequenceDetector",
    "TransformerContextAnalyzer",
    "GraphRelationshipAnalyzer",
    "InsiderThreatDetector",
    "APTBehavioralDetector",
    "AdvancedAnomalyDetectionEngine",
    # Phase 2: SOC Integration
    "SIEMFormat",
    "AlertSeverity",
    "IncidentStatus",
    "SIEMEvent",
    "Incident",
    "SIEMConnector",
    "IncidentManager",
    "ThreatHuntingEngine",
    "AlertingEngine",
    "ForensicCollector",
    "SOCIntegrationHub",
    # Phase 3: Compliance Framework
    "ComplianceFramework",
    "ComplianceStatus",
    "ControlSeverity",
    "ComplianceControl",
    "ComplianceEvidence",
    "ComplianceReport",
    "NIST80053ControlMapper",
    "HIPAAComplianceValidator",
    "FedRAMPMonitor",
    "ComplianceReportGenerator",
    "EvidenceCollector",
    # Phase 3: Audit Logging
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "BlockchainBlock",
    "TimestampAuthority",
    "DigitalSignature",
    "AuditBlockchain",
    "ForensicAnalyzer",
    "ChainOfCustodyManager",
    "EnhancedAuditLogger",
    # Phase 4: Zero Trust Architecture
    "TrustLevel",
    "DeviceHealthStatus",
    "NetworkSegment",
    "ServiceMeshConfig",
    "DeviceHealthCheck",
    "PolicyEnforcer",
    "ContinuousAuthEngine",
    "ZeroTrustController",
    # Phase 4: Secret Management
    "SecretType",
    "SecretRotationPolicy",
    "VaultConfig",
    "Secret",
    "SecretScanner",
    "DynamicSecretGenerator",
    "SecretRotationManager",
    "VaultIntegration",
    "SecretManagementSystem",
    # Regulatory Compliance Framework
    "AIRiskLevel",
    "RegulatoryFramework",
    "RegulatoryComplianceStatus",
    "ControlCategory",
    "RegulatoryRequirement",
    "RegulatoryMapping",
    "EUAIActCompliance",
    "UKLawCompliance",
    "USStandardsCompliance",
    "RegulatoryMappingGenerator",
    "generate_regulatory_mapping_table",
]
