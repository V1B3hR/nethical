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
]

