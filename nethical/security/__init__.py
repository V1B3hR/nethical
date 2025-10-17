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
]

