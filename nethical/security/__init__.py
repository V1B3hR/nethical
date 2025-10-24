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

from .middleware import (
    AuthMiddleware,
    require_auth,
    require_auth_and_role,
    require_auth_and_permission,
)

from .admin import (
    AdminInterface,
    UserInfo,
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
    # Middleware
    "AuthMiddleware",
    "require_auth",
    "require_auth_and_role",
    "require_auth_and_permission",
    # Admin Interface
    "AdminInterface",
    "UserInfo",
]

