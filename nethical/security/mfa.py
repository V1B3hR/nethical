"""
Multi-Factor Authentication (MFA) Support for Nethical

This module provides MFA capabilities including:
- TOTP (Time-based One-Time Password) support
- Backup codes generation and validation
- SMS-based verification (stub for external service integration)
- Admin operation MFA enforcement
- Brute-force protection with rate limiting

Security Features:
- Rate limiting to prevent brute-force attacks
- Account lockout after consecutive failed attempts
- Constant-time comparison for code validation

Dependencies:
    - pyotp (required for TOTP generation/validation)
    - qrcode (optional, for QR code generation)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "MFAMethod",
    "MFASetup",
    "MFAManager",
    "MFARequiredError",
    "InvalidMFACodeError",
    "MFALockedOutError",
    "MFADependencyError",
    "get_mfa_manager",
    "set_mfa_manager",
]

log = logging.getLogger(__name__)


class MFAMethod(str, Enum):
    """Multi-factor authentication methods"""

    TOTP = "totp"  # Time-based OTP (Google Authenticator, Authy, etc.)
    SMS = "sms"  # SMS-based verification (requires external service)
    BACKUP_CODE = "backup_code"  # Backup recovery codes


class MFARequiredError(Exception):
    """Raised when MFA is required but not provided"""


class InvalidMFACodeError(Exception):
    """Raised when MFA code is invalid"""


class MFALockedOutError(Exception):
    """Raised when user is locked out due to too many failed MFA attempts"""


class MFADependencyError(Exception):
    """Raised when a required MFA dependency (like pyotp) is not installed"""


@dataclass
class MFASetup:
    """MFA setup information for a user"""

    user_id: str
    enabled: bool = False
    methods: List[MFAMethod] = field(default_factory=list)
    totp_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)
    phone_number: Optional[str] = None  # For SMS
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None


class MFAManager:
    """
    Multi-Factor Authentication Manager

    Provides MFA capabilities for user authentication.
    For production use, consider integrating with services like:
    - Twilio (SMS)
    - Authy
    - Duo Security

    Security Features:
    - Rate limiting to prevent brute-force attacks on TOTP codes
    - Account lockout after consecutive failed attempts
    """

    # Rate limiting constants
    MAX_ATTEMPTS = 5  # Maximum failed attempts before lockout
    LOCKOUT_DURATION = timedelta(minutes=15)  # Lockout duration after max attempts
    ATTEMPT_WINDOW = timedelta(minutes=5)  # Window for counting failed attempts

    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration_minutes: int = 15,
        require_pyotp: bool = False,
    ):
        """
        Initialize MFA Manager

        Args:
            max_attempts: Maximum failed MFA attempts before lockout (default: 5)
            lockout_duration_minutes: Account lockout duration in minutes (default: 15)
            require_pyotp: If True, raise error when pyotp is not installed
        """
        self.user_mfa: Dict[str, MFASetup] = {}
        self.admin_mfa_required: bool = True  # Enforce MFA for admin operations
        self.require_pyotp = require_pyotp

        # Rate limiting configuration
        self.max_attempts = max_attempts
        self.lockout_duration = timedelta(minutes=lockout_duration_minutes)

        # Track failed attempts for rate limiting
        self._failed_attempts: Dict[str, List[datetime]] = {}
        self._lockouts: Dict[str, datetime] = {}

        log.info("MFAManager initialized")

    def setup_totp(self, user_id: str, issuer: str = "Nethical") -> Tuple[str, str, List[str]]:
        """
        Set up TOTP-based MFA for a user

        Args:
            user_id: User identifier
            issuer: Service name for QR code (default: "Nethical")

        Returns:
            Tuple of (totp_secret, provisioning_uri, backup_codes)
        """
        try:
            import pyotp
        except ImportError:
            log.warning("pyotp not installed, using fallback TOTP implementation")
            # Fallback: generate a base32 secret
            totp_secret = secrets.token_urlsafe(20).replace("-", "").replace("_", "")[:32].upper()
            # Create a basic provisioning URI
            provisioning_uri = (
                f"otpauth://totp/{issuer}:{user_id}?secret={totp_secret}&issuer={issuer}"
            )
        else:
            # Generate TOTP secret
            totp_secret = pyotp.random_base32()
            totp = pyotp.TOTP(totp_secret)
            provisioning_uri = totp.provisioning_uri(name=user_id, issuer_name=issuer)

        # Generate backup codes
        backup_codes = self._generate_backup_codes()

        # Store MFA setup
        if user_id not in self.user_mfa:
            self.user_mfa[user_id] = MFASetup(user_id=user_id)

        mfa_setup = self.user_mfa[user_id]
        mfa_setup.totp_secret = totp_secret
        mfa_setup.backup_codes = [self._hash_code(code) for code in backup_codes]

        if MFAMethod.TOTP not in mfa_setup.methods:
            mfa_setup.methods.append(MFAMethod.TOTP)

        log.info(f"TOTP MFA setup initiated for user {user_id}")

        return totp_secret, provisioning_uri, backup_codes

    def enable_mfa(self, user_id: str, method: MFAMethod = MFAMethod.TOTP) -> None:
        """
        Enable MFA for a user

        Args:
            user_id: User identifier
            method: MFA method to enable
        """
        if user_id not in self.user_mfa:
            raise ValueError(f"MFA not set up for user {user_id}")

        mfa_setup = self.user_mfa[user_id]
        mfa_setup.enabled = True

        if method not in mfa_setup.methods:
            mfa_setup.methods.append(method)

        log.info(f"MFA enabled for user {user_id} with method {method.value}")

    def disable_mfa(self, user_id: str) -> None:
        """
        Disable MFA for a user

        Args:
            user_id: User identifier
        """
        if user_id in self.user_mfa:
            self.user_mfa[user_id].enabled = False
            log.info(f"MFA disabled for user {user_id}")

    def _check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user is rate limited due to failed attempts.

        Args:
            user_id: User identifier

        Returns:
            True if user can attempt verification, False if rate limited

        Raises:
            MFALockedOutError: If user is locked out
        """
        now = datetime.now(timezone.utc)

        # Check if user is currently locked out
        if user_id in self._lockouts:
            lockout_end = self._lockouts[user_id]
            if now < lockout_end:
                remaining = (lockout_end - now).total_seconds()
                log.warning(f"User {user_id} is locked out for {remaining:.0f}s more")
                raise MFALockedOutError(
                    f"Account locked due to too many failed MFA attempts. "
                    f"Try again in {remaining:.0f} seconds."
                )
            else:
                # Lockout expired, remove it
                del self._lockouts[user_id]
                if user_id in self._failed_attempts:
                    del self._failed_attempts[user_id]

        return True

    def _record_failed_attempt(self, user_id: str) -> None:
        """
        Record a failed MFA attempt and potentially lock out the user.

        Args:
            user_id: User identifier
        """
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=5)

        # Initialize or clean up old attempts
        if user_id not in self._failed_attempts:
            self._failed_attempts[user_id] = []

        # Remove attempts outside the window
        self._failed_attempts[user_id] = [
            ts for ts in self._failed_attempts[user_id]
            if ts > window_start
        ]

        # Record this attempt
        self._failed_attempts[user_id].append(now)

        # Check if we should lock out the user
        if len(self._failed_attempts[user_id]) >= self.max_attempts:
            lockout_end = now + self.lockout_duration
            self._lockouts[user_id] = lockout_end
            log.warning(
                f"User {user_id} locked out until {lockout_end.isoformat()} "
                f"after {self.max_attempts} failed MFA attempts"
            )

    def _clear_failed_attempts(self, user_id: str) -> None:
        """Clear failed attempt counter on successful verification."""
        if user_id in self._failed_attempts:
            del self._failed_attempts[user_id]
        if user_id in self._lockouts:
            del self._lockouts[user_id]

    def get_remaining_attempts(self, user_id: str) -> int:
        """
        Get remaining MFA attempts before lockout.

        Args:
            user_id: User identifier

        Returns:
            Number of remaining attempts
        """
        if user_id not in self._failed_attempts:
            return self.max_attempts

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(minutes=5)
        recent_attempts = len([
            ts for ts in self._failed_attempts[user_id]
            if ts > window_start
        ])

        return max(0, self.max_attempts - recent_attempts)

    def verify_totp(self, user_id: str, code: str) -> bool:
        """
        Verify a TOTP code with brute-force protection.

        Args:
            user_id: User identifier
            code: 6-digit TOTP code

        Returns:
            True if code is valid

        Raises:
            MFALockedOutError: If user is locked out due to failed attempts
            MFADependencyError: If pyotp is not installed and require_pyotp is True
        """
        # Check rate limit first
        self._check_rate_limit(user_id)

        if user_id not in self.user_mfa:
            self._record_failed_attempt(user_id)
            return False

        mfa_setup = self.user_mfa[user_id]

        if not mfa_setup.enabled or MFAMethod.TOTP not in mfa_setup.methods:
            return False

        if not mfa_setup.totp_secret:
            return False

        try:
            import pyotp

            totp = pyotp.TOTP(mfa_setup.totp_secret)
            is_valid = totp.verify(code, valid_window=1)  # Allow 30s window
        except ImportError:
            if self.require_pyotp:
                log.error("pyotp is required but not installed")
                raise MFADependencyError(
                    "pyotp library is required for TOTP verification but is not installed. "
                    "Install it with: pip install pyotp"
                )
            # Fallback: reject all codes when pyotp is not available
            log.error(
                "pyotp not available for TOTP verification. "
                "Install pyotp for production use: pip install pyotp"
            )
            self._record_failed_attempt(user_id)
            return False

        if is_valid:
            self._clear_failed_attempts(user_id)
            mfa_setup.last_used_at = datetime.now(timezone.utc)
            log.info(f"TOTP code verified for user {user_id}")
        else:
            self._record_failed_attempt(user_id)
            remaining = self.get_remaining_attempts(user_id)
            log.warning(f"Invalid TOTP code for user {user_id}. {remaining} attempts remaining.")

        return is_valid

    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """
        Verify a backup recovery code

        Args:
            user_id: User identifier
            code: Backup code

        Returns:
            True if code is valid (code is consumed after use)
        """
        if user_id not in self.user_mfa:
            return False

        mfa_setup = self.user_mfa[user_id]

        if not mfa_setup.enabled:
            return False

        code_hash = self._hash_code(code)

        if code_hash in mfa_setup.backup_codes:
            # Remove used backup code
            mfa_setup.backup_codes.remove(code_hash)
            mfa_setup.last_used_at = datetime.now(timezone.utc)
            log.info(f"Backup code verified and consumed for user {user_id}")
            return True

        log.warning(f"Invalid backup code for user {user_id}")
        return False

    def verify_mfa(self, user_id: str, code: str, method: Optional[MFAMethod] = None) -> bool:
        """
        Verify MFA code (auto-detect method if not specified)

        Args:
            user_id: User identifier
            code: MFA code
            method: Optional specific method to try

        Returns:
            True if code is valid
        """
        if user_id not in self.user_mfa:
            return False

        mfa_setup = self.user_mfa[user_id]

        if not mfa_setup.enabled:
            return False

        # Try specific method if provided
        if method == MFAMethod.TOTP:
            return self.verify_totp(user_id, code)
        elif method == MFAMethod.BACKUP_CODE:
            return self.verify_backup_code(user_id, code)

        # Auto-detect: try TOTP first, then backup codes
        if MFAMethod.TOTP in mfa_setup.methods:
            if self.verify_totp(user_id, code):
                return True

        if MFAMethod.BACKUP_CODE in mfa_setup.methods:
            if self.verify_backup_code(user_id, code):
                return True

        return False

    def is_mfa_enabled(self, user_id: str) -> bool:
        """
        Check if MFA is enabled for a user

        Args:
            user_id: User identifier

        Returns:
            True if MFA is enabled
        """
        if user_id not in self.user_mfa:
            return False
        return self.user_mfa[user_id].enabled

    def require_mfa_for_admin(self, enabled: bool = True) -> None:
        """
        Set whether MFA is required for admin operations

        Args:
            enabled: Whether to require MFA for admin operations
        """
        self.admin_mfa_required = enabled
        log.info(f"Admin MFA requirement set to {enabled}")

    def check_admin_mfa_required(self, user_id: str, user_role: str) -> bool:
        """
        Check if MFA is required for this admin user

        Args:
            user_id: User identifier
            user_role: User's role (e.g., "admin")

        Returns:
            True if MFA is required and not satisfied
        """
        # Only enforce for admin role
        if user_role.lower() != "admin":
            return False

        # If admin MFA is not globally required, no check needed
        if not self.admin_mfa_required:
            return False

        # Check if user has MFA enabled
        return not self.is_mfa_enabled(user_id)

    def get_qr_code_data_uri(self, provisioning_uri: str) -> str:
        """
        Generate a QR code data URI for TOTP setup

        Args:
            provisioning_uri: TOTP provisioning URI

        Returns:
            Data URI string for QR code image
        """
        try:
            import qrcode
            import io
            import base64

            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        except ImportError:
            log.warning("qrcode library not installed, returning provisioning URI")
            return provisioning_uri

    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """
        Generate backup recovery codes

        Args:
            count: Number of codes to generate

        Returns:
            List of backup codes
        """
        codes = []
        # Exclude confusing characters: 0, O, 1, I, L
        allowed_chars = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = "".join(secrets.choice(allowed_chars) for _ in range(8))
            # Format as XXXX-XXXX for readability
            formatted = f"{code[:4]}-{code[4:]}"
            codes.append(formatted)
        return codes

    def _hash_code(self, code: str) -> str:
        """
        Hash a code for secure storage

        Args:
            code: Code to hash

        Returns:
            SHA-256 hash of the code
        """
        return hashlib.sha256(code.encode()).hexdigest()

    def regenerate_backup_codes(self, user_id: str) -> List[str]:
        """
        Regenerate backup codes for a user

        Args:
            user_id: User identifier

        Returns:
            New list of backup codes
        """
        if user_id not in self.user_mfa:
            raise ValueError(f"MFA not set up for user {user_id}")

        backup_codes = self._generate_backup_codes()
        self.user_mfa[user_id].backup_codes = [self._hash_code(code) for code in backup_codes]

        log.info(f"Backup codes regenerated for user {user_id}")
        return backup_codes


# Global MFA manager instance
_mfa_manager: Optional[MFAManager] = None


def get_mfa_manager() -> MFAManager:
    """Get or create the global MFA manager instance"""
    global _mfa_manager
    if _mfa_manager is None:
        _mfa_manager = MFAManager()
    return _mfa_manager


def set_mfa_manager(manager: MFAManager) -> None:
    """Set the global MFA manager instance"""
    global _mfa_manager
    _mfa_manager = manager
