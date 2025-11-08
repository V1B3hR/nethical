"""
Multi-Factor Authentication (MFA) Support for Nethical

This module provides MFA capabilities including:
- TOTP (Time-based One-Time Password) support
- Backup codes generation and validation
- SMS-based verification (stub for external service integration)
- Admin operation MFA enforcement

Dependencies:
    - pyotp (for TOTP generation/validation)
    - qrcode (for QR code generation)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "MFAMethod",
    "MFASetup",
    "MFAManager",
    "MFARequiredError",
    "InvalidMFACodeError",
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
    """

    def __init__(self):
        """Initialize MFA Manager"""
        self.user_mfa: Dict[str, MFASetup] = {}
        self.admin_mfa_required: bool = True  # Enforce MFA for admin operations
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

    def verify_totp(self, user_id: str, code: str) -> bool:
        """
        Verify a TOTP code

        Args:
            user_id: User identifier
            code: 6-digit TOTP code

        Returns:
            True if code is valid
        """
        if user_id not in self.user_mfa:
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
            # Fallback: basic validation (not recommended for production)
            log.warning("pyotp not available, using basic TOTP validation")
            # For security, we'll reject in fallback mode
            is_valid = False

        if is_valid:
            mfa_setup.last_used_at = datetime.now(timezone.utc)
            log.info(f"TOTP code verified for user {user_id}")
        else:
            log.warning(f"Invalid TOTP code for user {user_id}")

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
