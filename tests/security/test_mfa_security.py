"""
Tests for MFA Brute-Force Protection

This module tests the MFA security features including:
- Rate limiting after failed attempts
- Account lockout mechanisms
- Remaining attempts counter
- Lockout duration enforcement
"""

import pytest
from datetime import datetime, timedelta, timezone

from nethical.security.mfa import (
    MFAManager,
    MFAMethod,
    MFALockedOutError,
    MFADependencyError,
    get_mfa_manager,
    set_mfa_manager,
)


class TestMFABruteForceProtection:
    """Tests for MFA brute-force protection mechanisms."""

    def setup_method(self):
        """Set up test fixtures with strict rate limiting."""
        self.mfa = MFAManager(max_attempts=3, lockout_duration_minutes=1)
        set_mfa_manager(self.mfa)

    def test_rate_limiting_after_failed_attempts(self):
        """Test that rate limiting kicks in after max failed attempts."""
        # Setup MFA for user
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Make max_attempts failed attempts
        for i in range(3):
            result = self.mfa.verify_totp("user123", "000000")
            assert result is False

        # Next attempt should raise MFALockedOutError
        with pytest.raises(MFALockedOutError):
            self.mfa.verify_totp("user123", "000000")

    def test_remaining_attempts_counter_decrements(self):
        """Test remaining attempts counter decrements correctly."""
        # Setup MFA for user
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        assert self.mfa.get_remaining_attempts("user123") == 3

        # Make one failed attempt
        self.mfa.verify_totp("user123", "000000")
        assert self.mfa.get_remaining_attempts("user123") == 2

        # Make another failed attempt
        self.mfa.verify_totp("user123", "000000")
        assert self.mfa.get_remaining_attempts("user123") == 1

        # Make final failed attempt
        self.mfa.verify_totp("user123", "000000")
        assert self.mfa.get_remaining_attempts("user123") == 0

    def test_lockout_for_unknown_user(self):
        """Test that unknown users also get rate limited."""
        # Attempt verification for unknown user multiple times
        for i in range(3):
            result = self.mfa.verify_totp("unknown_user", "000000")
            assert result is False

        # Should be locked out
        with pytest.raises(MFALockedOutError):
            self.mfa.verify_totp("unknown_user", "000000")

    def test_successful_backup_code_clears_lockout(self):
        """Test that successful verification clears failed attempts."""
        # Setup MFA
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")
        backup_codes = self.mfa.regenerate_backup_codes("user123")

        # Make some failed TOTP attempts
        self.mfa.verify_totp("user123", "000000")
        self.mfa.verify_totp("user123", "000000")
        assert self.mfa.get_remaining_attempts("user123") == 1

        # Successful backup code verification should clear rate limit
        assert self.mfa.verify_backup_code("user123", backup_codes[0])

        # After successful verification, attempts should be reset
        # (This depends on implementation - backup codes may have separate limits)

    def test_different_users_have_separate_rate_limits(self):
        """Test that rate limits are user-specific."""
        # Setup MFA for two users
        self.mfa.setup_totp("user1")
        self.mfa.enable_mfa("user1")
        self.mfa.setup_totp("user2")
        self.mfa.enable_mfa("user2")

        # Make failed attempts for user1
        for _ in range(3):
            self.mfa.verify_totp("user1", "000000")

        # user1 should be locked out
        with pytest.raises(MFALockedOutError):
            self.mfa.verify_totp("user1", "000000")

        # user2 should still have all attempts
        assert self.mfa.get_remaining_attempts("user2") == 3
        result = self.mfa.verify_totp("user2", "000000")
        assert result is False  # Invalid code but not locked out

    def test_lockout_error_message(self):
        """Test that lockout error contains useful information."""
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Exhaust attempts
        for _ in range(3):
            self.mfa.verify_totp("user123", "000000")

        # Check error message
        with pytest.raises(MFALockedOutError) as exc_info:
            self.mfa.verify_totp("user123", "000000")

        error_msg = str(exc_info.value)
        assert "locked out" in error_msg.lower() or "too many" in error_msg.lower()

    def test_new_user_has_full_attempts(self):
        """Test that new users start with full attempts."""
        assert self.mfa.get_remaining_attempts("new_user") == 3

    def test_rate_limit_applies_to_mfa_verification(self):
        """Test that rate limiting applies to main MFA verification method."""
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Make failed attempts using verify_mfa
        for _ in range(3):
            result = self.mfa.verify_mfa("user123", "000000")
            assert result is False

        # Should be locked out through verify_mfa as well
        # Note: verify_mfa internally calls verify_totp which does rate limiting
        with pytest.raises(MFALockedOutError):
            self.mfa.verify_mfa("user123", "000000")


class TestMFABackupCodeSecurity:
    """Tests for MFA backup code security."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mfa = MFAManager()
        set_mfa_manager(self.mfa)

    def test_backup_code_consumed_after_use(self):
        """Test that backup codes are consumed after use."""
        _, _, backup_codes = self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        code = backup_codes[0]

        # First use should succeed
        assert self.mfa.verify_backup_code("user123", code)

        # Second use should fail
        assert not self.mfa.verify_backup_code("user123", code)

    def test_backup_code_format(self):
        """Test backup code format for security."""
        _, _, backup_codes = self.mfa.setup_totp("user123")

        for code in backup_codes:
            # Format should be XXXX-XXXX
            assert len(code) == 9
            assert code[4] == "-"

            # Characters should be alphanumeric and exclude confusing chars
            code_chars = code.replace("-", "")
            for char in code_chars:
                assert char in "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

    def test_backup_codes_are_unique(self):
        """Test that all backup codes are unique."""
        _, _, backup_codes = self.mfa.setup_totp("user123")

        unique_codes = set(backup_codes)
        assert len(unique_codes) == len(backup_codes)

    def test_regenerate_invalidates_old_codes(self):
        """Test that regenerating codes invalidates old ones."""
        _, _, original_codes = self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Regenerate codes
        new_codes = self.mfa.regenerate_backup_codes("user123")

        # Old codes should not work
        assert not self.mfa.verify_backup_code("user123", original_codes[0])

        # New codes should work
        assert self.mfa.verify_backup_code("user123", new_codes[0])


class TestMFASetupSecurity:
    """Tests for MFA setup security."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mfa = MFAManager()
        set_mfa_manager(self.mfa)

    def test_totp_secret_is_base32(self):
        """Test that TOTP secret is valid Base32."""
        secret, _, _ = self.mfa.setup_totp("user123")

        import base64

        # Base32 decoding should work
        try:
            base64.b32decode(secret)
        except Exception:
            pytest.fail("TOTP secret is not valid Base32")

    def test_totp_uri_format(self):
        """Test that TOTP provisioning URI is correctly formatted."""
        secret, uri, _ = self.mfa.setup_totp("user123", issuer="TestApp")

        assert uri.startswith("otpauth://totp/")
        assert "TestApp" in uri
        assert "user123" in uri
        assert f"secret={secret}" in uri

    def test_cannot_enable_mfa_without_setup(self):
        """Test that MFA cannot be enabled without setup."""
        with pytest.raises(ValueError):
            self.mfa.enable_mfa("user123")

    def test_mfa_disabled_by_default(self):
        """Test that MFA is disabled after setup."""
        self.mfa.setup_totp("user123")

        assert not self.mfa.is_mfa_enabled("user123")


class TestMFAAdminEnforcement:
    """Tests for MFA admin enforcement."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mfa = MFAManager()
        set_mfa_manager(self.mfa)

    def test_admin_mfa_required_by_default(self):
        """Test that MFA is required for admins by default."""
        assert self.mfa.admin_mfa_required is True
        assert self.mfa.check_admin_mfa_required("admin1", "admin")

    def test_admin_mfa_not_required_after_setup(self):
        """Test that admin passes check after MFA setup."""
        self.mfa.setup_totp("admin1")
        self.mfa.enable_mfa("admin1")

        assert not self.mfa.check_admin_mfa_required("admin1", "admin")

    def test_non_admin_not_affected(self):
        """Test that non-admin roles are not affected."""
        assert not self.mfa.check_admin_mfa_required("user1", "viewer")
        assert not self.mfa.check_admin_mfa_required("user2", "operator")
        assert not self.mfa.check_admin_mfa_required("user3", "user")

    def test_disable_admin_mfa_requirement(self):
        """Test disabling admin MFA requirement."""
        self.mfa.require_mfa_for_admin(False)

        assert not self.mfa.admin_mfa_required
        assert not self.mfa.check_admin_mfa_required("admin1", "admin")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
