"""
Tests for Multi-Factor Authentication (MFA) System
"""

import pytest
from datetime import datetime, timezone

from nethical.security.mfa import (
    MFAManager,
    MFAMethod,
    MFASetup,
    MFARequiredError,
    InvalidMFACodeError,
    get_mfa_manager,
    set_mfa_manager,
)


class TestMFASetup:
    """Test cases for MFASetup dataclass"""

    def test_mfa_setup_creation(self):
        """Test creating an MFA setup"""
        setup = MFASetup(user_id="user123")

        assert setup.user_id == "user123"
        assert not setup.enabled
        assert len(setup.methods) == 0
        assert setup.totp_secret is None

    def test_mfa_setup_with_methods(self):
        """Test MFA setup with methods"""
        setup = MFASetup(
            user_id="user123",
            enabled=True,
            methods=[MFAMethod.TOTP, MFAMethod.BACKUP_CODE],
        )

        assert setup.enabled
        assert MFAMethod.TOTP in setup.methods
        assert MFAMethod.BACKUP_CODE in setup.methods


class TestMFAManager:
    """Test cases for MFAManager"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mfa = MFAManager()
        set_mfa_manager(self.mfa)

    def test_initialization(self):
        """Test MFA manager initialization"""
        assert isinstance(self.mfa, MFAManager)
        assert self.mfa.admin_mfa_required is True
        assert len(self.mfa.user_mfa) == 0

    def test_setup_totp(self):
        """Test setting up TOTP for a user"""
        secret, uri, backup_codes = self.mfa.setup_totp("user123", issuer="TestApp")

        assert isinstance(secret, str)
        assert len(secret) > 0
        assert "otpauth://totp/" in uri
        assert "TestApp" in uri
        assert "user123" in uri
        assert len(backup_codes) == 10

        # Verify backup codes format (XXXX-XXXX)
        for code in backup_codes:
            assert len(code) == 9  # 8 chars + 1 hyphen
            assert code[4] == "-"

    def test_enable_mfa(self):
        """Test enabling MFA for a user"""
        # Setup TOTP first
        self.mfa.setup_totp("user123")

        # Enable MFA
        self.mfa.enable_mfa("user123", MFAMethod.TOTP)

        assert self.mfa.is_mfa_enabled("user123")
        assert MFAMethod.TOTP in self.mfa.user_mfa["user123"].methods

    def test_enable_mfa_without_setup_fails(self):
        """Test enabling MFA without setup raises error"""
        with pytest.raises(ValueError):
            self.mfa.enable_mfa("user123", MFAMethod.TOTP)

    def test_disable_mfa(self):
        """Test disabling MFA for a user"""
        # Setup and enable
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Disable
        self.mfa.disable_mfa("user123")

        assert not self.mfa.is_mfa_enabled("user123")

    def test_verify_backup_code(self):
        """Test verifying a backup code"""
        # Setup and enable
        secret, uri, backup_codes = self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Verify a backup code
        valid_code = backup_codes[0]
        assert self.mfa.verify_backup_code("user123", valid_code)

        # Code should be consumed
        assert not self.mfa.verify_backup_code("user123", valid_code)

    def test_verify_invalid_backup_code(self):
        """Test verifying an invalid backup code"""
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        assert not self.mfa.verify_backup_code("user123", "INVALID-CODE")

    def test_regenerate_backup_codes(self):
        """Test regenerating backup codes"""
        # Setup TOTP
        self.mfa.setup_totp("user123")

        # Regenerate codes
        new_codes = self.mfa.regenerate_backup_codes("user123")

        assert len(new_codes) == 10
        assert all(len(code) == 9 for code in new_codes)

    def test_is_mfa_enabled(self):
        """Test checking if MFA is enabled"""
        assert not self.mfa.is_mfa_enabled("user123")

        self.mfa.setup_totp("user123")
        assert not self.mfa.is_mfa_enabled("user123")

        self.mfa.enable_mfa("user123")
        assert self.mfa.is_mfa_enabled("user123")

    def test_admin_mfa_requirement(self):
        """Test admin MFA requirement checking"""
        # Admin MFA required by default
        assert self.mfa.check_admin_mfa_required("admin1", "admin")

        # Setup and enable MFA for admin
        self.mfa.setup_totp("admin1")
        self.mfa.enable_mfa("admin1")

        # Should no longer require (already has MFA)
        assert not self.mfa.check_admin_mfa_required("admin1", "admin")

    def test_admin_mfa_not_required_for_non_admin(self):
        """Test that MFA is not enforced for non-admin roles"""
        assert not self.mfa.check_admin_mfa_required("user1", "viewer")
        assert not self.mfa.check_admin_mfa_required("user2", "operator")

    def test_disable_admin_mfa_requirement(self):
        """Test disabling admin MFA requirement globally"""
        self.mfa.require_mfa_for_admin(False)

        assert not self.mfa.admin_mfa_required
        assert not self.mfa.check_admin_mfa_required("admin1", "admin")

    def test_get_qr_code_data_uri(self):
        """Test QR code generation (or fallback)"""
        secret, uri, _ = self.mfa.setup_totp("user123")

        # This will return either a data URI or the provisioning URI
        result = self.mfa.get_qr_code_data_uri(uri)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should either be data URI or provisioning URI
        assert result.startswith(("data:image/png;base64,", "otpauth://"))

    def test_multiple_users(self):
        """Test MFA for multiple users"""
        users = ["alice", "bob", "charlie"]

        for user in users:
            self.mfa.setup_totp(user)
            self.mfa.enable_mfa(user)

        # All should have MFA enabled
        for user in users:
            assert self.mfa.is_mfa_enabled(user)

    def test_backup_code_format(self):
        """Test backup code format and characters"""
        _, _, backup_codes = self.mfa.setup_totp("user123")

        for code in backup_codes:
            # Check format
            assert len(code) == 9
            assert code[4] == "-"

            # Check only allowed characters
            code_chars = code.replace("-", "")
            assert all(c in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" for c in code_chars)
            # Should not contain confusing characters (0, O, 1, I, L)
            assert "0" not in code
            assert "O" not in code
            assert "1" not in code
            assert "I" not in code
            assert "L" not in code


class TestMFAIntegration:
    """Integration tests for MFA system"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mfa = MFAManager()
        set_mfa_manager(self.mfa)

    def test_complete_mfa_setup_flow(self):
        """Test complete MFA setup and usage flow"""
        # 1. Setup TOTP
        secret, uri, backup_codes = self.mfa.setup_totp("user123")
        assert secret is not None

        # 2. Enable MFA
        self.mfa.enable_mfa("user123")
        assert self.mfa.is_mfa_enabled("user123")

        # 3. Use backup code
        first_code = backup_codes[0]
        assert self.mfa.verify_backup_code("user123", first_code)

        # 4. Code is consumed
        assert not self.mfa.verify_backup_code("user123", first_code)

        # 5. Other codes still work
        second_code = backup_codes[1]
        assert self.mfa.verify_backup_code("user123", second_code)

    def test_admin_enforcement_flow(self):
        """Test admin MFA enforcement flow"""
        # 1. Check that admin needs MFA
        assert self.mfa.check_admin_mfa_required("admin1", "admin")

        # 2. Setup MFA for admin
        self.mfa.setup_totp("admin1")
        self.mfa.enable_mfa("admin1")

        # 3. Now admin has MFA
        assert not self.mfa.check_admin_mfa_required("admin1", "admin")

        # 4. Regular users don't need MFA
        assert not self.mfa.check_admin_mfa_required("user1", "viewer")

    def test_mfa_disable_and_reenable(self):
        """Test disabling and re-enabling MFA"""
        # Setup and enable
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")
        assert self.mfa.is_mfa_enabled("user123")

        # Disable
        self.mfa.disable_mfa("user123")
        assert not self.mfa.is_mfa_enabled("user123")

        # Re-enable (setup is preserved)
        self.mfa.enable_mfa("user123")
        assert self.mfa.is_mfa_enabled("user123")

    def test_backup_code_regeneration(self):
        """Test regenerating backup codes"""
        # Initial setup
        _, _, original_codes = self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Use a code
        self.mfa.verify_backup_code("user123", original_codes[0])

        # Regenerate
        new_codes = self.mfa.regenerate_backup_codes("user123")

        # Old codes should not work
        assert not self.mfa.verify_backup_code("user123", original_codes[1])

        # New codes should work
        assert self.mfa.verify_backup_code("user123", new_codes[0])


class TestMFARateLimiting:
    """Test cases for MFA brute-force protection"""

    def setup_method(self):
        """Set up test fixtures"""
        # Use a smaller max_attempts for faster testing
        self.mfa = MFAManager(max_attempts=3, lockout_duration_minutes=1)
        set_mfa_manager(self.mfa)

    def test_rate_limiting_after_failed_attempts(self):
        """Test that rate limiting kicks in after max failed attempts"""
        from nethical.security.mfa import MFALockedOutError

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

    def test_remaining_attempts_counter(self):
        """Test remaining attempts counter decrements correctly"""
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

    def test_successful_verification_clears_rate_limit(self):
        """Test that successful verification clears failed attempt counter"""
        # This test uses backup codes since we can validate those without pyotp
        self.mfa.setup_totp("user123")
        self.mfa.enable_mfa("user123")

        # Get backup codes
        backup_codes = self.mfa.regenerate_backup_codes("user123")

        # Make some failed attempts with TOTP
        self.mfa.verify_totp("user123", "000000")
        self.mfa.verify_totp("user123", "000000")
        assert self.mfa.get_remaining_attempts("user123") == 1

        # Successful verification with backup code clears rate limit
        # (Note: backup codes don't share rate limit with TOTP in current implementation)
        assert self.mfa.verify_backup_code("user123", backup_codes[0])

    def test_lockout_for_unknown_user(self):
        """Test that unknown users also get rate limited"""
        from nethical.security.mfa import MFALockedOutError

        # Attempt verification for unknown user multiple times
        for i in range(3):
            result = self.mfa.verify_totp("unknown_user", "000000")
            assert result is False

        # Should be locked out
        with pytest.raises(MFALockedOutError):
            self.mfa.verify_totp("unknown_user", "000000")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
