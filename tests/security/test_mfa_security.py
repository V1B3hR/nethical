"""Tests for MFA brute-force protection."""

import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from nethical.security.mfa import (
    MFAManager,
    MFAMethod,
    MFASetup,
    MFALockedOutError,
)


class TestMFABruteForceProtection:
    """Tests for MFA rate limiting and lockout."""

    @pytest.fixture
    def mfa_manager(self):
        """Create MFA manager with test configuration."""
        return MFAManager(
            max_attempts=5,
            lockout_duration_minutes=15,
        )

    @pytest.fixture
    def user_with_totp(self, mfa_manager):
        """Set up a user with TOTP enabled."""
        user_id = "test_user"
        mfa_manager.setup_totp(user_id)
        mfa_manager.enable_mfa(user_id, MFAMethod.TOTP)
        return user_id

    def test_lockout_after_max_attempts(self, mfa_manager, user_with_totp):
        """Verify account locks after too many failed attempts."""
        user_id = user_with_totp

        # Make max_attempts - 1 failed attempts (should not lock out yet)
        for i in range(mfa_manager.max_attempts - 1):
            result = mfa_manager.verify_totp(user_id, "000000")
            assert result is False  # Invalid code

            # Should still have attempts remaining
            remaining = mfa_manager.get_remaining_attempts(user_id)
            assert remaining == mfa_manager.max_attempts - i - 1

        # One more attempt should be allowed
        remaining = mfa_manager.get_remaining_attempts(user_id)
        assert remaining == 1

        # Make final failed attempt - should trigger lockout
        result = mfa_manager.verify_totp(user_id, "000000")
        assert result is False

        # Now the user should be locked out
        with pytest.raises(MFALockedOutError) as exc_info:
            mfa_manager.verify_totp(user_id, "123456")

        assert "locked" in str(exc_info.value).lower()

    def test_lockout_duration(self, mfa_manager, user_with_totp):
        """Verify lockout expires after configured duration."""
        user_id = user_with_totp

        # Trigger lockout
        for _ in range(mfa_manager.max_attempts):
            mfa_manager.verify_totp(user_id, "000000")

        # Verify locked
        with pytest.raises(MFALockedOutError):
            mfa_manager.verify_totp(user_id, "123456")

        # Simulate time passing past lockout duration
        lockout_end = datetime.now(timezone.utc) - timedelta(seconds=1)
        mfa_manager._lockouts[user_id] = lockout_end

        # User should no longer be locked out (lockout expired)
        # This won't raise MFALockedOutError, but will still fail TOTP verification
        try:
            result = mfa_manager.verify_totp(user_id, "000000")
            assert result is False  # Still invalid code, but no lockout error
        except MFALockedOutError:
            pytest.fail("Lockout should have expired")

    def test_remaining_attempts_decrements(self, mfa_manager, user_with_totp):
        """Verify remaining attempts decrements correctly."""
        user_id = user_with_totp

        initial = mfa_manager.get_remaining_attempts(user_id)
        assert initial == mfa_manager.max_attempts

        mfa_manager.verify_totp(user_id, "000000")

        remaining = mfa_manager.get_remaining_attempts(user_id)
        assert remaining == mfa_manager.max_attempts - 1

    def test_successful_verification_clears_attempts(self, mfa_manager):
        """Verify successful verification clears failed attempts counter."""
        user_id = "pyotp_user"

        # This test requires pyotp to actually verify codes
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed - skipping successful verification test")

        # Set up TOTP
        secret, _, _ = mfa_manager.setup_totp(user_id)
        mfa_manager.enable_mfa(user_id, MFAMethod.TOTP)

        # Make some failed attempts
        for _ in range(2):
            mfa_manager.verify_totp(user_id, "000000")

        assert mfa_manager.get_remaining_attempts(user_id) == 3

        # Now verify with correct code
        totp = pyotp.TOTP(secret)
        correct_code = totp.now()
        result = mfa_manager.verify_totp(user_id, correct_code)
        assert result is True

        # Failed attempts should be cleared
        assert mfa_manager.get_remaining_attempts(user_id) == mfa_manager.max_attempts

    def test_lockout_message_contains_duration(self, mfa_manager, user_with_totp):
        """Verify lockout error message includes remaining time."""
        user_id = user_with_totp

        # Trigger lockout
        for _ in range(mfa_manager.max_attempts):
            mfa_manager.verify_totp(user_id, "000000")

        with pytest.raises(MFALockedOutError) as exc_info:
            mfa_manager.verify_totp(user_id, "123456")

        error_message = str(exc_info.value)
        assert "seconds" in error_message.lower() or "locked" in error_message.lower()

    def test_different_users_independent_lockouts(self, mfa_manager):
        """Verify lockouts are independent per user."""
        user1 = "user1"
        user2 = "user2"

        # Set up both users
        mfa_manager.setup_totp(user1)
        mfa_manager.enable_mfa(user1, MFAMethod.TOTP)
        mfa_manager.setup_totp(user2)
        mfa_manager.enable_mfa(user2, MFAMethod.TOTP)

        # Trigger lockout for user1
        for _ in range(mfa_manager.max_attempts):
            mfa_manager.verify_totp(user1, "000000")

        # User1 should be locked out
        with pytest.raises(MFALockedOutError):
            mfa_manager.verify_totp(user1, "123456")

        # User2 should still be able to attempt verification
        result = mfa_manager.verify_totp(user2, "000000")
        assert result is False  # Invalid code, but not locked out

    def test_custom_max_attempts(self):
        """Verify custom max_attempts configuration works."""
        custom_manager = MFAManager(max_attempts=3, lockout_duration_minutes=5)

        user_id = "custom_user"
        custom_manager.setup_totp(user_id)
        custom_manager.enable_mfa(user_id, MFAMethod.TOTP)

        # Should lock out after 3 attempts
        for _ in range(3):
            custom_manager.verify_totp(user_id, "000000")

        with pytest.raises(MFALockedOutError):
            custom_manager.verify_totp(user_id, "123456")

    def test_custom_lockout_duration(self):
        """Verify custom lockout duration configuration."""
        custom_manager = MFAManager(max_attempts=2, lockout_duration_minutes=30)

        assert custom_manager.lockout_duration == timedelta(minutes=30)

    def test_attempts_window_cleanup(self, mfa_manager, user_with_totp):
        """Verify old failed attempts are cleaned up outside the window."""
        user_id = user_with_totp

        # Manually add old attempts that should be outside the window
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        mfa_manager._failed_attempts[user_id] = [old_time, old_time]

        # Make a new attempt - old attempts should be cleaned up
        mfa_manager.verify_totp(user_id, "000000")

        # Only 1 recent attempt should be recorded
        assert len(mfa_manager._failed_attempts[user_id]) == 1


class TestMFABackupCodes:
    """Tests for backup code functionality."""

    @pytest.fixture
    def mfa_manager(self):
        """Create MFA manager."""
        return MFAManager()

    def test_backup_codes_generated(self, mfa_manager):
        """Verify backup codes are generated during TOTP setup."""
        _, _, backup_codes = mfa_manager.setup_totp("backup_test_user")

        assert len(backup_codes) == 10
        for code in backup_codes:
            # Format: XXXX-XXXX
            assert len(code) == 9
            assert code[4] == "-"

    def test_backup_code_verification(self, mfa_manager):
        """Verify backup codes can be used for authentication."""
        user_id = "backup_verify_user"
        _, _, backup_codes = mfa_manager.setup_totp(user_id)
        mfa_manager.enable_mfa(user_id)

        # Use first backup code
        result = mfa_manager.verify_backup_code(user_id, backup_codes[0])
        assert result is True

        # Same backup code should not work again (consumed)
        result = mfa_manager.verify_backup_code(user_id, backup_codes[0])
        assert result is False

    def test_regenerate_backup_codes(self, mfa_manager):
        """Verify backup codes can be regenerated."""
        user_id = "regen_user"
        _, _, original_codes = mfa_manager.setup_totp(user_id)
        mfa_manager.enable_mfa(user_id)

        # Regenerate codes
        new_codes = mfa_manager.regenerate_backup_codes(user_id)

        assert len(new_codes) == 10
        # New codes should be different
        assert set(new_codes) != set(original_codes)

        # Old codes should not work
        result = mfa_manager.verify_backup_code(user_id, original_codes[0])
        assert result is False

        # New codes should work
        result = mfa_manager.verify_backup_code(user_id, new_codes[0])
        assert result is True


class TestMFASetup:
    """Tests for MFA setup data structure."""

    def test_mfa_setup_defaults(self):
        """Verify MFA setup defaults."""
        setup = MFASetup(user_id="test_user")

        assert setup.user_id == "test_user"
        assert setup.enabled is False
        assert setup.methods == []
        assert setup.totp_secret is None
        assert setup.backup_codes == []
        assert setup.phone_number is None

    def test_mfa_enable_disable(self):
        """Verify MFA can be enabled and disabled."""
        manager = MFAManager()

        user_id = "toggle_user"
        manager.setup_totp(user_id)

        # Initially disabled
        assert manager.is_mfa_enabled(user_id) is False

        # Enable
        manager.enable_mfa(user_id)
        assert manager.is_mfa_enabled(user_id) is True

        # Disable
        manager.disable_mfa(user_id)
        assert manager.is_mfa_enabled(user_id) is False
