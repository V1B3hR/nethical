"""
Tests for OAuth CSRF Protection

This module tests the OAuth CSRF validation including:
- State parameter validation
- State expiry handling
- CSRF attack detection
"""

import pytest
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs

from nethical.security.sso import (
    SSOManager,
    SSOProvider,
    SSOConfig,
    SSOError,
    get_sso_manager,
    set_sso_manager,
)


class TestOAuthCSRFProtection:
    """Tests for OAuth CSRF (state parameter) protection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)

        # Configure OAuth for testing
        self.sso.configure_oauth(
            config_name="test_oauth",
            client_id="test_client_id",
            client_secret="test_client_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )

    def test_state_parameter_generated(self):
        """Test that state parameter is generated during login initiation."""
        auth_url, state = self.sso.initiate_oauth_login("test_oauth")

        assert state is not None
        assert len(state) > 0

        # State should be in the URL
        parsed = urlparse(auth_url)
        params = parse_qs(parsed.query)
        assert "state" in params
        assert params["state"][0] == state

    def test_state_parameter_unique(self):
        """Test that state parameters are unique for each login."""
        _, state1 = self.sso.initiate_oauth_login("test_oauth")
        _, state2 = self.sso.initiate_oauth_login("test_oauth")
        _, state3 = self.sso.initiate_oauth_login("test_oauth")

        states = {state1, state2, state3}
        assert len(states) == 3

    def test_missing_state_rejected(self):
        """Test that callbacks without state are rejected."""
        # Start OAuth flow to get a valid state
        _, _ = self.sso.initiate_oauth_login("test_oauth")

        # Callback without state should be rejected
        with pytest.raises(SSOError, match="[Ss]tate|CSRF"):
            self.sso.handle_oauth_callback(
                authorization_response="https://test.nethical.local/callback?code=auth_code",
                config_name="test_oauth",
                expected_state=None,  # No state
            )

    def test_invalid_state_rejected(self):
        """Test that invalid state values are rejected."""
        # Start OAuth flow
        _, valid_state = self.sso.initiate_oauth_login("test_oauth")

        # Try callback with wrong state
        with pytest.raises(SSOError, match="[Ss]tate|CSRF"):
            self.sso.handle_oauth_callback(
                authorization_response="https://test.nethical.local/callback?code=auth_code&state=wrong_state",
                config_name="test_oauth",
                expected_state="wrong_state",
            )

    def test_forged_state_rejected(self):
        """Test that forged/fabricated state values are rejected."""
        # Don't initiate login, just try with fabricated state
        fabricated_state = "forged_csrf_token_12345"

        with pytest.raises(SSOError, match="[Ss]tate|CSRF"):
            self.sso.handle_oauth_callback(
                authorization_response=f"https://test.nethical.local/callback?code=auth_code&state={fabricated_state}",
                config_name="test_oauth",
                expected_state=fabricated_state,
            )

    def test_state_mismatch_rejected(self):
        """Test that mismatched state values are rejected."""
        # Start two OAuth flows
        _, state1 = self.sso.initiate_oauth_login("test_oauth")
        _, state2 = self.sso.initiate_oauth_login("test_oauth")

        # Try to use state1 but provide state2 as expected
        with pytest.raises(SSOError, match="[Ss]tate|mismatch|CSRF"):
            self.sso.handle_oauth_callback(
                authorization_response=f"https://test.nethical.local/callback?code=auth_code&state={state1}",
                config_name="test_oauth",
                expected_state=state2,
            )


class TestOAuthStateManagement:
    """Tests for OAuth state management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)

        self.sso.configure_oauth(
            config_name="test_oauth",
            client_id="test_client_id",
            client_secret="test_client_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )

    def test_state_stored_after_initiation(self):
        """Test that state is stored in pending states after initiation."""
        _, state = self.sso.initiate_oauth_login("test_oauth")

        # State should be in pending states
        assert state in self.sso._pending_oauth_states

    def test_state_consumed_after_callback(self):
        """Test that state is consumed after successful callback."""
        _, state = self.sso.initiate_oauth_login("test_oauth")

        # State should be pending
        assert state in self.sso._pending_oauth_states

        # Handle callback (will use mock since libraries not available)
        try:
            self.sso.handle_oauth_callback(
                authorization_response=f"https://test.nethical.local/callback?code=auth_code&state={state}",
                config_name="test_oauth",
                expected_state=state,
            )
        except ImportError:
            # Libraries not installed, but state should still be consumed
            pass

        # State should be consumed (removed from pending)
        assert state not in self.sso._pending_oauth_states

    def test_state_has_timestamp(self):
        """Test that stored state has creation timestamp."""
        _, state = self.sso.initiate_oauth_login("test_oauth")

        state_info = self.sso._pending_oauth_states.get(state)
        assert state_info is not None
        assert "created_at" in state_info
        assert isinstance(state_info["created_at"], datetime)


class TestOAuthFlowSecurity:
    """Tests for complete OAuth flow security."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)

        self.sso.configure_oauth(
            config_name="test_oauth",
            client_id="test_client_id",
            client_secret="test_client_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )

    def test_auth_url_contains_required_params(self):
        """Test that authorization URL contains all required parameters."""
        auth_url, state = self.sso.initiate_oauth_login("test_oauth")

        parsed = urlparse(auth_url)
        params = parse_qs(parsed.query)

        # Required OAuth parameters
        assert "client_id" in params
        assert params["client_id"][0] == "test_client_id"
        assert "redirect_uri" in params
        assert "response_type" in params
        assert "state" in params

    def test_invalid_config_rejected(self):
        """Test that invalid config raises error."""
        with pytest.raises(SSOError):
            self.sso.initiate_oauth_login("nonexistent_config")

    def test_wrong_provider_type_rejected(self):
        """Test that OAuth login with SAML config is rejected."""
        self.sso.configure_saml(
            config_name="test_saml",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )

        with pytest.raises(SSOError):
            self.sso.initiate_oauth_login("test_saml")


class TestOAuthConfigurationSecurity:
    """Tests for OAuth configuration security."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)

    def test_oidc_provider_type(self):
        """Test that OIDC provider is correctly identified."""
        config = self.sso.configure_oauth(
            config_name="oidc_test",
            client_id="client",
            client_secret="secret",
            authorization_url="https://oidc.example.com/auth",
            token_url="https://oidc.example.com/token",
            userinfo_url="https://oidc.example.com/userinfo",
        )

        assert config.provider == SSOProvider.OIDC

    def test_oauth2_provider_type(self):
        """Test that OAuth2 provider is correctly identified."""
        config = self.sso.configure_oauth(
            config_name="oauth2_test",
            client_id="client",
            client_secret="secret",
            authorization_url="https://oauth.example.com/auth",
            token_url="https://oauth.example.com/token",
        )

        assert config.provider == SSOProvider.OAUTH2

    def test_redirect_uri_auto_generated(self):
        """Test that redirect URI is auto-generated if not provided."""
        config = self.sso.configure_oauth(
            config_name="test",
            client_id="client",
            client_secret="secret",
            authorization_url="https://oauth.example.com/auth",
            token_url="https://oauth.example.com/token",
        )

        assert config.oauth_config["redirect_uri"] == "https://test.nethical.local/auth/oauth/callback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
