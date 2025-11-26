"""Tests for OAuth CSRF protection."""

import pytest
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode, parse_qs, urlparse

from nethical.security.sso import (
    SSOManager,
    SSOProvider,
    SSOError,
)


class TestOAuthCSRF:
    """Tests for OAuth state parameter validation."""

    @pytest.fixture
    def sso_manager(self):
        """Create SSO manager for tests."""
        return SSOManager(base_url="https://test.example.com")

    @pytest.fixture
    def configured_manager(self, sso_manager):
        """Create SSO manager with OAuth configured."""
        sso_manager.configure_oauth(
            config_name="test_provider",
            client_id="test_client_id",
            client_secret="test_client_secret",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            userinfo_url="https://auth.example.com/userinfo",
        )
        return sso_manager

    def test_state_generation_on_login(self, configured_manager):
        """Verify state parameter is generated during OAuth login initiation."""
        auth_url, state = configured_manager.initiate_oauth_login("test_provider")
        
        # State should be a non-empty string
        assert state is not None
        assert len(state) > 0
        
        # State should be stored for later validation
        assert state in configured_manager._pending_oauth_states

    def test_state_included_in_authorization_url(self, configured_manager):
        """Verify state parameter is included in authorization URL."""
        auth_url, state = configured_manager.initiate_oauth_login("test_provider")
        
        parsed = urlparse(auth_url)
        query_params = parse_qs(parsed.query)
        
        assert "state" in query_params
        assert query_params["state"][0] == state

    def test_state_validation_required(self, configured_manager):
        """Verify requests without state are rejected."""
        # First initiate login to get a valid state
        _, valid_state = configured_manager.initiate_oauth_login("test_provider")
        
        # Create callback URL without state parameter
        callback_url = "https://test.example.com/auth/oauth/callback?code=test_code"
        
        # Should raise error due to missing state
        with pytest.raises(SSOError) as exc_info:
            configured_manager.handle_oauth_callback(
                callback_url,
                config_name="test_provider",
                expected_state=None,  # No expected state
            )
        
        assert "state" in str(exc_info.value).lower() or "csrf" in str(exc_info.value).lower()

    def test_state_mismatch_rejected(self, configured_manager):
        """Verify mismatched state values are rejected."""
        # Initiate login to get a valid state
        _, valid_state = configured_manager.initiate_oauth_login("test_provider")
        
        # Create callback URL with different state
        wrong_state = "wrong_state_value"
        callback_url = f"https://test.example.com/auth/oauth/callback?code=test_code&state={wrong_state}"
        
        # Should reject due to state mismatch
        with pytest.raises(SSOError) as exc_info:
            configured_manager.handle_oauth_callback(
                callback_url,
                config_name="test_provider",
                expected_state=valid_state,  # Expects valid_state but gets wrong_state
            )
        
        error_msg = str(exc_info.value).lower()
        assert "state" in error_msg or "csrf" in error_msg

    def test_state_reuse_prevented(self, configured_manager):
        """Verify state cannot be reused after callback."""
        # Initiate login and get state
        _, state = configured_manager.initiate_oauth_login("test_provider")
        
        callback_url = f"https://test.example.com/auth/oauth/callback?code=test_code&state={state}"
        
        # First callback attempt - this will consume the state
        # (may fail due to other reasons, but state should be consumed)
        try:
            configured_manager.handle_oauth_callback(
                callback_url,
                config_name="test_provider",
                expected_state=state,
            )
        except Exception:
            pass  # Expected due to mock environment
        
        # State should be removed from pending states
        assert state not in configured_manager._pending_oauth_states

    def test_state_expiry(self, configured_manager):
        """Verify expired states are rejected."""
        # Initiate login
        _, state = configured_manager.initiate_oauth_login("test_provider")
        
        # Manually expire the state
        configured_manager._pending_oauth_states[state]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(minutes=20)  # Expired
        )
        
        callback_url = f"https://test.example.com/auth/oauth/callback?code=test_code&state={state}"
        
        with pytest.raises(SSOError) as exc_info:
            configured_manager.handle_oauth_callback(
                callback_url,
                config_name="test_provider",
                expected_state=state,
            )
        
        assert "expired" in str(exc_info.value).lower()

    def test_custom_state_parameter(self, configured_manager):
        """Verify custom state parameter can be provided."""
        custom_state = "my_custom_state_value"
        
        auth_url, returned_state = configured_manager.initiate_oauth_login(
            "test_provider",
            state=custom_state,
        )
        
        # Should use the provided state
        assert returned_state == custom_state
        assert custom_state in configured_manager._pending_oauth_states

    def test_state_is_cryptographically_random(self, configured_manager):
        """Verify auto-generated states are cryptographically random."""
        states = set()
        
        # Generate multiple states
        for _ in range(10):
            _, state = configured_manager.initiate_oauth_login("test_provider")
            states.add(state)
        
        # All states should be unique
        assert len(states) == 10
        
        # States should be sufficiently long (for security)
        for state in states:
            assert len(state) >= 32  # 32 chars = 192 bits minimum

    def test_invalid_state_does_not_reveal_valid_states(self, configured_manager):
        """Verify error messages don't reveal valid states."""
        # Create a valid state
        _, valid_state = configured_manager.initiate_oauth_login("test_provider")
        
        # Try with invalid state
        callback_url = "https://test.example.com/auth/oauth/callback?code=test_code&state=invalid"
        
        with pytest.raises(SSOError) as exc_info:
            configured_manager.handle_oauth_callback(
                callback_url,
                config_name="test_provider",
            )
        
        # Error message should not contain the valid state
        assert valid_state not in str(exc_info.value)


class TestOAuthConfiguration:
    """Tests for OAuth/OIDC configuration."""

    @pytest.fixture
    def sso_manager(self):
        """Create SSO manager."""
        return SSOManager(base_url="https://app.example.com")

    def test_oauth_config_creation(self, sso_manager):
        """Verify OAuth configuration is created correctly."""
        config = sso_manager.configure_oauth(
            config_name="google",
            client_id="google_client_id",
            client_secret="google_client_secret",
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        )
        
        assert config is not None
        assert config.provider == SSOProvider.OIDC
        assert config.oauth_config["client_id"] == "google_client_id"
        assert "redirect_uri" in config.oauth_config

    def test_oidc_vs_oauth_detection(self, sso_manager):
        """Verify OIDC vs OAuth detection based on userinfo_url."""
        # With userinfo_url -> OIDC
        oidc_config = sso_manager.configure_oauth(
            config_name="oidc_provider",
            client_id="client",
            client_secret="secret",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            userinfo_url="https://auth.example.com/userinfo",
        )
        assert oidc_config.provider == SSOProvider.OIDC
        
        # Without userinfo_url -> OAuth2
        oauth_config = sso_manager.configure_oauth(
            config_name="oauth_provider",
            client_id="client",
            client_secret="secret",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            userinfo_url=None,
        )
        assert oauth_config.provider == SSOProvider.OAUTH2

    def test_list_configs(self, sso_manager):
        """Verify configs can be listed."""
        sso_manager.configure_oauth(
            config_name="provider1",
            client_id="c1",
            client_secret="s1",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        sso_manager.configure_oauth(
            config_name="provider2",
            client_id="c2",
            client_secret="s2",
            authorization_url="https://auth2.example.com/authorize",
            token_url="https://auth2.example.com/token",
        )
        
        configs = sso_manager.list_configs()
        assert "provider1" in configs
        assert "provider2" in configs

    def test_get_config(self, sso_manager):
        """Verify config can be retrieved by name."""
        sso_manager.configure_oauth(
            config_name="test_config",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        
        config = sso_manager.get_config("test_config")
        assert config is not None
        assert config.oauth_config["client_id"] == "test_client"
        
        # Non-existent config
        assert sso_manager.get_config("nonexistent") is None


class TestSSOManagerInitialization:
    """Tests for SSO manager initialization."""

    def test_default_initialization(self):
        """Verify default SSO manager initialization."""
        manager = SSOManager()
        
        assert manager.base_url == "https://nethical.local"
        assert len(manager.configs) == 0

    def test_custom_base_url(self):
        """Verify custom base URL is used."""
        manager = SSOManager(base_url="https://custom.example.com")
        
        assert manager.base_url == "https://custom.example.com"

    def test_redirect_uri_construction(self):
        """Verify redirect URI is properly constructed."""
        manager = SSOManager(base_url="https://myapp.example.com")
        
        config = manager.configure_oauth(
            config_name="test",
            client_id="client",
            client_secret="secret",
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
        )
        
        assert config.oauth_config["redirect_uri"] == "https://myapp.example.com/auth/oauth/callback"
