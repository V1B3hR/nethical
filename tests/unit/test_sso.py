"""
Tests for SSO/SAML Integration
"""

import pytest
from datetime import datetime, timezone

from nethical.security.sso import (
    SSOManager,
    SSOProvider,
    SSOConfig,
    SAMLConfig,
    SSOError,
    get_sso_manager,
    set_sso_manager,
)


class TestSAMLConfig:
    """Test cases for SAMLConfig dataclass"""
    
    def test_saml_config_creation(self):
        """Test creating a SAML configuration"""
        config = SAMLConfig(
            sp_entity_id="https://nethical.local",
            sp_acs_url="https://nethical.local/auth/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        assert config.sp_entity_id == "https://nethical.local"
        assert config.idp_entity_id == "https://idp.example.com"
        assert config.want_assertions_signed is True
    
    def test_saml_config_attribute_mapping(self):
        """Test SAML attribute mapping"""
        config = SAMLConfig(
            sp_entity_id="https://nethical.local",
            sp_acs_url="https://nethical.local/auth/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        # Default mappings should exist
        assert 'email' in config.attribute_mapping
        assert 'uid' in config.attribute_mapping
        assert config.attribute_mapping['email'] == 'email'


class TestSSOConfig:
    """Test cases for SSOConfig dataclass"""
    
    def test_sso_config_creation(self):
        """Test creating an SSO configuration"""
        config = SSOConfig(
            provider=SSOProvider.SAML,
            enabled=True,
        )
        
        assert config.provider == SSOProvider.SAML
        assert config.enabled is True
        assert config.auto_create_users is True
        assert config.default_role == "viewer"


class TestSSOManager:
    """Test cases for SSOManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)
    
    def test_initialization(self):
        """Test SSO manager initialization"""
        assert isinstance(self.sso, SSOManager)
        assert self.sso.base_url == "https://test.nethical.local"
        assert len(self.sso.configs) == 0
    
    def test_configure_saml(self):
        """Test configuring SAML authentication"""
        config = self.sso.configure_saml(
            config_name="test_saml",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        assert isinstance(config, SSOConfig)
        assert config.provider == SSOProvider.SAML
        assert config.saml_config is not None
        assert config.saml_config.sp_entity_id == "https://test.nethical.local"
        assert "test_saml" in self.sso.configs
    
    def test_configure_saml_with_custom_mapping(self):
        """Test SAML configuration with custom attribute mapping"""
        custom_mapping = {
            'emailAddress': 'email',
            'userId': 'user_id',
        }
        
        config = self.sso.configure_saml(
            config_name="custom_saml",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            attribute_mapping=custom_mapping,
        )
        
        # Custom mappings should be added to defaults
        assert 'emailAddress' in config.saml_config.attribute_mapping
        assert config.saml_config.attribute_mapping['emailAddress'] == 'email'
    
    def test_configure_oauth(self):
        """Test configuring OAuth 2.0 authentication"""
        config = self.sso.configure_oauth(
            config_name="test_oauth",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )
        
        assert isinstance(config, SSOConfig)
        assert config.provider == SSOProvider.OAUTH2
        assert config.oauth_config is not None
        assert config.oauth_config['client_id'] == "test_client"
        assert "test_oauth" in self.sso.configs
    
    def test_configure_oidc(self):
        """Test configuring OpenID Connect authentication"""
        config = self.sso.configure_oauth(
            config_name="test_oidc",
            client_id="oidc_client",
            client_secret="oidc_secret",
            authorization_url="https://oidc.example.com/authorize",
            token_url="https://oidc.example.com/token",
            userinfo_url="https://oidc.example.com/userinfo",
        )
        
        assert config.provider == SSOProvider.OIDC
        assert config.oauth_config['userinfo_url'] == "https://oidc.example.com/userinfo"
    
    def test_get_config(self):
        """Test retrieving a configuration"""
        self.sso.configure_saml(
            config_name="test",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        config = self.sso.get_config("test")
        assert config is not None
        assert config.provider == SSOProvider.SAML
        
        # Non-existent config
        assert self.sso.get_config("nonexistent") is None
    
    def test_list_configs(self):
        """Test listing all configurations"""
        assert len(self.sso.list_configs()) == 0
        
        self.sso.configure_saml(
            config_name="saml1",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        self.sso.configure_oauth(
            config_name="oauth1",
            client_id="client",
            client_secret="secret",
            authorization_url="https://oauth.example.com/auth",
            token_url="https://oauth.example.com/token",
        )
        
        configs = self.sso.list_configs()
        assert len(configs) == 2
        assert "saml1" in configs
        assert "oauth1" in configs
    
    def test_initiate_saml_login(self):
        """Test initiating SAML login"""
        self.sso.configure_saml(
            config_name="test",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        # This will return a stub URL since python3-saml is not installed
        login_url = self.sso.initiate_saml_login("test")
        
        assert isinstance(login_url, str)
        assert len(login_url) > 0
        assert "idp.example.com" in login_url
    
    def test_initiate_saml_login_invalid_config(self):
        """Test initiating SAML login with invalid config"""
        with pytest.raises(SSOError):
            self.sso.initiate_saml_login("nonexistent")
    
    def test_initiate_saml_login_wrong_provider(self):
        """Test initiating SAML login with OAuth config"""
        self.sso.configure_oauth(
            config_name="oauth",
            client_id="client",
            client_secret="secret",
            authorization_url="https://oauth.example.com/auth",
            token_url="https://oauth.example.com/token",
        )
        
        with pytest.raises(SSOError):
            self.sso.initiate_saml_login("oauth")
    
    def test_handle_saml_response(self):
        """Test handling SAML response"""
        self.sso.configure_saml(
            config_name="test",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
        )
        
        # Mock SAML response (will use fallback)
        user_data = self.sso.handle_saml_response(
            saml_response="mock_response",
            config_name="test"
        )
        
        assert isinstance(user_data, dict)
        assert 'user_id' in user_data or 'email' in user_data
    
    def test_initiate_oauth_login(self):
        """Test initiating OAuth login"""
        self.sso.configure_oauth(
            config_name="test",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )
        
        # This will return a stub URL
        auth_url = self.sso.initiate_oauth_login("test")
        
        assert isinstance(auth_url, str)
        assert "oauth.example.com" in auth_url
        assert "client_id=test_client" in auth_url
    
    def test_handle_oauth_callback(self):
        """Test handling OAuth callback"""
        self.sso.configure_oauth(
            config_name="test",
            client_id="test_client",
            client_secret="test_secret",
            authorization_url="https://oauth.example.com/authorize",
            token_url="https://oauth.example.com/token",
        )
        
        # Mock callback response (will use fallback)
        user_info = self.sso.handle_oauth_callback(
            authorization_response="https://test.nethical.local/callback?code=test_code",
            config_name="test"
        )
        
        assert isinstance(user_info, dict)
        assert 'sub' in user_info or 'email' in user_info
    
    def test_build_saml_settings(self):
        """Test building SAML settings dictionary"""
        saml_config = SAMLConfig(
            sp_entity_id="https://test.nethical.local",
            sp_acs_url="https://test.nethical.local/auth/saml/acs",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/sso",
            idp_x509_cert="MOCK_CERT",
        )
        
        settings = self.sso._build_saml_settings(saml_config)
        
        assert isinstance(settings, dict)
        assert 'sp' in settings
        assert 'idp' in settings
        assert 'security' in settings
        assert settings['sp']['entityId'] == "https://test.nethical.local"
        assert settings['idp']['entityId'] == "https://idp.example.com"
        assert settings['idp']['x509cert'] == "MOCK_CERT"
    
    def test_map_saml_attributes(self):
        """Test mapping SAML attributes to internal format"""
        attributes = {
            'email': ['user@example.com'],
            'uid': ['user123'],
            'cn': ['John Doe'],
            'memberOf': ['group1', 'group2'],
        }
        
        mapping = {
            'email': 'email',
            'uid': 'user_id',
            'cn': 'name',
            'memberOf': 'groups',
        }
        
        user_data = self.sso._map_saml_attributes(
            attributes,
            name_id="user@example.com",
            mapping=mapping
        )
        
        assert user_data['name_id'] == "user@example.com"
        assert user_data['email'] == "user@example.com"
        assert user_data['user_id'] == "user123"
        assert user_data['name'] == "John Doe"
        assert user_data['groups'] == "group1"  # First value from list


class TestSSOIntegration:
    """Integration tests for SSO system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.sso = SSOManager(base_url="https://test.nethical.local")
        set_sso_manager(self.sso)
    
    def test_multiple_sso_providers(self):
        """Test configuring multiple SSO providers"""
        # Configure SAML
        saml_config = self.sso.configure_saml(
            config_name="corporate_saml",
            sp_entity_id="https://test.nethical.local",
            idp_entity_id="https://corp-idp.example.com",
            idp_sso_url="https://corp-idp.example.com/sso",
        )
        
        # Configure OAuth
        oauth_config = self.sso.configure_oauth(
            config_name="google_oauth",
            client_id="google_client",
            client_secret="google_secret",
            authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
        )
        
        # Configure OIDC
        oidc_config = self.sso.configure_oauth(
            config_name="azure_oidc",
            client_id="azure_client",
            client_secret="azure_secret",
            authorization_url="https://login.microsoftonline.com/authorize",
            token_url="https://login.microsoftonline.com/token",
            userinfo_url="https://graph.microsoft.com/oidc/userinfo",
        )
        
        # Verify all configs exist
        configs = self.sso.list_configs()
        assert len(configs) == 3
        assert "corporate_saml" in configs
        assert "google_oauth" in configs
        assert "azure_oidc" in configs
        
        # Verify provider types
        assert saml_config.provider == SSOProvider.SAML
        assert oauth_config.provider == SSOProvider.OAUTH2
        assert oidc_config.provider == SSOProvider.OIDC
    
    def test_saml_configuration_flow(self):
        """Test complete SAML configuration flow"""
        # 1. Configure SAML
        config = self.sso.configure_saml(
            config_name="production",
            sp_entity_id="https://nethical.production.com",
            idp_entity_id="https://idp.company.com",
            idp_sso_url="https://idp.company.com/sso",
            idp_x509_cert="PRODUCTION_CERT",
            attribute_mapping={
                'employeeId': 'user_id',
                'mail': 'email',
            }
        )
        
        # 2. Verify configuration
        assert config.saml_config.sp_entity_id == "https://nethical.production.com"
        assert config.saml_config.idp_x509_cert == "PRODUCTION_CERT"
        assert 'employeeId' in config.saml_config.attribute_mapping
        
        # 3. Retrieve and verify
        retrieved = self.sso.get_config("production")
        assert retrieved is not None
        assert retrieved.provider == SSOProvider.SAML
    
    def test_oauth_configuration_flow(self):
        """Test complete OAuth configuration flow"""
        # 1. Configure OAuth
        config = self.sso.configure_oauth(
            config_name="github",
            client_id="github_client_123",
            client_secret="github_secret_456",
            authorization_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
        )
        
        # 2. Verify OAuth config
        assert config.oauth_config['client_id'] == "github_client_123"
        assert config.oauth_config['redirect_uri'] == "https://test.nethical.local/auth/oauth/callback"
        assert 'openid' in config.oauth_config['scope']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
