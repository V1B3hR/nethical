"""
SSO/SAML Integration Support for Nethical

This module provides Single Sign-On (SSO) and SAML 2.0 authentication support:
- SAML 2.0 Service Provider (SP) implementation
- OAuth 2.0 / OpenID Connect integration stubs
- Identity Provider (IdP) configuration
- User attribute mapping

Dependencies:
    - python3-saml (for SAML support)
    - requests-oauthlib (for OAuth/OIDC)

For production deployment:
1. Install dependencies: pip install python3-saml requests-oauthlib
2. Configure IdP metadata and certificates
3. Set up SP entity ID and endpoints
4. Map SAML attributes to user profiles
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin

__all__ = [
    "SSOProvider",
    "SSOConfig",
    "SAMLConfig",
    "SSOManager",
    "SSOError",
    "get_sso_manager",
    "set_sso_manager",
]

log = logging.getLogger(__name__)


class SSOProvider(str, Enum):
    """Supported SSO providers"""

    SAML = "saml"
    OAUTH2 = "oauth2"
    OIDC = "oidc"  # OpenID Connect
    LDAP = "ldap"


class SSOError(Exception):
    """Base exception for SSO errors"""


@dataclass
class SAMLConfig:
    """SAML 2.0 configuration"""

    # Service Provider (SP) configuration
    sp_entity_id: str
    sp_acs_url: str  # Assertion Consumer Service URL
    sp_sls_url: Optional[str] = None  # Single Logout Service URL

    # Identity Provider (IdP) configuration
    idp_entity_id: str = ""
    idp_sso_url: str = ""
    idp_slo_url: Optional[str] = None
    idp_x509_cert: Optional[str] = None

    # Security settings
    want_assertions_signed: bool = True
    want_messages_signed: bool = True
    want_name_id_encrypted: bool = False

    # Attribute mapping (SAML attribute -> internal field)
    attribute_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "email": "email",
            "uid": "user_id",
            "cn": "name",
            "givenName": "first_name",
            "sn": "last_name",
            "memberOf": "groups",
        }
    )

    # Organization info
    organization_name: str = "Nethical"
    organization_display_name: str = "Nethical AI Governance"
    organization_url: str = "https://nethical.ai"


@dataclass
class SSOConfig:
    """General SSO configuration"""

    provider: SSOProvider
    enabled: bool = True

    # Provider-specific config
    saml_config: Optional[SAMLConfig] = None
    oauth_config: Optional[Dict[str, Any]] = None

    # User provisioning
    auto_create_users: bool = True
    default_role: str = "viewer"

    # Session management
    session_timeout: int = 3600  # seconds

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SSOManager:
    """
    Single Sign-On Manager

    Provides SSO/SAML authentication integration.

    Note: This is a basic implementation scaffold.
    For production, use established libraries like:
    - python3-saml for SAML 2.0
    - authlib for OAuth/OIDC
    """

    def __init__(self, base_url: str = "https://nethical.local"):
        """
        Initialize SSO Manager

        Args:
            base_url: Base URL of the application
        """
        self.base_url = base_url
        self.configs: Dict[str, SSOConfig] = {}
        self.saml_auth = None
        log.info(f"SSOManager initialized with base_url: {base_url}")

    def configure_saml(
        self,
        config_name: str,
        sp_entity_id: str,
        idp_entity_id: str,
        idp_sso_url: str,
        idp_x509_cert: Optional[str] = None,
        attribute_mapping: Optional[Dict[str, str]] = None,
    ) -> SSOConfig:
        """
        Configure SAML 2.0 authentication

        Args:
            config_name: Configuration name/identifier
            sp_entity_id: Service Provider entity ID
            idp_entity_id: Identity Provider entity ID
            idp_sso_url: IdP Single Sign-On URL
            idp_x509_cert: IdP X.509 certificate (PEM format)
            attribute_mapping: Custom attribute mapping

        Returns:
            SSOConfig object
        """
        # Construct SP URLs
        sp_acs_url = urljoin(self.base_url, "/auth/saml/acs")
        sp_sls_url = urljoin(self.base_url, "/auth/saml/sls")

        saml_config = SAMLConfig(
            sp_entity_id=sp_entity_id,
            sp_acs_url=sp_acs_url,
            sp_sls_url=sp_sls_url,
            idp_entity_id=idp_entity_id,
            idp_sso_url=idp_sso_url,
            idp_x509_cert=idp_x509_cert,
        )

        if attribute_mapping:
            saml_config.attribute_mapping.update(attribute_mapping)

        sso_config = SSOConfig(
            provider=SSOProvider.SAML,
            saml_config=saml_config,
        )

        self.configs[config_name] = sso_config
        log.info(f"SAML configuration '{config_name}' created")

        return sso_config

    def configure_oauth(
        self,
        config_name: str,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        userinfo_url: Optional[str] = None,
    ) -> SSOConfig:
        """
        Configure OAuth 2.0 / OpenID Connect authentication

        Args:
            config_name: Configuration name/identifier
            client_id: OAuth client ID
            client_secret: OAuth client secret
            authorization_url: OAuth authorization endpoint
            token_url: OAuth token endpoint
            userinfo_url: Optional userinfo endpoint (for OIDC)

        Returns:
            SSOConfig object
        """
        oauth_config = {
            "client_id": client_id,
            "client_secret": client_secret,
            "authorization_url": authorization_url,
            "token_url": token_url,
            "userinfo_url": userinfo_url,
            "redirect_uri": urljoin(self.base_url, "/auth/oauth/callback"),
            "scope": ["openid", "email", "profile"],
        }

        provider = SSOProvider.OIDC if userinfo_url else SSOProvider.OAUTH2

        sso_config = SSOConfig(
            provider=provider,
            oauth_config=oauth_config,
        )

        self.configs[config_name] = sso_config
        log.info(f"OAuth/OIDC configuration '{config_name}' created")

        return sso_config

    def initiate_saml_login(self, config_name: str = "default") -> str:
        """
        Initiate SAML login flow

        Args:
            config_name: SSO configuration name

        Returns:
            SAML authentication request (redirect URL)

        Raises:
            SSOError: If configuration not found or SAML not available
        """
        if config_name not in self.configs:
            raise SSOError(f"SSO configuration '{config_name}' not found")

        config = self.configs[config_name]

        if config.provider != SSOProvider.SAML or not config.saml_config:
            raise SSOError("Configuration is not for SAML")

        try:
            from onelogin.saml2.auth import OneLogin_Saml2_Auth

            # Build SAML settings
            saml_settings = self._build_saml_settings(config.saml_config)

            # Create SAML auth object (requires request context in production)
            # This is a stub - in production, pass actual request object
            request_data = {
                "https": "on" if self.base_url.startswith("https") else "off",
                "http_host": self.base_url.split("://")[-1],
                "script_name": "/auth/saml/login",
                "server_port": 443 if self.base_url.startswith("https") else 80,
            }

            auth = OneLogin_Saml2_Auth(request_data, saml_settings)
            sso_url = auth.login()

            log.info(f"SAML login initiated for config '{config_name}'")
            return sso_url

        except ImportError:
            log.warning("python3-saml not installed, returning stub URL")
            # Return a stub for demonstration
            saml_cfg = config.saml_config
            return (
                f"{saml_cfg.idp_sso_url}?" f"SAMLRequest=<encoded_request>&" f"RelayState=<state>"
            )

    def handle_saml_response(
        self, saml_response: str, config_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Handle SAML authentication response

        Args:
            saml_response: Base64-encoded SAML response
            config_name: SSO configuration name

        Returns:
            User attributes dictionary

        Raises:
            SSOError: If response is invalid or authentication fails
        """
        if config_name not in self.configs:
            raise SSOError(f"SSO configuration '{config_name}' not found")

        config = self.configs[config_name]

        if config.provider != SSOProvider.SAML or not config.saml_config:
            raise SSOError("Configuration is not for SAML")

        try:
            from onelogin.saml2.auth import OneLogin_Saml2_Auth

            # Build SAML settings
            saml_settings = self._build_saml_settings(config.saml_config)

            # Process SAML response (requires request context in production)
            request_data = {
                "https": "on" if self.base_url.startswith("https") else "off",
                "http_host": self.base_url.split("://")[-1],
                "script_name": "/auth/saml/acs",
                "server_port": 443 if self.base_url.startswith("https") else 80,
                "post_data": {"SAMLResponse": saml_response},
            }

            auth = OneLogin_Saml2_Auth(request_data, saml_settings)
            auth.process_response()

            errors = auth.get_errors()
            if errors:
                raise SSOError(f"SAML authentication failed: {', '.join(errors)}")

            if not auth.is_authenticated():
                raise SSOError("SAML authentication failed: User not authenticated")

            # Extract user attributes
            attributes = auth.get_attributes()
            name_id = auth.get_nameid()

            # Map SAML attributes to internal format
            user_data = self._map_saml_attributes(
                attributes, name_id, config.saml_config.attribute_mapping
            )

            log.info(f"SAML response processed successfully for user {name_id}")
            return user_data

        except ImportError:
            log.warning("python3-saml not installed, returning mock user data")
            # Return mock data for demonstration
            return {
                "user_id": "demo_user",
                "email": "demo@example.com",
                "name": "Demo User",
                "groups": ["users"],
            }

    def initiate_oauth_login(
        self, config_name: str = "default", state: Optional[str] = None
    ) -> str:
        """
        Initiate OAuth 2.0 / OIDC login flow

        Args:
            config_name: SSO configuration name
            state: Optional state parameter for CSRF protection

        Returns:
            OAuth authorization URL

        Raises:
            SSOError: If configuration not found
        """
        if config_name not in self.configs:
            raise SSOError(f"SSO configuration '{config_name}' not found")

        config = self.configs[config_name]
        oauth_cfg = config.oauth_config

        if not oauth_cfg:
            raise SSOError("Configuration is not for OAuth/OIDC")

        try:
            from requests_oauthlib import OAuth2Session

            oauth = OAuth2Session(
                oauth_cfg["client_id"],
                redirect_uri=oauth_cfg["redirect_uri"],
                scope=oauth_cfg["scope"],
            )

            authorization_url, state = oauth.authorization_url(
                oauth_cfg["authorization_url"],
                state=state,
            )

            log.info(f"OAuth login initiated for config '{config_name}'")
            return authorization_url

        except ImportError:
            log.warning("requests-oauthlib not installed, returning stub URL")
            return (
                f"{oauth_cfg['authorization_url']}?"
                f"client_id={oauth_cfg['client_id']}&"
                f"redirect_uri={oauth_cfg['redirect_uri']}&"
                f"response_type=code&"
                f"scope=openid+email+profile"
            )

    def handle_oauth_callback(
        self, authorization_response: str, config_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and exchange code for token

        Args:
            authorization_response: Full callback URL with code
            config_name: SSO configuration name

        Returns:
            User information dictionary

        Raises:
            SSOError: If token exchange or userinfo fails
        """
        if config_name not in self.configs:
            raise SSOError(f"SSO configuration '{config_name}' not found")

        config = self.configs[config_name]
        oauth_cfg = config.oauth_config

        if not oauth_cfg:
            raise SSOError("Configuration is not for OAuth/OIDC")

        try:
            from requests_oauthlib import OAuth2Session

            oauth = OAuth2Session(
                oauth_cfg["client_id"],
                redirect_uri=oauth_cfg["redirect_uri"],
            )

            # Exchange code for token
            token = oauth.fetch_token(
                oauth_cfg["token_url"],
                authorization_response=authorization_response,
                client_secret=oauth_cfg["client_secret"],
            )

            # Fetch user info
            if oauth_cfg.get("userinfo_url"):
                userinfo = oauth.get(oauth_cfg["userinfo_url"]).json()
            else:
                # Decode ID token for OIDC
                import jwt

                userinfo = jwt.decode(token["id_token"], options={"verify_signature": False})

            log.info(f"OAuth callback processed for user {userinfo.get('sub')}")
            return userinfo

        except ImportError:
            log.warning("OAuth libraries not installed, returning mock user data")
            return {
                "sub": "demo_user",
                "email": "demo@example.com",
                "name": "Demo User",
            }

    def get_config(self, config_name: str) -> Optional[SSOConfig]:
        """
        Get SSO configuration by name

        Args:
            config_name: Configuration name

        Returns:
            SSOConfig or None if not found
        """
        return self.configs.get(config_name)

    def list_configs(self) -> List[str]:
        """
        List all SSO configuration names

        Returns:
            List of configuration names
        """
        return list(self.configs.keys())

    def _build_saml_settings(self, saml_config: SAMLConfig) -> Dict[str, Any]:
        """
        Build python3-saml settings dictionary

        Args:
            saml_config: SAML configuration

        Returns:
            Settings dictionary for python3-saml
        """
        settings = {
            "strict": True,
            "debug": False,
            "sp": {
                "entityId": saml_config.sp_entity_id,
                "assertionConsumerService": {
                    "url": saml_config.sp_acs_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST",
                },
                "singleLogoutService": {
                    "url": saml_config.sp_sls_url or "",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
            },
            "idp": {
                "entityId": saml_config.idp_entity_id,
                "singleSignOnService": {
                    "url": saml_config.idp_sso_url,
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "singleLogoutService": {
                    "url": saml_config.idp_slo_url or "",
                    "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect",
                },
                "x509cert": saml_config.idp_x509_cert or "",
            },
            "security": {
                "wantAssertionsSigned": saml_config.want_assertions_signed,
                "wantMessagesSigned": saml_config.want_messages_signed,
                "wantNameIdEncrypted": saml_config.want_name_id_encrypted,
            },
            "organization": {
                "en-US": {
                    "name": saml_config.organization_name,
                    "displayname": saml_config.organization_display_name,
                    "url": saml_config.organization_url,
                }
            },
        }

        return settings

    def _map_saml_attributes(
        self, attributes: Dict[str, List[str]], name_id: str, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Map SAML attributes to internal user data format

        Args:
            attributes: SAML attributes (dict of lists)
            name_id: SAML NameID
            mapping: Attribute mapping dictionary

        Returns:
            Mapped user data dictionary
        """
        user_data = {"name_id": name_id}

        for saml_attr, internal_field in mapping.items():
            if saml_attr in attributes:
                # Get first value from list
                value = attributes[saml_attr][0] if attributes[saml_attr] else None
                user_data[internal_field] = value

        return user_data


# Global SSO manager instance
_sso_manager: Optional[SSOManager] = None


def get_sso_manager() -> SSOManager:
    """Get or create the global SSO manager instance"""
    global _sso_manager
    if _sso_manager is None:
        _sso_manager = SSOManager()
    return _sso_manager


def set_sso_manager(manager: SSOManager) -> None:
    """Set the global SSO manager instance"""
    global _sso_manager
    _sso_manager = manager
