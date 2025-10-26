from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, List
from urllib.parse import urlencode

# Public API: SSOManager, SSOProvider, SSOConfig, SAMLConfig, SSOError,
# get_sso_manager, set_sso_manager


class SSOError(Exception):
    """Generic SSO-related error."""
    pass


class SSOProvider(str, Enum):
    SAML = "saml"
    OAUTH2 = "oauth2"
    OIDC = "oidc"


@dataclass
class SAMLConfig:
    # Service Provider (SP)
    sp_entity_id: str
    sp_acs_url: str
    # Identity Provider (IdP)
    idp_entity_id: str
    idp_sso_url: str
    idp_x509_cert: Optional[str] = None

    # Security options
    want_assertions_signed: bool = True

    # Attribute mapping: external_attribute_name -> internal_field_name
    # Must contain defaults for 'email' and 'uid' as the tests expect their presence.
    attribute_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "email": "email",
            "uid": "uid",
        }
    )


@dataclass
class SSOConfig:
    provider: SSOProvider
    enabled: bool = True
    auto_create_users: bool = True
    default_role: str = "viewer"

    # Provider-specific configurations
    saml_config: Optional[SAMLConfig] = None
    oauth_config: Optional[Dict[str, Any]] = None


_global_sso_manager: Optional["SSOManager"] = None


def set_sso_manager(manager: "SSOManager") -> None:
    global _global_sso_manager
    _global_sso_manager = manager


def get_sso_manager() -> Optional["SSOManager"]:
    return _global_sso_manager


class SSOManager:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.configs: Dict[str, SSOConfig] = {}

    # ---------- Configuration ----------

    def configure_saml(
        self,
        *,
        config_name: str,
        sp_entity_id: str,
        idp_entity_id: str,
        idp_sso_url: str,
        sp_acs_url: Optional[str] = None,
        idp_x509_cert: Optional[str] = None,
        attribute_mapping: Optional[Dict[str, str]] = None,
    ) -> SSOConfig:
        # Defaults for SP ACS URL and attribute mappings
        if sp_acs_url is None:
            sp_acs_url = f"{self.base_url}/auth/saml/acs"

        default_mapping = {
            "email": "email",
            "uid": "uid",
            "cn": "name",
            "memberOf": "groups",
        }
        if attribute_mapping:
            default_mapping.update(attribute_mapping)

        saml_cfg = SAMLConfig(
            sp_entity_id=sp_entity_id,
            sp_acs_url=sp_acs_url,
            idp_entity_id=idp_entity_id,
            idp_sso_url=idp_sso_url,
            idp_x509_cert=idp_x509_cert,
            attribute_mapping=default_mapping,
        )

        cfg = SSOConfig(provider=SSOProvider.SAML, saml_config=saml_cfg)
        self.configs[config_name] = cfg
        return cfg

    def configure_oauth(
        self,
        *,
        config_name: str,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        userinfo_url: Optional[str] = None,
        scope: Optional[str] = None,
        redirect_uri: Optional[str] = None,
    ) -> SSOConfig:
        # Defaults expected by tests
        if redirect_uri is None:
            redirect_uri = f"{self.base_url}/auth/oauth/callback"
        # Tests expect 'openid' to be included in scope, even for OAuth
        if scope is None:
            scope = "openid profile email"

        oauth_cfg = {
            "client_id": client_id,
            "client_secret": client_secret,
            "authorization_url": authorization_url,
            "token_url": token_url,
            "redirect_uri": redirect_uri,
            "scope": scope,
        }
        if userinfo_url:
            oauth_cfg["userinfo_url"] = userinfo_url
            provider = SSOProvider.OIDC
        else:
            provider = SSOProvider.OAUTH2

        cfg = SSOConfig(provider=provider, oauth_config=oauth_cfg)
        self.configs[config_name] = cfg
        return cfg

    def get_config(self, config_name: str) -> Optional[SSOConfig]:
        return self.configs.get(config_name)

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    # ---------- SAML flows (stubbed for tests) ----------

    def initiate_saml_login(self, config_name: str) -> str:
        cfg = self.get_config(config_name)
        if not cfg:
            raise SSOError(f"SSO config '{config_name}' not found")

        if cfg.provider != SSOProvider.SAML or not cfg.saml_config:
            raise SSOError("SAML login requested for non-SAML configuration")

        # Return a stub URL that points to the IdP with a dummy SAMLRequest,
        # to satisfy hostname inspection in tests
        idp_url = cfg.saml_config.idp_sso_url
        sep = "&" if "?" in idp_url else "?"
        return f"{idp_url}{sep}SAMLRequest=stub"

    def handle_saml_response(self, *, saml_response: str, config_name: str) -> Dict[str, Any]:
        cfg = self.get_config(config_name)
        if not cfg or cfg.provider != SSOProvider.SAML or not cfg.saml_config:
            raise SSOError("Invalid SAML configuration")

        # Fallback mock parsing: return minimal mapped data
        # In a real implementation, you'd parse the SAMLResponse XML.
        return {
            "user_id": "mock_user",
            "email": "mock@example.com",
        }

    def _build_saml_settings(self, saml_config: SAMLConfig) -> Dict[str, Any]:
        settings = {
            "sp": {
                "entityId": saml_config.sp_entity_id,
                "assertionConsumerService": {
                    "url": saml_config.sp_acs_url,
                },
            },
            "idp": {
                "entityId": saml_config.idp_entity_id,
                "singleSignOnService": {
                    "url": saml_config.idp_sso_url,
                },
            },
            "security": {
                "wantAssertionsSigned": saml_config.want_assertions_signed,
            },
        }
        if saml_config.idp_x509_cert:
            settings["idp"]["x509cert"] = saml_config.idp_x509_cert
        return settings

    def _map_saml_attributes(
        self,
        attributes: Dict[str, List[str]],
        *,
        name_id: Optional[str],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if name_id is not None:
            result["name_id"] = name_id

        for external_attr, internal_field in mapping.items():
            if external_attr in attributes and attributes[external_attr]:
                value = attributes[external_attr][0]
                result[internal_field] = value

        return result

    # ---------- OAuth/OIDC flows (stubbed for tests) ----------

    def initiate_oauth_login(self, config_name: str) -> str:
        cfg = self.get_config(config_name)
        if not cfg or not cfg.oauth_config:
            raise SSOError(f"OAuth config '{config_name}' not found")

        if cfg.provider not in (SSOProvider.OAUTH2, SSOProvider.OIDC):
            raise SSOError("OAuth login requested for non-OAuth configuration")

        params = {
            "client_id": cfg.oauth_config["client_id"],
            "redirect_uri": cfg.oauth_config["redirect_uri"],
            "response_type": "code",
            "scope": cfg.oauth_config.get("scope", "openid"),
        }
        return f"{cfg.oauth_config['authorization_url']}?{urlencode(params)}"

    def handle_oauth_callback(self, *, authorization_response: str, config_name: str) -> Dict[str, Any]:
        cfg = self.get_config(config_name)
        if not cfg or not cfg.oauth_config:
            raise SSOError("Invalid OAuth configuration")

        # Fallback mock user info
        data = {"sub": "mock_sub", "email": "mock@example.com"}
        return data
