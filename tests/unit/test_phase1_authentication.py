"""
Unit tests for Phase 1: Military-Grade Authentication Module
"""

import pytest
from datetime import datetime, timedelta, timezone
from nethical.security.authentication import (
    AuthCredentials,
    AuthResult,
    ClearanceLevel,
    PKICertificateValidator,
    MultiFactorAuthEngine,
    SecureSessionManager,
    LDAPConnector,
    MilitaryGradeAuthProvider,
)


class TestPKICertificateValidator:
    """Test PKI Certificate Validator"""
    
    def test_initialization(self):
        """Test validator initialization"""
        validator = PKICertificateValidator()
        assert validator.enable_crl_check is True
        assert validator.enable_ocsp is True
    
    @pytest.mark.asyncio
    async def test_validate_no_certificate(self):
        """Test validation with no certificate"""
        validator = PKICertificateValidator()
        result = await validator.validate(None)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_validate_certificate(self):
        """Test certificate validation (stub)"""
        validator = PKICertificateValidator()
        cert = b"fake_certificate_data"
        result = await validator.validate(cert)
        assert result is True  # Stub implementation always returns True
    
    def test_extract_user_info(self):
        """Test user info extraction"""
        validator = PKICertificateValidator()
        cert = b"fake_certificate"
        info = validator.extract_user_info(cert)
        
        assert "common_name" in info
        assert "email" in info
        assert "organization" in info


class TestMultiFactorAuthEngine:
    """Test Multi-Factor Authentication Engine"""
    
    def test_initialization(self):
        """Test MFA engine initialization"""
        engine = MultiFactorAuthEngine()
        assert engine.require_mfa_for_critical is True
    
    @pytest.mark.asyncio
    async def test_challenge_no_mfa_enabled(self):
        """Test challenge when MFA not enabled"""
        engine = MultiFactorAuthEngine()
        result = await engine.challenge("user123")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_challenge_with_valid_totp(self):
        """Test challenge with valid TOTP code"""
        engine = MultiFactorAuthEngine()
        engine.setup_mfa("user123", method="totp")
        
        result = await engine.challenge("user123", mfa_code="123456")
        assert result is True  # Stub accepts 6-digit codes
    
    @pytest.mark.asyncio
    async def test_challenge_without_code(self):
        """Test challenge without MFA code"""
        engine = MultiFactorAuthEngine()
        engine.setup_mfa("user123")
        
        result = await engine.challenge("user123", mfa_code=None)
        assert result is False
    
    def test_setup_mfa(self):
        """Test MFA setup"""
        engine = MultiFactorAuthEngine()
        setup_info = engine.setup_mfa("user123", method="totp")
        
        assert setup_info["method"] == "totp"
        assert "secret" in setup_info
        assert "qr_code_url" in setup_info


class TestSecureSessionManager:
    """Test Secure Session Manager"""
    
    def test_initialization(self):
        """Test session manager initialization"""
        manager = SecureSessionManager(timeout=900)
        assert manager.timeout == 900
        assert manager.require_reauth_for_critical is True
    
    def test_create_session(self):
        """Test session creation"""
        manager = SecureSessionManager()
        token = manager.create_session(
            user_id="user123",
            clearance_level=ClearanceLevel.SECRET,
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_validate_session(self):
        """Test session validation"""
        manager = SecureSessionManager(timeout=900)
        token = manager.create_session(
            user_id="user123",
            clearance_level=ClearanceLevel.SECRET,
        )
        
        session = manager.validate_session(token)
        assert session is not None
        assert session["user_id"] == "user123"
        assert session["clearance_level"] == ClearanceLevel.SECRET
    
    def test_validate_invalid_session(self):
        """Test validation of invalid session"""
        manager = SecureSessionManager()
        session = manager.validate_session("invalid_token")
        assert session is None
    
    def test_revoke_session(self):
        """Test session revocation"""
        manager = SecureSessionManager()
        token = manager.create_session(
            user_id="user123",
            clearance_level=ClearanceLevel.SECRET,
        )
        
        result = manager.revoke_session(token)
        assert result is True
        
        # Should not be valid after revocation
        session = manager.validate_session(token)
        assert session is None
    
    def test_concurrent_session_limit(self):
        """Test concurrent session limiting"""
        manager = SecureSessionManager(max_concurrent_sessions=2)
        
        # Create 3 sessions for same user
        token1 = manager.create_session("user123", ClearanceLevel.SECRET)
        token2 = manager.create_session("user123", ClearanceLevel.SECRET)
        token3 = manager.create_session("user123", ClearanceLevel.SECRET)
        
        # First token should be revoked
        session1 = manager.validate_session(token1)
        assert session1 is None
        
        # Second and third should be valid
        session2 = manager.validate_session(token2)
        session3 = manager.validate_session(token3)
        assert session2 is not None
        assert session3 is not None


class TestLDAPConnector:
    """Test LDAP/Active Directory Connector"""
    
    def test_initialization(self):
        """Test LDAP connector initialization"""
        connector = LDAPConnector(
            server_url="ldaps://ldap.example.gov:636",
            base_dn="dc=example,dc=gov",
        )
        
        assert connector.server_url == "ldaps://ldap.example.gov:636"
        assert connector.base_dn == "dc=example,dc=gov"
    
    @pytest.mark.asyncio
    async def test_authenticate(self):
        """Test LDAP authentication (stub)"""
        connector = LDAPConnector(
            server_url="ldaps://ldap.example.gov:636",
            base_dn="dc=example,dc=gov",
        )
        
        result = await connector.authenticate("testuser", "password123")
        assert result is True  # Stub implementation accepts 8+ char passwords
    
    @pytest.mark.asyncio
    async def test_get_user_groups(self):
        """Test getting user groups"""
        connector = LDAPConnector(
            server_url="ldaps://ldap.example.gov:636",
            base_dn="dc=example,dc=gov",
        )
        
        groups = await connector.get_user_groups("testuser")
        assert isinstance(groups, list)
        assert len(groups) > 0
    
    @pytest.mark.asyncio
    async def test_get_clearance_level(self):
        """Test clearance level determination"""
        connector = LDAPConnector(
            server_url="ldaps://ldap.example.gov:636",
            base_dn="dc=example,dc=gov",
        )
        
        level = await connector.get_clearance_level("testuser")
        assert isinstance(level, ClearanceLevel)


class TestMilitaryGradeAuthProvider:
    """Test Military-Grade Authentication Provider"""
    
    def test_initialization(self):
        """Test provider initialization"""
        provider = MilitaryGradeAuthProvider()
        
        assert provider.pki_validator is not None
        assert provider.mfa_engine is not None
        assert provider.session_manager is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_with_certificate(self):
        """Test authentication with PKI certificate"""
        provider = MilitaryGradeAuthProvider()
        
        credentials = AuthCredentials(
            user_id="user123",
            certificate=b"fake_certificate",
        )
        
        result = await provider.authenticate(credentials)
        
        assert isinstance(result, AuthResult)
        assert result.authenticated is True
        assert result.user_id == "user123"
        assert result.session_token is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_with_mfa_required(self):
        """Test authentication requiring MFA"""
        provider = MilitaryGradeAuthProvider()
        provider.mfa_engine.setup_mfa("user123")
        
        credentials = AuthCredentials(
            user_id="user123",
            certificate=b"fake_certificate",
            mfa_code=None,  # MFA not provided
        )
        
        result = await provider.authenticate(credentials)
        
        assert result.authenticated is False
        assert result.requires_mfa is True
    
    @pytest.mark.asyncio
    async def test_authenticate_with_mfa_success(self):
        """Test successful authentication with MFA"""
        provider = MilitaryGradeAuthProvider()
        provider.mfa_engine.setup_mfa("user123")
        
        credentials = AuthCredentials(
            user_id="user123",
            certificate=b"fake_certificate",
            mfa_code="123456",
        )
        
        result = await provider.authenticate(credentials)
        
        assert result.authenticated is True
        assert result.user_id == "user123"
        assert result.session_token is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_with_ldap(self):
        """Test authentication with LDAP"""
        ldap = LDAPConnector(
            server_url="ldaps://ldap.example.gov:636",
            base_dn="dc=example,dc=gov",
        )
        provider = MilitaryGradeAuthProvider(ldap_connector=ldap)
        
        credentials = AuthCredentials(
            user_id="user123",
            certificate=b"fake_certificate",
            ldap_credentials={
                "username": "user123",
                "password": "validpass123",  # 8+ chars for stub validation
            },
        )
        
        result = await provider.authenticate(credentials)
        
        assert result.authenticated is True
        assert result.clearance_level is not None
    
    @pytest.mark.asyncio
    async def test_authenticate_certificate_failure(self):
        """Test authentication with invalid certificate"""
        provider = MilitaryGradeAuthProvider()
        
        # Mock validator to return False
        async def mock_validate(cert):
            return False
        provider.pki_validator.validate = mock_validate
        
        credentials = AuthCredentials(
            user_id="user123",
            certificate=b"invalid_certificate",
        )
        
        result = await provider.authenticate(credentials)
        
        assert result.authenticated is False
        assert result.error_message is not None
    
    def test_audit_log(self):
        """Test audit logging"""
        provider = MilitaryGradeAuthProvider()
        
        provider._log_auth_event(
            user_id="user123",
            event_type="authentication_success",
            clearance_level=ClearanceLevel.SECRET,
        )
        
        logs = provider.get_audit_log()
        assert len(logs) > 0
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["event_type"] == "authentication_success"
    
    def test_audit_log_filtering(self):
        """Test audit log filtering"""
        provider = MilitaryGradeAuthProvider()
        
        provider._log_auth_event("user1", "login")
        provider._log_auth_event("user2", "logout")
        provider._log_auth_event("user1", "access")
        
        # Filter by user
        user_logs = provider.get_audit_log(user_id="user1")
        assert len(user_logs) == 2
        assert all(log["user_id"] == "user1" for log in user_logs)
        
        # Filter by event type
        login_logs = provider.get_audit_log(event_type="login")
        assert len(login_logs) == 1
        assert login_logs[0]["event_type"] == "login"


class TestClearanceLevel:
    """Test Clearance Level Enum"""
    
    def test_clearance_levels(self):
        """Test all clearance levels"""
        assert ClearanceLevel.UNCLASSIFIED == "unclassified"
        assert ClearanceLevel.CONFIDENTIAL == "confidential"
        assert ClearanceLevel.SECRET == "secret"
        assert ClearanceLevel.TOP_SECRET == "top_secret"
        assert ClearanceLevel.ADMIN == "admin"


class TestAuthCredentials:
    """Test Authentication Credentials"""
    
    def test_credentials_creation(self):
        """Test credentials creation"""
        creds = AuthCredentials(
            user_id="user123",
            certificate=b"cert_data",
            password="password",
            mfa_code="123456",
        )
        
        assert creds.user_id == "user123"
        assert creds.certificate == b"cert_data"
        assert creds.password == "password"
        assert creds.mfa_code == "123456"
    
    def test_credentials_with_metadata(self):
        """Test credentials with metadata"""
        creds = AuthCredentials(
            user_id="user123",
            metadata={"ip_address": "192.168.1.1", "user_agent": "Mozilla/5.0"},
        )
        
        assert creds.metadata["ip_address"] == "192.168.1.1"


class TestAuthResult:
    """Test Authentication Result"""
    
    def test_successful_result(self):
        """Test successful authentication result"""
        result = AuthResult(
            authenticated=True,
            user_id="user123",
            clearance_level=ClearanceLevel.SECRET,
            session_token="token123",
        )
        
        assert result.is_success() is True
        assert result.authenticated is True
        assert result.requires_mfa is False
    
    def test_failed_result(self):
        """Test failed authentication result"""
        result = AuthResult(
            authenticated=False,
            error_message="Invalid credentials",
        )
        
        assert result.is_success() is False
        assert result.error_message == "Invalid credentials"
    
    def test_mfa_required_result(self):
        """Test MFA required result"""
        result = AuthResult(
            authenticated=False,
            user_id="user123",
            requires_mfa=True,
        )
        
        assert result.is_success() is False
        assert result.requires_mfa is True
