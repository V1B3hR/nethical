"""
Tests for JWT Authentication System
"""

import pytest
from datetime import datetime, timedelta, timezone

from nethical.security.auth import (
    AuthManager,
    TokenPayload,
    TokenType,
    APIKey,
    TokenExpiredError,
    InvalidTokenError,
)


class TestTokenPayload:
    """Test cases for TokenPayload"""
    
    def test_token_payload_creation(self):
        """Test creating a token payload"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        
        payload = TokenPayload(
            user_id="user123",
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=expires,
        )
        
        assert payload.user_id == "user123"
        assert payload.token_type == TokenType.ACCESS
        assert not payload.is_expired()
    
    def test_token_expiry(self):
        """Test token expiry checking"""
        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        
        payload = TokenPayload(
            user_id="user123",
            token_type=TokenType.ACCESS,
            issued_at=past,
            expires_at=past + timedelta(minutes=30),
        )
        
        assert payload.is_expired()
    
    def test_token_to_dict(self):
        """Test converting token to dictionary"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        
        payload = TokenPayload(
            user_id="user123",
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=expires,
            scope="read write",
        )
        
        data = payload.to_dict()
        assert data["sub"] == "user123"
        assert data["type"] == "access"
        assert data["scope"] == "read write"
        assert "jti" in data
    
    def test_token_from_dict(self):
        """Test creating token from dictionary"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=1)
        
        data = {
            "sub": "user123",
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "jti": "test-jti",
            "scope": "read",
        }
        
        payload = TokenPayload.from_dict(data)
        assert payload.user_id == "user123"
        assert payload.token_type == TokenType.ACCESS
        assert payload.jti == "test-jti"


class TestAPIKey:
    """Test cases for APIKey"""
    
    def test_api_key_creation(self):
        """Test creating an API key"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=30)
        
        api_key = APIKey(
            key_id="key123",
            user_id="user123",
            key_hash="abcdef1234567890",
            name="Test Key",
            created_at=now,
            expires_at=expires,
        )
        
        assert api_key.key_id == "key123"
        assert api_key.user_id == "user123"
        assert api_key.is_valid()
    
    def test_api_key_expiry(self):
        """Test API key expiry checking"""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=1)
        
        api_key = APIKey(
            key_id="key123",
            user_id="user123",
            key_hash="abcdef1234567890",
            name="Test Key",
            created_at=past,
            expires_at=now - timedelta(hours=1),
        )
        
        assert api_key.is_expired()
        assert not api_key.is_valid()
    
    def test_api_key_disabled(self):
        """Test disabled API key"""
        now = datetime.now(timezone.utc)
        
        api_key = APIKey(
            key_id="key123",
            user_id="user123",
            key_hash="abcdef1234567890",
            name="Test Key",
            created_at=now,
            enabled=False,
        )
        
        assert not api_key.is_valid()


class TestAuthManager:
    """Test cases for AuthManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-secret-key")
    
    def test_initialization(self):
        """Test auth manager initialization"""
        assert isinstance(self.auth, AuthManager)
        assert self.auth.secret_key is not None
        assert len(self.auth.api_keys) == 0
    
    def test_create_access_token(self):
        """Test creating an access token"""
        token, payload = self.auth.create_access_token("user123")
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert payload.user_id == "user123"
        assert payload.token_type == TokenType.ACCESS
        assert not payload.is_expired()
    
    def test_create_refresh_token(self):
        """Test creating a refresh token"""
        token, payload = self.auth.create_refresh_token("user123")
        
        assert isinstance(token, str)
        assert payload.user_id == "user123"
        assert payload.token_type == TokenType.REFRESH
    
    def test_verify_valid_token(self):
        """Test verifying a valid token"""
        token, original_payload = self.auth.create_access_token("user123", scope="read")
        
        verified_payload = self.auth.verify_token(token)
        
        assert verified_payload.user_id == original_payload.user_id
        assert verified_payload.token_type == original_payload.token_type
        assert verified_payload.jti == original_payload.jti
    
    def test_verify_expired_token(self):
        """Test verifying an expired token"""
        # Create auth manager with very short expiry
        auth = AuthManager(
            secret_key="test-key",
            access_token_expiry=timedelta(seconds=0)
        )
        
        token, _ = auth.create_access_token("user123")
        
        # Token should be expired immediately
        with pytest.raises(TokenExpiredError):
            auth.verify_token(token)
    
    def test_revoke_token(self):
        """Test revoking a token"""
        token, payload = self.auth.create_access_token("user123")
        
        # Verify it works before revocation
        self.auth.verify_token(token)
        
        # Revoke the token
        self.auth.revoke_token(token)
        
        # Should fail after revocation
        with pytest.raises(InvalidTokenError):
            self.auth.verify_token(token)
    
    def test_refresh_access_token(self):
        """Test refreshing an access token"""
        refresh_token, _ = self.auth.create_refresh_token("user123")
        
        new_access_token, new_payload = self.auth.refresh_access_token(refresh_token)
        
        assert new_payload.user_id == "user123"
        assert new_payload.token_type == TokenType.ACCESS
    
    def test_refresh_with_access_token_fails(self):
        """Test that refreshing with an access token fails"""
        access_token, _ = self.auth.create_access_token("user123")
        
        with pytest.raises(InvalidTokenError):
            self.auth.refresh_access_token(access_token)
    
    def test_create_api_key(self):
        """Test creating an API key"""
        key_string, api_key = self.auth.create_api_key(
            user_id="user123",
            name="Test API Key"
        )
        
        assert isinstance(key_string, str)
        assert "." in key_string
        assert api_key.user_id == "user123"
        assert api_key.name == "Test API Key"
        assert api_key.is_valid()
    
    def test_verify_api_key(self):
        """Test verifying an API key"""
        key_string, original_key = self.auth.create_api_key(
            user_id="user123",
            name="Test Key"
        )
        
        verified_key = self.auth.verify_api_key(key_string)
        
        assert verified_key.key_id == original_key.key_id
        assert verified_key.user_id == "user123"
        assert verified_key.last_used_at is not None
    
    def test_verify_invalid_api_key(self):
        """Test verifying an invalid API key"""
        with pytest.raises(InvalidTokenError):
            self.auth.verify_api_key("invalid.key.format.wrong")
    
    def test_revoke_api_key(self):
        """Test revoking an API key"""
        key_string, api_key = self.auth.create_api_key(
            user_id="user123",
            name="Test Key"
        )
        
        # Verify it works before revocation
        self.auth.verify_api_key(key_string)
        
        # Revoke the key
        self.auth.revoke_api_key(api_key.key_id)
        
        # Should fail after revocation
        with pytest.raises(InvalidTokenError):
            self.auth.verify_api_key(key_string)
    
    def test_list_api_keys(self):
        """Test listing API keys"""
        self.auth.create_api_key("user1", "Key 1")
        self.auth.create_api_key("user1", "Key 2")
        self.auth.create_api_key("user2", "Key 3")
        
        # List all keys
        all_keys = self.auth.list_api_keys()
        assert len(all_keys) == 3
        
        # List keys for specific user
        user1_keys = self.auth.list_api_keys(user_id="user1")
        assert len(user1_keys) == 2
        assert all(k.user_id == "user1" for k in user1_keys)
    
    def test_api_key_with_expiry(self):
        """Test API key with expiration"""
        expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        key_string, api_key = self.auth.create_api_key(
            user_id="user123",
            name="Temporary Key",
            expires_at=expires_at
        )
        
        assert api_key.expires_at == expires_at
        assert not api_key.is_expired()
    
    def test_token_scope(self):
        """Test token with custom scope"""
        token, payload = self.auth.create_access_token(
            "user123",
            scope="read:data write:data"
        )
        
        verified = self.auth.verify_token(token)
        assert verified.scope == "read:data write:data"


class TestAuthenticationRequest:
    """Test cases for authenticate_request helper"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-key")
        set_auth_manager(self.auth)
    
    def test_authenticate_with_jwt(self):
        """Test authentication with JWT token"""
        token, _ = self.auth.create_access_token("user123")
        auth_header = f"Bearer {token}"
        
        user_id = authenticate_request(authorization_header=auth_header)
        assert user_id == "user123"
    
    def test_authenticate_with_api_key(self):
        """Test authentication with API key"""
        key_string, _ = self.auth.create_api_key("user456", "Test Key")
        
        user_id = authenticate_request(api_key_header=key_string)
        assert user_id == "user456"
    
    def test_authenticate_no_credentials(self):
        """Test authentication without credentials"""
        with pytest.raises(AuthenticationError):
            authenticate_request()
    
    def test_authenticate_invalid_header_format(self):
        """Test authentication with invalid header format"""
        with pytest.raises(AuthenticationError):
            authenticate_request(authorization_header="InvalidFormat token")
    
    def test_authenticate_expired_token(self):
        """Test authentication with expired token"""
        auth = AuthManager(
            secret_key="test-key",
            access_token_expiry=timedelta(seconds=0)
        )
        set_auth_manager(auth)
        
        token, _ = auth.create_access_token("user123")
        auth_header = f"Bearer {token}"
        
        with pytest.raises(AuthenticationError):
            authenticate_request(authorization_header=auth_header)


class TestAuthIntegration:
    """Integration tests for authentication system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(
            secret_key="integration-test-key",
            access_token_expiry=timedelta(hours=1),
            refresh_token_expiry=timedelta(days=7),
        )
        set_auth_manager(self.auth)
    
    def test_complete_auth_flow(self):
        """Test complete authentication flow"""
        # 1. Create access and refresh tokens
        access_token, access_payload = self.auth.create_access_token("user123")
        refresh_token, _ = self.auth.create_refresh_token("user123")
        
        # 2. Verify access token works
        verified = self.auth.verify_token(access_token)
        assert verified.user_id == "user123"
        
        # 3. Use refresh token to get new access token
        new_access_token, new_payload = self.auth.refresh_access_token(refresh_token)
        assert new_payload.user_id == "user123"
        
        # 4. Verify new token works
        verified_new = self.auth.verify_token(new_access_token)
        assert verified_new.user_id == "user123"
    
    def test_api_key_workflow(self):
        """Test API key workflow"""
        # 1. Create API key
        key_string, api_key = self.auth.create_api_key(
            user_id="service_account",
            name="Service API Key"
        )
        
        # 2. Verify key works
        verified = self.auth.verify_api_key(key_string)
        assert verified.user_id == "service_account"
        assert verified.last_used_at is not None
        
        # 3. List keys
        keys = self.auth.list_api_keys(user_id="service_account")
        assert len(keys) == 1
        
        # 4. Revoke key
        self.auth.revoke_api_key(api_key.key_id)
        
        # 5. Verify key no longer works
        with pytest.raises(InvalidTokenError):
            self.auth.verify_api_key(key_string)
    
    def test_multi_user_tokens(self):
        """Test tokens for multiple users"""
        users = ["alice", "bob", "charlie"]
        tokens = {}
        
        # Create tokens for each user
        for user in users:
            token, _ = self.auth.create_access_token(user)
            tokens[user] = token
        
        # Verify each token
        for user, token in tokens.items():
            payload = self.auth.verify_token(token)
            assert payload.user_id == user


class TestAuthManagerSecurity:
    """Test cases for AuthManager security features"""
    
    def test_blocks_insecure_literal_secret(self):
        """Test that AuthManager blocks the literal string 'secret'"""
        with pytest.raises(ValueError) as exc_info:
            AuthManager(secret_key="secret")
        
        assert "insecure literal secret" in str(exc_info.value).lower()
        assert "secret" in str(exc_info.value)
    
    def test_blocks_short_secret_key(self):
        """Test that AuthManager blocks keys shorter than 16 characters"""
        with pytest.raises(ValueError) as exc_info:
            AuthManager(secret_key="short")
        
        assert "too short" in str(exc_info.value).lower()
        assert "16" in str(exc_info.value)
    
    def test_accepts_valid_secret_key(self):
        """Test that AuthManager accepts valid secret keys"""
        # 16 character key (minimum)
        auth = AuthManager(secret_key="1234567890123456")
        assert auth.secret_key == "1234567890123456"
        
        # Longer key (recommended)
        long_key = "this-is-a-very-secure-and-long-secret-key-12345678"
        auth = AuthManager(secret_key=long_key)
        assert auth.secret_key == long_key
    
    def test_environment_variable_jwt_secret(self, monkeypatch):
        """Test that AuthManager reads JWT_SECRET from environment"""
        test_secret = "environment-secret-key-at-least-16-chars"
        monkeypatch.setenv("JWT_SECRET", test_secret)
        
        # Create AuthManager without explicit secret_key
        auth = AuthManager()
        assert auth.secret_key == test_secret
    
    def test_environment_variable_blocks_insecure_secret(self, monkeypatch):
        """Test that insecure secret is blocked even from environment"""
        monkeypatch.setenv("JWT_SECRET", "secret")
        
        with pytest.raises(ValueError) as exc_info:
            AuthManager()
        
        assert "insecure literal secret" in str(exc_info.value).lower()
    
    def test_auto_generates_ephemeral_key_with_warning(self, monkeypatch):
        """Test that AuthManager auto-generates key and warns when no secret provided"""
        # Remove JWT_SECRET if it exists
        monkeypatch.delenv("JWT_SECRET", raising=False)
        
        # Should generate ephemeral key and issue warning
        with pytest.warns(UserWarning, match="Auto-generated key will be lost on restart"):
            auth = AuthManager()
        
        # Should have a valid key
        assert auth.secret_key is not None
        assert len(auth.secret_key) >= 16
    
    def test_jti_generation_in_tokens(self):
        """Test that tokens include secure JTI (JWT ID)"""
        auth = AuthManager(secret_key="test-secret-key-12345678")
        
        token1, payload1 = auth.create_access_token("user1")
        token2, payload2 = auth.create_access_token("user1")
        
        # Both tokens should have JTI
        assert payload1.jti is not None
        assert payload2.jti is not None
        
        # JTIs should be unique
        assert payload1.jti != payload2.jti
        
        # JTI should be URL-safe (alphanumeric plus - and _) and non-empty
        assert len(payload1.jti) > 0
        import re
        assert re.match(r'^[A-Za-z0-9_-]+$', payload1.jti), f"JTI contains invalid characters: {payload1.jti}"
        assert re.match(r'^[A-Za-z0-9_-]+$', payload2.jti), f"JTI contains invalid characters: {payload2.jti}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
