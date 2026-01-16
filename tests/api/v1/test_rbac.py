"""Tests for RBAC functionality."""

from datetime import timedelta
from nethical.api.rbac import (
    Role,
    create_access_token,
    get_password_hash,
    verify_password,
    )


class TestPasswordHashing:
    """Test password hashing and verification."""
    
    def test_password_hashing(self):
        """Test that passwords are hashed correctly."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 0
    
    def test_password_verification(self):
        """Test password verification."""
        password = "test_password_123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed)
        assert not verify_password("wrong_password", hashed)


class TestTokenCreation:
    """Test JWT token creation."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        data = {
            "sub": "testuser",
            "role": "admin",
            "user_id": 1,
        }
        
        token = create_access_token(data)
        
        assert token is not None
        assert len(token) > 0
        assert isinstance(token, str)
    
    def test_create_token_with_expiration(self):
        """Test token creation with custom expiration."""
        data = {"sub": "testuser", "role": "admin"}
        expires_delta = timedelta(minutes=15)
        
        token = create_access_token(data, expires_delta=expires_delta)
        
        assert token is not None
        assert isinstance(token, str)


class TestRoles:
    """Test role enumeration."""
    
    def test_roles_defined(self):
        """Test that all roles are defined."""
        assert Role.ADMIN == "admin"
        assert Role.AUDITOR == "auditor"
        assert Role.OPERATOR == "operator"
    
    def test_role_values(self):
        """Test role string values."""
        assert Role.ADMIN.value == "admin"
        assert Role.AUDITOR.value == "auditor"
        assert Role.OPERATOR.value == "operator"
