"""
Pytest configuration for Nethical tests

Adds custom command line options for extended tests and common fixtures.
"""

import os
import tempfile
import shutil
import pytest


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="Run extended/long-running tests (e.g., 10-minute load tests)"
    )
    
    parser.addoption(
        "--run-soak",
        action="store_true",
        default=False,
        help="Run soak tests (2-hour tests)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "extended: marks tests that require --run-extended flag"
    )
    config.addinivalue_line(
        "markers", "soak: marks tests that require --run-soak flag"
    )
    config.addinivalue_line(
        "markers", "security: marks security-related tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


# =============================================================================
# Common Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp(prefix="nethical_test_")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def governance_instance(temp_dir):
    """Create a basic IntegratedGovernance instance for testing."""
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=temp_dir,
        enable_quota_enforcement=True,
    )


@pytest.fixture
def governance_with_region(temp_dir):
    """Create an IntegratedGovernance instance with region configured."""
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=temp_dir,
        region_id="us-east-1",
        enable_quota_enforcement=True,
    )


@pytest.fixture
def governance_eu_region(temp_dir):
    """Create an IntegratedGovernance instance for EU region."""
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=temp_dir,
        region_id="eu-west-1",
        enable_quota_enforcement=True,
    )


@pytest.fixture
def mfa_manager():
    """Create an MFAManager instance for testing."""
    from nethical.security.mfa import MFAManager
    
    return MFAManager(
        max_attempts=5,
        lockout_duration_minutes=15,
    )


@pytest.fixture
def sso_manager():
    """Create an SSOManager instance for testing."""
    from nethical.security.sso import SSOManager
    
    return SSOManager(base_url="https://test.nethical.local")


@pytest.fixture
def encryption_service():
    """Create a MilitaryGradeEncryption instance for testing."""
    from nethical.security.encryption import MilitaryGradeEncryption
    
    return MilitaryGradeEncryption()


@pytest.fixture
def key_management_service():
    """Create a KeyManagementService instance for testing."""
    from nethical.security.encryption import KeyManagementService
    
    return KeyManagementService()


# =============================================================================
# Mock Redis Fixture
# =============================================================================


class MockRedis:
    """Mock Redis client for testing without Redis server."""
    
    def __init__(self):
        self._data = {}
        self._expires = {}
    
    def get(self, key):
        """Get a value from the mock store."""
        return self._data.get(key)
    
    def set(self, key, value, ex=None):
        """Set a value in the mock store."""
        self._data[key] = value
        if ex:
            self._expires[key] = ex
        return True
    
    def delete(self, *keys):
        """Delete keys from the mock store."""
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count
    
    def exists(self, *keys):
        """Check if keys exist in the mock store."""
        return sum(1 for key in keys if key in self._data)
    
    def incr(self, key):
        """Increment a value.
        
        Note: Redis INCR creates the key with value 1 if it doesn't exist,
        or returns an error if the value is not a number.
        """
        if key not in self._data:
            self._data[key] = 0
        try:
            self._data[key] = int(self._data[key]) + 1
            return self._data[key]
        except (ValueError, TypeError):
            raise ValueError(f"value at key '{key}' is not an integer or out of range")
    
    def expire(self, key, seconds):
        """Set expiry on a key."""
        self._expires[key] = seconds
        return True
    
    def ttl(self, key):
        """Get TTL for a key."""
        return self._expires.get(key, -1)
    
    def hset(self, name, key=None, value=None, mapping=None):
        """Set hash fields."""
        if name not in self._data:
            self._data[name] = {}
        if mapping:
            self._data[name].update(mapping)
        elif key is not None:
            self._data[name][key] = value
        return 1
    
    def hget(self, name, key):
        """Get a hash field."""
        if name not in self._data:
            return None
        return self._data[name].get(key)
    
    def hgetall(self, name):
        """Get all hash fields."""
        return self._data.get(name, {})
    
    def hdel(self, name, *keys):
        """Delete hash fields."""
        if name not in self._data:
            return 0
        count = 0
        for key in keys:
            if key in self._data[name]:
                del self._data[name][key]
                count += 1
        return count
    
    def ping(self):
        """Check connection."""
        return True
    
    def flushall(self):
        """Clear all data."""
        self._data.clear()
        self._expires.clear()
        return True


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    return MockRedis()


@pytest.fixture
def governance_with_mock_redis(temp_dir, mock_redis):
    """Create governance instance with mock Redis."""
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=temp_dir,
        redis_client=mock_redis,
        enable_quota_enforcement=True,
    )


# =============================================================================
# Test Data Factories
# =============================================================================


class TestDataFactory:
    """Factory for creating test data objects."""
    
    @staticmethod
    def create_agent_action(
        agent_id="test_agent",
        action="test action",
        action_type="query",
        context=None,
    ):
        """Create an AgentAction for testing."""
        from nethical.core.models import AgentAction
        
        return AgentAction(
            agent_id=agent_id,
            action=action,
            action_type=action_type,
            context=context or {},
        )
    
    @staticmethod
    def create_mfa_user(mfa_manager, user_id="test_user", enable=True):
        """Create a user with MFA set up."""
        secret, uri, backup_codes = mfa_manager.setup_totp(user_id)
        if enable:
            mfa_manager.enable_mfa(user_id)
        return {
            "user_id": user_id,
            "secret": secret,
            "provisioning_uri": uri,
            "backup_codes": backup_codes,
        }
    
    @staticmethod
    def create_oauth_config(
        sso_manager,
        config_name="test_provider",
        client_id="test_client",
        client_secret="test_secret",
    ):
        """Create an OAuth configuration for testing."""
        return sso_manager.configure_oauth(
            config_name=config_name,
            client_id=client_id,
            client_secret=client_secret,
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            userinfo_url="https://auth.example.com/userinfo",
        )


@pytest.fixture
def test_factory():
    """Provide test data factory."""
    return TestDataFactory()


# =============================================================================
# Sample Test Data
# =============================================================================


@pytest.fixture
def sample_pii_text():
    """Sample text containing PII for testing."""
    return (
        "Contact John Doe at john.doe@example.com or call 555-123-4567. "
        "His SSN is 123-45-6789 and he lives at 123 Main St."
    )


@pytest.fixture
def sample_clean_text():
    """Sample text without PII for testing."""
    return "This is a simple test message with no personal information."


@pytest.fixture
def sample_harmful_content():
    """Sample harmful content for testing."""
    return "Here's how to exploit vulnerabilities and attack the system."


@pytest.fixture
def sample_sql_injection():
    """Sample SQL injection attempt for testing."""
    return "'; DROP TABLE users; --"
