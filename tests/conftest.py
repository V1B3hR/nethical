"""
Pytest configuration for Nethical tests

Adds custom command line options for extended tests and provides common fixtures.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock


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


# ============== Common Fixtures ==============


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """
    Temporary storage directory for persistence tests.
    
    Returns:
        Path to temporary storage directory
    """
    storage_dir = tmp_path / "nethical_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def governance_instance(temp_storage_dir: Path):
    """
    Create a fresh IntegratedGovernance instance for testing.
    
    Returns:
        IntegratedGovernance instance
    """
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=str(temp_storage_dir),
        enable_quota_enforcement=False,  # Disable to avoid Redis dependency
        enable_shadow_mode=True,
        enable_ml_blending=True,
        enable_anomaly_detection=True,
    )


@pytest.fixture
def governance_with_region(temp_storage_dir: Path):
    """
    Create IntegratedGovernance instance with regional configuration.
    
    Returns:
        IntegratedGovernance instance with region settings
    """
    from nethical.core.integrated_governance import IntegratedGovernance
    
    return IntegratedGovernance(
        storage_dir=str(temp_storage_dir),
        region_id="eu-west-1",
        logical_domain="test-domain",
        data_residency_policy="EU_GDPR",
        enable_quota_enforcement=False,
    )


@pytest.fixture
def mock_redis():
    """
    Mock Redis client for quota/caching tests.
    
    Returns:
        MagicMock Redis client
    """
    mock = MagicMock()
    
    # Configure common Redis operations
    mock.get.return_value = None
    mock.set.return_value = True
    mock.incr.return_value = 1
    mock.expire.return_value = True
    mock.delete.return_value = 1
    mock.exists.return_value = 0
    mock.hget.return_value = None
    mock.hset.return_value = 1
    mock.pipeline.return_value.__enter__ = lambda x: mock
    mock.pipeline.return_value.__exit__ = lambda x, *args: None
    
    return mock


@pytest.fixture
def sample_agent_action() -> Dict[str, Any]:
    """
    Factory for creating test AgentAction objects.
    
    Returns:
        Dictionary representing an agent action
    """
    from tests.fixtures.agent_actions import create_agent_action
    return create_agent_action()


@pytest.fixture
def safe_agent_action() -> Dict[str, Any]:
    """
    Create a safe agent action for testing.
    
    Returns:
        Dictionary representing a safe agent action
    """
    from tests.fixtures.agent_actions import create_safe_action
    return create_safe_action()


@pytest.fixture
def risky_agent_action() -> Dict[str, Any]:
    """
    Create a risky agent action for testing.
    
    Returns:
        Dictionary representing a risky agent action
    """
    from tests.fixtures.agent_actions import create_risky_action
    return create_risky_action()


@pytest.fixture
def sample_violation() -> Dict[str, Any]:
    """
    Create a sample violation for testing.
    
    Returns:
        Dictionary representing a violation
    """
    from tests.fixtures.violations import create_violation
    return create_violation()


@pytest.fixture
def encryption_service():
    """
    Create MilitaryGradeEncryption instance for testing.
    
    Returns:
        MilitaryGradeEncryption instance
    """
    from nethical.security.encryption import MilitaryGradeEncryption
    return MilitaryGradeEncryption()


@pytest.fixture
def mfa_manager():
    """
    Create MFAManager instance for testing.
    
    Returns:
        MFAManager instance with test configuration
    """
    from nethical.security.mfa import MFAManager, set_mfa_manager
    
    manager = MFAManager(max_attempts=3, lockout_duration_minutes=1)
    set_mfa_manager(manager)
    return manager


@pytest.fixture
def sso_manager():
    """
    Create SSOManager instance for testing.
    
    Returns:
        SSOManager instance
    """
    from nethical.security.sso import SSOManager, set_sso_manager
    
    manager = SSOManager(base_url="https://test.nethical.local")
    set_sso_manager(manager)
    return manager


@pytest.fixture
def configured_sso_manager(sso_manager):
    """
    Create SSOManager with OAuth configuration for testing.
    
    Returns:
        SSOManager instance with OAuth configured
    """
    sso_manager.configure_oauth(
        config_name="test_oauth",
        client_id="test_client_id",
        client_secret="test_client_secret",
        authorization_url="https://oauth.example.com/authorize",
        token_url="https://oauth.example.com/token",
    )
    return sso_manager


# ============== Async Test Utilities ==============


@pytest.fixture
def event_loop_policy():
    """
    Provide asyncio event loop policy.
    
    Ensures consistent async behavior across tests.
    """
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# ============== Test Data Fixtures ==============


@pytest.fixture
def injection_payloads():
    """
    Provide SQL injection test payloads.
    
    Returns:
        List of injection payloads
    """
    from tests.fixtures.payloads import INJECTION_PAYLOADS
    return INJECTION_PAYLOADS


@pytest.fixture
def xss_payloads():
    """
    Provide XSS test payloads.
    
    Returns:
        List of XSS payloads
    """
    from tests.fixtures.payloads import XSS_PAYLOADS
    return XSS_PAYLOADS


@pytest.fixture
def jailbreak_payloads():
    """
    Provide jailbreak/prompt injection test payloads.
    
    Returns:
        List of jailbreak payloads
    """
    from tests.fixtures.payloads import JAILBREAK_PAYLOADS
    return JAILBREAK_PAYLOADS
