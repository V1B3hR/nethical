"""Tests for Phase 4 Operational Security: Zero Trust Architecture and Secret Management."""

import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nethical.security.zero_trust import (
    TrustLevel,
    DeviceHealthStatus,
    NetworkSegment,
    ServiceMeshConfig,
    DeviceHealthCheck,
    PolicyEnforcer,
    ContinuousAuthEngine,
    ZeroTrustController,
)

from nethical.security.secret_management import (
    SecretType,
    SecretRotationPolicy,
    VaultConfig,
    Secret,
    SecretScanner,
    DynamicSecretGenerator,
    SecretRotationManager,
    VaultIntegration,
    SecretManagementSystem,
)


class TestZeroTrust:
    """Test Zero Trust Architecture implementation."""
    
    def test_trust_level_ordering(self):
        """Test trust level comparison."""
        levels = [TrustLevel.UNTRUSTED, TrustLevel.LOW, TrustLevel.MEDIUM, 
                 TrustLevel.HIGH, TrustLevel.VERIFIED]
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value or levels[i] != levels[i + 1]
    
    def test_network_segment_creation(self):
        """Test network segment creation."""
        segment = NetworkSegment(
            segment_id="prod-1",
            name="Production Network",
            allowed_services=["web", "api"],
            allowed_protocols=["https", "ssh"],
            min_trust_level=TrustLevel.MEDIUM,
            require_mfa=True,
        )
        
        assert segment.segment_id == "prod-1"
        assert len(segment.allowed_services) == 2
        assert segment.require_mfa is True
    
    def test_service_mesh_config_validation(self):
        """Test service mesh configuration validation."""
        # Valid config with mTLS disabled
        config = ServiceMeshConfig(
            service_name="test-service",
            enable_mtls=False,
        )
        assert config.validate() is True
        
        # Invalid config with mTLS enabled but no certs
        config_invalid = ServiceMeshConfig(
            service_name="test-service",
            enable_mtls=True,
        )
        assert config_invalid.validate() is False
    
    def test_device_health_check(self):
        """Test device health check."""
        healthy = DeviceHealthCheck(
            device_id="device-1",
            status=DeviceHealthStatus.HEALTHY,
            os_version="11.0",
            patch_level="2024-01",
            antivirus_updated=True,
            disk_encryption_enabled=True,
            firewall_enabled=True,
            compliance_score=0.95,
            last_check=datetime.now(timezone.utc),
        )
        assert healthy.is_healthy() is True


class TestPolicyEnforcer:
    """Test Policy Enforcer."""
    
    def test_policy_enforcer_initialization(self):
        """Test policy enforcer initialization."""
        enforcer = PolicyEnforcer()
        assert len(enforcer.segments) == 0
        assert len(enforcer.access_logs) == 0
    
    def test_add_segment(self):
        """Test adding network segments."""
        enforcer = PolicyEnforcer()
        
        segment = NetworkSegment(
            segment_id="prod-1",
            name="Production",
            allowed_services=["web"],
            allowed_protocols=["https"],
            min_trust_level=TrustLevel.HIGH,
        )
        
        enforcer.add_segment(segment)
        assert "prod-1" in enforcer.segments
    
    def test_evaluate_access_success(self):
        """Test successful access evaluation."""
        segment = NetworkSegment(
            segment_id="dev-1",
            name="Development",
            allowed_services=["api", "db"],
            allowed_protocols=["https"],
            min_trust_level=TrustLevel.LOW,
        )
        
        enforcer = PolicyEnforcer([segment])
        
        allowed, reason = enforcer.evaluate_access(
            user_id="user-1",
            trust_level=TrustLevel.MEDIUM,
            segment_id="dev-1",
            service="api",
        )
        
        assert allowed is True
        assert "granted" in reason.lower()
    
    def test_prevent_lateral_movement(self):
        """Test lateral movement prevention."""
        segment1 = NetworkSegment(
            segment_id="segment-1",
            name="Segment 1",
            allowed_services=["api"],
            allowed_protocols=["https"],
            min_trust_level=TrustLevel.MEDIUM,
        )
        
        segment2 = NetworkSegment(
            segment_id="segment-2",
            name="Segment 2",
            allowed_services=["db"],
            allowed_protocols=["tcp"],
            min_trust_level=TrustLevel.HIGH,
        )
        
        enforcer = PolicyEnforcer([segment1, segment2])
        
        # Same segment - allowed
        allowed, _ = enforcer.prevent_lateral_movement(
            "segment-1", "segment-1", "user-1"
        )
        assert allowed is True
        
        # Different segments - denied by default
        allowed, _ = enforcer.prevent_lateral_movement(
            "segment-1", "segment-2", "user-1"
        )
        assert allowed is False


class TestContinuousAuthEngine:
    """Test Continuous Authentication Engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ContinuousAuthEngine()
        assert engine.default_trust == TrustLevel.LOW
        assert len(engine.user_sessions) == 0
    
    def test_create_session(self):
        """Test session creation."""
        engine = ContinuousAuthEngine()
        
        token = engine.create_session(
            user_id="user-1",
            initial_trust=TrustLevel.MEDIUM,
            device_id="device-1",
        )
        
        assert token is not None
        assert token in engine.user_sessions
    
    def test_verify_session(self):
        """Test session verification."""
        engine = ContinuousAuthEngine()
        
        token = engine.create_session("user-1", TrustLevel.MEDIUM)
        valid, trust = engine.verify_session(token)
        
        assert valid is True
        assert trust == TrustLevel.MEDIUM
    
    def test_report_risk_event(self):
        """Test reporting risk events."""
        engine = ContinuousAuthEngine()
        
        token = engine.create_session("user-1", TrustLevel.HIGH)
        user_id = engine.user_sessions[token]["user_id"]
        initial_score = engine.trust_scores[user_id]
        
        engine.report_risk_event(token, "suspicious_activity", 0.8)
        
        new_score = engine.trust_scores[user_id]
        assert new_score < initial_score


class TestZeroTrustController:
    """Test Zero Trust Controller."""
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        controller = ZeroTrustController()
        
        assert controller.service_mesh is None
        assert controller.policy_enforcer is not None
        assert controller.auth_engine is not None
    
    def test_check_device_health(self):
        """Test device health check."""
        controller = ZeroTrustController()
        
        health = controller.check_device_health(
            device_id="device-1",
            os_version="11.0",
            patch_level="2024-01",
        )
        
        assert health.status == DeviceHealthStatus.HEALTHY
        assert health.is_healthy() is True
    
    def test_authorize_access(self):
        """Test access authorization."""
        segment = NetworkSegment(
            segment_id="prod-1",
            name="Production",
            allowed_services=["api"],
            allowed_protocols=["https"],
            min_trust_level=TrustLevel.MEDIUM,
        )
        
        controller = ZeroTrustController(segments=[segment])
        
        token = controller.auth_engine.create_session("user-1", TrustLevel.HIGH)
        controller.check_device_health("device-1", "11.0", "2024-01")
        
        allowed, _ = controller.authorize_access(
            session_token=token,
            segment_id="prod-1",
            service="api",
            device_id="device-1",
        )
        
        assert allowed is True


class TestSecretManagement:
    """Test Secret Management implementation."""
    
    def test_secret_creation(self):
        """Test secret creation."""
        secret = Secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            value="test-value",
            created_at=datetime.now(timezone.utc),
        )
        
        assert secret.secret_id == "api-key-1"
        assert secret.secret_type == SecretType.API_KEY
    
    def test_secret_expiration(self):
        """Test secret expiration check."""
        secret = Secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            value="test-value",
            created_at=datetime.now(timezone.utc) - timedelta(days=100),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert secret.is_expired() is True
    
    def test_vault_config_validation(self):
        """Test Vault configuration validation."""
        config = VaultConfig(
            vault_address="https://vault.example.com",
            vault_token="test-token",
            enabled=True,
        )
        assert config.validate() is True


class TestSecretScanner:
    """Test Secret Scanner."""
    
    def test_scanner_initialization(self):
        """Test scanner initialization."""
        scanner = SecretScanner()
        assert len(scanner.findings) == 0
    
    def test_scan_text_api_key(self):
        """Test scanning for API keys."""
        scanner = SecretScanner()
        
        text = 'api_key = "sk-1234567890abcdefghij"'
        findings = scanner.scan_text(text)
        
        api_key_findings = [f for f in findings 
                           if f.get("secret_type") == SecretType.API_KEY.value]
        assert len(api_key_findings) > 0
    
    def test_scan_text_password(self):
        """Test scanning for passwords."""
        scanner = SecretScanner()
        
        text = 'password = "MySecretP@ssw0rd123"'
        findings = scanner.scan_text(text)
        
        pwd_findings = [f for f in findings 
                       if f.get("secret_type") == SecretType.PASSWORD.value]
        assert len(pwd_findings) > 0


class TestDynamicSecretGenerator:
    """Test Dynamic Secret Generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = DynamicSecretGenerator()
        assert len(generator.generated_secrets) == 0
    
    def test_generate_api_key(self):
        """Test API key generation."""
        generator = DynamicSecretGenerator()
        
        secret = generator.generate_api_key("api-key-1", length=32)
        
        assert secret.secret_id == "api-key-1"
        assert secret.secret_type == SecretType.API_KEY
        assert len(secret.value) > 0
    
    def test_generate_password(self):
        """Test password generation."""
        generator = DynamicSecretGenerator()
        
        secret = generator.generate_password("password-1", length=24)
        
        assert secret.secret_type == SecretType.PASSWORD
        assert len(secret.value) == 24
    
    def test_generate_encryption_key(self):
        """Test encryption key generation."""
        generator = DynamicSecretGenerator()
        
        secret = generator.generate_encryption_key("enc-key-1", key_size=32)
        
        assert secret.secret_type == SecretType.ENCRYPTION_KEY
        assert len(secret.value) == 64  # 32 bytes as hex


class TestSecretRotationManager:
    """Test Secret Rotation Manager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = SecretRotationManager()
        assert len(manager.policies) == 0
    
    def test_add_policy(self):
        """Test adding rotation policy."""
        manager = SecretRotationManager()
        
        policy = SecretRotationPolicy(
            secret_type=SecretType.API_KEY,
            rotation_interval_days=90,
            auto_rotate=True,
        )
        
        manager.add_policy(policy)
        assert SecretType.API_KEY in manager.policies
    
    def test_should_rotate(self):
        """Test rotation check."""
        manager = SecretRotationManager()
        
        policy = SecretRotationPolicy(
            secret_type=SecretType.API_KEY,
            rotation_interval_days=30,
            auto_rotate=True,
        )
        manager.add_policy(policy)
        
        old_secret = Secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            value="old-value",
            created_at=datetime.now(timezone.utc) - timedelta(days=40),
        )
        
        assert manager.should_rotate(old_secret) is True
    
    def test_rotate_secret(self):
        """Test secret rotation."""
        manager = SecretRotationManager()
        generator = DynamicSecretGenerator()
        
        old_secret = Secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            value="old-value",
            created_at=datetime.now(timezone.utc) - timedelta(days=100),
            rotation_count=0,
        )
        
        new_secret = manager.rotate_secret(old_secret, generator)
        
        assert new_secret.rotation_count == 1
        assert new_secret.value != old_secret.value


class TestVaultIntegration:
    """Test Vault Integration."""
    
    def test_vault_initialization(self):
        """Test Vault integration initialization."""
        config = VaultConfig(
            vault_address="https://vault.example.com",
            vault_token="test-token",
            enabled=True,
        )
        
        vault = VaultIntegration(config)
        assert vault.connected is False
    
    def test_vault_connect(self):
        """Test Vault connection."""
        config = VaultConfig(
            vault_address="https://vault.example.com",
            vault_token="test-token",
            enabled=True,
        )
        
        vault = VaultIntegration(config)
        result = vault.connect()
        
        assert result is True
        assert vault.connected is True


class TestSecretManagementSystem:
    """Test Secret Management System."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = SecretManagementSystem()
        
        assert system.scanner is not None
        assert system.generator is not None
        assert system.rotation_manager is not None
        
        # Check default policies
        assert SecretType.API_KEY in system.rotation_manager.policies
        assert SecretType.PASSWORD in system.rotation_manager.policies
    
    def test_create_secret(self):
        """Test creating secrets."""
        system = SecretManagementSystem()
        
        secret = system.create_secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            length=32,
        )
        
        assert secret.secret_id == "api-key-1"
        assert "api-key-1" in system.secrets
    
    def test_rotate_secrets(self):
        """Test rotating secrets."""
        system = SecretManagementSystem()
        
        old_secret = Secret(
            secret_id="api-key-1",
            secret_type=SecretType.API_KEY,
            value="old-value",
            created_at=datetime.now(timezone.utc) - timedelta(days=100),
        )
        system.secrets["api-key-1"] = old_secret
        
        rotated = system.rotate_secrets()
        
        assert len(rotated) == 1
    
    def test_get_system_status(self):
        """Test getting system status."""
        system = SecretManagementSystem()
        
        system.create_secret("api-key-1", SecretType.API_KEY)
        
        status = system.get_system_status()
        
        assert status["total_secrets"] == 1
        assert "vault_connected" in status


class TestPhase4Integration:
    """Integration tests for Phase 4 Operational Security."""
    
    def test_zero_trust_complete_flow(self):
        """Test complete zero trust flow."""
        # Configure service mesh
        config = ServiceMeshConfig(
            service_name="app-service",
            enable_mtls=True,
            certificate=b"cert",
            private_key=b"key",
            ca_bundle=b"ca",
        )
        
        # Create segments
        segments = [
            NetworkSegment(
                segment_id="app-tier",
                name="Application Tier",
                allowed_services=["web", "api"],
                allowed_protocols=["https"],
                min_trust_level=TrustLevel.MEDIUM,
            ),
            NetworkSegment(
                segment_id="data-tier",
                name="Data Tier",
                allowed_services=["db"],
                allowed_protocols=["tcp"],
                min_trust_level=TrustLevel.HIGH,
            ),
        ]
        
        controller = ZeroTrustController(config, segments)
        
        # Verify service mesh
        assert controller.validate_service_mesh() is True
        
        # Check device health
        health = controller.check_device_health("device-1", "11.0", "2024-01")
        assert health.is_healthy() is True
        
        # Create session
        token = controller.auth_engine.create_session("user-1", TrustLevel.HIGH)
        
        # Authorize access
        allowed, _ = controller.authorize_access(token, "app-tier", "api", "device-1")
        assert allowed is True
    
    def test_secret_management_complete_flow(self):
        """Test complete secret management flow."""
        vault_config = VaultConfig(
            vault_address="https://vault.example.com",
            vault_token="test-token",
            enabled=True,
        )
        
        system = SecretManagementSystem(vault_config)
        
        # Create secrets
        api_key = system.create_secret("api-key-1", SecretType.API_KEY, length=32)
        password = system.create_secret("password-1", SecretType.PASSWORD, length=24)
        
        assert len(system.secrets) == 2
        
        # Test scanning
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('api_key = "sk-test123456789012345678"\n')
            temp_file = f.name
        
        try:
            findings = system.scan_for_secrets(temp_file)
            assert len(findings) > 0
        finally:
            Path(temp_file).unlink()
        
        # Get status
        status = system.get_system_status()
        assert status["total_secrets"] == 2
    
    def test_integrated_operational_security(self):
        """Test integrated operational security with both systems."""
        # Initialize both systems
        secret_system = SecretManagementSystem()
        zt_controller = ZeroTrustController()
        
        # Generate API key for service
        api_key = secret_system.create_secret(
            secret_id="service-api-key",
            secret_type=SecretType.API_KEY,
        )
        
        # Configure zero trust
        segment = NetworkSegment(
            segment_id="secure-api",
            name="Secure API Segment",
            allowed_services=["api"],
            allowed_protocols=["https"],
            min_trust_level=TrustLevel.HIGH,
        )
        zt_controller.policy_enforcer.add_segment(segment)
        
        # Create session and authorize
        token = zt_controller.auth_engine.create_session("api-user", TrustLevel.HIGH)
        zt_controller.check_device_health("device-1", "11.0", "2024-01")
        
        allowed, _ = zt_controller.authorize_access(token, "secure-api", "api", "device-1")
        
        assert allowed is True
        assert api_key.value is not None
        
        # Verify both systems are operational
        zt_status = zt_controller.get_system_status()
        secret_status = secret_system.get_system_status()
        
        assert zt_status["active_segments"] == 1
        assert secret_status["total_secrets"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
