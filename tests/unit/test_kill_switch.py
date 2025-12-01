"""Tests for the Kill Switch Protocol module.

This module tests:
- GlobalKillSwitch: Emergency shutdown for ALL agents
- ActuatorSevering: Disconnection of Agent-to-Actuator connections
- CryptoSignedCommands: Multi-signature verification
- HardwareIsolation: Hardware-level isolation
- KillSwitchProtocol: Unified facade

Author: Nethical Core Team
Version: 1.0.0
"""

import pytest
import time
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from typing import Optional

from nethical.core.kill_switch import (
    # Enums
    ShutdownMode,
    CommandType,
    KeyType,
    ConnectionType,
    IsolationLevel,
    ActuatorState,
    # Data Classes
    KillSwitchConfig,
    AgentRecord,
    ActuatorRecord,
    SignedCommand,
    AuditLogEntry,
    KillSwitchResult,
    # Interfaces
    KillSwitchCallback,
    # Main Classes
    GlobalKillSwitch,
    ActuatorSevering,
    CryptoSignedCommands,
    HardwareIsolation,
    KillSwitchProtocol,
)


# ========================== Test Helpers ==========================


class MockCallback(KillSwitchCallback):
    """Mock callback for testing."""

    def __init__(self, should_abort: bool = False):
        self.pre_shutdown_called = False
        self.post_shutdown_called = False
        self.should_abort = should_abort
        self.last_mode: Optional[ShutdownMode] = None
        self.last_result: Optional[KillSwitchResult] = None

    def on_pre_shutdown(self, mode: ShutdownMode, target: Optional[str]) -> bool:
        self.pre_shutdown_called = True
        self.last_mode = mode
        return not self.should_abort

    def on_post_shutdown(self, result: KillSwitchResult) -> None:
        self.post_shutdown_called = True
        self.last_result = result


def create_test_signature(data: bytes, key: bytes) -> bytes:
    """Create a test signature using HMAC-SHA256."""
    return hmac.new(key, data, hashlib.sha256).digest()


# ========================== Test Enums ==========================


class TestEnums:
    """Test cases for enums."""

    def test_shutdown_mode_values(self):
        """Test ShutdownMode enum values."""
        assert ShutdownMode.IMMEDIATE.value == "immediate"
        assert ShutdownMode.GRACEFUL.value == "graceful"
        assert ShutdownMode.STAGED.value == "staged"

    def test_command_type_values(self):
        """Test CommandType enum values."""
        assert CommandType.KILL_ALL.value == "kill_all"
        assert CommandType.KILL_COHORT.value == "kill_cohort"
        assert CommandType.KILL_AGENT.value == "kill_agent"
        assert CommandType.SEVER_ACTUATORS.value == "sever_actuators"
        assert CommandType.HARDWARE_ISOLATE.value == "hardware_isolate"

    def test_connection_type_values(self):
        """Test ConnectionType enum values."""
        assert ConnectionType.NETWORK_TCP.value == "network_tcp"
        assert ConnectionType.SERIAL.value == "serial"
        assert ConnectionType.GPIO.value == "gpio"

    def test_isolation_level_values(self):
        """Test IsolationLevel enum values."""
        assert IsolationLevel.NETWORK_ONLY.value == "network_only"
        assert IsolationLevel.FULL_ISOLATION.value == "full_isolation"
        assert IsolationLevel.AIRGAP.value == "airgap"


# ========================== Test KillSwitchConfig ==========================


class TestKillSwitchConfig:
    """Test cases for KillSwitchConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KillSwitchConfig()

        assert config.enabled is True
        assert config.sla_target_ms == 1000
        assert config.default_mode == ShutdownMode.GRACEFUL
        assert config.graceful_timeout_s == 5.0
        assert config.multi_sig_enabled is True
        assert config.threshold == 2
        assert config.total_signers == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = KillSwitchConfig(
            enabled=False,
            sla_target_ms=500,
            default_mode=ShutdownMode.IMMEDIATE,
            threshold=3,
            total_signers=5,
        )

        assert config.enabled is False
        assert config.sla_target_ms == 500
        assert config.default_mode == ShutdownMode.IMMEDIATE
        assert config.threshold == 3
        assert config.total_signers == 5


# ========================== Test GlobalKillSwitch ==========================


class TestGlobalKillSwitch:
    """Test cases for GlobalKillSwitch."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = KillSwitchConfig(sla_target_ms=1000)
        self.kill_switch = GlobalKillSwitch(config=self.config)

    def test_initialization(self):
        """Test GlobalKillSwitch initialization."""
        assert self.kill_switch.config.enabled is True
        assert self.kill_switch.is_activated is False
        assert len(self.kill_switch._agents) == 0

    def test_register_agent(self):
        """Test agent registration."""
        record = self.kill_switch.register_agent(
            agent_id="agent-1",
            cohort="cohort-a",
            priority=10,
            metadata={"type": "worker"},
        )

        assert record.agent_id == "agent-1"
        assert record.cohort == "cohort-a"
        assert record.priority == 10
        assert record.is_active is True
        assert "agent-1" in self.kill_switch._agents

    def test_unregister_agent(self):
        """Test agent unregistration."""
        self.kill_switch.register_agent("agent-1", "cohort-a")

        success = self.kill_switch.unregister_agent("agent-1")
        assert success is True
        assert "agent-1" not in self.kill_switch._agents

        # Unregistering non-existent agent
        success = self.kill_switch.unregister_agent("agent-999")
        assert success is False

    def test_activate_global_shutdown(self):
        """Test global shutdown activation."""
        # Register some agents
        self.kill_switch.register_agent("agent-1", "cohort-a")
        self.kill_switch.register_agent("agent-2", "cohort-a")
        self.kill_switch.register_agent("agent-3", "cohort-b")

        result = self.kill_switch.activate(mode=ShutdownMode.IMMEDIATE)

        assert result.success is True
        assert result.agents_affected == 3
        assert self.kill_switch.is_activated is True

    def test_activate_cohort_shutdown(self):
        """Test cohort-specific shutdown."""
        self.kill_switch.register_agent("agent-1", "cohort-a")
        self.kill_switch.register_agent("agent-2", "cohort-a")
        self.kill_switch.register_agent("agent-3", "cohort-b")

        result = self.kill_switch.activate(cohort="cohort-a")

        assert result.success is True
        assert result.agents_affected == 2
        assert result.metadata["target_cohort"] == "cohort-a"

    def test_activate_single_agent_shutdown(self):
        """Test single agent shutdown."""
        self.kill_switch.register_agent("agent-1", "cohort-a")
        self.kill_switch.register_agent("agent-2", "cohort-a")

        result = self.kill_switch.activate(agent_id="agent-1")

        assert result.success is True
        assert result.agents_affected == 1
        assert result.metadata["target_agent"] == "agent-1"

    def test_activate_staged_shutdown(self):
        """Test staged shutdown (priority order)."""
        self.kill_switch.register_agent("agent-low", "cohort-a", priority=1)
        self.kill_switch.register_agent("agent-high", "cohort-a", priority=10)
        self.kill_switch.register_agent("agent-mid", "cohort-a", priority=5)

        result = self.kill_switch.activate(mode=ShutdownMode.STAGED)

        assert result.success is True
        assert result.agents_affected == 3

    def test_activate_disabled(self):
        """Test activation when kill switch is disabled."""
        config = KillSwitchConfig(enabled=False)
        kill_switch = GlobalKillSwitch(config=config)
        kill_switch.register_agent("agent-1", "cohort-a")

        result = kill_switch.activate()

        assert result.success is False
        assert "disabled" in result.errors[0].lower()

    def test_reset(self):
        """Test kill switch reset."""
        self.kill_switch.register_agent("agent-1", "cohort-a")
        self.kill_switch.activate()

        assert self.kill_switch.is_activated is True

        success = self.kill_switch.reset()

        assert success is True
        assert self.kill_switch.is_activated is False
        assert self.kill_switch._agents["agent-1"].is_active is True

    def test_reset_not_activated(self):
        """Test reset when not activated."""
        success = self.kill_switch.reset()
        assert success is False

    def test_callback_execution(self):
        """Test callback execution during activation."""
        callback = MockCallback()
        self.kill_switch.register_callback(callback)
        self.kill_switch.register_agent("agent-1", "cohort-a")

        result = self.kill_switch.activate()

        assert callback.pre_shutdown_called is True
        assert callback.post_shutdown_called is True
        assert callback.last_mode == ShutdownMode.GRACEFUL
        assert callback.last_result is not None

    def test_callback_abort(self):
        """Test callback can abort shutdown."""
        callback = MockCallback(should_abort=True)
        self.kill_switch.register_callback(callback)
        self.kill_switch.register_agent("agent-1", "cohort-a")

        result = self.kill_switch.activate()

        assert result.success is False
        assert "aborted" in result.errors[0].lower()
        assert callback.pre_shutdown_called is True
        assert callback.post_shutdown_called is False

    def test_sla_compliance(self):
        """Test SLA compliance (<1 second activation)."""
        # Register many agents to test performance
        for i in range(100):
            self.kill_switch.register_agent(f"agent-{i}", f"cohort-{i % 5}")

        result = self.kill_switch.activate()

        assert result.success is True
        assert result.activation_time_ms < 1000  # SLA target
        assert result.metadata["meets_sla"] is True

    def test_get_statistics(self):
        """Test statistics gathering."""
        self.kill_switch.register_agent("agent-1", "cohort-a")
        self.kill_switch.register_agent("agent-2", "cohort-b")
        self.kill_switch.activate()

        stats = self.kill_switch.get_statistics()

        assert stats["is_activated"] is True
        assert stats["activation_count"] == 1
        assert stats["registered_agents"] == 2
        assert stats["registered_cohorts"] == 2
        assert stats["active_agents"] == 0  # All shut down


# ========================== Test ActuatorSevering ==========================


class TestActuatorSevering:
    """Test cases for ActuatorSevering."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = KillSwitchConfig()
        self.severing = ActuatorSevering(config=self.config)

    def test_initialization(self):
        """Test ActuatorSevering initialization."""
        assert len(self.severing._actuators) == 0
        assert len(self.severing._audit_log) == 0

    def test_register_actuator(self):
        """Test actuator registration."""
        record = self.severing.register_actuator(
            actuator_id="actuator-1",
            connection_type=ConnectionType.NETWORK_TCP,
            agent_id="agent-1",
            safe_state_config={"position": 0},
            metadata={"location": "room-1"},
        )

        assert record.actuator_id == "actuator-1"
        assert record.connection_type == ConnectionType.NETWORK_TCP
        assert record.agent_id == "agent-1"
        assert record.state == ActuatorState.CONNECTED
        assert "actuator-1" in self.severing._actuators

    def test_unregister_actuator(self):
        """Test actuator unregistration."""
        self.severing.register_actuator("actuator-1", ConnectionType.SERIAL, "agent-1")

        success = self.severing.unregister_actuator("actuator-1")
        assert success is True
        assert "actuator-1" not in self.severing._actuators

        success = self.severing.unregister_actuator("actuator-999")
        assert success is False

    def test_sever_actuator(self):
        """Test severing a single actuator."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")

        success, error = self.severing.sever_actuator("actuator-1", "test-user")

        assert success is True
        assert error is None
        assert self.severing._actuators["actuator-1"].state == ActuatorState.SEVERED
        assert self.severing._actuators["actuator-1"].last_severed_at is not None

    def test_sever_actuator_not_found(self):
        """Test severing non-existent actuator."""
        success, error = self.severing.sever_actuator("actuator-999")

        assert success is False
        assert "not found" in error.lower()

    def test_sever_already_severed(self):
        """Test severing already severed actuator."""
        self.severing.register_actuator("actuator-1", ConnectionType.SERIAL, "agent-1")
        self.severing.sever_actuator("actuator-1")

        success, error = self.severing.sever_actuator("actuator-1")

        assert success is True  # Already severed is considered success
        assert error is None

    def test_sever_agent_actuators(self):
        """Test severing all actuators for an agent."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        self.severing.register_actuator("actuator-2", ConnectionType.SERIAL, "agent-1")
        self.severing.register_actuator("actuator-3", ConnectionType.GPIO, "agent-2")

        result = self.severing.sever_agent_actuators("agent-1")

        assert result.success is True
        assert result.actuators_severed == 2
        assert result.metadata["agent_id"] == "agent-1"

    def test_sever_all(self):
        """Test severing all actuators."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        self.severing.register_actuator("actuator-2", ConnectionType.SERIAL, "agent-2")
        self.severing.register_actuator("actuator-3", ConnectionType.GPIO, "agent-3")

        result = self.severing.sever_all()

        assert result.success is True
        assert result.actuators_severed == 3

    def test_authorize_reconnection(self):
        """Test authorizing reconnection."""
        config = KillSwitchConfig(reconnection_cooldown_s=0)  # No cooldown for test
        severing = ActuatorSevering(config=config)

        severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        severing.sever_actuator("actuator-1")

        success = severing.authorize_reconnection("actuator-1")

        assert success is True
        assert severing._actuators["actuator-1"].state == ActuatorState.CONNECTED

    def test_authorize_reconnection_cooldown(self):
        """Test reconnection with cooldown."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        self.severing.sever_actuator("actuator-1")

        # Cooldown should prevent immediate reconnection
        success = self.severing.authorize_reconnection("actuator-1")

        assert success is False  # Still in cooldown

    def test_audit_log(self):
        """Test audit logging."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        self.severing.sever_actuator("actuator-1", "test-user")

        log = self.severing.get_audit_log()

        assert len(log) == 1
        assert log[0].operation == "sever_actuator"
        assert log[0].actor == "test-user"
        assert log[0].target == "actuator-1"
        assert log[0].success is True
        assert log[0].signature is not None  # Signed by default

    def test_get_statistics(self):
        """Test statistics gathering."""
        self.severing.register_actuator("actuator-1", ConnectionType.NETWORK_TCP, "agent-1")
        self.severing.register_actuator("actuator-2", ConnectionType.SERIAL, "agent-1")
        self.severing.sever_actuator("actuator-1")

        stats = self.severing.get_statistics()

        assert stats["total_actuators"] == 2
        assert stats["total_agents"] == 1
        assert stats["state_counts"]["severed"] == 1
        assert stats["state_counts"]["connected"] == 1


# ========================== Test CryptoSignedCommands ==========================


class TestCryptoSignedCommands:
    """Test cases for CryptoSignedCommands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = KillSwitchConfig(threshold=2, total_signers=3)
        self.crypto = CryptoSignedCommands(config=self.config)

        # Register test signers
        self.signer1_key = b"signer1-key-12345678901234567890"
        self.signer2_key = b"signer2-key-12345678901234567890"
        self.signer3_key = b"signer3-key-12345678901234567890"

        self.crypto.register_signer("signer-1", self.signer1_key)
        self.crypto.register_signer("signer-2", self.signer2_key)
        self.crypto.register_signer("signer-3", self.signer3_key)

    def test_initialization(self):
        """Test CryptoSignedCommands initialization."""
        crypto = CryptoSignedCommands()
        assert len(crypto._signers) == 0
        assert len(crypto._used_nonces) == 0

    def test_register_signer(self):
        """Test signer registration."""
        crypto = CryptoSignedCommands()
        success = crypto.register_signer("test-signer", b"test-key")

        assert success is True
        assert "test-signer" in crypto._signers

    def test_register_signer_max_reached(self):
        """Test signer registration when max is reached."""
        success = self.crypto.register_signer("signer-4", b"key-4")

        assert success is False  # Already have 3 signers

    def test_unregister_signer(self):
        """Test signer unregistration."""
        success = self.crypto.unregister_signer("signer-1")
        assert success is True
        assert "signer-1" not in self.crypto._signers

        success = self.crypto.unregister_signer("non-existent")
        assert success is False

    def test_create_command(self):
        """Test command creation."""
        command = self.crypto.create_command(
            command_type=CommandType.KILL_ALL,
            ttl_seconds=300,
        )

        assert command.command_type == CommandType.KILL_ALL
        assert command.target is None
        assert command.nonce is not None
        assert command.expires_at is not None
        assert len(command.signatures) == 0

    def test_add_signature(self):
        """Test adding signatures to command."""
        command = self.crypto.create_command(CommandType.KILL_ALL)

        # Create valid signature
        signature = create_test_signature(command.get_signing_data(), self.signer1_key)

        success = self.crypto.add_signature(command, "signer-1", signature)

        assert success is True
        assert len(command.signatures) == 1

    def test_add_signature_unknown_signer(self):
        """Test adding signature from unknown signer."""
        command = self.crypto.create_command(CommandType.KILL_ALL)

        success = self.crypto.add_signature(command, "unknown-signer", b"fake-sig")

        assert success is False
        assert len(command.signatures) == 0

    def test_add_signature_duplicate(self):
        """Test adding duplicate signature from same signer."""
        command = self.crypto.create_command(CommandType.KILL_ALL)
        signature = create_test_signature(command.get_signing_data(), self.signer1_key)

        self.crypto.add_signature(command, "signer-1", signature)
        success = self.crypto.add_signature(command, "signer-1", signature)

        assert success is False
        assert len(command.signatures) == 1

    def test_verify_command_success(self):
        """Test successful command verification."""
        command = self.crypto.create_command(CommandType.KILL_ALL)

        # Add two valid signatures (meets threshold)
        sig1 = create_test_signature(command.get_signing_data(), self.signer1_key)
        sig2 = create_test_signature(command.get_signing_data(), self.signer2_key)

        self.crypto.add_signature(command, "signer-1", sig1)
        self.crypto.add_signature(command, "signer-2", sig2)

        is_valid, errors = self.crypto.verify_command(command)

        assert is_valid is True
        assert len(errors) == 0

    def test_verify_command_insufficient_signatures(self):
        """Test verification with insufficient signatures."""
        command = self.crypto.create_command(CommandType.KILL_ALL)

        # Only one signature (threshold is 2)
        sig1 = create_test_signature(command.get_signing_data(), self.signer1_key)
        self.crypto.add_signature(command, "signer-1", sig1)

        is_valid, errors = self.crypto.verify_command(command)

        assert is_valid is False
        assert any("insufficient" in e.lower() for e in errors)

    def test_verify_command_expired(self):
        """Test verification of expired command."""
        command = self.crypto.create_command(CommandType.KILL_ALL, ttl_seconds=-1)

        sig1 = create_test_signature(command.get_signing_data(), self.signer1_key)
        sig2 = create_test_signature(command.get_signing_data(), self.signer2_key)

        self.crypto.add_signature(command, "signer-1", sig1)
        self.crypto.add_signature(command, "signer-2", sig2)

        is_valid, errors = self.crypto.verify_command(command)

        assert is_valid is False
        assert any("expired" in e.lower() for e in errors)

    def test_verify_command_replay_protection(self):
        """Test replay attack protection."""
        command = self.crypto.create_command(CommandType.KILL_ALL)

        sig1 = create_test_signature(command.get_signing_data(), self.signer1_key)
        sig2 = create_test_signature(command.get_signing_data(), self.signer2_key)

        self.crypto.add_signature(command, "signer-1", sig1)
        self.crypto.add_signature(command, "signer-2", sig2)

        # First verification should succeed
        is_valid1, _ = self.crypto.verify_command(command)
        assert is_valid1 is True

        # Second verification should fail (nonce already used)
        is_valid2, errors = self.crypto.verify_command(command)
        assert is_valid2 is False
        assert any("replay" in e.lower() for e in errors)

    def test_execute_command(self):
        """Test command execution."""
        kill_switch = GlobalKillSwitch()
        kill_switch.register_agent("agent-1", "cohort-a")

        severing = ActuatorSevering()

        command = self.crypto.create_command(CommandType.KILL_ALL)

        sig1 = create_test_signature(command.get_signing_data(), self.signer1_key)
        sig2 = create_test_signature(command.get_signing_data(), self.signer2_key)

        self.crypto.add_signature(command, "signer-1", sig1)
        self.crypto.add_signature(command, "signer-2", sig2)

        result = self.crypto.execute_command(command, kill_switch, severing)

        assert result.success is True
        assert result.agents_affected == 1

    def test_get_statistics(self):
        """Test statistics gathering."""
        stats = self.crypto.get_statistics()

        assert stats["registered_signers"] == 3
        assert stats["threshold"] == 2
        assert stats["total_signers"] == 3


# ========================== Test HardwareIsolation ==========================


class TestHardwareIsolation:
    """Test cases for HardwareIsolation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = KillSwitchConfig()
        self.isolation = HardwareIsolation(config=self.config)

    def test_initialization(self):
        """Test HardwareIsolation initialization."""
        assert self.isolation.is_isolated is False
        assert len(self.isolation._disabled_interfaces) == 0
        assert len(self.isolation._injected_rules) == 0

    def test_isolate_network_only(self):
        """Test network-only isolation."""
        result = self.isolation.isolate(
            level=IsolationLevel.NETWORK_ONLY,
            dry_run=True,  # Dry run for testing
        )

        assert result.success is True
        assert result.operation == "hardware_isolate"
        assert result.metadata["level"] == "network_only"
        assert result.metadata["dry_run"] is True

    def test_isolate_full_isolation(self):
        """Test full isolation."""
        result = self.isolation.isolate(
            level=IsolationLevel.FULL_ISOLATION,
            dry_run=True,
        )

        assert result.success is True
        assert result.metadata["level"] == "full_isolation"

    def test_isolate_airgap(self):
        """Test airgap isolation."""
        result = self.isolation.isolate(
            level=IsolationLevel.AIRGAP,
            dry_run=True,
        )

        assert result.success is True
        assert result.metadata["level"] == "airgap"

    def test_isolate_already_isolated(self):
        """Test isolation when already isolated."""
        self.isolation.isolate(dry_run=True)

        result = self.isolation.isolate(dry_run=True)

        assert result.success is True
        assert result.metadata.get("already_isolated") is True

    def test_restore(self):
        """Test restoration from isolation."""
        self.isolation.isolate(dry_run=True)

        result = self.isolation.restore()

        assert result.success is True
        assert result.operation == "hardware_restore"
        assert self.isolation.is_isolated is False

    def test_restore_not_isolated(self):
        """Test restoration when not isolated."""
        result = self.isolation.restore()

        assert result.success is True
        assert result.metadata.get("was_isolated") is False

    def test_secure_memory_wipe(self):
        """Test secure memory wiping."""
        success = self.isolation.secure_memory_wipe()
        assert success is True

    def test_destroy_encryption_keys(self):
        """Test encryption key destruction."""
        success = self.isolation.destroy_encryption_keys()
        assert success is True

    def test_check_tpm_available(self):
        """Test TPM availability check."""
        # This just verifies the method runs without error
        available = self.isolation.check_tpm_available()
        assert isinstance(available, bool)

    def test_get_statistics(self):
        """Test statistics gathering."""
        stats = self.isolation.get_statistics()

        assert "is_isolated" in stats
        assert "isolation_level" in stats
        assert "tpm_available" in stats


# ========================== Test KillSwitchProtocol ==========================


class TestKillSwitchProtocol:
    """Test cases for KillSwitchProtocol."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = KillSwitchConfig()
        self.protocol = KillSwitchProtocol(config=self.config)

    def test_initialization(self):
        """Test KillSwitchProtocol initialization."""
        assert self.protocol.global_kill_switch is not None
        assert self.protocol.actuator_severing is not None
        assert self.protocol.crypto_commands is not None
        assert self.protocol.hardware_isolation is not None

    def test_emergency_shutdown_basic(self):
        """Test basic emergency shutdown."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")
        self.protocol.actuator_severing.register_actuator(
            "actuator-1", ConnectionType.NETWORK_TCP, "agent-1"
        )

        result = self.protocol.emergency_shutdown()

        assert result.success is True
        assert result.agents_affected == 1
        assert result.actuators_severed == 1
        assert result.operation == "emergency_shutdown"

    def test_emergency_shutdown_with_mode(self):
        """Test emergency shutdown with specific mode."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")

        result = self.protocol.emergency_shutdown(mode=ShutdownMode.IMMEDIATE)

        assert result.success is True
        assert result.metadata["mode"] == "immediate"

    def test_emergency_shutdown_cohort(self):
        """Test emergency shutdown for specific cohort."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")
        self.protocol.global_kill_switch.register_agent("agent-2", "cohort-b")

        result = self.protocol.emergency_shutdown(cohort="cohort-a")

        assert result.success is True
        assert result.agents_affected == 1
        assert result.metadata["cohort"] == "cohort-a"

    def test_emergency_shutdown_no_actuators(self):
        """Test emergency shutdown without severing actuators."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")
        self.protocol.actuator_severing.register_actuator(
            "actuator-1", ConnectionType.NETWORK_TCP, "agent-1"
        )

        result = self.protocol.emergency_shutdown(sever_actuators=False)

        assert result.success is True
        assert result.actuators_severed == 0

    def test_emergency_shutdown_with_hardware_isolation(self):
        """Test emergency shutdown with hardware isolation."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")

        result = self.protocol.emergency_shutdown(isolate_hardware=True)

        assert result.success is True
        assert result.metadata["isolate_hardware"] is True

    def test_reset(self):
        """Test protocol reset."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")
        self.protocol.emergency_shutdown(isolate_hardware=True)

        success = self.protocol.reset()

        assert success is True
        assert self.protocol.global_kill_switch.is_activated is False
        assert self.protocol.hardware_isolation.is_isolated is False

    def test_get_status(self):
        """Test status retrieval."""
        self.protocol.global_kill_switch.register_agent("agent-1", "cohort-a")

        status = self.protocol.get_status()

        assert "enabled" in status
        assert "kill_switch" in status
        assert "actuator_severing" in status
        assert "crypto_commands" in status
        assert "hardware_isolation" in status

    def test_sla_compliance(self):
        """Test SLA compliance (<1 second)."""
        # Register multiple agents and actuators
        for i in range(50):
            self.protocol.global_kill_switch.register_agent(f"agent-{i}", f"cohort-{i % 5}")
            self.protocol.actuator_severing.register_actuator(
                f"actuator-{i}", ConnectionType.NETWORK_TCP, f"agent-{i}"
            )

        result = self.protocol.emergency_shutdown()

        assert result.success is True
        assert result.activation_time_ms < 1000  # SLA target
        assert result.metadata["meets_sla"] is True


# ========================== Test Integration with Quarantine ==========================


class TestQuarantineIntegration:
    """Test integration with QuarantineManager."""

    def test_hardware_isolate_cohort(self):
        """Test hardware isolation of cohort via QuarantineManager."""
        from nethical.core.quarantine import (
            QuarantineManager,
            QuarantineReason,
            HardwareIsolationLevel,
        )

        manager = QuarantineManager()
        manager.register_agent_cohort("agent-1", "test-cohort")

        record = manager.hardware_isolate_cohort(
            cohort="test-cohort",
            isolation_level=HardwareIsolationLevel.NETWORK_ONLY,
        )

        assert record.reason == QuarantineReason.HARDWARE_ISOLATION_REQUIRED
        assert "hardware_isolation_level" in record.metadata
        assert record.metadata["hardware_isolation_level"] == "network_only"

    def test_get_hardware_isolation_status(self):
        """Test getting hardware isolation status."""
        from nethical.core.quarantine import (
            QuarantineManager,
            HardwareIsolationLevel,
        )

        manager = QuarantineManager()
        manager.register_agent_cohort("agent-1", "test-cohort")

        manager.hardware_isolate_cohort(
            cohort="test-cohort",
            isolation_level=HardwareIsolationLevel.FULL_ISOLATION,
        )

        status = manager.get_hardware_isolation_status("test-cohort")

        assert status["is_hardware_isolated"] is True
        assert status["isolation_level"] == "full_isolation"


# ========================== Test Data Classes ==========================


class TestDataClasses:
    """Test cases for data classes."""

    def test_signed_command_get_signing_data(self):
        """Test SignedCommand signing data generation."""
        command = SignedCommand(
            command_id="cmd-123",
            command_type=CommandType.KILL_ALL,
            target="cohort-a",
        )

        data = command.get_signing_data()

        assert b"cmd-123" in data
        assert b"kill_all" in data
        assert b"cohort-a" in data

    def test_signed_command_is_expired(self):
        """Test SignedCommand expiration check."""
        from datetime import timedelta

        # Not expired
        command = SignedCommand(
            command_id="cmd-123",
            command_type=CommandType.KILL_ALL,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert command.is_expired() is False

        # Expired
        command_expired = SignedCommand(
            command_id="cmd-124",
            command_type=CommandType.KILL_ALL,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert command_expired.is_expired() is True

        # No expiration
        command_no_exp = SignedCommand(
            command_id="cmd-125",
            command_type=CommandType.KILL_ALL,
        )
        assert command_no_exp.is_expired() is False

    def test_agent_record(self):
        """Test AgentRecord dataclass."""
        record = AgentRecord(
            agent_id="agent-1",
            cohort="cohort-a",
            priority=10,
            metadata={"type": "worker"},
        )

        assert record.agent_id == "agent-1"
        assert record.cohort == "cohort-a"
        assert record.priority == 10
        assert record.is_active is True

    def test_actuator_record(self):
        """Test ActuatorRecord dataclass."""
        record = ActuatorRecord(
            actuator_id="actuator-1",
            connection_type=ConnectionType.NETWORK_TCP,
            agent_id="agent-1",
        )

        assert record.actuator_id == "actuator-1"
        assert record.state == ActuatorState.CONNECTED

    def test_kill_switch_result(self):
        """Test KillSwitchResult dataclass."""
        result = KillSwitchResult(
            success=True,
            operation="test",
            activation_time_ms=100.5,
            agents_affected=5,
            actuators_severed=10,
        )

        assert result.success is True
        assert result.activation_time_ms == 100.5
        assert result.agents_affected == 5
        assert result.actuators_severed == 10


# ========================== Performance Tests ==========================


class TestPerformance:
    """Performance tests for SLA compliance."""

    def test_global_kill_switch_performance(self):
        """Test GlobalKillSwitch meets SLA target."""
        config = KillSwitchConfig(sla_target_ms=1000)
        kill_switch = GlobalKillSwitch(config=config)

        # Register 1000 agents across 10 cohorts
        for i in range(1000):
            kill_switch.register_agent(f"agent-{i}", f"cohort-{i % 10}", priority=i)

        # Measure activation time
        start = time.time()
        result = kill_switch.activate()
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert result.agents_affected == 1000
        assert elapsed_ms < 1000  # SLA: <1 second
        assert result.metadata["meets_sla"] is True

    def test_actuator_severing_performance(self):
        """Test ActuatorSevering meets SLA target."""
        severing = ActuatorSevering()

        # Register 500 actuators
        for i in range(500):
            severing.register_actuator(
                f"actuator-{i}",
                ConnectionType.NETWORK_TCP,
                f"agent-{i % 100}",
            )

        # Measure severing time
        start = time.time()
        result = severing.sever_all()
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert result.actuators_severed == 500
        assert elapsed_ms < 1000  # SLA: <1 second

    def test_full_protocol_performance(self):
        """Test full KillSwitchProtocol meets SLA target."""
        protocol = KillSwitchProtocol()

        # Register agents and actuators
        for i in range(100):
            protocol.global_kill_switch.register_agent(f"agent-{i}", f"cohort-{i % 10}")
            protocol.actuator_severing.register_actuator(
                f"actuator-{i}",
                ConnectionType.NETWORK_TCP,
                f"agent-{i}",
            )

        # Measure full emergency shutdown
        start = time.time()
        result = protocol.emergency_shutdown(
            mode=ShutdownMode.IMMEDIATE,
            sever_actuators=True,
            isolate_hardware=False,  # Skip hardware isolation for speed test
        )
        elapsed_ms = (time.time() - start) * 1000

        assert result.success is True
        assert result.agents_affected == 100
        assert result.actuators_severed == 100
        assert elapsed_ms < 1000  # SLA: <1 second
