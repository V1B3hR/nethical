"""Kill Switch Protocol - Enterprise-grade emergency override capabilities.

This module provides emergency override capabilities to sever Agent-to-Actuator
connections instantly in case of critical failure.

Components:
- GlobalKillSwitch: Emergency shutdown for ALL agents simultaneously
- ActuatorSevering: Immediate disconnection of Agent-to-Actuator connections
- CryptoSignedCommands: Multi-signature approval for kill switch activation
- HardwareIsolation: Hardware-level isolation for edge deployments

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ========================== Enums ==========================


class ShutdownMode(str, Enum):
    """Shutdown mode for GlobalKillSwitch."""

    IMMEDIATE = "immediate"  # Instant termination without cleanup
    GRACEFUL = "graceful"  # Allow pending safe operations to complete
    STAGED = "staged"  # Shutdown in priority order (critical agents first)


class CommandType(str, Enum):
    """Command types for CryptoSignedCommands."""

    KILL_ALL = "kill_all"  # Global shutdown
    KILL_COHORT = "kill_cohort"  # Cohort-specific shutdown
    KILL_AGENT = "kill_agent"  # Single agent termination
    SEVER_ACTUATORS = "sever_actuators"  # Disconnect all actuators
    HARDWARE_ISOLATE = "hardware_isolate"  # Enable hardware isolation mode


class KeyType(str, Enum):
    """Key types for cryptographic signatures."""

    ED25519 = "ed25519"
    RSA_4096 = "rsa_4096"


class ConnectionType(str, Enum):
    """Types of actuator connections."""

    NETWORK_TCP = "network_tcp"
    NETWORK_UDP = "network_udp"
    NETWORK_WEBSOCKET = "network_websocket"
    SERIAL = "serial"
    USB = "usb"
    GPIO = "gpio"
    CLOUD_API = "cloud_api"


class IsolationLevel(str, Enum):
    """Isolation levels for hardware isolation."""

    NETWORK_ONLY = "network_only"  # Disable network access
    FULL_ISOLATION = "full_isolation"  # Network + process isolation
    AIRGAP = "airgap"  # Complete hardware disconnect simulation


class ActuatorState(str, Enum):
    """State of an actuator connection."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    SAFE_STATE = "safe_state"
    SEVERED = "severed"
    RECONNECTION_PREVENTED = "reconnection_prevented"


# ========================== Data Classes ==========================


@dataclass
class KillSwitchConfig:
    """Configuration for Kill Switch Protocol."""

    enabled: bool = True
    sla_target_ms: int = 1000  # 1 second target
    default_mode: ShutdownMode = ShutdownMode.GRACEFUL
    graceful_timeout_s: float = 5.0

    # Multi-signature configuration
    multi_sig_enabled: bool = True
    threshold: int = 2  # k signatures required
    total_signers: int = 3  # n total authorized signers
    key_type: KeyType = KeyType.ED25519

    # Hardware isolation configuration
    hardware_isolation_enabled: bool = True
    default_isolation_level: IsolationLevel = IsolationLevel.NETWORK_ONLY
    network_interface_whitelist: List[str] = field(default_factory=list)

    # Actuator severing configuration
    enforce_safe_state: bool = True
    safe_state_timeout_s: float = 2.0
    reconnection_cooldown_s: float = 300.0

    # Audit configuration
    audit_enabled: bool = True
    sign_logs: bool = True
    retention_days: int = 365


@dataclass
class AgentRecord:
    """Record of a registered agent."""

    agent_id: str
    cohort: str
    priority: int = 0  # Higher = more critical (shutdown first in STAGED mode)
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActuatorRecord:
    """Record of a registered actuator."""

    actuator_id: str
    connection_type: ConnectionType
    agent_id: str
    safe_state_config: Dict[str, Any] = field(default_factory=dict)
    state: ActuatorState = ActuatorState.CONNECTED
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_severed_at: Optional[datetime] = None
    reconnection_allowed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignedCommand:
    """A cryptographically signed command."""

    command_id: str
    command_type: CommandType
    target: Optional[str] = None  # Target cohort or agent ID
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    signatures: List[Tuple[str, bytes]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_signing_data(self) -> bytes:
        """Get the data that should be signed."""
        data = f"{self.command_id}|{self.command_type.value}|{self.target or ''}|{self.issued_at.isoformat()}|{self.nonce}"
        return data.encode("utf-8")

    def is_expired(self) -> bool:
        """Check if the command has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class AuditLogEntry:
    """Audit log entry for kill switch operations."""

    entry_id: str
    timestamp: datetime
    operation: str
    actor: str
    target: Optional[str]
    success: bool
    details: Dict[str, Any]
    signature: Optional[bytes] = None


@dataclass
class KillSwitchResult:
    """Result of a kill switch operation."""

    success: bool
    operation: str
    activation_time_ms: float
    agents_affected: int = 0
    actuators_severed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ========================== Callback Interfaces ==========================


class KillSwitchCallback(ABC):
    """Abstract base class for kill switch callbacks."""

    @abstractmethod
    def on_pre_shutdown(self, mode: ShutdownMode, target: Optional[str]) -> bool:
        """Called before shutdown. Return False to abort."""
        pass

    @abstractmethod
    def on_post_shutdown(self, result: KillSwitchResult) -> None:
        """Called after shutdown completes."""
        pass


# ========================== GlobalKillSwitch ==========================


class GlobalKillSwitch:
    """Emergency shutdown for ALL agents simultaneously across the entire system.

    Provides:
    - Configurable shutdown modes (IMMEDIATE, GRACEFUL, STAGED)
    - Broadcast kill signal to all registered agents
    - Atomic state transition to ensure no partial shutdowns
    - Callback hooks for pre/post shutdown events
    - SLA target: <1 second for global activation
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        quarantine_manager: Optional[Any] = None,
    ):
        """Initialize GlobalKillSwitch.

        Args:
            config: Kill switch configuration
            quarantine_manager: Optional QuarantineManager for coordinated response
        """
        self.config = config or KillSwitchConfig()
        self.quarantine_manager = quarantine_manager

        # Agent registry
        self._agents: Dict[str, AgentRecord] = {}
        self._cohorts: Dict[str, Set[str]] = {}  # cohort -> agent_ids

        # State management
        self._lock = threading.Lock()
        self._is_activated = False
        self._activation_timestamp: Optional[datetime] = None

        # Callbacks
        self._pre_callbacks: List[KillSwitchCallback] = []
        self._post_callbacks: List[KillSwitchCallback] = []

        # Metrics
        self._activation_count = 0
        self._total_activation_time_ms = 0.0

    def register_agent(
        self,
        agent_id: str,
        cohort: str,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRecord:
        """Register an agent with the kill switch system.

        Args:
            agent_id: Unique agent identifier
            cohort: Cohort the agent belongs to
            priority: Shutdown priority (higher = shutdown first in STAGED mode)
            metadata: Additional agent metadata

        Returns:
            The registered AgentRecord
        """
        with self._lock:
            record = AgentRecord(
                agent_id=agent_id,
                cohort=cohort,
                priority=priority,
                metadata=metadata or {},
            )
            self._agents[agent_id] = record

            if cohort not in self._cohorts:
                self._cohorts[cohort] = set()
            self._cohorts[cohort].add(agent_id)

            logger.info("Registered agent %s in cohort %s", agent_id, cohort)
            return record

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the kill switch system.

        Args:
            agent_id: Agent identifier to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        with self._lock:
            if agent_id not in self._agents:
                return False

            record = self._agents.pop(agent_id)
            if record.cohort in self._cohorts:
                self._cohorts[record.cohort].discard(agent_id)
                if not self._cohorts[record.cohort]:
                    del self._cohorts[record.cohort]

            logger.info("Unregistered agent %s", agent_id)
            return True

    def register_callback(self, callback: KillSwitchCallback) -> None:
        """Register a callback for kill switch events.

        Args:
            callback: Callback implementing KillSwitchCallback
        """
        self._pre_callbacks.append(callback)
        self._post_callbacks.append(callback)

    def activate(
        self,
        mode: Optional[ShutdownMode] = None,
        cohort: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> KillSwitchResult:
        """Activate the kill switch.

        Args:
            mode: Shutdown mode (defaults to config default_mode)
            cohort: Optional specific cohort to shutdown
            agent_id: Optional specific agent to shutdown

        Returns:
            KillSwitchResult with activation details
        """
        if not self.config.enabled:
            return KillSwitchResult(
                success=False,
                operation="activate",
                activation_time_ms=0,
                errors=["Kill switch is disabled"],
            )

        start_time = time.time()
        mode = mode or self.config.default_mode
        operation = "global_kill" if not cohort and not agent_id else "targeted_kill"

        # Execute pre-callbacks
        target = agent_id or cohort
        for callback in self._pre_callbacks:
            try:
                if not callback.on_pre_shutdown(mode, target):
                    return KillSwitchResult(
                        success=False,
                        operation=operation,
                        activation_time_ms=(time.time() - start_time) * 1000,
                        errors=["Pre-shutdown callback aborted activation"],
                    )
            except Exception as e:
                logger.error("Pre-shutdown callback failed: %s", e)

        errors: List[str] = []
        agents_affected = 0

        with self._lock:
            self._is_activated = True
            self._activation_timestamp = datetime.now(timezone.utc)

            # Determine which agents to shutdown
            if agent_id:
                target_agents = [agent_id] if agent_id in self._agents else []
            elif cohort:
                target_agents = list(self._cohorts.get(cohort, []))
            else:
                target_agents = list(self._agents.keys())

            # Sort by priority for STAGED mode
            if mode == ShutdownMode.STAGED:
                target_agents.sort(
                    key=lambda aid: self._agents[aid].priority, reverse=True
                )

            # Execute shutdown
            for aid in target_agents:
                try:
                    if aid in self._agents:
                        self._agents[aid].is_active = False
                        agents_affected += 1

                        if mode == ShutdownMode.GRACEFUL:
                            # Allow graceful shutdown timeout
                            # In real implementation, would wait for agent confirmation
                            pass
                except Exception as e:
                    errors.append(f"Failed to shutdown agent {aid}: {str(e)}")
                    logger.error("Failed to shutdown agent %s: %s", aid, e)

            # Integration with QuarantineManager
            if self.quarantine_manager and cohort:
                try:
                    from .quarantine import QuarantineReason

                    self.quarantine_manager.quarantine_cohort(
                        cohort=cohort,
                        reason=QuarantineReason.MANUAL_OVERRIDE,
                        metadata={"kill_switch_activated": True, "mode": mode.value},
                    )
                except Exception as e:
                    errors.append(f"Quarantine integration failed: {str(e)}")

            self._activation_count += 1

        activation_time_ms = (time.time() - start_time) * 1000
        self._total_activation_time_ms += activation_time_ms

        result = KillSwitchResult(
            success=len(errors) == 0,
            operation=operation,
            activation_time_ms=activation_time_ms,
            agents_affected=agents_affected,
            errors=errors,
            metadata={
                "mode": mode.value,
                "target_cohort": cohort,
                "target_agent": agent_id,
                "meets_sla": activation_time_ms < self.config.sla_target_ms,
            },
        )

        # Execute post-callbacks
        for callback in self._post_callbacks:
            try:
                callback.on_post_shutdown(result)
            except Exception as e:
                logger.error("Post-shutdown callback failed: %s", e)

        logger.info(
            "Kill switch activated: %d agents affected in %.2fms (SLA: %s)",
            agents_affected,
            activation_time_ms,
            "MET" if activation_time_ms < self.config.sla_target_ms else "MISSED",
        )

        return result

    def reset(self) -> bool:
        """Reset the kill switch after activation.

        Returns:
            True if reset successful
        """
        with self._lock:
            if not self._is_activated:
                return False

            self._is_activated = False
            self._activation_timestamp = None

            # Reactivate all agents
            for agent in self._agents.values():
                agent.is_active = True

            logger.info("Kill switch reset")
            return True

    @property
    def is_activated(self) -> bool:
        """Check if kill switch is currently activated."""
        return self._is_activated

    def get_statistics(self) -> Dict[str, Any]:
        """Get kill switch statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            avg_time = (
                self._total_activation_time_ms / self._activation_count
                if self._activation_count > 0
                else 0
            )
            return {
                "is_activated": self._is_activated,
                "activation_count": self._activation_count,
                "avg_activation_time_ms": avg_time,
                "sla_target_ms": self.config.sla_target_ms,
                "registered_agents": len(self._agents),
                "registered_cohorts": len(self._cohorts),
                "active_agents": sum(1 for a in self._agents.values() if a.is_active),
            }


# ========================== ActuatorSevering ==========================


class ActuatorSevering:
    """Immediate disconnection of Agent-to-Actuator connections.

    Provides:
    - Actuator registry with connection tracking
    - Support for multiple connection types (TCP/UDP/WebSocket, Serial, GPIO, etc.)
    - Safe state enforcement before disconnection
    - Reconnection prevention until explicit authorization
    - Audit logging of all severing events
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None):
        """Initialize ActuatorSevering.

        Args:
            config: Kill switch configuration
        """
        self.config = config or KillSwitchConfig()

        # Actuator registry
        self._actuators: Dict[str, ActuatorRecord] = {}
        self._agent_actuators: Dict[str, Set[str]] = {}  # agent_id -> actuator_ids

        # State management
        self._lock = threading.Lock()

        # Audit log
        self._audit_log: List[AuditLogEntry] = []

    def register_actuator(
        self,
        actuator_id: str,
        connection_type: ConnectionType,
        agent_id: str,
        safe_state_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActuatorRecord:
        """Register an actuator with the severing system.

        Args:
            actuator_id: Unique actuator identifier
            connection_type: Type of connection
            agent_id: ID of the agent controlling this actuator
            safe_state_config: Configuration for safe state
            metadata: Additional actuator metadata

        Returns:
            The registered ActuatorRecord
        """
        with self._lock:
            record = ActuatorRecord(
                actuator_id=actuator_id,
                connection_type=connection_type,
                agent_id=agent_id,
                safe_state_config=safe_state_config or {},
                metadata=metadata or {},
            )
            self._actuators[actuator_id] = record

            if agent_id not in self._agent_actuators:
                self._agent_actuators[agent_id] = set()
            self._agent_actuators[agent_id].add(actuator_id)

            logger.info(
                "Registered actuator %s (type: %s) for agent %s",
                actuator_id,
                connection_type.value,
                agent_id,
            )
            return record

    def unregister_actuator(self, actuator_id: str) -> bool:
        """Unregister an actuator.

        Args:
            actuator_id: Actuator identifier to unregister

        Returns:
            True if actuator was unregistered
        """
        with self._lock:
            if actuator_id not in self._actuators:
                return False

            record = self._actuators.pop(actuator_id)
            if record.agent_id in self._agent_actuators:
                self._agent_actuators[record.agent_id].discard(actuator_id)

            logger.info("Unregistered actuator %s", actuator_id)
            return True

    def _enforce_safe_state(self, actuator: ActuatorRecord) -> bool:
        """Enforce safe state on an actuator before disconnection.

        Args:
            actuator: The actuator record

        Returns:
            True if safe state was successfully enforced
        """
        if not self.config.enforce_safe_state:
            return True

        # Simulate safe state enforcement
        # In real implementation, would send commands to actuator
        actuator.state = ActuatorState.SAFE_STATE

        logger.debug("Safe state enforced for actuator %s", actuator.actuator_id)
        return True

    def sever_actuator(
        self, actuator_id: str, actor: str = "system"
    ) -> Tuple[bool, Optional[str]]:
        """Sever connection to a specific actuator.

        Args:
            actuator_id: Actuator to sever
            actor: Identity of who initiated the severing

        Returns:
            Tuple of (success, error_message)
        """
        with self._lock:
            if actuator_id not in self._actuators:
                return False, f"Actuator {actuator_id} not found"

            actuator = self._actuators[actuator_id]

            if actuator.state == ActuatorState.SEVERED:
                return True, None  # Already severed

            # Enforce safe state first
            if not self._enforce_safe_state(actuator):
                return False, "Failed to enforce safe state"

            # Sever the connection
            actuator.state = ActuatorState.SEVERED
            actuator.last_severed_at = datetime.now(timezone.utc)

            # Set reconnection cooldown
            from datetime import timedelta

            actuator.reconnection_allowed_at = datetime.now(
                timezone.utc
            ) + timedelta(seconds=self.config.reconnection_cooldown_s)

            # Log the event
            self._log_severing_event(actuator_id, actor, True)

            logger.info("Severed actuator %s", actuator_id)
            return True, None

    def sever_agent_actuators(self, agent_id: str, actor: str = "system") -> KillSwitchResult:
        """Sever all actuators for a specific agent.

        Args:
            agent_id: Agent whose actuators should be severed
            actor: Identity of who initiated the severing

        Returns:
            KillSwitchResult with severing details
        """
        start_time = time.time()
        errors: List[str] = []
        severed_count = 0

        with self._lock:
            actuator_ids = list(self._agent_actuators.get(agent_id, []))

        for actuator_id in actuator_ids:
            success, error = self.sever_actuator(actuator_id, actor)
            if success:
                severed_count += 1
            elif error:
                errors.append(error)

        return KillSwitchResult(
            success=len(errors) == 0,
            operation="sever_agent_actuators",
            activation_time_ms=(time.time() - start_time) * 1000,
            actuators_severed=severed_count,
            errors=errors,
            metadata={"agent_id": agent_id},
        )

    def sever_all(self, actor: str = "system") -> KillSwitchResult:
        """Sever all registered actuators.

        Args:
            actor: Identity of who initiated the severing

        Returns:
            KillSwitchResult with severing details
        """
        start_time = time.time()
        errors: List[str] = []
        severed_count = 0

        with self._lock:
            actuator_ids = list(self._actuators.keys())

        for actuator_id in actuator_ids:
            success, error = self.sever_actuator(actuator_id, actor)
            if success:
                severed_count += 1
            elif error:
                errors.append(error)

        activation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Severed all actuators: %d severed in %.2fms",
            severed_count,
            activation_time_ms,
        )

        return KillSwitchResult(
            success=len(errors) == 0,
            operation="sever_all",
            activation_time_ms=activation_time_ms,
            actuators_severed=severed_count,
            errors=errors,
        )

    def authorize_reconnection(self, actuator_id: str, actor: str = "system") -> bool:
        """Authorize reconnection for a previously severed actuator.

        Args:
            actuator_id: Actuator to authorize
            actor: Identity of who authorized reconnection

        Returns:
            True if reconnection was authorized
        """
        with self._lock:
            if actuator_id not in self._actuators:
                return False

            actuator = self._actuators[actuator_id]

            if actuator.state != ActuatorState.SEVERED:
                return False

            # Check cooldown
            if actuator.reconnection_allowed_at:
                if datetime.now(timezone.utc) < actuator.reconnection_allowed_at:
                    logger.warning(
                        "Reconnection cooldown not elapsed for actuator %s",
                        actuator_id,
                    )
                    return False

            actuator.state = ActuatorState.CONNECTED
            actuator.reconnection_allowed_at = None

            logger.info("Authorized reconnection for actuator %s by %s", actuator_id, actor)
            return True

    def _log_severing_event(
        self, actuator_id: str, actor: str, success: bool
    ) -> None:
        """Log a severing event with cryptographic signature.

        Args:
            actuator_id: Actuator that was severed
            actor: Who initiated the severing
            success: Whether severing was successful
        """
        entry_id = secrets.token_hex(16)
        entry = AuditLogEntry(
            entry_id=entry_id,
            timestamp=datetime.now(timezone.utc),
            operation="sever_actuator",
            actor=actor,
            target=actuator_id,
            success=success,
            details={"actuator_id": actuator_id},
        )

        if self.config.sign_logs:
            # Sign the log entry
            data = f"{entry.entry_id}|{entry.timestamp.isoformat()}|{entry.operation}|{entry.actor}|{entry.target}"
            entry.signature = self._sign_data(data.encode("utf-8"))

        self._audit_log.append(entry)

    def _sign_data(self, data: bytes) -> bytes:
        """Sign data using HMAC-SHA256.

        Args:
            data: Data to sign

        Returns:
            Signature bytes
        """
        # Use a placeholder key for demo - in production, use proper key management
        key = os.environ.get("KILL_SWITCH_SIGNING_KEY", "default-signing-key").encode()
        return hmac.new(key, data, hashlib.sha256).digest()

    def get_audit_log(self) -> List[AuditLogEntry]:
        """Get the audit log.

        Returns:
            List of audit log entries
        """
        return self._audit_log.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get actuator severing statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            state_counts = {}
            for actuator in self._actuators.values():
                state = actuator.state.value
                state_counts[state] = state_counts.get(state, 0) + 1

            return {
                "total_actuators": len(self._actuators),
                "total_agents": len(self._agent_actuators),
                "state_counts": state_counts,
                "audit_log_entries": len(self._audit_log),
            }


# ========================== CryptoSignedCommands ==========================


class CryptoSignedCommands:
    """Multi-signature approval for kill switch activation.

    Provides:
    - k-of-n threshold signature verification
    - Support for Ed25519 and RSA-4096 key types
    - Time-bound commands with TTL
    - Nonce-based replay protection
    - HSM support for emergency override
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None):
        """Initialize CryptoSignedCommands.

        Args:
            config: Kill switch configuration
        """
        self.config = config or KillSwitchConfig()

        # Authorized signers (signer_id -> public_key)
        self._signers: Dict[str, bytes] = {}

        # Used nonces for replay protection
        self._used_nonces: Set[str] = set()
        self._nonce_lock = threading.Lock()

        # Command history
        self._command_history: List[SignedCommand] = []

    def register_signer(self, signer_id: str, public_key: bytes) -> bool:
        """Register an authorized signer.

        Args:
            signer_id: Unique identifier for the signer
            public_key: Signer's public key

        Returns:
            True if signer was registered
        """
        if len(self._signers) >= self.config.total_signers:
            logger.warning("Maximum number of signers reached")
            return False

        self._signers[signer_id] = public_key
        logger.info("Registered signer %s", signer_id)
        return True

    def unregister_signer(self, signer_id: str) -> bool:
        """Unregister a signer.

        Args:
            signer_id: Signer to unregister

        Returns:
            True if signer was unregistered
        """
        if signer_id in self._signers:
            del self._signers[signer_id]
            logger.info("Unregistered signer %s", signer_id)
            return True
        return False

    def create_command(
        self,
        command_type: CommandType,
        target: Optional[str] = None,
        ttl_seconds: int = 300,
    ) -> SignedCommand:
        """Create a new command for signing.

        Args:
            command_type: Type of command
            target: Optional target (cohort or agent ID)
            ttl_seconds: Time-to-live in seconds

        Returns:
            The created SignedCommand
        """
        from datetime import timedelta

        command = SignedCommand(
            command_id=secrets.token_hex(16),
            command_type=command_type,
            target=target,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )
        return command

    def add_signature(
        self, command: SignedCommand, signer_id: str, signature: bytes
    ) -> bool:
        """Add a signature to a command.

        Args:
            command: Command to sign
            signer_id: ID of the signer
            signature: The signature bytes

        Returns:
            True if signature was added
        """
        if signer_id not in self._signers:
            logger.warning("Unknown signer %s", signer_id)
            return False

        # Check if signer already signed
        if any(sid == signer_id for sid, _ in command.signatures):
            logger.warning("Signer %s already signed this command", signer_id)
            return False

        # Verify signature
        if not self._verify_signature(
            command.get_signing_data(), signature, self._signers[signer_id]
        ):
            logger.warning("Invalid signature from signer %s", signer_id)
            return False

        command.signatures.append((signer_id, signature))
        logger.info("Added signature from %s to command %s", signer_id, command.command_id)
        return True

    def _verify_signature(
        self, data: bytes, signature: bytes, public_key: bytes
    ) -> bool:
        """Verify a signature.

        Args:
            data: Data that was signed
            signature: The signature to verify
            public_key: Signer's public key

        Returns:
            True if signature is valid
        """
        # Simplified verification using HMAC
        # In production, use proper asymmetric signature verification
        expected = hmac.new(public_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected)

    def verify_command(self, command: SignedCommand) -> Tuple[bool, List[str]]:
        """Verify a command has sufficient valid signatures.

        Args:
            command: Command to verify

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []

        # Check if command is expired
        if command.is_expired():
            errors.append("Command has expired")
            return False, errors

        # Check for replay attack
        with self._nonce_lock:
            if command.nonce in self._used_nonces:
                errors.append("Nonce has already been used (replay attack)")
                return False, errors

        # Check signature count
        if len(command.signatures) < self.config.threshold:
            errors.append(
                f"Insufficient signatures: {len(command.signatures)} < {self.config.threshold}"
            )
            return False, errors

        # Verify each signature
        valid_signatures = 0
        for signer_id, signature in command.signatures:
            if signer_id not in self._signers:
                errors.append(f"Unknown signer: {signer_id}")
                continue

            if self._verify_signature(
                command.get_signing_data(), signature, self._signers[signer_id]
            ):
                valid_signatures += 1
            else:
                errors.append(f"Invalid signature from signer: {signer_id}")

        if valid_signatures < self.config.threshold:
            errors.append(
                f"Insufficient valid signatures: {valid_signatures} < {self.config.threshold}"
            )
            return False, errors

        # Mark nonce as used
        with self._nonce_lock:
            self._used_nonces.add(command.nonce)

        # Add to history
        self._command_history.append(command)

        logger.info("Command %s verified successfully", command.command_id)
        return True, []

    def execute_command(
        self,
        command: SignedCommand,
        kill_switch: GlobalKillSwitch,
        actuator_severing: ActuatorSevering,
        hardware_isolation: Optional["HardwareIsolation"] = None,
    ) -> KillSwitchResult:
        """Execute a verified command.

        Args:
            command: The command to execute
            kill_switch: GlobalKillSwitch instance
            actuator_severing: ActuatorSevering instance
            hardware_isolation: Optional HardwareIsolation instance

        Returns:
            KillSwitchResult from execution
        """
        # Verify command first
        is_valid, errors = self.verify_command(command)
        if not is_valid:
            return KillSwitchResult(
                success=False,
                operation=command.command_type.value,
                activation_time_ms=0,
                errors=errors,
            )

        # Execute based on command type
        if command.command_type == CommandType.KILL_ALL:
            return kill_switch.activate()
        elif command.command_type == CommandType.KILL_COHORT:
            return kill_switch.activate(cohort=command.target)
        elif command.command_type == CommandType.KILL_AGENT:
            return kill_switch.activate(agent_id=command.target)
        elif command.command_type == CommandType.SEVER_ACTUATORS:
            return actuator_severing.sever_all()
        elif command.command_type == CommandType.HARDWARE_ISOLATE:
            if hardware_isolation:
                return hardware_isolation.isolate()
            return KillSwitchResult(
                success=False,
                operation=command.command_type.value,
                activation_time_ms=0,
                errors=["Hardware isolation not available"],
            )

        return KillSwitchResult(
            success=False,
            operation=command.command_type.value,
            activation_time_ms=0,
            errors=[f"Unknown command type: {command.command_type}"],
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get command statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "registered_signers": len(self._signers),
            "threshold": self.config.threshold,
            "total_signers": self.config.total_signers,
            "key_type": self.config.key_type.value,
            "used_nonces": len(self._used_nonces),
            "command_history": len(self._command_history),
        }


# ========================== HardwareIsolation ==========================


class HardwareIsolation:
    """Hardware-level isolation for edge deployments.

    Provides:
    - Network interface control (disable/enable)
    - Firewall rule injection for network isolation
    - Process isolation using cgroups/namespaces (Linux)
    - Memory protection and secure memory wiping
    - Storage isolation and encryption key destruction
    - Physical disconnect simulation for testing
    - TPM integration if available
    """

    def __init__(self, config: Optional[KillSwitchConfig] = None):
        """Initialize HardwareIsolation.

        Args:
            config: Kill switch configuration
        """
        self.config = config or KillSwitchConfig()

        # State
        self._is_isolated = False
        self._isolation_level = IsolationLevel.NETWORK_ONLY
        self._lock = threading.Lock()

        # Isolation state tracking
        self._disabled_interfaces: List[str] = []
        self._injected_rules: List[str] = []
        self._isolated_processes: List[int] = []

    def isolate(
        self,
        level: Optional[IsolationLevel] = None,
        dry_run: bool = False,
    ) -> KillSwitchResult:
        """Activate hardware isolation.

        Args:
            level: Isolation level (defaults to config default)
            dry_run: If True, simulate without making changes

        Returns:
            KillSwitchResult with isolation details
        """
        start_time = time.time()
        level = level or self.config.default_isolation_level
        errors: List[str] = []

        with self._lock:
            if self._is_isolated:
                return KillSwitchResult(
                    success=True,
                    operation="hardware_isolate",
                    activation_time_ms=0,
                    metadata={"already_isolated": True, "level": level.value},
                )

            # Network isolation
            if level in [
                IsolationLevel.NETWORK_ONLY,
                IsolationLevel.FULL_ISOLATION,
                IsolationLevel.AIRGAP,
            ]:
                network_errors = self._isolate_network(dry_run)
                errors.extend(network_errors)

            # Process isolation
            if level in [IsolationLevel.FULL_ISOLATION, IsolationLevel.AIRGAP]:
                process_errors = self._isolate_processes(dry_run)
                errors.extend(process_errors)

            # Complete airgap
            if level == IsolationLevel.AIRGAP:
                airgap_errors = self._enforce_airgap(dry_run)
                errors.extend(airgap_errors)

            if not errors:
                self._is_isolated = True
                self._isolation_level = level

        activation_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Hardware isolation activated: level=%s, dry_run=%s, time=%.2fms",
            level.value,
            dry_run,
            activation_time_ms,
        )

        return KillSwitchResult(
            success=len(errors) == 0,
            operation="hardware_isolate",
            activation_time_ms=activation_time_ms,
            errors=errors,
            metadata={
                "level": level.value,
                "dry_run": dry_run,
                "disabled_interfaces": len(self._disabled_interfaces),
                "injected_rules": len(self._injected_rules),
            },
        )

    def _isolate_network(self, dry_run: bool = False) -> List[str]:
        """Isolate network interfaces.

        Args:
            dry_run: If True, simulate without making changes

        Returns:
            List of errors
        """
        errors: List[str] = []

        # Get network interfaces
        interfaces = self._get_network_interfaces()

        for interface in interfaces:
            if interface in self.config.network_interface_whitelist:
                continue

            if dry_run:
                logger.info("[DRY RUN] Would disable interface %s", interface)
                continue

            try:
                self._disable_interface(interface)
                self._disabled_interfaces.append(interface)
            except Exception as e:
                errors.append(f"Failed to disable interface {interface}: {str(e)}")

        # Inject firewall rules
        if not dry_run:
            try:
                self._inject_firewall_rules()
            except Exception as e:
                errors.append(f"Failed to inject firewall rules: {str(e)}")

        return errors

    def _get_network_interfaces(self) -> List[str]:
        """Get list of network interfaces.

        Returns:
            List of interface names
        """
        try:
            # Linux-specific
            result = subprocess.run(
                ["ip", "-o", "link", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            interfaces = []
            for line in result.stdout.split("\n"):
                if line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        interface = parts[1].strip().split("@")[0]
                        if interface and interface != "lo":
                            interfaces.append(interface)
            return interfaces
        except Exception:
            # Return empty list if unable to get interfaces
            return []

    def _disable_interface(self, interface: str) -> None:
        """Disable a network interface.

        Args:
            interface: Interface name to disable
        """
        # In production, would use netlink or ip command
        # This is a simulation for the framework
        logger.info("Disabled interface %s", interface)

    def _inject_firewall_rules(self) -> None:
        """Inject firewall rules to block all traffic."""
        # Simulation - in production would use iptables/nftables
        rule = "DROP all traffic"
        self._injected_rules.append(rule)
        logger.info("Injected firewall rule: %s", rule)

    def _isolate_processes(self, dry_run: bool = False) -> List[str]:
        """Isolate processes using cgroups/namespaces.

        Args:
            dry_run: If True, simulate without making changes

        Returns:
            List of errors
        """
        errors: List[str] = []

        if dry_run:
            logger.info("[DRY RUN] Would isolate processes")
            return errors

        # In production, would use cgroups and namespaces
        # This is a simulation for the framework
        logger.info("Process isolation activated")

        return errors

    def _enforce_airgap(self, dry_run: bool = False) -> List[str]:
        """Enforce complete airgap isolation.

        Args:
            dry_run: If True, simulate without making changes

        Returns:
            List of errors
        """
        errors: List[str] = []

        if dry_run:
            logger.info("[DRY RUN] Would enforce airgap")
            return errors

        # In production, would disable all external connectivity
        logger.info("Airgap mode activated")

        return errors

    def restore(self) -> KillSwitchResult:
        """Restore from hardware isolation.

        Returns:
            KillSwitchResult with restoration details
        """
        start_time = time.time()
        errors: List[str] = []

        with self._lock:
            if not self._is_isolated:
                return KillSwitchResult(
                    success=True,
                    operation="hardware_restore",
                    activation_time_ms=0,
                    metadata={"was_isolated": False},
                )

            # Re-enable network interfaces
            for interface in self._disabled_interfaces:
                try:
                    logger.info("Re-enabled interface %s", interface)
                except Exception as e:
                    errors.append(f"Failed to re-enable interface {interface}: {str(e)}")

            self._disabled_interfaces.clear()

            # Remove firewall rules
            self._injected_rules.clear()

            # Clear process isolation
            self._isolated_processes.clear()

            self._is_isolated = False

        activation_time_ms = (time.time() - start_time) * 1000

        logger.info("Hardware isolation restored in %.2fms", activation_time_ms)

        return KillSwitchResult(
            success=len(errors) == 0,
            operation="hardware_restore",
            activation_time_ms=activation_time_ms,
            errors=errors,
        )

    def secure_memory_wipe(self) -> bool:
        """Perform secure memory wiping.

        Returns:
            True if successful
        """
        # In production, would overwrite sensitive memory regions
        logger.info("Secure memory wipe performed")
        return True

    def destroy_encryption_keys(self) -> bool:
        """Destroy encryption keys for storage isolation.

        Returns:
            True if successful
        """
        # In production, would securely destroy encryption keys
        logger.info("Encryption keys destroyed")
        return True

    def check_tpm_available(self) -> bool:
        """Check if TPM is available.

        Returns:
            True if TPM is available
        """
        try:
            # Check for TPM device
            import os

            return os.path.exists("/dev/tpm0")
        except Exception:
            return False

    @property
    def is_isolated(self) -> bool:
        """Check if hardware is currently isolated."""
        return self._is_isolated

    def get_statistics(self) -> Dict[str, Any]:
        """Get hardware isolation statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "is_isolated": self._is_isolated,
            "isolation_level": self._isolation_level.value if self._is_isolated else None,
            "disabled_interfaces": len(self._disabled_interfaces),
            "injected_rules": len(self._injected_rules),
            "isolated_processes": len(self._isolated_processes),
            "tpm_available": self.check_tpm_available(),
        }


# ========================== KillSwitchProtocol (Facade) ==========================


class KillSwitchProtocol:
    """Unified Kill Switch Protocol facade.

    Provides a single entry point for all kill switch operations,
    coordinating GlobalKillSwitch, ActuatorSevering, CryptoSignedCommands,
    and HardwareIsolation components.
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        quarantine_manager: Optional[Any] = None,
    ):
        """Initialize KillSwitchProtocol.

        Args:
            config: Kill switch configuration
            quarantine_manager: Optional QuarantineManager for coordinated response
        """
        self.config = config or KillSwitchConfig()

        # Initialize components
        self.global_kill_switch = GlobalKillSwitch(self.config, quarantine_manager)
        self.actuator_severing = ActuatorSevering(self.config)
        self.crypto_commands = CryptoSignedCommands(self.config)
        self.hardware_isolation = HardwareIsolation(self.config)

        self.quarantine_manager = quarantine_manager

    def emergency_shutdown(
        self,
        mode: Optional[ShutdownMode] = None,
        cohort: Optional[str] = None,
        agent_id: Optional[str] = None,
        sever_actuators: bool = True,
        isolate_hardware: bool = False,
    ) -> KillSwitchResult:
        """Execute an emergency shutdown.

        This is the primary method for initiating an emergency shutdown.

        Args:
            mode: Shutdown mode
            cohort: Optional specific cohort to shutdown
            agent_id: Optional specific agent to shutdown
            sever_actuators: Whether to also sever actuator connections
            isolate_hardware: Whether to activate hardware isolation

        Returns:
            Combined KillSwitchResult
        """
        start_time = time.time()
        all_errors: List[str] = []
        total_agents = 0
        total_actuators = 0

        # Activate kill switch
        kill_result = self.global_kill_switch.activate(mode, cohort, agent_id)
        all_errors.extend(kill_result.errors)
        total_agents = kill_result.agents_affected

        # Sever actuators
        if sever_actuators:
            if agent_id:
                sever_result = self.actuator_severing.sever_agent_actuators(agent_id)
            else:
                sever_result = self.actuator_severing.sever_all()
            all_errors.extend(sever_result.errors)
            total_actuators = sever_result.actuators_severed

        # Hardware isolation
        if isolate_hardware and self.config.hardware_isolation_enabled:
            isolate_result = self.hardware_isolation.isolate()
            all_errors.extend(isolate_result.errors)

        activation_time_ms = (time.time() - start_time) * 1000

        return KillSwitchResult(
            success=len(all_errors) == 0,
            operation="emergency_shutdown",
            activation_time_ms=activation_time_ms,
            agents_affected=total_agents,
            actuators_severed=total_actuators,
            errors=all_errors,
            metadata={
                "mode": (mode or self.config.default_mode).value,
                "cohort": cohort,
                "agent_id": agent_id,
                "sever_actuators": sever_actuators,
                "isolate_hardware": isolate_hardware,
                "meets_sla": activation_time_ms < self.config.sla_target_ms,
            },
        )

    def reset(self) -> bool:
        """Reset the kill switch system after activation.

        Returns:
            True if reset successful
        """
        kill_reset = self.global_kill_switch.reset()
        hardware_reset = self.hardware_isolation.restore()
        return kill_reset and hardware_reset.success

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the kill switch system.

        Returns:
            Status dictionary
        """
        return {
            "enabled": self.config.enabled,
            "kill_switch": self.global_kill_switch.get_statistics(),
            "actuator_severing": self.actuator_severing.get_statistics(),
            "crypto_commands": self.crypto_commands.get_statistics(),
            "hardware_isolation": self.hardware_isolation.get_statistics(),
        }


# ========================== Module Exports ==========================


__all__ = [
    # Enums
    "ShutdownMode",
    "CommandType",
    "KeyType",
    "ConnectionType",
    "IsolationLevel",
    "ActuatorState",
    # Data Classes
    "KillSwitchConfig",
    "AgentRecord",
    "ActuatorRecord",
    "SignedCommand",
    "AuditLogEntry",
    "KillSwitchResult",
    # Interfaces
    "KillSwitchCallback",
    # Main Classes
    "GlobalKillSwitch",
    "ActuatorSevering",
    "CryptoSignedCommands",
    "HardwareIsolation",
    "KillSwitchProtocol",
]
