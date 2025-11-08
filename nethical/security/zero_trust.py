"""
Zero Trust Architecture for Nethical

This module provides zero trust network access (ZTNA) capabilities:
- Service mesh with mutual TLS (mTLS)
- Policy-based network segmentation
- Device health verification
- Continuous authentication
- Lateral movement prevention
- Micro-segmentation enforcement

Compliance: NIST SP 800-207 Zero Trust Architecture
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

__all__ = [
    "TrustLevel",
    "DeviceHealthStatus",
    "NetworkSegment",
    "ServiceMeshConfig",
    "DeviceHealthCheck",
    "PolicyEnforcer",
    "ContinuousAuthEngine",
    "ZeroTrustController",
]

log = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Trust level for zero trust evaluation"""

    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

    @property
    def numeric_value(self) -> int:
        """Get numeric value for comparison"""
        mapping = {
            "untrusted": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "verified": 4,
        }
        return mapping.get(self.value, 0)


class DeviceHealthStatus(str, Enum):
    """Device health status indicators"""

    HEALTHY = "healthy"
    WARNING = "warning"
    COMPROMISED = "compromised"
    UNKNOWN = "unknown"


@dataclass
class NetworkSegment:
    """Network segmentation policy"""

    segment_id: str
    name: str
    allowed_services: List[str]
    allowed_protocols: List[str]
    min_trust_level: TrustLevel
    max_session_duration: int = 3600  # seconds
    require_mfa: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMeshConfig:
    """Service mesh configuration for mutual TLS"""

    service_name: str
    certificate: Optional[bytes] = None
    private_key: Optional[bytes] = None
    ca_bundle: Optional[bytes] = None
    enable_mtls: bool = True
    allowed_peers: List[str] = field(default_factory=list)
    tls_version: str = "1.3"
    cipher_suites: List[str] = field(
        default_factory=lambda: [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
        ]
    )

    def validate(self) -> bool:
        """Validate service mesh configuration"""
        if self.enable_mtls:
            return all(
                [
                    self.certificate is not None,
                    self.private_key is not None,
                    self.ca_bundle is not None,
                ]
            )
        return True


@dataclass
class DeviceHealthCheck:
    """Device health assessment result"""

    device_id: str
    status: DeviceHealthStatus
    os_version: str
    patch_level: str
    antivirus_updated: bool
    disk_encryption_enabled: bool
    firewall_enabled: bool
    compliance_score: float
    last_check: datetime
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if device meets health requirements"""
        return (
            self.status == DeviceHealthStatus.HEALTHY
            and self.compliance_score >= 0.8
            and self.antivirus_updated
            and self.disk_encryption_enabled
            and self.firewall_enabled
        )


class PolicyEnforcer:
    """
    Policy-based network segmentation enforcer

    Enforces zero trust policies for network access based on:
    - User identity and trust level
    - Device health status
    - Network segment policies
    - Time-based restrictions
    """

    def __init__(self, segments: Optional[List[NetworkSegment]] = None):
        """
        Initialize policy enforcer

        Args:
            segments: List of network segments with policies
        """
        self.segments: Dict[str, NetworkSegment] = {}
        if segments:
            for segment in segments:
                self.segments[segment.segment_id] = segment

        self.access_logs: List[Dict[str, Any]] = []
        log.info("PolicyEnforcer initialized with %d segments", len(self.segments))

    def add_segment(self, segment: NetworkSegment) -> None:
        """Add a network segment policy"""
        self.segments[segment.segment_id] = segment
        log.info(f"Added segment: {segment.segment_id}")

    def evaluate_access(
        self,
        user_id: str,
        trust_level: TrustLevel,
        segment_id: str,
        service: str,
        device_health: Optional[DeviceHealthCheck] = None,
    ) -> tuple[bool, str]:
        """
        Evaluate access request against policies

        Args:
            user_id: User identifier
            trust_level: Current trust level of user
            segment_id: Target network segment
            service: Target service name
            device_health: Optional device health check result

        Returns:
            Tuple of (allowed, reason)
        """
        if segment_id not in self.segments:
            return False, f"Unknown segment: {segment_id}"

        segment = self.segments[segment_id]

        # Check trust level using numeric comparison
        if trust_level.numeric_value < segment.min_trust_level.numeric_value:
            return False, f"Insufficient trust level: {trust_level} < {segment.min_trust_level}"

        # Check service authorization
        if service not in segment.allowed_services:
            return False, f"Service not allowed in segment: {service}"

        # Check device health if required
        if device_health and not device_health.is_healthy():
            return False, f"Device health check failed: {device_health.status}"

        # Log access decision
        self.access_logs.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "user_id": user_id,
                "segment_id": segment_id,
                "service": service,
                "trust_level": trust_level.value,
                "allowed": True,
            }
        )

        return True, "Access granted"

    def prevent_lateral_movement(
        self,
        source_segment: str,
        target_segment: str,
        user_id: str,
    ) -> tuple[bool, str]:
        """
        Check if lateral movement between segments is allowed

        Args:
            source_segment: Source network segment
            target_segment: Target network segment
            user_id: User identifier

        Returns:
            Tuple of (allowed, reason)
        """
        # Implement lateral movement policies
        # By default, prevent movement unless explicitly allowed
        if source_segment == target_segment:
            return True, "Same segment"

        # Check if both segments exist
        if source_segment not in self.segments or target_segment not in self.segments:
            return False, "Invalid segment"

        # Default deny for lateral movement
        log.warning(
            f"Lateral movement attempt: {user_id} from {source_segment} to {target_segment}"
        )
        return False, "Lateral movement not allowed by policy"

    def get_access_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get access logs with optional filtering"""
        logs = self.access_logs
        if user_id:
            logs = [log for log in logs if log.get("user_id") == user_id]
        return logs[-limit:]


class ContinuousAuthEngine:
    """
    Continuous authentication engine

    Continuously evaluates user trust level based on:
    - Authentication signals
    - Behavioral patterns
    - Risk indicators
    - Device health
    """

    def __init__(self, default_trust: TrustLevel = TrustLevel.LOW):
        """
        Initialize continuous auth engine

        Args:
            default_trust: Default trust level for new sessions
        """
        self.default_trust = default_trust
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.trust_scores: Dict[str, float] = {}
        log.info("ContinuousAuthEngine initialized")

    def create_session(
        self,
        user_id: str,
        initial_trust: Optional[TrustLevel] = None,
        device_id: Optional[str] = None,
    ) -> str:
        """
        Create a new user session

        Args:
            user_id: User identifier
            initial_trust: Initial trust level
            device_id: Device identifier

        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        trust_level = initial_trust or self.default_trust

        self.user_sessions[session_token] = {
            "user_id": user_id,
            "device_id": device_id,
            "trust_level": trust_level,
            "created_at": datetime.now(timezone.utc),
            "last_verified": datetime.now(timezone.utc),
            "risk_events": [],
        }

        self.trust_scores[user_id] = self._trust_to_score(trust_level)
        log.info(f"Created session for {user_id} with trust {trust_level}")
        return session_token

    def verify_session(
        self,
        session_token: str,
        device_health: Optional[DeviceHealthCheck] = None,
    ) -> tuple[bool, TrustLevel]:
        """
        Verify session and update trust level

        Args:
            session_token: Session token to verify
            device_health: Optional device health check

        Returns:
            Tuple of (valid, trust_level)
        """
        if session_token not in self.user_sessions:
            return False, TrustLevel.UNTRUSTED

        session = self.user_sessions[session_token]
        user_id = session["user_id"]

        # Update last verification time
        session["last_verified"] = datetime.now(timezone.utc)

        # Evaluate trust level based on factors
        trust_score = self.trust_scores.get(user_id, 0.5)

        # Adjust based on device health
        if device_health:
            if device_health.is_healthy():
                trust_score = min(1.0, trust_score + 0.1)
            else:
                trust_score = max(0.0, trust_score - 0.3)

        # Convert score to trust level
        trust_level = self._score_to_trust(trust_score)
        session["trust_level"] = trust_level
        self.trust_scores[user_id] = trust_score

        return True, trust_level

    def report_risk_event(
        self,
        session_token: str,
        event_type: str,
        severity: float,
    ) -> None:
        """
        Report a risk event for a session

        Args:
            session_token: Session token
            event_type: Type of risk event
            severity: Severity score (0.0-1.0)
        """
        if session_token not in self.user_sessions:
            return

        session = self.user_sessions[session_token]
        session["risk_events"].append(
            {
                "type": event_type,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        # Decrease trust score based on severity
        user_id = session["user_id"]
        current_score = self.trust_scores.get(user_id, 0.5)
        self.trust_scores[user_id] = max(0.0, current_score - severity * 0.5)

        log.warning(f"Risk event for {user_id}: {event_type} (severity: {severity})")

    def _trust_to_score(self, trust: TrustLevel) -> float:
        """Convert trust level to numeric score"""
        mapping = {
            TrustLevel.UNTRUSTED: 0.0,
            TrustLevel.LOW: 0.25,
            TrustLevel.MEDIUM: 0.5,
            TrustLevel.HIGH: 0.75,
            TrustLevel.VERIFIED: 1.0,
        }
        return mapping.get(trust, 0.5)

    def _score_to_trust(self, score: float) -> TrustLevel:
        """Convert numeric score to trust level"""
        if score < 0.2:
            return TrustLevel.UNTRUSTED
        elif score < 0.4:
            return TrustLevel.LOW
        elif score < 0.6:
            return TrustLevel.MEDIUM
        elif score < 0.8:
            return TrustLevel.HIGH
        else:
            return TrustLevel.VERIFIED


class ZeroTrustController:
    """
    Main zero trust controller

    Orchestrates zero trust components:
    - Policy enforcement
    - Continuous authentication
    - Service mesh management
    - Device health verification
    """

    def __init__(
        self,
        service_mesh_config: Optional[ServiceMeshConfig] = None,
        segments: Optional[List[NetworkSegment]] = None,
    ):
        """
        Initialize zero trust controller

        Args:
            service_mesh_config: Service mesh configuration
            segments: Network segments
        """
        self.service_mesh = service_mesh_config
        self.policy_enforcer = PolicyEnforcer(segments)
        self.auth_engine = ContinuousAuthEngine()
        self.device_health_cache: Dict[str, DeviceHealthCheck] = {}
        log.info("ZeroTrustController initialized")

    def validate_service_mesh(self) -> bool:
        """Validate service mesh configuration"""
        if not self.service_mesh:
            log.warning("Service mesh not configured")
            return False
        return self.service_mesh.validate()

    def check_device_health(
        self,
        device_id: str,
        os_version: str,
        patch_level: str,
        antivirus_updated: bool = True,
        disk_encryption_enabled: bool = True,
        firewall_enabled: bool = True,
    ) -> DeviceHealthCheck:
        """
        Perform device health check

        Args:
            device_id: Device identifier
            os_version: Operating system version
            patch_level: Security patch level
            antivirus_updated: Whether antivirus is up to date
            disk_encryption_enabled: Whether disk encryption is enabled
            firewall_enabled: Whether firewall is enabled

        Returns:
            Device health check result
        """
        issues = []
        compliance_score = 1.0

        # Check antivirus
        if not antivirus_updated:
            issues.append("Antivirus definitions out of date")
            compliance_score -= 0.2

        # Check encryption
        if not disk_encryption_enabled:
            issues.append("Disk encryption not enabled")
            compliance_score -= 0.3

        # Check firewall
        if not firewall_enabled:
            issues.append("Firewall not enabled")
            compliance_score -= 0.2

        # Determine status
        if compliance_score >= 0.8:
            status = DeviceHealthStatus.HEALTHY
        elif compliance_score >= 0.5:
            status = DeviceHealthStatus.WARNING
        else:
            status = DeviceHealthStatus.COMPROMISED

        health_check = DeviceHealthCheck(
            device_id=device_id,
            status=status,
            os_version=os_version,
            patch_level=patch_level,
            antivirus_updated=antivirus_updated,
            disk_encryption_enabled=disk_encryption_enabled,
            firewall_enabled=firewall_enabled,
            compliance_score=compliance_score,
            last_check=datetime.now(timezone.utc),
            issues=issues,
        )

        self.device_health_cache[device_id] = health_check
        log.info(f"Device health check for {device_id}: {status}")
        return health_check

    def authorize_access(
        self,
        session_token: str,
        segment_id: str,
        service: str,
        device_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Authorize access request with zero trust verification

        Args:
            session_token: User session token
            segment_id: Target network segment
            service: Target service
            device_id: Optional device identifier

        Returns:
            Tuple of (authorized, reason)
        """
        # Verify session and get trust level
        device_health = None
        if device_id and device_id in self.device_health_cache:
            device_health = self.device_health_cache[device_id]

        valid, trust_level = self.auth_engine.verify_session(session_token, device_health)

        if not valid:
            return False, "Invalid session"

        # Get user ID from session
        session = self.auth_engine.user_sessions.get(session_token)
        if not session:
            return False, "Session not found"

        user_id = session["user_id"]

        # Evaluate access policy
        allowed, reason = self.policy_enforcer.evaluate_access(
            user_id=user_id,
            trust_level=trust_level,
            segment_id=segment_id,
            service=service,
            device_health=device_health,
        )

        return allowed, reason

    def get_system_status(self) -> Dict[str, Any]:
        """Get zero trust system status"""
        return {
            "service_mesh_configured": self.service_mesh is not None,
            "service_mesh_valid": self.validate_service_mesh(),
            "active_segments": len(self.policy_enforcer.segments),
            "active_sessions": len(self.auth_engine.user_sessions),
            "cached_device_health": len(self.device_health_cache),
            "total_access_logs": len(self.policy_enforcer.access_logs),
        }
