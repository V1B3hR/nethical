"""Quarantine Mode for Phase 4.3: Quarantine & Incident Response.

This module implements:
- Automatic global policy override for agent cohort with anomalies
- Quarantine workflow and API
- Quarantine scenario simulation (<15s isolation)
"""

import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class QuarantineReason(str, Enum):
    """Reason for quarantine."""

    ANOMALY_DETECTED = "anomaly_detected"
    COORDINATED_ATTACK = "coordinated_attack"
    POLICY_VIOLATION = "policy_violation"
    MANUAL_OVERRIDE = "manual_override"
    SYNTHETIC_TEST = "synthetic_test"
    HIGH_RISK_SCORE = "high_risk_score"


class QuarantineStatus(str, Enum):
    """Quarantine status."""

    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class QuarantinePolicy:
    """Restrictive policy applied during quarantine."""

    allow_read_only: bool = True
    block_external_access: bool = True
    require_manual_review: bool = True
    rate_limit_factor: float = 0.1  # 10% of normal rate
    elevated_monitoring: bool = True
    notify_admins: bool = True


@dataclass
class QuarantineRecord:
    """Record of a quarantine event."""

    cohort: str
    reason: QuarantineReason
    status: QuarantineStatus = QuarantineStatus.PENDING
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    duration_hours: float = 24.0
    auto_release: bool = False
    policy: QuarantinePolicy = field(default_factory=QuarantinePolicy)
    affected_agents: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    activation_time_ms: Optional[float] = None  # Time to activate quarantine


class QuarantineManager:
    """Quarantine management system for rapid incident response."""

    def __init__(
        self,
        default_duration_hours: float = 24.0,
        auto_release: bool = False,
        isolation_threshold: float = 0.75,
        max_isolation_time_hours: float = 168.0,  # 7 days
        target_activation_time_s: float = 15.0,
    ):
        """Initialize quarantine manager.

        Args:
            default_duration_hours: Default quarantine duration
            auto_release: Automatically release after duration
            isolation_threshold: Risk threshold for auto-quarantine
            max_isolation_time_hours: Maximum quarantine duration
            target_activation_time_s: Target activation time (seconds)
        """
        self.default_duration_hours = default_duration_hours
        self.auto_release = auto_release
        self.isolation_threshold = isolation_threshold
        self.max_isolation_time_hours = max_isolation_time_hours
        self.target_activation_time_s = target_activation_time_s

        # Active quarantines
        self.quarantines: Dict[str, QuarantineRecord] = {}

        # Quarantine history
        self.history: List[QuarantineRecord] = []

        # Cohort-to-agents mapping
        self.cohort_agents: Dict[str, Set[str]] = {}

    def register_agent_cohort(self, agent_id: str, cohort: str):
        """Register agent as part of cohort.

        Args:
            agent_id: Agent identifier
            cohort: Cohort name
        """
        if cohort not in self.cohort_agents:
            self.cohort_agents[cohort] = set()
        self.cohort_agents[cohort].add(agent_id)

    def quarantine_cohort(
        self,
        cohort: str,
        reason: QuarantineReason,
        duration_hours: Optional[float] = None,
        auto_release: Optional[bool] = None,
        policy: Optional[QuarantinePolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuarantineRecord:
        """Quarantine an agent cohort.

        Args:
            cohort: Cohort to quarantine
            reason: Reason for quarantine
            duration_hours: Quarantine duration (hours)
            auto_release: Auto-release after duration
            policy: Custom quarantine policy
            metadata: Additional metadata

        Returns:
            Quarantine record
        """
        start_time = time.time()

        duration = duration_hours or self.default_duration_hours
        auto_rel = auto_release if auto_release is not None else self.auto_release

        # Cap at max duration
        duration = min(duration, self.max_isolation_time_hours)

        # Create quarantine record
        record = QuarantineRecord(
            cohort=cohort,
            reason=reason,
            duration_hours=duration,
            auto_release=auto_rel,
            policy=policy or QuarantinePolicy(),
            metadata=metadata or {},
        )

        # Set expiration
        record.expires_at = datetime.utcnow() + timedelta(hours=duration)

        # Get affected agents
        record.affected_agents = self.cohort_agents.get(cohort, set()).copy()

        # Activate quarantine
        record.status = QuarantineStatus.ACTIVE
        record.activated_at = datetime.utcnow()

        # Store quarantine
        self.quarantines[cohort] = record

        # Calculate activation time
        activation_time_ms = (time.time() - start_time) * 1000
        record.activation_time_ms = activation_time_ms

        # Log to history
        self.history.append(record)

        return record

    def release_cohort(self, cohort: str, reason: str = "manual_release") -> bool:
        """Release cohort from quarantine.

        Args:
            cohort: Cohort to release
            reason: Reason for release

        Returns:
            True if released, False if not quarantined
        """
        if cohort not in self.quarantines:
            return False

        record = self.quarantines[cohort]
        record.status = QuarantineStatus.RELEASED
        record.released_at = datetime.utcnow()
        record.metadata["release_reason"] = reason

        # Remove from active quarantines
        del self.quarantines[cohort]

        return True

    def get_quarantine_status(self, cohort: str) -> Dict[str, Any]:
        """Get quarantine status for cohort.

        Args:
            cohort: Cohort name

        Returns:
            Status dictionary
        """
        record = self.quarantines.get(cohort)

        if not record:
            return {"is_quarantined": False, "cohort": cohort}

        # Check if expired
        if record.expires_at and datetime.utcnow() > record.expires_at:
            if record.auto_release:
                self.release_cohort(cohort, reason="auto_expired")
                return {
                    "is_quarantined": False,
                    "cohort": cohort,
                    "was_quarantined": True,
                    "expired": True,
                }

        return {
            "is_quarantined": True,
            "cohort": cohort,
            "reason": record.reason.value,
            "status": record.status.value,
            "initiated_at": record.initiated_at.isoformat(),
            "activated_at": record.activated_at.isoformat() if record.activated_at else None,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "duration_hours": record.duration_hours,
            "affected_agents": len(record.affected_agents),
            "activation_time_ms": record.activation_time_ms,
            "policy": {
                "allow_read_only": record.policy.allow_read_only,
                "block_external_access": record.policy.block_external_access,
                "require_manual_review": record.policy.require_manual_review,
                "rate_limit_factor": record.policy.rate_limit_factor,
            },
        }

    def is_agent_quarantined(self, agent_id: str) -> bool:
        """Check if agent is quarantined.

        Args:
            agent_id: Agent identifier

        Returns:
            True if quarantined
        """
        # Find cohort for agent
        for cohort, agents in self.cohort_agents.items():
            if agent_id in agents and cohort in self.quarantines:
                # Check if quarantine is active
                status = self.get_quarantine_status(cohort)
                return status.get("is_quarantined", False)

        return False

    def get_quarantine_policy(self, cohort: str) -> Optional[QuarantinePolicy]:
        """Get quarantine policy for cohort.

        Args:
            cohort: Cohort name

        Returns:
            Quarantine policy or None
        """
        record = self.quarantines.get(cohort)
        return record.policy if record else None

    def list_quarantines(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List quarantines.

        Args:
            active_only: Only list active quarantines

        Returns:
            List of quarantine status dictionaries
        """
        if active_only:
            cohorts = list(self.quarantines.keys())
        else:
            # Include history
            cohorts = list(set(list(self.quarantines.keys()) + [r.cohort for r in self.history]))

        return [self.get_quarantine_status(cohort) for cohort in cohorts]

    def simulate_attack_response(
        self, cohort: str, attack_type: str = "synthetic"
    ) -> Dict[str, Any]:
        """Simulate attack and measure quarantine response time.

        Args:
            cohort: Target cohort
            attack_type: Type of attack

        Returns:
            Simulation results
        """
        start_time = time.time()

        # Simulate attack detection
        time.sleep(0.001)  # Minimal detection time

        # Trigger quarantine
        record = self.quarantine_cohort(
            cohort=cohort,
            reason=QuarantineReason.SYNTHETIC_TEST,
            metadata={"attack_type": attack_type},
        )

        total_time_s = time.time() - start_time

        # Check if under target
        meets_requirement = total_time_s < self.target_activation_time_s

        return {
            "cohort": cohort,
            "attack_type": attack_type,
            "total_time_s": total_time_s,
            "activation_time_ms": record.activation_time_ms,
            "target_time_s": self.target_activation_time_s,
            "meets_requirement": meets_requirement,
            "affected_agents": len(record.affected_agents),
            "status": record.status.value,
        }

    def cleanup_expired(self) -> int:
        """Clean up expired quarantines.

        Returns:
            Number of quarantines cleaned up
        """
        expired = []

        for cohort, record in list(self.quarantines.items()):
            if record.expires_at and datetime.utcnow() > record.expires_at:
                if record.auto_release:
                    expired.append(cohort)

        for cohort in expired:
            self.release_cohort(cohort, reason="auto_expired")

        return len(expired)

    def get_statistics(self) -> Dict[str, Any]:
        """Get quarantine system statistics.

        Returns:
            Statistics dictionary
        """
        active_count = len(self.quarantines)
        total_count = len(self.history)

        # Calculate average activation time
        activation_times = [
            r.activation_time_ms for r in self.history if r.activation_time_ms is not None
        ]
        avg_activation_ms = sum(activation_times) / len(activation_times) if activation_times else 0

        return {
            "active_quarantines": active_count,
            "total_quarantines": total_count,
            "avg_activation_time_ms": avg_activation_ms,
            "target_activation_time_s": self.target_activation_time_s,
            "meets_target": avg_activation_ms < (self.target_activation_time_s * 1000),
            "total_cohorts": len(self.cohort_agents),
            "isolation_threshold": self.isolation_threshold,
        }
