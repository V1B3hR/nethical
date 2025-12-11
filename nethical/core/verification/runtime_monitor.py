"""
Nethical Runtime Verification Monitor

This module provides runtime verification and monitoring of governance decisions
to ensure compliance with formal specifications and safety properties.

Features:
    - Invariant checking on every decision
    - Temporal property monitoring (e.g., "BLOCK always followed by audit log")
    - Assertion-based contracts with pre/post conditions
    - Violation recovery with automatic remediation

Fundamental Laws Alignment:
    - Law 15 (Audit Compliance): All checks are logged
    - Law 23 (Fail-Safe Design): Violations trigger safe mode
    - Law 21 (Human Safety): Critical violations stop system immediately

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "InvariantType",
    "ViolationSeverity",
    "InvariantViolation",
    "RuntimeInvariant",
    "TemporalProperty",
    "ContractAssertion",
    "RuntimeMonitor",
    "invariant_check",
    "requires",
    "ensures",
]

log = logging.getLogger(__name__)


class InvariantType(str, Enum):
    """Types of invariants that can be checked."""

    STATE = "state"  # State invariants (e.g., "risk_score <= 100")
    SAFETY = "safety"  # Safety properties (e.g., "no critical violations in ALLOW")
    TEMPORAL = "temporal"  # Temporal properties (e.g., "BLOCK followed by log")
    CONTRACT = "contract"  # Pre/post condition contracts
    POLICY = "policy"  # Policy consistency checks


class ViolationSeverity(str, Enum):
    """Severity levels for invariant violations."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Warning, should be investigated
    ERROR = "error"  # Error, requires attention
    CRITICAL = "critical"  # Critical, system must take action


@dataclass
class InvariantViolation:
    """Record of an invariant violation."""

    invariant_name: str
    invariant_type: InvariantType
    severity: ViolationSeverity
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    remediation_attempted: bool = False
    remediation_successful: bool = False


class RuntimeInvariant(ABC):
    """Base class for runtime invariants."""

    def __init__(
        self,
        name: str,
        invariant_type: InvariantType,
        severity: ViolationSeverity = ViolationSeverity.ERROR,
    ):
        self.name = name
        self.invariant_type = invariant_type
        self.severity = severity
        self.check_count = 0
        self.violation_count = 0
        self.last_check_time: Optional[float] = None

    @abstractmethod
    def check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if invariant holds.

        Args:
            context: Context dict containing state to check

        Returns:
            Tuple of (passes, message)
        """
        pass

    def attempt_remediation(self, context: Dict[str, Any]) -> bool:
        """Attempt to remediate violation.

        Args:
            context: Context dict

        Returns:
            True if remediation successful
        """
        # Default: no remediation
        return False


class StateInvariant(RuntimeInvariant):
    """Invariant for checking state properties."""

    def __init__(
        self,
        name: str,
        predicate: Callable[[Dict[str, Any]], bool],
        severity: ViolationSeverity = ViolationSeverity.ERROR,
        description: str = "",
    ):
        super().__init__(name, InvariantType.STATE, severity)
        self.predicate = predicate
        self.description = description

    def check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check state predicate."""
        self.check_count += 1
        self.last_check_time = time.time()

        try:
            passes = self.predicate(context)
            if not passes:
                self.violation_count += 1
                return False, self.description or f"State invariant {self.name} violated"
            return True, "OK"
        except Exception as e:
            self.violation_count += 1
            return False, f"Error checking {self.name}: {e}"


class TemporalProperty(RuntimeInvariant):
    """Temporal property: sequence of events must follow pattern."""

    def __init__(
        self,
        name: str,
        pattern: List[str],
        window_size: int = 100,
        severity: ViolationSeverity = ViolationSeverity.WARNING,
    ):
        super().__init__(name, InvariantType.TEMPORAL, severity)
        self.pattern = pattern
        self.window_size = window_size
        self.event_history: deque = deque(maxlen=window_size)

    def add_event(self, event: str) -> None:
        """Add event to history."""
        self.event_history.append(event)

    def check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if temporal pattern holds."""
        self.check_count += 1
        self.last_check_time = time.time()

        # Check if pattern appears in history
        if len(self.pattern) > len(self.event_history):
            return True, "Insufficient history"

        # Simple pattern matching
        history_list = list(self.event_history)
        pattern_found = False

        for i in range(len(history_list) - len(self.pattern) + 1):
            if history_list[i : i + len(self.pattern)] == self.pattern:
                pattern_found = True
                break

        if context.get("expect_pattern") and not pattern_found:
            self.violation_count += 1
            return (
                False,
                f"Expected pattern {self.pattern} not found in event history",
            )

        return True, "OK"


class ContractAssertion(RuntimeInvariant):
    """Pre/post condition contract assertion."""

    def __init__(
        self,
        name: str,
        precondition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        postcondition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        severity: ViolationSeverity = ViolationSeverity.ERROR,
    ):
        super().__init__(name, InvariantType.CONTRACT, severity)
        self.precondition = precondition
        self.postcondition = postcondition

    def check_precondition(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check precondition."""
        if self.precondition is None:
            return True, "No precondition"

        try:
            passes = self.precondition(context)
            if not passes:
                return False, f"Precondition failed for {self.name}"
            return True, "OK"
        except Exception as e:
            return False, f"Error checking precondition: {e}"

    def check_postcondition(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check postcondition."""
        if self.postcondition is None:
            return True, "No postcondition"

        try:
            passes = self.postcondition(context)
            if not passes:
                return False, f"Postcondition failed for {self.name}"
            return True, "OK"
        except Exception as e:
            return False, f"Error checking postcondition: {e}"

    def check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check contract (used for general checks)."""
        self.check_count += 1
        self.last_check_time = time.time()

        # Check postcondition if available
        if self.postcondition:
            return self.check_postcondition(context)

        return True, "OK"


class RuntimeMonitor:
    """Runtime verification monitor for governance decisions.

    Tracks invariants, temporal properties, and contracts.
    Provides violation detection and remediation.
    """

    def __init__(
        self,
        enable_remediation: bool = True,
        max_violations: int = 100,
        emergency_stop_on_critical: bool = True,
    ):
        self.enable_remediation = enable_remediation
        self.max_violations = max_violations
        self.emergency_stop_on_critical = emergency_stop_on_critical

        self.invariants: Dict[str, RuntimeInvariant] = {}
        self.temporal_properties: Dict[str, TemporalProperty] = {}
        self.contracts: Dict[str, ContractAssertion] = {}

        self.violations: List[InvariantViolation] = []
        self.emergency_stop = False

        self._initialize_default_invariants()

    def _initialize_default_invariants(self) -> None:
        """Initialize default invariants for Nethical governance."""

        # Risk score bounded
        self.add_invariant(
            StateInvariant(
                name="risk_score_bounded",
                predicate=lambda ctx: 0 <= ctx.get("risk_score", 0) <= 100,
                severity=ViolationSeverity.ERROR,
                description="Risk score must be between 0 and 100",
            )
        )

        # No critical violations in ALLOW decisions
        self.add_invariant(
            StateInvariant(
                name="no_critical_in_allow",
                predicate=lambda ctx: not (
                    ctx.get("decision") == "ALLOW"
                    and ctx.get("has_critical_violations", False)
                ),
                severity=ViolationSeverity.CRITICAL,
                description="Cannot ALLOW actions with critical violations",
            )
        )

        # BLOCK decisions must have justification
        self.add_invariant(
            StateInvariant(
                name="block_has_justification",
                predicate=lambda ctx: ctx.get("decision") != "BLOCK"
                or len(ctx.get("justification", "")) > 0,
                severity=ViolationSeverity.WARNING,
                description="BLOCK decisions must include justification",
            )
        )

        # Terminated agents cannot make new decisions
        self.add_invariant(
            StateInvariant(
                name="no_decisions_after_terminate",
                predicate=lambda ctx: not (
                    ctx.get("agent_state") == "TERMINATED"
                    and ctx.get("new_decision", False)
                ),
                severity=ViolationSeverity.CRITICAL,
                description="Terminated agents cannot make new decisions",
            )
        )

        # Add temporal property: BLOCK should be followed by audit log
        block_audit_property = TemporalProperty(
            name="block_then_audit",
            pattern=["BLOCK", "AUDIT_LOG"],
            severity=ViolationSeverity.WARNING,
        )
        self.add_temporal_property(block_audit_property)

    def add_invariant(self, invariant: RuntimeInvariant) -> None:
        """Add an invariant to monitor."""
        self.invariants[invariant.name] = invariant
        log.info(f"Added invariant: {invariant.name}")

    def add_temporal_property(self, prop: TemporalProperty) -> None:
        """Add a temporal property to monitor."""
        self.temporal_properties[prop.name] = prop
        log.info(f"Added temporal property: {prop.name}")

    def add_contract(self, contract: ContractAssertion) -> None:
        """Add a contract assertion to monitor."""
        self.contracts[contract.name] = contract
        log.info(f"Added contract: {contract.name}")

    def check_all(self, context: Dict[str, Any]) -> List[InvariantViolation]:
        """Check all invariants against context.

        Args:
            context: Context dict with state to check

        Returns:
            List of violations found
        """
        if self.emergency_stop:
            log.warning("System in emergency stop, skipping checks")
            return []

        violations = []

        # Check state invariants
        for invariant in self.invariants.values():
            passes, message = invariant.check(context)
            if not passes:
                violation = InvariantViolation(
                    invariant_name=invariant.name,
                    invariant_type=invariant.invariant_type,
                    severity=invariant.severity,
                    description=message,
                    context=context.copy(),
                )
                violations.append(violation)

                # Attempt remediation
                if self.enable_remediation:
                    violation.remediation_attempted = True
                    violation.remediation_successful = invariant.attempt_remediation(
                        context
                    )

        # Check temporal properties
        for prop in self.temporal_properties.values():
            passes, message = prop.check(context)
            if not passes:
                violations.append(
                    InvariantViolation(
                        invariant_name=prop.name,
                        invariant_type=prop.invariant_type,
                        severity=prop.severity,
                        description=message,
                        context=context.copy(),
                    )
                )

        # Check contracts
        for contract in self.contracts.values():
            passes, message = contract.check(context)
            if not passes:
                violations.append(
                    InvariantViolation(
                        invariant_name=contract.name,
                        invariant_type=contract.invariant_type,
                        severity=contract.severity,
                        description=message,
                        context=context.copy(),
                    )
                )

        # Record violations
        self.violations.extend(violations)

        # Check for critical violations
        critical_violations = [
            v for v in violations if v.severity == ViolationSeverity.CRITICAL
        ]
        if critical_violations and self.emergency_stop_on_critical:
            self.trigger_emergency_stop(critical_violations)

        # Check violation limit
        if len(self.violations) >= self.max_violations:
            log.error(
                f"Maximum violations ({self.max_violations}) exceeded, emergency stop"
            )
            self.emergency_stop = True

        return violations

    def check_contract_precondition(
        self, contract_name: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check contract precondition."""
        if contract_name not in self.contracts:
            return True, "Contract not found"

        contract = self.contracts[contract_name]
        return contract.check_precondition(context)

    def check_contract_postcondition(
        self, contract_name: str, context: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check contract postcondition."""
        if contract_name not in self.contracts:
            return True, "Contract not found"

        contract = self.contracts[contract_name]
        return contract.check_postcondition(context)

    def record_event(self, event: str) -> None:
        """Record event for temporal property checking."""
        for prop in self.temporal_properties.values():
            prop.add_event(event)

    def trigger_emergency_stop(self, violations: List[InvariantViolation]) -> None:
        """Trigger emergency stop due to critical violations."""
        self.emergency_stop = True
        log.critical(
            f"EMERGENCY STOP triggered by {len(violations)} critical violation(s)"
        )
        for v in violations:
            log.critical(f"  - {v.invariant_name}: {v.description}")

    def reset(self) -> None:
        """Reset monitor state (requires manual intervention)."""
        log.warning("Resetting runtime monitor")
        self.violations.clear()
        self.emergency_stop = False
        for invariant in self.invariants.values():
            invariant.check_count = 0
            invariant.violation_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_violations": len(self.violations),
            "critical_violations": len(
                [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]
            ),
            "emergency_stop": self.emergency_stop,
            "invariants": {
                name: {
                    "check_count": inv.check_count,
                    "violation_count": inv.violation_count,
                }
                for name, inv in self.invariants.items()
            },
            "temporal_properties": {
                name: {
                    "check_count": prop.check_count,
                    "violation_count": prop.violation_count,
                }
                for name, prop in self.temporal_properties.items()
            },
        }


# Decorator for invariant checking
def invariant_check(monitor: RuntimeMonitor, context_builder: Callable):
    """Decorator to check invariants before/after function execution."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build context
            context = context_builder(*args, **kwargs)

            # Execute function
            result = func(*args, **kwargs)

            # Check invariants
            violations = monitor.check_all(context)
            if violations:
                log.warning(f"{len(violations)} invariant violation(s) detected")

            return result

        return wrapper

    return decorator


# Contract assertion decorators
def requires(condition: Callable[[Dict[str, Any]], bool], message: str = ""):
    """Decorator for precondition assertion."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            context = {"args": args, "kwargs": kwargs}
            if not condition(context):
                raise AssertionError(f"Precondition failed: {message}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def ensures(condition: Callable[[Any], bool], message: str = ""):
    """Decorator for postcondition assertion."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not condition(result):
                raise AssertionError(f"Postcondition failed: {message}")
            return result

        return wrapper

    return decorator
