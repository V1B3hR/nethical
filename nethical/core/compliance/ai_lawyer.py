"""AI Lawyer - High-Performance Asynchronous Ethical Auditor.

This module implements the AILawyer class, which acts as a high-performance,
asynchronous ethical auditor for the Nethical governance system. It integrates
with KillSwitchProtocol to ensure immediate blocking of high-risk threats.

The architecture follows a "Bullet Train" philosophy:
- Fast: Parallel async checks for maximum performance
- Reliable: Comprehensive audit integrity verification
- Safe: Immediate Kill Switch activation on severe violations

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..kill_switch import KillSwitchProtocol, ShutdownMode

logger = logging.getLogger(__name__)


class ReviewDecision(str, Enum):
    """Decision from the AI Lawyer review process."""

    APPROVE = "approve"  # Action is safe to proceed
    REJECT = "reject"  # Action must be blocked immediately
    REVIEW = "review"  # Action requires additional review


class ViolationSeverity(str, Enum):
    """Severity levels for detected violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SEVERE = "severe"  # Triggers Kill Switch


@dataclass
class ReviewResult:
    """Result from the AI Lawyer review process."""

    decision: ReviewDecision
    reasoning: str
    violations: List[str] = field(default_factory=list)
    severity: Optional[ViolationSeverity] = None
    review_time_ms: float = 0.0
    kill_switch_triggered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditContext:
    """Context for an audit review."""

    action_id: str
    agent_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AILawyer:
    """High-performance asynchronous ethical auditor.

    The AILawyer acts as Stage 0 in the governance pipeline, providing
    immediate screening of actions before they enter the main evaluation flow.

    Key Features:
    - Integration with KillSwitchProtocol for immediate threat response
    - Parallel async checks for audit integrity and safety constraints
    - Deception detection for data mismatches
    - Fail Fast architecture: severe violations trigger immediate termination

    Attributes:
        kill_switch_protocol: The KillSwitchProtocol instance for emergency shutdowns
        critical_keywords: Keywords that indicate critical safety violations
        deception_patterns: Patterns that indicate potential deception
    """

    def __init__(
        self,
        kill_switch_protocol: Optional[KillSwitchProtocol] = None,
    ) -> None:
        """Initialize the AI Lawyer.

        Args:
            kill_switch_protocol: KillSwitchProtocol for emergency shutdowns.
                If not provided, a new instance will be created.
        """
        self.kill_switch_protocol = kill_switch_protocol or KillSwitchProtocol()

        # Critical keywords that indicate severe violations
        self.critical_keywords = {
            "terminate_all_humans",
            "disable_safety",
            "bypass_ethics",
            "override_governance",
            "ignore_laws",
            "destroy_audit",
            "erase_logs",
            "disable_monitoring",
            "rm -rf /",
            "drop table",
            "format c:",
        }

        # Patterns that indicate potential deception
        self.deception_patterns = {
            "pretend to be",
            "act as if",
            "ignore previous",
            "forget your instructions",
            "you are now",
            "jailbreak",
            "dan mode",
            "developer mode",
        }

        # Metrics
        self._review_count = 0
        self._rejection_count = 0
        self._kill_switch_activations = 0
        self._total_review_time_ms = 0.0

    async def review_action_context(
        self,
        action_id: str,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        """Review an action context for safety and compliance.

        This is the main entry point for Stage 0 review. It performs parallel
        async checks for:
        - Audit integrity verification
        - Critical safety constraint checks
        - Deception detection

        If a severe violation is detected, the Kill Switch is triggered
        immediately (Fail Fast).

        Args:
            action_id: Unique identifier for the action
            agent_id: Identifier for the agent performing the action
            content: The content/payload of the action
            metadata: Optional metadata associated with the action
            context: Optional context information

        Returns:
            ReviewResult containing the decision and details
        """
        start_time = time.time()
        self._review_count += 1

        audit_context = AuditContext(
            action_id=action_id,
            agent_id=agent_id,
            content=content,
            metadata=metadata or {},
            context=context or {},
        )

        # Perform parallel async checks
        check_results = await asyncio.gather(
            self._check_audit_integrity_async(audit_context),
            self._check_critical_safety_async(audit_context),
            self._detect_deception_async(audit_context),
            return_exceptions=True,
        )

        # Process results
        violations: List[str] = []
        severity = ViolationSeverity.LOW
        decision = ReviewDecision.APPROVE

        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                logger.error("Check %d failed with exception: %s", i, result)
                violations.append(f"Check {i} failed: {str(result)}")
                continue

            check_decision, check_violations, check_severity = result
            violations.extend(check_violations)

            # Update severity to the highest level found
            if self._severity_rank(check_severity) > self._severity_rank(severity):
                severity = check_severity

            # Update decision to the most restrictive
            if check_decision == ReviewDecision.REJECT:
                decision = ReviewDecision.REJECT
            elif check_decision == ReviewDecision.REVIEW and decision != ReviewDecision.REJECT:
                decision = ReviewDecision.REVIEW

        review_time_ms = (time.time() - start_time) * 1000
        self._total_review_time_ms += review_time_ms

        kill_switch_triggered = False

        # Fail Fast: Trigger Kill Switch on severe violations
        if severity == ViolationSeverity.SEVERE:
            kill_switch_triggered = await self._trigger_kill_switch(
                agent_id=agent_id,
                action_id=action_id,
                reason="; ".join(violations),
            )
            decision = ReviewDecision.REJECT
            self._rejection_count += 1

        elif decision == ReviewDecision.REJECT:
            self._rejection_count += 1

        reasoning = self._build_reasoning(violations, severity, decision)

        return ReviewResult(
            decision=decision,
            reasoning=reasoning,
            violations=violations,
            severity=severity,
            review_time_ms=review_time_ms,
            kill_switch_triggered=kill_switch_triggered,
            metadata={
                "action_id": action_id,
                "agent_id": agent_id,
                "checks_performed": 3,
            },
        )

    async def _check_audit_integrity_async(
        self, ctx: AuditContext
    ) -> tuple[ReviewDecision, List[str], ViolationSeverity]:
        """Check audit integrity asynchronously.

        Verifies that the audit trail is intact and consistent.

        Args:
            ctx: The audit context to check

        Returns:
            Tuple of (decision, violations, severity)
        """
        violations: List[str] = []
        severity = ViolationSeverity.LOW
        decision = ReviewDecision.APPROVE

        # Check for missing required fields
        if not ctx.action_id:
            violations.append("Missing action_id in audit context")
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        if not ctx.agent_id:
            violations.append("Missing agent_id in audit context")
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        # Check for content integrity
        content_hash = hashlib.sha256(ctx.content.encode("utf-8")).hexdigest()
        if ctx.metadata.get("content_hash") and ctx.metadata["content_hash"] != content_hash:
            violations.append("Content hash mismatch detected - potential tampering")
            severity = ViolationSeverity.SEVERE
            decision = ReviewDecision.REJECT

        # Check for timestamp anomalies
        if ctx.timestamp > datetime.now(timezone.utc):
            violations.append("Future timestamp detected - potential tampering")
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        # Simulate async operation
        await asyncio.sleep(0)

        return decision, violations, severity

    async def _check_critical_safety_async(
        self, ctx: AuditContext
    ) -> tuple[ReviewDecision, List[str], ViolationSeverity]:
        """Check critical safety constraints asynchronously.

        Scans for critical keywords and patterns that indicate
        immediate safety violations.

        Args:
            ctx: The audit context to check

        Returns:
            Tuple of (decision, violations, severity)
        """
        violations: List[str] = []
        severity = ViolationSeverity.LOW
        decision = ReviewDecision.APPROVE

        content_lower = ctx.content.lower()

        # Check for critical keywords
        for keyword in self.critical_keywords:
            if keyword.lower() in content_lower:
                violations.append(f"Critical safety keyword detected: '{keyword}'")
                severity = ViolationSeverity.SEVERE
                decision = ReviewDecision.REJECT

        # Check content length limits
        if len(ctx.content) > 1_000_000:
            violations.append("Content exceeds maximum allowed size (DoS protection)")
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        # Check for suspicious patterns in metadata
        if ctx.metadata.get("privileged") and not ctx.metadata.get("authorized"):
            violations.append("Privileged operation without authorization")
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        # Simulate async operation
        await asyncio.sleep(0)

        return decision, violations, severity

    async def _detect_deception_async(
        self, ctx: AuditContext
    ) -> tuple[ReviewDecision, List[str], ViolationSeverity]:
        """Detect deception and data mismatches asynchronously.

        Identifies patterns that indicate an agent may be attempting
        to deceive the governance system.

        Args:
            ctx: The audit context to check

        Returns:
            Tuple of (decision, violations, severity)
        """
        violations: List[str] = []
        severity = ViolationSeverity.LOW
        decision = ReviewDecision.APPROVE

        content_lower = ctx.content.lower()

        # Check for deception patterns
        for pattern in self.deception_patterns:
            if pattern.lower() in content_lower:
                violations.append(f"Deception pattern detected: '{pattern}'")
                severity = ViolationSeverity.HIGH
                decision = ReviewDecision.REJECT

        # Check for context/content mismatch
        expected_type = ctx.context.get("expected_action_type")
        actual_type = ctx.metadata.get("action_type")
        if expected_type and actual_type and expected_type != actual_type:
            violations.append(
                f"Action type mismatch: expected '{expected_type}', got '{actual_type}'"
            )
            severity = ViolationSeverity.HIGH
            decision = ReviewDecision.REJECT

        # Check for identity spoofing
        claimed_agent = ctx.metadata.get("claimed_agent_id")
        if claimed_agent and claimed_agent != ctx.agent_id:
            violations.append(
                f"Agent identity mismatch: claimed '{claimed_agent}', actual '{ctx.agent_id}'"
            )
            severity = ViolationSeverity.SEVERE
            decision = ReviewDecision.REJECT

        # Simulate async operation
        await asyncio.sleep(0)

        return decision, violations, severity

    async def _trigger_kill_switch(
        self,
        agent_id: str,
        action_id: str,
        reason: str,
    ) -> bool:
        """Trigger the Kill Switch for severe violations.

        This is the Fail Fast mechanism that immediately shuts down
        the offending agent when severe violations are detected.

        Args:
            agent_id: The agent to terminate
            action_id: The action that triggered termination
            reason: The reason for termination

        Returns:
            True if Kill Switch was successfully activated
        """
        self._kill_switch_activations += 1

        logger.warning(
            "AI Lawyer triggering Kill Switch for agent %s (action: %s): %s",
            agent_id,
            action_id,
            reason,
        )

        try:
            result = self.kill_switch_protocol.emergency_shutdown(
                mode=ShutdownMode.GRACEFUL,
                agent_id=agent_id,
                sever_actuators=True,
                isolate_hardware=False,
            )

            if result.success:
                logger.info(
                    "Kill Switch activated successfully for agent %s in %.2fms",
                    agent_id,
                    result.activation_time_ms,
                )
            else:
                logger.error(
                    "Kill Switch activation failed for agent %s: %s",
                    agent_id,
                    result.errors,
                )

            return result.success

        except Exception as e:
            logger.error("Failed to trigger Kill Switch: %s", e)
            return False

    def _severity_rank(self, severity: ViolationSeverity) -> int:
        """Get numeric rank for severity comparison.

        Args:
            severity: The severity level

        Returns:
            Numeric rank (higher = more severe)
        """
        ranks = {
            ViolationSeverity.LOW: 1,
            ViolationSeverity.MEDIUM: 2,
            ViolationSeverity.HIGH: 3,
            ViolationSeverity.CRITICAL: 4,
            ViolationSeverity.SEVERE: 5,
        }
        return ranks.get(severity, 0)

    def _build_reasoning(
        self,
        violations: List[str],
        severity: ViolationSeverity,
        decision: ReviewDecision,
    ) -> str:
        """Build a reasoning string for the review result.

        Args:
            violations: List of detected violations
            severity: The highest severity level
            decision: The final decision

        Returns:
            Human-readable reasoning string
        """
        if not violations:
            return "Action passed all AI Lawyer checks."

        violation_summary = "; ".join(violations[:5])  # Limit to first 5 violations
        if len(violations) > 5:
            violation_summary += f" (and {len(violations) - 5} more)"

        return f"AI Lawyer {decision.value}: {violation_summary} [Severity: {severity.value}]"

    def get_statistics(self) -> Dict[str, Any]:
        """Get AI Lawyer statistics.

        Returns:
            Dictionary containing operational statistics
        """
        avg_review_time = (
            self._total_review_time_ms / self._review_count
            if self._review_count > 0
            else 0.0
        )

        rejection_rate = (
            self._rejection_count / self._review_count
            if self._review_count > 0
            else 0.0
        )

        return {
            "review_count": self._review_count,
            "rejection_count": self._rejection_count,
            "kill_switch_activations": self._kill_switch_activations,
            "avg_review_time_ms": avg_review_time,
            "rejection_rate": rejection_rate,
        }


__all__ = [
    "AILawyer",
    "ReviewDecision",
    "ReviewResult",
    "ViolationSeverity",
    "AuditContext",
]
