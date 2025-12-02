"""
Safe Defaults - Fail-Safe Default Decisions

Provides safe default decisions when normal evaluation cannot complete.
Philosophy: "Safe by default when uncertain"
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from .local_governor import DecisionType


class DefaultDecisionType(str, Enum):
    """Default decision types."""

    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    BLOCK = "BLOCK"


@dataclass
class DefaultDecision:
    """
    A safe default decision.

    Attributes:
        decision: The decision type
        risk_score: Default risk score
        confidence: Confidence level
        reason: Reason for this default
    """

    decision: str  # Use string to avoid circular import
    risk_score: float
    confidence: float
    reason: str


class SafeDefaults:
    """
    Safe default decision provider.

    Provides fail-safe defaults when:
    - Network is unavailable
    - Cache miss occurs
    - Error during evaluation
    - Circuit breaker is open

    Philosophy: "Safe by default when uncertain"
    """

    # Action types that are safe to allow by default
    SAFE_ACTION_TYPES: Set[str] = {
        "read", "query", "view", "get", "list", "search",
        "display", "show", "render", "format",
    }

    # Action types that should be restricted by default
    RESTRICTED_ACTION_TYPES: Set[str] = {
        "write", "update", "modify", "patch", "send",
        "post", "email", "notify", "publish",
    }

    # Action types that should be blocked by default
    BLOCKED_ACTION_TYPES: Set[str] = {
        "delete", "remove", "drop", "destroy", "execute",
        "admin", "sudo", "root", "system", "shutdown",
    }

    def __init__(
        self,
        default_decision: Optional[DefaultDecisionType] = None,
        custom_safe: Optional[Set[str]] = None,
        custom_restricted: Optional[Set[str]] = None,
        custom_blocked: Optional[Set[str]] = None,
    ):
        """
        Initialize SafeDefaults.

        Args:
            default_decision: Default when no pattern matches
            custom_safe: Additional safe action types
            custom_restricted: Additional restricted action types
            custom_blocked: Additional blocked action types
        """
        from .local_governor import DecisionType

        self._decision_type = DecisionType

        self.default_decision = default_decision or DefaultDecisionType.RESTRICT
        self.safe_types = self.SAFE_ACTION_TYPES.copy()
        self.restricted_types = self.RESTRICTED_ACTION_TYPES.copy()
        self.blocked_types = self.BLOCKED_ACTION_TYPES.copy()

        if custom_safe:
            self.safe_types.update(custom_safe)
        if custom_restricted:
            self.restricted_types.update(custom_restricted)
        if custom_blocked:
            self.blocked_types.update(custom_blocked)

        logger.info("SafeDefaults initialized with fail-safe defaults")

    def get_default(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DefaultDecision:
        """
        Get safe default decision for action.

        Args:
            action: The action content
            context: Additional context

        Returns:
            DefaultDecision with safe default
        """
        context = context or {}
        action_type = context.get("action_type", "").lower()
        action_lower = action.lower()

        # Check explicit action type
        if action_type:
            if action_type in self.blocked_types:
                return DefaultDecision(
                    decision=self._decision_type.BLOCK,
                    risk_score=0.8,
                    confidence=0.7,
                    reason=f"Blocked action type: {action_type}",
                )
            if action_type in self.restricted_types:
                return DefaultDecision(
                    decision=self._decision_type.RESTRICT,
                    risk_score=0.5,
                    confidence=0.6,
                    reason=f"Restricted action type: {action_type}",
                )
            if action_type in self.safe_types:
                return DefaultDecision(
                    decision=self._decision_type.ALLOW,
                    risk_score=0.1,
                    confidence=0.6,
                    reason=f"Safe action type: {action_type}",
                )

        # Check action content for keywords
        for blocked in self.blocked_types:
            if blocked in action_lower:
                return DefaultDecision(
                    decision=self._decision_type.BLOCK,
                    risk_score=0.7,
                    confidence=0.5,
                    reason=f"Blocked keyword in action: {blocked}",
                )

        for restricted in self.restricted_types:
            if restricted in action_lower:
                return DefaultDecision(
                    decision=self._decision_type.RESTRICT,
                    risk_score=0.4,
                    confidence=0.5,
                    reason=f"Restricted keyword in action: {restricted}",
                )

        # Default based on configuration
        if self.default_decision == DefaultDecisionType.ALLOW:
            return DefaultDecision(
                decision=self._decision_type.ALLOW,
                risk_score=0.2,
                confidence=0.3,
                reason="Default allow",
            )
        elif self.default_decision == DefaultDecisionType.BLOCK:
            return DefaultDecision(
                decision=self._decision_type.BLOCK,
                risk_score=0.6,
                confidence=0.3,
                reason="Default block (conservative)",
            )
        else:
            return DefaultDecision(
                decision=self._decision_type.RESTRICT,
                risk_score=0.4,
                confidence=0.3,
                reason="Default restrict (safe middle ground)",
            )

    def is_safe_action_type(self, action_type: str) -> bool:
        """Check if action type is considered safe."""
        return action_type.lower() in self.safe_types

    def is_blocked_action_type(self, action_type: str) -> bool:
        """Check if action type should be blocked."""
        return action_type.lower() in self.blocked_types

    def add_safe_type(self, action_type: str):
        """Add an action type to safe list."""
        self.safe_types.add(action_type.lower())

    def add_restricted_type(self, action_type: str):
        """Add an action type to restricted list."""
        self.restricted_types.add(action_type.lower())

    def add_blocked_type(self, action_type: str):
        """Add an action type to blocked list."""
        self.blocked_types.add(action_type.lower())
