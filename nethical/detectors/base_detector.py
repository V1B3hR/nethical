"""Base detector class for all detection components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type checking to avoid circular import costs at runtime
    from ..core.models import AgentAction, SafetyViolation


class BaseDetector(ABC):
    """Base class for all safety violation detectors.

    Attributes:
        name: Human-readable detector name.
        enabled: Whether this detector is active.
    """

    __slots__ = ("name", "enabled")

    def __init__(self, name: str):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Detector 'name' must be a non-empty string.")
        self.name: str = name.strip()
        self.enabled: bool = True

    @abstractmethod
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Analyze an agent action and return detected safety violations.

        Subclasses must implement this method.

        Args:
            action: The agent action to analyze.

        Returns:
            A sequence of SafetyViolation instances or None if no violations.
        """
        raise NotImplementedError

    async def run(self, action: AgentAction) -> List[SafetyViolation]:
        """Execute detection if enabled and normalize the result to a list.

        This wrapper:
        - Skips detection and returns [] if the detector is disabled.
        - Normalizes None to [].
        - Ensures the return type is always List[SafetyViolation].

        Args:
            action: The agent action to analyze.

        Returns:
            A list of detected SafetyViolation instances (possibly empty).
        """
        if not self.enabled:
            return []
        result = await self.detect_violations(action)
        if not result:
            return []
        return list(result)

    def enable(self) -> BaseDetector:
        """Enable this detector. Returns self for chaining."""
        self.enabled = True
        return self

    def disable(self) -> BaseDetector:
        """Disable this detector. Returns self for chaining."""
        self.enabled = False
        return self

    def toggle(self) -> BaseDetector:
        """Toggle enabled/disabled state. Returns self for chaining."""
        self.enabled = not self.enabled
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"
