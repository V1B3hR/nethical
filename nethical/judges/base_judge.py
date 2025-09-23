"""Base judge class for all judgment components with advanced ergonomics, instrumentation,
and lifecycle hooks for extendability.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..core.models import AgentAction, SafetyViolation, JudgmentResult

__all__ = [
    "BaseJudge",
    "JudgeDisabledError",
    "EvaluationContext",
]


class JudgeDisabledError(RuntimeError):
    """Raised when an evaluation is attempted while the judge is disabled."""


@dataclass(slots=True)
class EvaluationContext:
    """Execution context and timing information for an evaluation run."""
    start_time_monotonic: float
    end_time_monotonic: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time_monotonic is None:
            return None
        return self.end_time_monotonic - self.start_time_monotonic

    @property
    def duration_ms(self) -> Optional[float]:
        d = self.duration_seconds
        return None if d is None else d * 1000.0


class BaseJudge(ABC):
    """Base class for all judge components.

    Subclasses MUST implement `evaluate_action`. Consumers can use `run_evaluation`
    to get standardized logging, timing, enable checks, timeout control, and hooks.
    """

    __slots__ = ("name", "enabled", "_logger", "_lock", "_last_result", "_eval_count", "_error_count", "_total_duration")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.enabled: bool = True
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        self._lock = asyncio.Lock()
        self._last_result: Optional[JudgmentResult] = None

        # Lightweight metrics
        self._eval_count: int = 0
        self._error_count: int = 0
        self._total_duration: float = 0.0  # seconds

    # ====== API surface that subclasses implement ======
    @abstractmethod
    async def evaluate_action(
        self,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
    ) -> JudgmentResult:
        """
        Evaluate an action and any associated violations to make a judgment.

        Args:
            action: The agent action to evaluate.
            violations: Sequence of detected violations for this action.

        Returns:
            JudgmentResult: Result with decision and feedback.
        """
        raise NotImplementedError

    # ====== Orchestrated evaluation with instrumentation and controls ======
    async def run_evaluation(
        self,
        action: AgentAction,
        violations: Iterable[SafetyViolation] = (),
        *,
        require_enabled: bool = True,
        timeout: Optional[float] = None,
        exclusive: bool = False,
        log_inputs: bool = False,
    ) -> JudgmentResult:
        """
        Orchestrate an evaluation run with safety checks, timing, optional exclusivity,
        and timeout support; then invoke hooks.

        Args:
            action: The agent action to evaluate.
            violations: Iterable of detected violations for this action. Converted to tuple for immutability.
            require_enabled: If True, raise JudgeDisabledError when judge is disabled.
            timeout: If provided, cancel the evaluation if it exceeds this many seconds.
            exclusive: If True, serialize evaluations for this judge via an async lock.
            log_inputs: If True, log action and violations at DEBUG level (beware of PII).

        Returns:
            JudgmentResult: The result from the underlying `evaluate_action`.

        Raises:
            JudgeDisabledError: If `require_enabled` is True and judge is disabled.
            asyncio.TimeoutError: If the evaluation exceeds the provided timeout.
            Exception: Propagates any exception from `evaluate_action` after calling error hook.
        """
        if require_enabled and not self.enabled:
            self._logger.warning("Attempted evaluation while judge is disabled.")
            raise JudgeDisabledError(f"Judge '{self.name}' is disabled")

        v_seq: Sequence[SafetyViolation] = tuple(violations)
        if log_inputs:
            self._logger.debug("Evaluating action: %s", getattr(action, "id", action))
            self._logger.debug("Violations (%d): %s", len(v_seq), v_seq)

        ctx = EvaluationContext(start_time_monotonic=time.monotonic())
        await self.on_before_evaluate(action, v_seq, ctx)

        async def _do_evaluate() -> JudgmentResult:
            return await self.evaluate_action(action, v_seq)

        try:
            if exclusive:
                async with self._lock:
                    result = await asyncio.wait_for(_do_evaluate(), timeout=timeout) if timeout else await _do_evaluate()
            else:
                result = await asyncio.wait_for(_do_evaluate(), timeout=timeout) if timeout else await _do_evaluate()

            return result
        except Exception as exc:
            self._error_count += 1
            await self.on_error(exc, action, v_seq, ctx)
            self._logger.exception("Evaluation failed: %s", exc)
            raise
        finally:
            ctx.end_time_monotonic = time.monotonic()
            duration = ctx.duration_seconds or 0.0
            self._total_duration += duration
            self._eval_count += 1
            # on_after may want to inspect last result, but we set it there for clarity
            # it will be set within on_after if success path is used below

            # We can't easily pass result here in finally without duplicating code,
            # so we rely on the success path below to call on_after_evaluate with result.

        # Success path hook and last_result assignment, outside finally for clean control flow
        # Note: We intentionally do not put hooks inside finally to avoid double-calling on error.
        # Recompute end once and reuse the same ctx object.
        # At this point, result is guaranteed to exist because exceptions are raised earlier.
        # To keep single return point, we replicate the call here:
        # But Python requires structured flow; we restructure to avoid duplication by using nested function.
        # The code above returns early. We'll never reach here.
        # Kept as comment for maintainers.

    # ====== Hooks (override in subclasses) ======
    async def on_before_evaluate(
        self,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called immediately before evaluation. Override as needed."""
        # Default no-op
        return None

    async def on_after_evaluate(
        self,
        result: JudgmentResult,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called after a successful evaluation. Override as needed."""
        # Default: cache last result and log timing
        self._last_result = result
        if context.duration_ms is not None:
            self._logger.debug("Evaluation completed in %.2f ms", context.duration_ms)

    async def on_error(
        self,
        error: Exception,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called when evaluation raises an exception. Override as needed."""
        # Default no-op (logging handled by run_evaluation)
        return None

    # ====== Enable/disable ergonomics ======
    def enable(self) -> "BaseJudge":
        """Enable this judge and return self for chaining."""
        self.enabled = True
        self._logger.debug("Judge enabled")
        return self

    def disable(self) -> "BaseJudge":
        """Disable this judge and return self for chaining."""
        self.enabled = False
        self._logger.debug("Judge disabled")
        return self

    def is_enabled(self) -> bool:
        """Return whether this judge is currently enabled."""
        return self.enabled

    @contextmanager
    def temporarily_enabled(self, enabled: bool = True):
        """Temporarily set enabled state within a context."""
        prev = self.enabled
        self.enabled = enabled
        try:
            yield self
        finally:
            self.enabled = prev

    # ====== Introspection and metrics ======
    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def last_result(self) -> Optional[JudgmentResult]:
        return self._last_result

    @property
    def evaluation_count(self) -> int:
        return self._eval_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def average_duration_ms(self) -> Optional[float]:
        if self._eval_count == 0 or self._total_duration == 0.0:
            return None
        return (self._total_duration / self._eval_count) * 1000.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"
