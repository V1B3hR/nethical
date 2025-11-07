"""Base judge class for all judgment components with advanced ergonomics, rich
instrumentation, and lifecycle hooks for extensibility within the Nethical system.

Key Improvements:
- Correctly invokes on_after_evaluate (bug fix from previous version).
- Adds EvaluationStats dataclass for richer metrics & reset capability.
- Adds optional evaluation_id and extra metadata flowing through hooks.
- Adds structured / contextual logging (action id, evaluation id, duration).
- Adds run_many helper for batch evaluations with optional concurrency limits.
- Adds optional tracing callback support (e.g., OpenTelemetry) via trace_hook.
- Adds timeout using asyncio.timeout when available (Python 3.11+) with graceful fallback.
- Adds total_duration_ms and reset_metrics ergonomics.
- Ensures metrics are incremented deterministically (success vs error).
- Enhances error path logging with metadata context.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import (
    Iterable,
    Optional,
    Sequence,
    Any,
    Dict,
    List,
    Callable,
    TypeVar,
    Generic,
    Awaitable,
)

from ..core.models import AgentAction, SafetyViolation, JudgmentResult

__all__ = [
    "BaseJudge",
    "JudgeDisabledError",
    "EvaluationContext",
    "EvaluationStats",
]


class JudgeDisabledError(RuntimeError):
    """Raised when an evaluation is attempted while the judge is disabled."""


TAction = TypeVar("TAction", bound=AgentAction)
TViolation = TypeVar("TViolation", bound=SafetyViolation)
TResult = TypeVar("TResult", bound=JudgmentResult)


@dataclass(slots=True)
class EvaluationContext:
    """Execution context and timing information for an evaluation run."""
    start_time_monotonic: float
    end_time_monotonic: Optional[float] = None
    evaluation_id: Optional[str] = None
    extra: Dict[str, Any] | None = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time_monotonic is None:
            return None
        return self.end_time_monotonic - self.start_time_monotonic

    @property
    def duration_ms(self) -> Optional[float]:
        d = self.duration_seconds
        return None if d is None else d * 1000.0


@dataclass(slots=True)
class EvaluationStats:
    """Aggregate metrics for judge operation."""
    evaluations: int = 0
    errors: int = 0
    total_duration_seconds: float = 0.0

    @property
    def average_duration_seconds(self) -> Optional[float]:
        if self.evaluations == 0:
            return None
        return self.total_duration_seconds / self.evaluations

    @property
    def average_duration_ms(self) -> Optional[float]:
        avg_s = self.average_duration_seconds
        return None if avg_s is None else avg_s * 1000.0

    @property
    def total_duration_ms(self) -> float:
        return self.total_duration_seconds * 1000.0

    def reset(self) -> None:
        self.evaluations = 0
        self.errors = 0
        self.total_duration_seconds = 0.0


class BaseJudge(ABC, Generic[TAction, TViolation, TResult]):
    """Base class for all judge components.

    Subclasses MUST implement `evaluate_action` to produce a `JudgmentResult`.

    Features:
      - Enable/disable switching & context manager.
      - Concurrency control via async lock (exclusive mode).
      - Timeout support (native asyncio.timeout if Python 3.11+, else wait_for).
      - Pre/post/error hooks for extensibility.
      - Rich metrics with reset & average durations.
      - Structured logging with optional action/evaluation IDs.
      - Batch evaluation helper (`run_many`) with concurrency limits.
      - Optional tracing hook: call trace_hook(stage=..., **details).

    Hook Stages (trace_hook):
      before, success, error

    Recommended subclass pattern:
    class MyJudge(BaseJudge):
        async def evaluate_action(self, action, violations) -> JudgmentResult:
            ...

    """

    __slots__ = (
        "name",
        "enabled",
        "_logger",
        "_lock",
        "_last_result",
        "_stats",
        "_trace_hook",
    )

    def __init__(
        self,
        name: str,
        *,
        trace_hook: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[None] | None]
        ] = None,
    ) -> None:
        self.name: str = name
        self.enabled: bool = True
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        self._lock = asyncio.Lock()
        self._last_result: Optional[JudgmentResult] = None
        self._stats = EvaluationStats()
        self._trace_hook = trace_hook

    # ====== API surface that subclasses implement ======
    @abstractmethod
    async def evaluate_action(
        self,
        action: TAction,
        violations: Sequence[TViolation],
    ) -> TResult:
        """
        Evaluate an action and any associated violations to make a judgment.

        Args:
            action: The agent action to evaluate.
            violations: Sequence of detected violations for this action.

        Returns:
            JudgmentResult: Result with decision and feedback.

        Raises:
            Exception: Any domain-specific error (propagated through run_evaluation).
        """
        raise NotImplementedError

    # ====== Orchestrated evaluation with instrumentation and controls ======
    async def run_evaluation(
        self,
        action: TAction,
        violations: Iterable[TViolation] = (),
        *,
        require_enabled: bool = True,
        timeout: Optional[float] = None,
        exclusive: bool = False,
        log_inputs: bool = False,
        evaluation_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> TResult:
        """
        Orchestrate an evaluation run with safety checks, timing, optional exclusivity,
        timeout support, and lifecycle hooks.

        Args:
            action: The agent action to evaluate.
            violations: Iterable of detected violations for this action. Converted to tuple for immutability.
            require_enabled: If True, raise JudgeDisabledError when judge is disabled.
            timeout: If provided, cancel the evaluation if it exceeds this many seconds.
            exclusive: If True, serialize evaluations for this judge via an async lock.
            log_inputs: If True, log action and violations at DEBUG level (beware of PII).
            evaluation_id: Optional identifier for correlation / tracing.
            extra: Optional metadata dict passed to hooks and trace_hook.

        Returns:
            JudgmentResult: The result from the underlying `evaluate_action`.

        Raises:
            JudgeDisabledError: If `require_enabled` is True and judge is disabled.
            asyncio.TimeoutError: If the evaluation exceeds the provided timeout.
            Exception: Propagates any exception from `evaluate_action` after calling error hook.
        """
        if require_enabled and not self.enabled:
            self._logger.warning(
                "Attempted evaluation while judge '%s' is disabled (evaluation_id=%s)",
                self.name,
                evaluation_id,
            )
            raise JudgeDisabledError(f"Judge '{self.name}' is disabled")

        v_seq: Sequence[TViolation] = tuple(violations)
        if log_inputs:
            self._logger.debug(
                "Evaluating action(id=%s) with %d violations (evaluation_id=%s, judge=%s)",
                getattr(action, "id", "<no-id>"),
                len(v_seq),
                evaluation_id,
                self.name,
            )
            self._logger.debug("Violations detail: %s", v_seq)

        ctx = EvaluationContext(
            start_time_monotonic=time.monotonic(),
            evaluation_id=evaluation_id,
            extra=extra,
        )

        await self.on_before_evaluate(action, v_seq, ctx)
        await self._maybe_trace("before", action=action, violations=v_seq, context=ctx)

        async def _invoke() -> TResult:
            return await self.evaluate_action(action, v_seq)

        # Handle timeout compatibility (Python 3.11 adds asyncio.timeout)
        async def _with_timeout(coro: Awaitable[TResult]) -> TResult:
            if timeout is None:
                return await coro
            if hasattr(asyncio, "timeout"):
                # Python 3.11+
                with asyncio.timeout(timeout):
                    return await coro
            else:
                # Fallback
                return await asyncio.wait_for(coro, timeout=timeout)

        try:
            if exclusive:
                async with self._lock:
                    result = await _with_timeout(_invoke())
            else:
                result = await _with_timeout(_invoke())

        except Exception as exc:
            ctx.end_time_monotonic = time.monotonic()
            duration = ctx.duration_seconds or 0.0
            # Metrics: increment evaluations and errors (we still count attempts)
            self._stats.evaluations += 1
            self._stats.errors += 1
            self._stats.total_duration_seconds += duration

            await self.on_error(exc, action, v_seq, ctx)
            await self._maybe_trace(
                "error",
                action=action,
                violations=v_seq,
                context=ctx,
                error=exc,
            )
            self._logger.exception(
                "Evaluation failed (judge=%s, evaluation_id=%s, duration_ms=%.2f): %s",
                self.name,
                evaluation_id,
                (duration * 1000.0),
                exc,
            )
            raise
        else:
            # Success path
            ctx.end_time_monotonic = time.monotonic()
            duration = ctx.duration_seconds or 0.0
            self._stats.evaluations += 1
            self._stats.total_duration_seconds += duration

            await self.on_after_evaluate(result, action, v_seq, ctx)
            await self._maybe_trace(
                "success", action=action, violations=v_seq, context=ctx, result=result
            )

            # Structured success log
            self._logger.debug(
                "Evaluation success (judge=%s, evaluation_id=%s, action_id=%s, violations=%d, duration_ms=%.2f)",
                self.name,
                evaluation_id,
                getattr(action, "id", "<no-id>"),
                len(v_seq),
                (duration * 1000.0),
            )
            return result

    # ====== Batch Evaluation Helper ======
    async def run_many(
        self,
        actions: Sequence[TAction],
        violations_list: Sequence[Iterable[TViolation]] | None = None,
        *,
        concurrency: int = 5,
        propagate_errors: bool = True,
        timeout: Optional[float] = None,
    ) -> List[Optional[TResult]]:
        """
        Evaluate many actions concurrently with controlled parallelism.

        Args:
            actions: Sequence of actions to evaluate.
            violations_list: Optional sequence parallel to actions providing violations.
                             If None, uses empty violations for each action.
            concurrency: Max simultaneous evaluations.
            propagate_errors: If False, errors are logged and result position becomes None.
            timeout: Optional timeout applied to each individual evaluation.

        Returns:
            List of results (or None where evaluation failed and propagate_errors=False).
        """
        if violations_list is not None and len(violations_list) != len(actions):
            raise ValueError("violations_list length must match actions length")

        semaphore = asyncio.Semaphore(concurrency)
        results: List[Optional[TResult]] = [None] * len(actions)

        async def _eval(i: int):
            action = actions[i]
            vio_iter = (
                violations_list[i]
                if violations_list is not None
                else ()
            )
            async with semaphore:
                evaluation_id = f"{self.name}-{i}-{int(time.time()*1000)}"
                try:
                    res = await self.run_evaluation(
                        action,
                        vio_iter,
                        timeout=timeout,
                        evaluation_id=evaluation_id,
                    )
                    results[i] = res
                except Exception:
                    if propagate_errors:
                        raise
                    else:
                        self._logger.error(
                            "run_many suppressed error (index=%d, evaluation_id=%s)",
                            i,
                            evaluation_id,
                        )
                        results[i] = None

        tasks = [asyncio.create_task(_eval(i)) for i in range(len(actions))]
        # Gather with propagate errors semantics
        if propagate_errors:
            await asyncio.gather(*tasks)
        else:
            # Suppress exceptions individually
            for t in tasks:
                with suppress(Exception):
                    await t
        return results

    # ====== Hooks (override in subclasses) ======
    async def on_before_evaluate(
        self,
        action: TAction,
        violations: Sequence[TViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called immediately before evaluation. Override as needed."""
        return None

    async def on_after_evaluate(
        self,
        result: TResult,
        action: TAction,
        violations: Sequence[TViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called after a successful evaluation. Override as needed."""
        self._last_result = result
        if context.duration_ms is not None:
            self._logger.debug(
                "Evaluation completed (evaluation_id=%s) in %.2f ms",
                context.evaluation_id,
                context.duration_ms,
            )

    async def on_error(
        self,
        error: Exception,
        action: TAction,
        violations: Sequence[TViolation],
        context: EvaluationContext,
    ) -> None:
        """Hook called when evaluation raises an exception. Override as needed."""
        return None

    async def _maybe_trace(self, stage: str, **payload: Any) -> None:
        """Internal helper to invoke tracing callback if provided."""
        if self._trace_hook is None:
            return
        try:
            maybe = self._trace_hook(stage, payload)
            if asyncio.iscoroutine(maybe):
                await maybe
        except Exception as exc:  # Never let tracing break judge logic
            self._logger.warning("Trace hook error (stage=%s): %s", stage, exc)

    # ====== Enable/disable ergonomics ======
    def enable(self) -> "BaseJudge":
        self.enabled = True
        self._logger.debug("Judge enabled")
        return self

    def disable(self) -> "BaseJudge":
        self.enabled = False
        self._logger.debug("Judge disabled")
        return self

    def is_enabled(self) -> bool:
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
        return self._stats.evaluations

    @property
    def error_count(self) -> int:
        return self._stats.errors

    @property
    def average_duration_ms(self) -> Optional[float]:
        return self._stats.average_duration_ms

    @property
    def total_duration_ms(self) -> float:
        return self._stats.total_duration_ms

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._stats.reset()
        self._logger.debug("Metrics reset")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled}, "
            f"evaluations={self._stats.evaluations}, errors={self._stats.errors})"
        )
