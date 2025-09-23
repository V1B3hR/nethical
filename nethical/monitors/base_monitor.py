"""Base monitor class for all monitoring components."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.models import AgentAction, SafetyViolation


class BaseMonitor(ABC):
    """Base class for all monitoring components.

    Advanced capabilities:
    - Enable/disable lifecycle with context managers.
    - Optional timeout and robust error handling.
    - Pre/post hooks and error hook for extensibility.
    - Concurrency-safe metrics (counts, latency, violations).
    - Optional evaluation caching with pluggable keys.
    - "supports" gate to skip non-applicable actions.
    - Rich configuration with chainable setters.
    """

    __slots__ = (
        "name",
        "_enabled",
        "priority",
        "tags",
        "timeout",
        "strict_errors",
        "max_violations",
        "_logger",
        "_metrics",
        "_lock",
        "_cache_enabled",
        "_cache_maxsize",
        "_cache",
    )

    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        priority: int = 100,
        tags: Optional[Iterable[str]] = None,
        timeout: Optional[float] = None,
        strict_errors: bool = False,
        max_violations: Optional[int] = None,
        cache_enabled: bool = False,
        cache_maxsize: int = 256,
    ) -> None:
        """
        Initialize a monitor.

        Args:
            name: Human-readable identifier for the monitor.
            enabled: Whether this monitor starts enabled.
            priority: Scheduling priority hint for orchestration (lower runs earlier).
            tags: Free-form labels to describe this monitor.
            timeout: Optional per-evaluation timeout in seconds.
            strict_errors: If True, re-raise analyze errors; otherwise log and continue.
            max_violations: If set, cap the number of violations returned.
            cache_enabled: If True, cache evaluate() results by action key.
            cache_maxsize: Maximum size of the evaluation cache (if enabled).
        """
        self.name = name
        self._enabled = bool(enabled)
        self.priority = int(priority)
        self.tags: Set[str] = set(tags or [])
        self.timeout = timeout if timeout is None or timeout >= 0 else 0.0
        self.strict_errors = bool(strict_errors)
        self.max_violations = max_violations if (max_violations is None or max_violations >= 0) else None

        # Logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Metrics
        self._metrics: Dict[str, int | float] = {
            "evaluations": 0,
            "skipped_disabled": 0,
            "skipped_unsupported": 0,
            "errors": 0,
            "timeouts": 0,
            "violations": 0,
            "latency_ns_total": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._lock = asyncio.Lock()

        # Simple LRU cache for evaluate() results, keyed by get_action_key(action)
        self._cache_enabled = bool(cache_enabled)
        self._cache_maxsize = max(1, int(cache_maxsize))
        self._cache: "OrderedDict[Hashable, Tuple[float, List[SafetyViolation]]]" = OrderedDict()

    # ========== ABSTRACT API ==========

    @abstractmethod
    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        """
        Analyze an agent action and return any detected violations.

        Implementations should be deterministic and side-effect free with respect to
        the provided action, and MUST return a list (possibly empty).

        Args:
            action: The agent action to analyze.

        Returns:
            A list of detected safety violations (empty if none).
        """
        raise NotImplementedError

    # ========== EVALUATION ==========

    async def evaluate(
        self,
        action: AgentAction,
        *,
        respect_enabled: bool = True,
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> List[SafetyViolation]:
        """
        Evaluate an action and return violations, handling enabled state, timeouts, errors, hooks, and caching.

        Args:
            action: The agent action to analyze.
            respect_enabled: If True, return [] when monitor is disabled.
            use_cache: If True and cache is enabled, attempt to reuse prior result by key.
            timeout: Optional override for this call's timeout; falls back to self.timeout.

        Returns:
            List of detected safety violations (possibly empty).
        """
        # Enabled-state gate
        if respect_enabled and not self._enabled:
            async with self._lock:
                self._metrics["skipped_disabled"] += 1
            return []

        # Supports gate
        if not self.supports(action):
            async with self._lock:
                self._metrics["skipped_unsupported"] += 1
            return []

        # Cache lookup
        key: Optional[Hashable] = None
        if self._cache_enabled and use_cache:
            key = self.get_action_key(action)
            if key is not None:
                cached = await self._cache_get(key)
                if cached is not None:
                    async with self._lock:
                        self._metrics["cache_hits"] += 1
                    # Return a shallow copy to protect cache integrity
                    return list(cached)

                async with self._lock:
                    self._metrics["cache_misses"] += 1

        await self.on_before_analyze(action)

        t0 = time.monotonic_ns()
        eff_timeout = self.timeout if timeout is None else timeout

        try:
            if eff_timeout and eff_timeout > 0:
                result = await asyncio.wait_for(self.analyze_action(action), timeout=eff_timeout)
            else:
                result = await self.analyze_action(action)

        except asyncio.TimeoutError as ex:
            await self._record_error(timeout=True)
            await self.on_error(action, ex)
            if self.strict_errors:
                raise
            self._logger.warning("Monitor '%s' timed out after %.3fs", self.name, eff_timeout or -1.0)
            await self._record_latency_since(t0)
            return []

        except Exception as ex:  # pylint: disable=broad-except
            await self._record_error(timeout=False)
            await self.on_error(action, ex)
            if self.strict_errors:
                raise
            self._logger.exception("Monitor '%s' analyze_action failed: %s", self.name, ex)
            await self._record_latency_since(t0)
            return []

        # Normalize
        violations = self._ensure_violations_list(result)

        # Apply cap
        if self.max_violations is not None and len(violations) > self.max_violations:
            violations = violations[: self.max_violations]

        await self.on_after_analyze(action, violations)

        # Metrics
        await self._record_latency_since(t0)
        async with self._lock:
            self._metrics["evaluations"] += 1
            self._metrics["violations"] += len(violations)

        # Cache store
        if key is not None and self._cache_enabled and use_cache:
            await self._cache_set(key, violations)

        return violations

    # ========== HOOKS & EXTENSIBILITY ==========

    def supports(self, action: AgentAction) -> bool:
        """
        Return True if this monitor can/should analyze the given action.
        Override in subclasses to restrict applicability (e.g., by action type/tool).
        """
        return True

    async def on_before_analyze(self, action: AgentAction) -> None:
        """Hook called immediately before analyze_action(). Default: no-op."""
        return None

    async def on_after_analyze(self, action: AgentAction, violations: Sequence[SafetyViolation]) -> None:
        """Hook called after analyze_action() with the computed violations. Default: no-op."""
        return None

    async def on_error(self, action: AgentAction, exc: BaseException) -> None:
        """Hook called when analyze_action() raises an error. Default: no-op."""
        return None

    def get_action_key(self, action: AgentAction) -> Optional[Hashable]:
        """
        Return a stable, hashable key for caching evaluations of this action.
        Default: None (no caching). Subclasses can override to enable caching.
        """
        return None

    # ========== LIFECYCLE ==========

    def enable(self) -> "BaseMonitor":
        """Enable this monitor. Returns self for chaining."""
        self._enabled = True
        return self

    def disable(self) -> "BaseMonitor":
        """Disable this monitor. Returns self for chaining."""
        self._enabled = False
        return self

    def set_enabled(self, value: bool) -> "BaseMonitor":
        """Set enabled state. Returns self for chaining."""
        self._enabled = bool(value)
        return self

    @property
    def is_enabled(self) -> bool:
        """Whether this monitor is currently enabled."""
        return self._enabled

    @contextmanager
    def temporarily_disabled(self):
        """Context manager to temporarily disable this monitor (sync)."""
        prev = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = prev

    @asynccontextmanager
    async def temporarily_disabled_async(self):
        """Async context manager to temporarily disable this monitor."""
        prev = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = prev

    async def aclose(self) -> None:
        """Asynchronous cleanup hook for monitors that hold resources."""
        return None

    # ========== METRICS & INTROSPECTION ==========

    @property
    def metrics(self) -> Dict[str, int | float]:
        """Return a shallow copy of current metrics."""
        return dict(self._metrics)

    def snapshot(self) -> Dict[str, Any]:
        """Return a descriptive snapshot of this monitor's configuration and health."""
        avg_latency_ms = self.average_latency_ms
        return {
            "name": self.name,
            "enabled": self._enabled,
            "priority": self.priority,
            "tags": sorted(self.tags),
            "timeout": self.timeout,
            "strict_errors": self.strict_errors,
            "max_violations": self.max_violations,
            "cache_enabled": self._cache_enabled,
            "cache_maxsize": self._cache_maxsize,
            "metrics": dict(self._metrics),
            "average_latency_ms": avg_latency_ms,
            "violation_rate": self.violation_rate,
        }

    @property
    def average_latency_ms(self) -> float:
        """Average evaluation latency in milliseconds."""
        evals = max(1, int(self._metrics["evaluations"]))
        total_ns = int(self._metrics["latency_ns_total"])
        return (total_ns / evals) / 1_000_000.0

    @property
    def violation_rate(self) -> float:
        """Average violations per evaluation."""
        evals = max(1, int(self._metrics["evaluations"]))
        return float(self._metrics["violations"]) / float(evals)

    # ========== CONFIG HELPERS ==========

    def set_timeout(self, seconds: Optional[float]) -> "BaseMonitor":
        """Set or clear the per-evaluation timeout. Returns self."""
        if seconds is not None and seconds < 0:
            seconds = 0.0
        self.timeout = seconds
        return self

    def set_max_violations(self, cap: Optional[int]) -> "BaseMonitor":
        """Set or clear the cap on returned violations. Returns self."""
        if cap is not None and cap < 0:
            cap = None
        self.max_violations = cap
        return self

    def set_strict_errors(self, strict: bool) -> "BaseMonitor":
        """Control whether analyze errors are raised. Returns self."""
        self.strict_errors = bool(strict)
        return self

    def add_tags(self, *tags: str) -> "BaseMonitor":
        """Add tags to this monitor."""
        self.tags.update(t for t in tags if t)
        return self

    def remove_tags(self, *tags: str) -> "BaseMonitor":
        """Remove tags from this monitor."""
        for t in tags:
            self.tags.discard(t)
        return self

    def configure(
        self,
        *,
        enabled: Optional[bool] = None,
        priority: Optional[int] = None,
        timeout: Optional[float] = None,
        strict_errors: Optional[bool] = None,
        max_violations: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        cache_maxsize: Optional[int] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> "BaseMonitor":
        """Batch update configuration. Returns self."""
        if enabled is not None:
            self._enabled = bool(enabled)
        if priority is not None:
            self.priority = int(priority)
        if timeout is not None:
            self.set_timeout(timeout)
        if strict_errors is not None:
            self.strict_errors = bool(strict_errors)
        if max_violations is not None:
            self.set_max_violations(max_violations)
        if cache_enabled is not None:
            self._cache_enabled = bool(cache_enabled)
        if cache_maxsize is not None:
            self._cache_maxsize = max(1, int(cache_maxsize))
            # Truncate cache if needed
            while len(self._cache) > self._cache_maxsize:
                self._cache.popitem(last=False)
        if tags is not None:
            self.tags = set(tags)
        return self

    # ========== INTERNALS ==========

    def _ensure_violations_list(self, result: Optional[Iterable[SafetyViolation]]) -> List[SafetyViolation]:
        if result is None:
            return []
        if isinstance(result, list):
            return result
        return list(result)

    async def _record_error(self, *, timeout: bool) -> None:
        async with self._lock:
            self._metrics["errors"] += 1
            if timeout:
                self._metrics["timeouts"] += 1

    async def _record_latency_since(self, t0_ns: int) -> None:
        dt = time.monotonic_ns() - t0_ns
        async with self._lock:
            self._metrics["latency_ns_total"] += dt

    async def _cache_get(self, key: Hashable) -> Optional[List[SafetyViolation]]:
        # Simple LRU behavior
        val = self._cache.get(key)
        if val is None:
            return None
        _ts, violations = val
        # move to end (most recently used)
        self._cache.move_to_end(key, last=True)
        # Return a copy to protect cache content
        return list(violations)

    async def _cache_set(self, key: Hashable, violations: List[SafetyViolation]) -> None:
        self._cache[key] = (time.time(), list(violations))
        self._cache.move_to_end(key, last=True)
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

    # ========== REPRS ==========

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, enabled={self._enabled}, priority={self.priority}, "
            f"timeout={self.timeout}, strict_errors={self.strict_errors}, "
            f"max_violations={self.max_violations}, cache_enabled={self._cache_enabled}, "
            f"tags={sorted(self.tags)!r})"
        )

    def __str__(self) -> str:
        state = "enabled" if self._enabled else "disabled"
        return f"{self.name} ({state}, priority={self.priority})"
