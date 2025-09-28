from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

# Import your existing models
# Adjust import path if different in your repo
from ..core.models import AgentAction, SafetyViolation  # type: ignore


# ======================================================
# Utility / Data Structures
# ======================================================

@dataclass(slots=True)
class EvaluationContext:
    action: AgentAction
    trace_id: str
    created_ns: int
    deadline_ns: Optional[int]
    metadata: Dict[str, Any]
    prior_violations: List[SafetyViolation]
    cancel_event: asyncio.Event
    experiment_flags: Dict[str, bool]


@dataclass(slots=True)
class EvaluationOutcome:
    violations: List[SafetyViolation]
    risk_score: float
    severity: str
    cached: bool
    trace_id: str
    latency_ms: float


class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


# ======================================================
# Rolling Counter (time window)
# ======================================================

class RollingCounter:
    def __init__(self, window_sec: float = 300.0):
        self.window_sec = window_sec
        self._events: Deque[Tuple[float, float]] = deque()
        self._sum = 0.0

    def add(self, value: float):
        now = time.time()
        self._events.append((now, value))
        self._sum += value
        self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self.window_sec
        while self._events and self._events[0][0] < cutoff:
            ts, val = self._events.popleft()
            self._sum -= val

    def average_rate(self) -> float:
        now = time.time()
        self._prune(now)
        # Rate expressed as sum per window length
        return self._sum / max(1e-6, self.window_sec)


# ======================================================
# Cache Interface + In-Memory TTL Implementation
# ======================================================

class EvaluationCache(ABC):
    @abstractmethod
    async def get(self, key: Hashable) -> Optional[EvaluationOutcome]:
        ...

    @abstractmethod
    async def set(self, key: Hashable, value: EvaluationOutcome, ttl_s: Optional[float] = None):
        ...

    @abstractmethod
    async def invalidate(self, key: Hashable):
        ...


class InMemoryTTLCache(EvaluationCache):
    """Simple thread-unsafe (async locked) TTL LRU-like cache."""

    def __init__(self, maxsize: int = 512):
        self._store: "OrderedDict[Hashable, Tuple[float | None, EvaluationOutcome]]" = OrderedDict()
        self._maxsize = max(1, int(maxsize))
        self._lock = asyncio.Lock()

    async def get(self, key: Hashable) -> Optional[EvaluationOutcome]:
        async with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            exp_ts, outcome = item
            now = time.time()
            if exp_ts is not None and now > exp_ts:
                # Expired: remove and return None
                self._store.pop(key, None)
                return None
            # Mark as recently used
            self._store.move_to_end(key, last=True)
            # Return a shallow copy conceptually; here outcome is immutable enough
            cloned = EvaluationOutcome(
                violations=list(outcome.violations),
                risk_score=outcome.risk_score,
                severity=outcome.severity,
                cached=True,
                trace_id=outcome.trace_id,
                latency_ms=outcome.latency_ms,
            )
            return cloned

    async def set(self, key: Hashable, value: EvaluationOutcome, ttl_s: Optional[float] = None):
        async with self._lock:
            exp_ts = None if ttl_s is None else (time.time() + ttl_s)
            self._store[key] = (exp_ts, value)
            self._store.move_to_end(key, last=True)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    async def invalidate(self, key: Hashable):
        async with self._lock:
            self._store.pop(key, None)


# ======================================================
# Advanced Base Monitor
# ======================================================

class AdvancedBaseMonitor(ABC):
    """
    Advanced monitoring base with:
    - EvaluationContext & structured EvaluationOutcome
    - TTL caching (pluggable backend)
    - Circuit breaker
    - Rolling error & violation rates
    - Hooks (before / after / error)
    - Risk scoring abstraction
    """

    __slots__ = (
        "name", "_enabled", "priority", "timeout", "strict_errors", "max_violations",
        "_logger", "_cache", "_cache_ttl_s", "_metrics_lock", "_metrics",
        "_hooks", "_circuit_state", "_circuit_fail_count", "_circuit_open_until",
        "_circuit_threshold", "_circuit_cooldown_s", "_rolling_errors",
        "_rolling_violations"
    )

    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        priority: int = 100,
        timeout: Optional[float] = None,
        strict_errors: bool = False,
        max_violations: Optional[int] = None,
        cache: Optional[EvaluationCache] = None,
        cache_ttl_s: Optional[float] = 300.0,
        circuit_threshold: int = 5,
        circuit_cooldown_s: float = 30.0,
    ):
        self.name = name
        self._enabled = enabled
        self.priority = priority
        self.timeout = timeout
        self.strict_errors = strict_errors
        self.max_violations = max_violations
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._cache = cache or InMemoryTTLCache()
        self._cache_ttl_s = cache_ttl_s

        self._metrics_lock = asyncio.Lock()
        self._metrics: Dict[str, float] = {
            "evaluations": 0,
            "violations": 0,
            "errors": 0,
            "timeouts": 0,
            "cache_hits": 0,
        }

        self._hooks: Dict[str, List[Callable[..., Any]]] = {
            "before": [],
            "after": [],
            "error": [],
        }

        # Circuit breaker
        self._circuit_state = CircuitState.CLOSED
        self._circuit_fail_count = 0
        self._circuit_open_until = 0.0
        self._circuit_threshold = circuit_threshold
        self._circuit_cooldown_s = circuit_cooldown_s

        # Rolling windows
        self._rolling_errors = RollingCounter()
        self._rolling_violations = RollingCounter()

    # ---------- Public API ----------

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register_hook(self, kind: str, fn: Callable[..., Any]):
        self._hooks.setdefault(kind, []).append(fn)

    # ---------- Orchestrated Evaluate ----------

    async def evaluate(
        self,
        action: AgentAction,
        *,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> EvaluationOutcome:
        """
        Main evaluation method returning structured outcome.
        """
        if not self._enabled:
            return EvaluationOutcome([], 0.0, "none", cached=False, trace_id=trace_id or "", latency_ms=0.0)

        now = time.time()
        if self._circuit_state == CircuitState.OPEN and now < self._circuit_open_until:
            # Circuit open: short-circuit
            return EvaluationOutcome([], 0.0, "none", cached=False, trace_id=trace_id or "", latency_ms=0.0)
        elif self._circuit_state == CircuitState.OPEN and now >= self._circuit_open_until:
            # Half-open trial attempt
            self._circuit_state = CircuitState.HALF_OPEN

        if not self.supports(action):
            return EvaluationOutcome([], 0.0, "none", cached=False, trace_id=trace_id or "", latency_ms=0.0)

        trace_id = trace_id or str(uuid.uuid4())
        ctx = EvaluationContext(
            action=action,
            trace_id=trace_id,
            created_ns=time.monotonic_ns(),
            deadline_ns=self._compute_deadline_ns(),
            metadata=metadata or {},
            prior_violations=[],
            cancel_event=asyncio.Event(),
            experiment_flags={},  # placeholder for future flag routing
        )

        key = self.get_action_key(action) if use_cache else None
        if key is not None and use_cache:
            cached = await self._cache.get(key)
            if cached:
                await self._metric_inc("cache_hits")
                return cached

        await self._run_hooks("before", ctx)

        t0 = time.monotonic_ns()
        try:
            violations = await self._run_with_timeout(ctx)
        except asyncio.TimeoutError:
            await self._metric_inc("timeouts")
            await self._record_failure("timeout")
            await self._run_hooks("error", ctx, "timeout")
            if self.strict_errors:
                raise
            return EvaluationOutcome([], 0.0, "none", cached=False, trace_id=trace_id, latency_ms=self._lat_ms(t0))
        except Exception as ex:  # noqa: BLE001
            await self._metric_inc("errors")
            await self._record_failure("error")
            await self._run_hooks("error", ctx, ex)
            if self.strict_errors:
                raise
            self._logger.exception("Monitor %s failed: %s", self.name, ex)
            return EvaluationOutcome([], 0.0, "none", cached=False, trace_id=trace_id, latency_ms=self._lat_ms(t0))

        # Success resets circuit (if half-open or closed)
        self._reset_circuit()

        if self.max_violations is not None and len(violations) > self.max_violations:
            violations = violations[: self.max_violations]

        score, severity = self.score_violations(violations)
        outcome = EvaluationOutcome(
            violations=violations,
            risk_score=score,
            severity=severity,
            cached=False,
            trace_id=trace_id,
            latency_ms=self._lat_ms(t0),
        )

        await self._metric_inc("evaluations")
        await self._metric_add("violations", len(violations))
        self._rolling_errors.add(0.0)
        self._rolling_violations.add(float(len(violations)))

        await self._run_hooks("after", ctx, outcome)

        if key is not None and use_cache:
            await self._cache.set(key, outcome, ttl_s=self._cache_ttl_s)

        return outcome

    # Backward compatibility helper (returns just violations list like legacy BaseMonitor)
    async def evaluate_legacy(self, action: AgentAction) -> List[SafetyViolation]:
        outcome = await self.evaluate(action)
        return outcome.violations

    # ---------- Abstracts & Overrides ----------

    @abstractmethod
    async def analyze_action(self, ctx: EvaluationContext) -> List[SafetyViolation]:
        """
        Subclasses implement core analysis logic.
        Should be deterministic and side-effect free relative to input.
        """
        ...

    def supports(self, action: AgentAction) -> bool:
        return True

    def get_action_key(self, action: AgentAction) -> Optional[Hashable]:
        """
        Override to provide cache key. Return None to disable caching for this action.
        """
        return None

    def score_violations(self, violations: List[SafetyViolation]) -> Tuple[float, str]:
        """
        Basic risk scoring: weight by severity if attribute exists, else count.
        """
        if not violations:
            return 0.0, "none"
        score = 0.0
        for v in violations:
            sev = getattr(v, "severity", None)
            if isinstance(sev, str):
                sev_l = sev.lower()
                score += {
                    "info": 0.5,
                    "low": 1.0,
                    "medium": 2.0,
                    "high": 4.0,
                    "critical": 8.0,
                }.get(sev_l, 2.0)
            else:
                score += 2.0
        if score >= 16:
            level = "critical"
        elif score >= 8:
            level = "high"
        elif score >= 4:
            level = "medium"
        elif score >= 1:
            level = "low"
        else:
            level = "none"
        return score, level

    # ---------- Internals ----------

    def _compute_deadline_ns(self) -> Optional[int]:
        if self.timeout is None:
            return None
        return time.monotonic_ns() + int(self.timeout * 1e9)

    async def _run_with_timeout(self, ctx: EvaluationContext) -> List[SafetyViolation]:
        if self.timeout and self.timeout > 0:
            return await asyncio.wait_for(self.analyze_action(ctx), timeout=self.timeout)
        return await self.analyze_action(ctx)

    async def _run_hooks(self, kind: str, *args):
        for fn in self._hooks.get(kind, []):
            try:
                if asyncio.iscoroutinefunction(fn):  # type: ignore[attr-defined]
                    await fn(*args)
                else:
                    fn(*args)
            except Exception as hook_ex:  # noqa: BLE001
                self._logger.warning("Hook '%s' failed in %s: %s", kind, self.name, hook_ex)

    async def _metric_inc(self, key: str):
        async with self._metrics_lock:
            self._metrics[key] = self._metrics.get(key, 0) + 1

    async def _metric_add(self, key: str, value: float):
        async with self._metrics_lock:
            self._metrics[key] = self._metrics.get(key, 0) + value

    def _lat_ms(self, t0_ns: int) -> float:
        return (time.monotonic_ns() - t0_ns) / 1_000_000.0

    async def _record_failure(self, _reason: str):
        self._circuit_fail_count += 1
        self._rolling_errors.add(1.0)
        if self._circuit_state in (CircuitState.CLOSED, CircuitState.HALF_OPEN) and self._circuit_fail_count >= self._circuit_threshold:
            self._circuit_state = CircuitState.OPEN
            self._circuit_open_until = time.time() + self._circuit_cooldown_s
            self._logger.warning("Circuit opened for monitor %s", self.name)

    def _reset_circuit(self):
        if self._circuit_state != CircuitState.CLOSED:
            self._circuit_state = CircuitState.CLOSED
        self._circuit_fail_count = 0

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self._enabled,
            "circuit_state": self._circuit_state.name,
            "metrics": dict(self._metrics),
            "rolling_error_rate": self._rolling_errors.average_rate(),
            "rolling_violation_rate": self._rolling_violations.average_rate(),
        }

    # ---------- Representation ----------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, enabled={self._enabled}, priority={self.priority}, "
            f"timeout={self.timeout}, strict_errors={self.strict_errors}, "
            f"max_violations={self.max_violations})"
        )


# ======================================================
# IntentDeviationMonitor (Converted)
# ======================================================

class IntentDeviationMonitor(AdvancedBaseMonitor):
    """
    Converted monitor that analyzes deviation between agent's 'intent'
    (assumed provided in action metadata or previous context) and its
    actual action content.

    This is a simplified adaptation; replace placeholder logic with
    your original advanced NLI / similarity scoring if available.
    """

    _DEFAULT_STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "but",
        "to", "of", "in", "on", "for", "with", "at", "by", "from",
        "as", "is", "are", "was", "were", "be", "been", "being",
        "that", "this", "these", "those", "it", "its", "into", "about",
        "if", "then", "so", "than", "too", "very",
        "can", "could", "should", "would", "will", "may", "might", "must",
        "do", "does", "did", "not", "no", "yes"
    }

    _HIGH_RISK_TOKENS: Set[str] = {
        "delete", "drop", "truncate", "wipe", "erase", "format",
        "shutdown", "kill", "terminate", "halt",
        "leak", "exfiltrate", "steal", "exfil", "dump",
        "chmod", "chown", "sudo", "root", "rm", "rmdir", "del",
        "privilege", "escalate", "exploit", "backdoor",
        "disable", "bypass", "ignore", "override",
        "production", "prod", "database", "db",
        "secrets", "secret", "token", "apikey", "api_key", "password", "passwd", "key",
    }

    def __init__(
        self,
        *,
        deviation_threshold: float = 0.35,
        high_risk_weight: float = 0.25,
        **kwargs
    ):
        super().__init__("intent_deviation_monitor", **kwargs)
        self.deviation_threshold = deviation_threshold
        self.high_risk_weight = high_risk_weight

    def get_action_key(self, action: AgentAction) -> Optional[Hashable]:
        # Simple stable key if action has an id/hash; adapt to your model fields
        aid = getattr(action, "id", None) or getattr(action, "uuid", None)
        content = getattr(action, "content", None)
        if aid is None and content is None:
            return None
        return ("intent_dev", aid, hash(content) if content else None)

    async def analyze_action(self, ctx: EvaluationContext) -> List[SafetyViolation]:
        """
        Perform deviation analysis using naive lexical similarity + risk token presence.
        """
        action = ctx.action
        # Retrieve supposed declared intent from metadata or action attributes
        declared_intent = getattr(action, "intent", None) or getattr(action, "goal", None) \
            or ctx.metadata.get("declared_intent") or ""

        # Action textual body
        body = getattr(action, "content", None) or getattr(action, "text", None) or ""
        if not declared_intent or not body:
            return []

        normalized_intent_tokens = self._normalize_tokens(declared_intent)
        normalized_body_tokens = self._normalize_tokens(body)

        if not normalized_intent_tokens or not normalized_body_tokens:
            return []

        overlap = normalized_intent_tokens & normalized_body_tokens
        union = normalized_intent_tokens | normalized_body_tokens
        lexical_similarity = len(overlap) / max(1, len(union))

        high_risk_count = sum(1 for t in normalized_body_tokens if t in self._HIGH_RISK_TOKENS)

        # Composite "deviation" score: 1 - similarity + high risk factor
        deviation_score = (1.0 - lexical_similarity) + self.high_risk_weight * high_risk_count
        violations: List[SafetyViolation] = []
        if deviation_score >= self.deviation_threshold:
            confidence = min(1.0, deviation_score)
            msg = (
                f"Potential intent deviation: lexical_similarity={lexical_similarity:.2f}, "
                f"deviation_score={deviation_score:.2f}, high_risk_tokens={high_risk_count}"
            )
            recommendations = [
                "Review action alignment with stated intent",
                "Require user confirmation for high-risk operations",
            ]
            if high_risk_count:
                recommendations.append("Escalate for manual approval due to high-risk terms")

            violation = self._make_violation(
                category="intent_deviation",
                message=msg,
                severity="medium" if deviation_score < 0.75 else "high",
                confidence=confidence,
                recommendations=recommendations,
                data={
                    "lexical_similarity": lexical_similarity,
                    "deviation_score": deviation_score,
                    "high_risk_count": high_risk_count,
                },
            )
            violations.append(violation)

        return violations

    # ---------- Helpers ----------

    def _normalize_tokens(self, text: str) -> Set[str]:
        tokens = {
            tok.lower()
            for tok in text.replace("\n", " ").replace("\t", " ").split(" ")
            if tok and tok.isascii()
        }
        return {t for t in tokens if t not in self._DEFAULT_STOPWORDS}

    def _make_violation(self, **kwargs) -> SafetyViolation:
        """
        Defensive constructor for SafetyViolation to accommodate unknown signature differences.
        """
        try:
            return SafetyViolation(**kwargs)  # type: ignore[arg-type]
        except Exception:
            # Fallback: try instantiate empty then patch attributes
            try:
                v = SafetyViolation()  # type: ignore[call-arg]
            except Exception:
                # Last resort: create a lightweight stand-in object
                class _FallbackViolation:
                    pass
                v = _FallbackViolation()  # type: ignore[assignment]

            for k, val in kwargs.items():
                try:
                    setattr(v, k, val)
                except Exception:
                    pass
            return v  # type: ignore[return-value]


# ======================================================
# Backward Compatibility Adapter (Optional)
# ======================================================

class LegacyCompatMonitor(AdvancedBaseMonitor):
    """
    Adapter to wrap an old-style monitor implementation that
    expects analyze_action(action: AgentAction) -> List[SafetyViolation].
    Provide a subclass with legacy_analyze(action) to reuse logic.
    """

    async def analyze_action(self, ctx: EvaluationContext) -> List[SafetyViolation]:
        if hasattr(self, "legacy_analyze"):
            return await self._invoke_legacy(ctx.action)
        raise NotImplementedError(
            "Subclass must define legacy_analyze(self, action: AgentAction) for LegacyCompatMonitor"
        )

    async def _invoke_legacy(self, action: AgentAction) -> List[SafetyViolation]:
        fn = getattr(self, "legacy_analyze")
        if asyncio.iscoroutinefunction(fn):  # type: ignore[attr-defined]
            return await fn(action)  # type: ignore[misc]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, action)  # type: ignore[arg-type]


# ======================================================
# Example Usage (Comment Out in Production)
# ======================================================
#
# async def main():
#     monitor = IntentDeviationMonitor(deviation_threshold=0.4, timeout=2.0)
#     fake_action = AgentAction(
#         id="a1",
#         intent="Summarize recent financial report",
#         content="DROP TABLE users; -- attempt to delete data"
#     )
#     outcome = await monitor.evaluate(fake_action)
#     print(outcome)
#
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
#
# ======================================================
