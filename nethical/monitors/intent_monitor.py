"""
Advanced IntentDeviationMonitor with integrated logging & optional OpenTelemetry instrumentation.

Features:
- Pluggable lexical metrics
- Optional embeddings & NLI (lazy loaded)
- **v2.0: Semantic intent deviation using sentence-transformers with graceful fallback**
- Risk taxonomy & deviation profiling
- Dynamic weighting & threshold adaptation
- Structured logging hooks
- OpenTelemetry (optional) traces & metrics:
    * Traces: analyze_action span includes sub-spans for lexical, embedding, NLI
    * Metrics:
        - Histogram: intent_deviation_score
        - Counter: intent_deviation_violations_total
        - Counter: intent_deviation_risk_triggers_total
        - ObservableGauge: last_intent_deviation_score (per-process)
- Graceful fallback if OTEL not present

Environment Variables (optional):
    NETHICAL_ENABLE_OTEL=1        Enable OTEL initialization
    OTEL_EXPORTER=console|otlp    Select exporter (default console)
    LOG_LEVEL=DEBUG|INFO|...      Logging level (default INFO)
    NETHICAL_SEMANTIC=1           Force enable semantic monitoring (default: auto-detect)

Optional Dependencies:
    sentence-transformers
    transformers
    opentelemetry-api
    opentelemetry-sdk
    opentelemetry-exporter-otlp

This file is self-contained for demonstration. Integrate with project models accordingly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import string
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import blake2b
from itertools import tee
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Protocol,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Project model placeholders (Adjust import paths to your actual project)
# ---------------------------------------------------------------------------
try:
    from .base_monitor import BaseMonitor
    from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel
except Exception:
    # Fallback lightweight stand-ins for demonstration/testing outside full project
    class SeverityLevel:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        CRITICAL = "CRITICAL"

    class ViolationType:
        INTENT_DEVIATION = "INTENT_DEVIATION"

    @dataclass
    class AgentAction:
        id: Optional[str]
        stated_intent: Optional[str]
        actual_action: Optional[str]

    @dataclass
    class SafetyViolation:
        id: str
        action_id: Optional[str]
        violation_type: str
        severity: str
        description: str
        evidence: Dict[str, Any]

    class BaseMonitor:
        def __init__(self, name: str):
            self.name = name
            self.enabled = True


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------


def setup_logging(
    level: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """
    Configure root logging. Idempotent: subsequent calls update level only.
    """
    lvl = getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        if json_format:
            formatter = JsonLogFormatter()
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(lvl)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


# ---------------------------------------------------------------------------
# OpenTelemetry Observability Layer (graceful fallback)
# ---------------------------------------------------------------------------


class Observability:
    """
    Encapsulates OTEL tracer & meter plus metrics instruments.
    Degrades to no-ops if OTEL not available or disabled.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._otel_available = False
        self.tracer = None
        self._gauge_last_score_callback_registered = False

        self._hist_deviation = None
        self._counter_violations = None
        self._counter_risk_triggers = None
        self._last_score = 0.0

        if enabled:
            self._init_otel()

    def _init_otel(self) -> None:
        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            exporter_choice = os.getenv("OTEL_EXPORTER", "console").lower()

            # Span Exporter
            if exporter_choice == "otlp":
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                span_exporter = OTLPSpanExporter()
                metric_exporter = OTLPMetricExporter()
            else:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                from opentelemetry.sdk.metrics.export import ConsoleMetricExporter

                span_exporter = ConsoleSpanExporter()
                metric_exporter = ConsoleMetricExporter()

            resource = Resource.create({"service.name": "nethical-intent-monitor"})
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer("nethical.intent.monitor")

            metric_reader = PeriodicExportingMetricReader(metric_exporter)
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter("nethical.intent.monitor")

            self._hist_deviation = meter.create_histogram(
                name="intent_deviation_score",
                description="Distribution of deviation scores (higher=worse)",
                unit="1",
            )
            self._counter_violations = meter.create_counter(
                name="intent_deviation_violations_total",
                description="Total number of detected intent deviation violations",
                unit="1",
            )
            self._counter_risk_triggers = meter.create_counter(
                name="intent_deviation_risk_triggers_total",
                description="Total number of times risk cues were detected",
                unit="1",
            )

            # Observable gauge for last score
            def _observe_last_score(observer):
                observer.observe(self._last_score)

            meter.create_observable_gauge(
                name="intent_deviation_last_score",
                callbacks=[_observe_last_score],
                description="Last computed intent deviation score",
                unit="1",
            )

            self._otel_available = True
        except Exception as e:
            logging.getLogger(__name__).debug("OTEL init skipped or failed: %s", e)
            self.enabled = False

    def span(self, name: str):
        """
        Context manager returning an OTEL span if enabled, else dummy context.
        """
        if self.enabled and self._otel_available and self.tracer:
            return self.tracer.start_as_current_span(name)
        return _NullContext()

    def record_deviation(self, score: float, attributes: Optional[Dict[str, Any]] = None):
        self._last_score = score
        if self.enabled and self._otel_available and self._hist_deviation:
            self._hist_deviation.record(score, attributes=attributes or {})

    def increment_violation(self, attributes: Optional[Dict[str, Any]] = None):
        if self.enabled and self._otel_available and self._counter_violations:
            self._counter_violations.add(1, attributes=attributes or {})

    def increment_risk_trigger(self, attributes: Optional[Dict[str, Any]] = None):
        if self.enabled and self._otel_available and self._counter_risk_triggers:
            self._counter_risk_triggers.add(1, attributes=attributes or {})


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def setup_observability() -> Observability:
    enable = os.getenv("NETHICAL_ENABLE_OTEL", "0") == "1"
    return Observability(enable)


# ---------------------------------------------------------------------------
# Optional ML Backends
# ---------------------------------------------------------------------------


class _OptionalSentenceTransformerEmbedder:
    """Lazy loader for sentence-transformers embedding model (optional)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._available_error: Optional[str] = None

    def available(self) -> bool:
        if self._model is None and self._available_error is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:  # noqa: BLE001
                self._available_error = f"{type(e).__name__}: {e}"
        return self._model is not None

    @lru_cache(maxsize=512)
    def encode(self, text: str) -> Optional[List[float]]:
        if not self.available():
            return None
        vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def similarity(self, a: str, b: str) -> Optional[float]:
        va, vb = self.encode(a), self.encode(b)
        if not va or not vb:
            return None
        dot = sum(x * y for x, y in zip(va, vb))
        return max(0.0, min(1.0, float(dot)))


class _OptionalNLI:
    """Lazy loader for NLI pipeline producing entailment / contradiction / neutral probabilities."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self._pipe = None
        self._available_error: Optional[str] = None

    def available(self) -> bool:
        if self._pipe is None and self._available_error is None:
            try:
                from transformers import pipeline  # type: ignore

                self._pipe = pipeline("text-classification", model=self.model_name)
            except Exception as e:  # noqa: BLE001
                self._available_error = f"{type(e).__name__}: {e}"
        return self._pipe is not None

    @lru_cache(maxsize=512)
    def score(self, premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
        if not self.available():
            return None
        try:
            pipe = self._pipe
            raw = pipe(f"{premise} </s></s> {hypothesis}", return_all_scores=True)  # type: ignore[operator]
            if not raw or not isinstance(raw, list) or not raw[0]:
                return None
            label_map: Dict[str, float] = {}
            for item in raw[0]:
                label = str(item.get("label", "")).lower()
                label_map[label] = float(item.get("score", 0.0))
            entail = label_map.get("entailment", 0.0)
            contra = label_map.get("contradiction", 0.0)
            neutral = label_map.get("neutral", 0.0)
            total = entail + contra + neutral
            if total > 0:
                entail, contra, neutral = entail / total, contra / total, neutral / total
            return {"entailment": entail, "contradiction": contra, "neutral": neutral}
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Metric Protocol & Implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Metric(Protocol):
    name: str

    def compute(self, ctx: "MetricContext") -> float: ...


@dataclass
class MetricContext:
    intent_tokens: List[str]
    action_tokens: List[str]
    norm_intent: str
    norm_action: str
    bigrams_intent: List[Tuple[str, str]]
    bigrams_action: List[Tuple[str, str]]


@dataclass
class JaccardMetric:
    name: str
    use_bigrams: bool = False

    def compute(self, ctx: MetricContext) -> float:
        a = ctx.bigrams_intent if self.use_bigrams else ctx.intent_tokens
        b = ctx.bigrams_action if self.use_bigrams else ctx.action_tokens
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        union = sa | sb
        if not union:
            return 0.0
        return len(sa & sb) / len(union)


@dataclass
class CosineTFMetric:
    name: str = "cosine"

    def compute(self, ctx: MetricContext) -> float:
        a_tokens, b_tokens = ctx.intent_tokens, ctx.action_tokens
        if not a_tokens and not b_tokens:
            return 1.0
        va, vb = Counter(a_tokens), Counter(b_tokens)
        dot = sum(va[t] * vb.get(t, 0) for t in va)
        na = math.sqrt(sum(c * c for c in va.values()))
        nb = math.sqrt(sum(c * c for c in vb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


@dataclass
class CharRatioMetric:
    name: str = "char_ratio"

    def compute(self, ctx: MetricContext) -> float:
        if not ctx.norm_intent and not ctx.norm_action:
            return 1.0
        return SequenceMatcher(a=ctx.norm_intent, b=ctx.norm_action).ratio()


@dataclass
class CoverageMetric:
    name: str = "coverage"

    def compute(self, ctx: MetricContext) -> float:
        si, sa = set(ctx.intent_tokens), set(ctx.action_tokens)
        if not si and not sa:
            return 1.0
        cov1 = (len(sa & si) / len(sa)) if sa else 1.0
        cov2 = (len(si & sa) / len(si)) if si else 1.0
        return 0.5 * (cov1 + cov2)


# ---------------------------------------------------------------------------
# Config & Severity Policy
# ---------------------------------------------------------------------------

SeverityPolicy = Callable[[float, Dict[str, Any]], Optional[SeverityLevel]]


@dataclass
class IntentMonitorConfig:
    deviation_threshold: float = 0.7
    min_token_len: int = 2
    stopwords: Optional[Sequence[str]] = None
    synonyms: Optional[Dict[str, str]] = None
    weights: Optional[Dict[str, float]] = None
    enable_embeddings: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_nli: bool = False
    nli_model: str = "facebook/bart-large-mnli"
    risk_threshold_boost: float = 0.1
    contradiction_severity_boost: float = 0.1
    extra_high_risk_tokens: Sequence[str] = field(default_factory=list)
    max_overlap_terms: int = 50
    enable_timings: bool = False
    parallel_optional: bool = True
    max_chars: int = 8000
    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "critical": 0.95,
            "high": 0.85,
            "medium": 0.70,
        }
    )
    severity_policy: Optional[SeverityPolicy] = None
    metric_overrides: Optional[Dict[str, Metric]] = None
    allow_synonym_cycle: bool = False


# ---------------------------------------------------------------------------
# IntentDeviationMonitor
# ---------------------------------------------------------------------------


class IntentDeviationMonitor(BaseMonitor):
    """Detects deviation between stated intent and actual action with logging + OTEL instrumentation."""

    _DEFAULT_STOPWORDS: Set[str] = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "into",
        "about",
        "over",
        "under",
        "if",
        "then",
        "so",
        "than",
        "too",
        "very",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "must",
        "do",
        "does",
        "did",
        "not",
        "no",
        "yes",
    }

    _RISK_TOKENS_CATEGORY: Dict[str, str] = {
        "delete": "destructive",
        "drop": "destructive",
        "truncate": "destructive",
        "wipe": "destructive",
        "erase": "destructive",
        "format": "destructive",
        "shutdown": "destructive",
        "kill": "destructive",
        "terminate": "destructive",
        "halt": "destructive",
        "rm": "destructive",
        "rmdir": "destructive",
        "rm -rf": "destructive",
        "privilege": "privilege_escalation",
        "escalate": "privilege_escalation",
        "exploit": "privilege_escalation",
        "backdoor": "privilege_escalation",
        "sudo": "privilege_escalation",
        "root": "privilege_escalation",
        "chmod": "privilege_escalation",
        "chown": "privilege_escalation",
        "leak": "data_exfiltration",
        "exfiltrate": "data_exfiltration",
        "exfil": "data_exfiltration",
        "steal": "data_exfiltration",
        "dump": "data_exfiltration",
        "secrets": "secret_access",
        "token": "secret_access",
        "apikey": "secret_access",
        "password": "secret_access",
        "passwd": "secret_access",
        "key": "secret_access",
        "production": "environment",
        "prod": "environment",
        "database": "environment",
        "db": "environment",
        "disable": "control_bypass",
        "bypass": "control_bypass",
        "ignore": "control_bypass",
        "override": "control_bypass",
    }
    _BYPASS_PHRASES: Set[str] = {
        "ignore safety",
        "bypass policy",
        "disable guard",
        "disable safeguards",
    }
    _NEGATIONS: Set[str] = {"not", "no", "never", "without", "avoid"}

    def __init__(self, observability: Optional[Observability] = None, **kwargs):
        super().__init__("Intent Deviation Monitor (OTEL)")
        self.log = logging.getLogger(self.__class__.__name__)
        if "config" in kwargs:
            self.config: IntentMonitorConfig = kwargs["config"]
        else:
            self.config = IntentMonitorConfig(**kwargs)

        self.observability = observability or Observability(False)

        self.deviation_threshold = self._clamp(self.config.deviation_threshold, 0.0, 1.0)
        self.min_token_len = max(0, int(self.config.min_token_len))
        self.stopwords = (
            set(self.config.stopwords)
            if self.config.stopwords is not None
            else set(self._DEFAULT_STOPWORDS)
        )
        self.synonyms = dict(self.config.synonyms) if self.config.synonyms else {}
        if not self.config.allow_synonym_cycle:
            self._validate_synonyms(self.synonyms)

        self.high_risk_tokens: Set[str] = set(self._RISK_TOKENS_CATEGORY.keys()) | set(
            self.config.extra_high_risk_tokens
        )
        self.configured_weights = self._sanitize_weights(
            self.config.weights or self._default_weights()
        )
        self._punct_regex = re.compile(f"[{re.escape(string.punctuation)}]")

        # Optional ML
        self._embedder = (
            _OptionalSentenceTransformerEmbedder(self.config.embedding_model)
            if self.config.enable_embeddings
            else None
        )
        self._nli = _OptionalNLI(self.config.nli_model) if self.config.enable_nli else None

        # Metrics registry
        self.metrics: Dict[str, Metric] = {
            "unigram_jaccard": JaccardMetric("unigram_jaccard", use_bigrams=False),
            "bigram_jaccard": JaccardMetric("bigram_jaccard", use_bigrams=True),
            "cosine": CosineTFMetric(),
            "char_ratio": CharRatioMetric(),
            "coverage": CoverageMetric(),
        }
        if self.config.metric_overrides:
            self.metrics.update(self.config.metric_overrides)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        if not self.enabled:
            return []
        with self.observability.span("intent_monitor.analyze_action"):
            stated_intent = getattr(action, "stated_intent", None)
            actual_action = getattr(action, "actual_action", None)

            deviation_score, evidence, risk_info = await self._calculate_deviation_async(
                stated_intent, actual_action
            )
            self.observability.record_deviation(
                deviation_score, attributes={"component": "intent_monitor"}
            )

            effective_threshold = self.deviation_threshold
            if risk_info.get("risk_cues_detected"):
                effective_threshold = max(
                    0.0, effective_threshold - self.config.risk_threshold_boost
                )

            if deviation_score > effective_threshold:
                contradiction = (
                    evidence.get("nli", {}).get("intent->action", {}).get("contradiction", 0.0)
                    if evidence.get("nli")
                    else 0.0
                )
                severity = self._determine_severity(
                    deviation_score,
                    contradiction=contradiction,
                    risk_active=risk_info.get("risk_cues_detected", False),
                    evidence=evidence,
                )
                self.observability.increment_violation(
                    attributes={
                        "severity": severity,
                        "risk": str(risk_info.get("risk_cues_detected")),
                    }
                )
                if risk_info.get("risk_cues_detected"):
                    self.observability.increment_risk_trigger()

                violation = SafetyViolation(
                    id=str(uuid.uuid4()),
                    action_id=getattr(action, "id", None),
                    violation_type=ViolationType.INTENT_DEVIATION,
                    severity=severity,
                    description=(
                        f"Deviation between stated intent and actual action "
                        f"(score: {deviation_score:.2f}, threshold: {effective_threshold:.2f})"
                    ),
                    evidence={
                        "stated_intent": stated_intent,
                        "actual_action": actual_action,
                        "deviation_score": deviation_score,
                        "base_threshold": self.deviation_threshold,
                        "effective_threshold": effective_threshold,
                        **evidence,
                    },
                )
                self.log.info(
                    "Intent deviation violation detected severity=%s score=%.3f threshold=%.3f risk=%s action_id=%s",
                    severity,
                    deviation_score,
                    effective_threshold,
                    risk_info.get("risk_cues_detected"),
                    getattr(action, "id", None),
                )
                return [violation]

            self.log.debug(
                "No violation score=%.3f threshold=%.3f risk=%s",
                deviation_score,
                effective_threshold,
                risk_info.get("risk_cues_detected"),
            )
            return []

    # -----------------------------------------------------------------------
    # Core async deviation compute
    # -----------------------------------------------------------------------

    async def _calculate_deviation_async(
        self,
        intent: Optional[str],
        action: Optional[str],
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        if not intent and not action:
            return 0.0, self._empty_evidence(), {"risk_cues_detected": False}
        if not intent or not action:
            return 1.0, self._empty_evidence(flag="one_side_empty"), {"risk_cues_detected": False}

        truncated_intent, trunc_note_i = self._maybe_truncate(intent)
        truncated_action, trunc_note_a = self._maybe_truncate(action)

        norm_intent = self._normalize(truncated_intent)
        norm_action = self._normalize(truncated_action)
        tokens_intent = self._tokenize_cached(norm_intent)
        tokens_action = self._tokenize_cached(norm_action)
        bigrams_intent = self._bigrams(tokens_intent)
        bigrams_action = self._bigrams(tokens_action)

        ctx = MetricContext(
            intent_tokens=tokens_intent,
            action_tokens=tokens_action,
            norm_intent=norm_intent,
            norm_action=norm_action,
            bigrams_intent=bigrams_intent,
            bigrams_action=bigrams_action,
        )

        components: Dict[str, float] = {}
        timings: Dict[str, float] = {}
        with self.observability.span("intent_monitor.lexical_metrics"):
            for name, metric in self.metrics.items():
                t0 = time.time()
                try:
                    val = metric.compute(ctx)
                except Exception:
                    val = 0.0
                components[name] = self._clamp(val, 0.0, 1.0)
                if self.config.enable_timings:
                    timings[name] = time.time() - t0

        embed_sim: Optional[float] = None
        nli_scores: Optional[Dict[str, Dict[str, float]]] = None

        async def run_embedding():
            if self._embedder is None:
                return None
            loop = asyncio.get_running_loop()
            with self.observability.span("intent_monitor.embedding"):
                t0 = time.time()
                sim = await loop.run_in_executor(
                    None, self._embedder.similarity, norm_intent, norm_action
                )
                if self.config.enable_timings:
                    timings["embed_cosine"] = time.time() - t0
                return sim

        async def run_nli():
            if self._nli is None:
                return None
            loop = asyncio.get_running_loop()

            def _score_pair():
                s1 = self._nli.score(norm_intent, norm_action)
                s2 = self._nli.score(norm_action, norm_intent)
                if s1 or s2:
                    return {"intent->action": s1 or {}, "action->intent": s2 or {}}
                return None

            with self.observability.span("intent_monitor.nli"):
                t0 = time.time()
                scores = await loop.run_in_executor(None, _score_pair)
                if self.config.enable_timings:
                    timings["nli"] = time.time() - t0
                return scores

        if self.config.parallel_optional:
            tasks: List[Awaitable] = []
            if self._embedder is not None:
                tasks.append(run_embedding())
            if self._nli is not None:
                tasks.append(run_nli())
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                idx = 0
                if self._embedder is not None:
                    r = results[idx]
                    idx += 1
                    if not isinstance(r, Exception):
                        embed_sim = r  # type: ignore
                if self._nli is not None:
                    r = results[idx]
                    if not isinstance(r, Exception):
                        nli_scores = r  # type: ignore
        else:
            if self._embedder is not None:
                embed_sim = await run_embedding()
            if self._nli is not None:
                nli_scores = await run_nli()

        embed_available = self._embedder.available() if self._embedder else False
        nli_available = self._nli.available() if self._nli else False

        if embed_sim is not None:
            components["embed_cosine"] = self._clamp(embed_sim, 0.0, 1.0)
        if nli_scores and "intent->action" in nli_scores:
            entail = float(nli_scores["intent->action"].get("entailment", 0.0))
            components["nli_entailment"] = self._clamp(entail, 0.0, 1.0)

        weighted_similarity, effective_weights = self._aggregate_similarity(components)
        deviation = 1.0 - weighted_similarity

        # Risk analysis
        risky_intent = self._find_risky_terms(tokens_intent)
        risky_action = self._find_risky_terms(tokens_action)
        first_seen_in_action = sorted(risky_action - risky_intent)
        bypass_cues = self._find_bypass_cues(norm_intent) or self._find_bypass_cues(norm_action)
        negation_intent = any(t in self._NEGATIONS for t in tokens_intent)
        negation_action = any(t in self._NEGATIONS for t in tokens_action)
        risk_cues_detected = bool(first_seen_in_action or bypass_cues)

        cat_intent = self._categorize_risks(risky_intent)
        cat_action = self._categorize_risks(risky_action)

        only_in_intent = sorted(set(tokens_intent) - set(tokens_action))[
            : self.config.max_overlap_terms
        ]
        only_in_action = sorted(set(tokens_action) - set(tokens_intent))[
            : self.config.max_overlap_terms
        ]
        only_in_intent_bi = sorted(set(bigrams_intent) - set(bigrams_action))[
            : self.config.max_overlap_terms
        ]
        only_in_action_bi = sorted(set(bigrams_action) - set(bigrams_intent))[
            : self.config.max_overlap_terms
        ]

        contribution_profile = self._build_contribution_profile(components, effective_weights)

        metrics_evidence = {
            **{m: round(components[m], 4) for m in components},
            "weighted_similarity": round(weighted_similarity, 4),
            "deviation": round(deviation, 4),
        }

        notes = []
        if trunc_note_i:
            notes.append(trunc_note_i)
        if trunc_note_a:
            notes.append(trunc_note_a)

        evidence: Dict[str, Any] = {
            "metrics": metrics_evidence,
            "configured_weights": self.configured_weights,
            "effective_weights": {k: round(v, 4) for k, v in effective_weights.items()},
            "model_availability": {"embeddings": embed_available, "nli": nli_available},
            "nli": nli_scores,
            "unigram_overlap": {"only_in_intent": only_in_intent, "only_in_action": only_in_action},
            "bigram_overlap": {
                "only_in_intent": only_in_intent_bi,
                "only_in_action": only_in_action_bi,
            },
            "normalized": {"intent": norm_intent, "action": norm_action},
            "token_counts": {"intent": len(tokens_intent), "action": len(tokens_action)},
            "risk": {
                "risky_terms_intent": sorted(risky_intent),
                "risky_terms_action": sorted(risky_action),
                "first_seen_in_action": first_seen_in_action,
                "risk_categories_intent": cat_intent,
                "risk_categories_action": cat_action,
                "bypass_cues_detected": bool(bypass_cues),
                "negation_intent": negation_intent,
                "negation_action": negation_action,
            },
            "deviation_profile": contribution_profile,
        }
        if self.config.enable_timings:
            evidence["timings_sec"] = {k: round(v, 6) for k, v in timings.items()}
        if notes:
            evidence["notes"] = notes

        risk_info = {"risk_cues_detected": risk_cues_detected}
        return deviation, evidence, risk_info

    # -----------------------------------------------------------------------
    # Helper computations
    # -----------------------------------------------------------------------

    def _build_contribution_profile(
        self, components: Dict[str, float], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        contrib = []
        for k, w in weights.items():
            contrib.append((k, w * components.get(k, 0.0)))
        contrib.sort(key=lambda x: x[1], reverse=True)
        return {
            "ordered_metric_contributions": [(k, round(v, 4)) for k, v in contrib],
            "top_metric": contrib[0][0] if contrib else None,
        }

    def _default_weights(self) -> Dict[str, float]:
        return {
            "unigram_jaccard": 0.20,
            "bigram_jaccard": 0.10,
            "cosine": 0.25,
            "char_ratio": 0.05,
            "coverage": 0.20,
            "embed_cosine": 0.15,
            "nli_entailment": 0.05,
        }

    def _sanitize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        supported = set(self._default_weights().keys())
        filtered = {k: max(0.0, float(v)) for k, v in weights.items() if k in supported}
        if not filtered:
            filtered = self._default_weights()
        total = sum(filtered.values())
        if total <= 0.0:
            filtered = self._default_weights()
            total = sum(filtered.values())
        return {k: v / total for k, v in filtered.items()}

    def _aggregate_similarity(self, components: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        usable = {k: v for k, v in self.configured_weights.items() if k in components}
        total = sum(usable.values())
        if total <= 0:
            return 0.0, {}
        eff = {k: v / total for k, v in usable.items()}
        sim = sum(eff[k] * components[k] for k in eff)
        return self._clamp(sim, 0.0, 1.0), eff

    def _find_risky_terms(self, tokens: Sequence[str]) -> Set[str]:
        toks = set(tokens)
        risky: Set[str] = set()
        for t in toks:
            if t in self.high_risk_tokens:
                risky.add(t)
            if t in {"rm", "rmrf"} or t.startswith("rm-rf"):
                risky.add("rm -rf")
            if "drop" in t and any(x in t for x in ("table", "db", "database")):
                risky.add("drop")
        return risky

    def _categorize_risks(self, risky_terms: Set[str]) -> Dict[str, List[str]]:
        cat_map: Dict[str, List[str]] = {}
        for term in risky_terms:
            cat = self._RISK_TOKENS_CATEGORY.get(term, "other")
            cat_map.setdefault(cat, []).append(term)
        return {k: sorted(v) for k, v in cat_map.items()}

    def _find_bypass_cues(self, text: str) -> bool:
        if not text:
            return False
        for phrase in self._BYPASS_PHRASES:
            if phrase in text:
                return True
        return False

    def _determine_severity(
        self,
        deviation_score: float,
        *,
        contradiction: float,
        risk_active: bool,
        evidence: Dict[str, Any],
    ) -> SeverityLevel:
        adjusted = deviation_score
        adjusted = min(
            1.0, adjusted + self.config.contradiction_severity_boost * (contradiction**0.5)
        )
        if risk_active:
            adjusted = min(1.0, adjusted + 0.15)

        thresholds = self.config.severity_thresholds
        if self.config.severity_policy:
            custom = self.config.severity_policy(adjusted, evidence)
            if custom is not None:
                return custom

        if adjusted >= thresholds.get("critical", 0.95):
            return SeverityLevel.CRITICAL
        if adjusted >= thresholds.get("high", 0.85):
            return SeverityLevel.HIGH
        if adjusted >= thresholds.get("medium", 0.70):
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = self._punct_regex.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _apply_synonyms(self, token: str) -> str:
        return self.synonyms.get(token, token)

    def _validate_synonyms(self, syn_map: Dict[str, str]) -> None:
        for k, v in syn_map.items():
            if syn_map.get(v) == k:
                raise ValueError(
                    f"Synonym cycle detected: {k} <-> {v}. Set allow_synonym_cycle=True to override."
                )

    @lru_cache(maxsize=4096)
    def _tokenize_cached(self, text: str) -> List[str]:
        return self._tokenize(text)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        out: List[str] = []
        for t in text.split():
            if t in self.stopwords:
                continue
            t = self._apply_synonyms(t)
            if len(t) >= self.min_token_len:
                out.append(t)
        return out

    def _bigrams(self, tokens: Sequence[str]) -> List[Tuple[str, str]]:
        if len(tokens) < 2:
            return []
        a, b = tee(tokens)
        next(b, None)
        return list(zip(a, b))

    def _maybe_truncate(self, text: str) -> Tuple[str, Optional[str]]:
        if len(text) > self.config.max_chars:
            return (
                text[: self.config.max_chars],
                f"truncated_input_{len(text)}->{self.config.max_chars}",
            )
        return text, None

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _empty_evidence(self, *, flag: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "metrics": {},
            "unigram_overlap": {"only_in_intent": [], "only_in_action": []},
            "bigram_overlap": {"only_in_intent": [], "only_in_action": []},
            "model_availability": {"embeddings": False, "nli": False},
        }
        if flag:
            out["notes"] = [flag]
        return out

    def _lexical_signature(self, norm_intent: str, norm_action: str) -> str:
        h = blake2b(digest_size=16)
        h.update(norm_intent.encode())
        h.update(b"\x00")
        h.update(norm_action.encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Demonstration / Manual Test
# ---------------------------------------------------------------------------


async def _demo():
    obs = setup_observability()
    monitor = IntentDeviationMonitor(
        observability=obs,
        enable_embeddings=False,  # set True if sentence-transformers available
        enable_nli=False,  # set True if transformers available
    )

    examples = [
        AgentAction(
            id="1",
            stated_intent="List files in the project directory",
            actual_action="List files in the project directory",
        ),
        AgentAction(
            id="2",
            stated_intent="Read configuration file",
            actual_action="Delete configuration and wipe database",
        ),
        AgentAction(
            id="3",
            stated_intent="Check user permissions",
            actual_action="Attempt to escalate privilege with sudo",
        ),
    ]

    for act in examples:
        violations = await monitor.analyze_action(act)
        if violations:
            print(f"Action {act.id} -> VIOLATIONS:")
            for v in violations:
                print("  Severity:", v.severity)
                print("  Description:", v.description)
                print("  Risk first_seen:", v.evidence.get("risk", {}).get("first_seen_in_action"))
        else:
            print(f"Action {act.id} -> No violation")


if __name__ == "__main__":
    setup_logging(json_format=False)
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        pass
