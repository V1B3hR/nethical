"""Distributed Tracing System

OpenTelemetry-based tracing with:
- 10% baseline sampling for normal operations
- 100% sampling for errors
- Span creation and context propagation
"""

from __future__ import annotations

import time
import random
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

# Optional OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


@dataclass
class SpanInfo:
    """Information about a trace span (fallback implementation)."""

    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: list = field(default_factory=list)

    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000.0
        return None


class TracingManager:
    """Manages distributed tracing for governance operations."""

    def __init__(
        self,
        service_name: str = "nethical-governance",
        baseline_sample_rate: float = 0.1,
        error_sample_rate: float = 1.0,
        enable_otel: bool = True,
    ):
        """Initialize tracing manager.

        Args:
            service_name: Name of the service for tracing
            baseline_sample_rate: Sample rate for normal operations (0.0-1.0)
            error_sample_rate: Sample rate for error traces (0.0-1.0)
            enable_otel: Whether to use OpenTelemetry if available
        """
        self.service_name = service_name
        self.baseline_sample_rate = baseline_sample_rate
        self.error_sample_rate = error_sample_rate
        self.enable_otel = enable_otel and OTEL_AVAILABLE

        self._lock = threading.RLock()
        self._spans: Dict[str, SpanInfo] = {}
        self._current_span = threading.local()

        # Setup OpenTelemetry if available
        if self.enable_otel:
            self._setup_otel()
            self.tracer = trace.get_tracer(service_name)
        else:
            self.tracer = None

    def _setup_otel(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not OTEL_AVAILABLE:
            return

        # Create sampler with baseline rate
        sampler = ParentBased(root=TraceIdRatioBased(self.baseline_sample_rate))

        # Setup tracer provider
        provider = TracerProvider(sampler=sampler)

        # Add console exporter for debugging
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

        # Set as global default
        trace.set_tracer_provider(provider)

    def should_sample(self, is_error: bool = False) -> bool:
        """Determine if current operation should be sampled.

        Args:
            is_error: Whether this is an error trace

        Returns:
            True if should sample
        """
        rate = self.error_sample_rate if is_error else self.baseline_sample_rate
        return random.random() < rate

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = None,
    ):
        """Start a new trace span (context manager).

        Args:
            name: Name of the span
            attributes: Optional span attributes
            kind: Span kind (e.g., 'CLIENT', 'SERVER', 'INTERNAL')

        Yields:
            Span object (OpenTelemetry) or SpanInfo (fallback)
        """
        attributes = attributes or {}

        # Use OpenTelemetry if available
        if self.enable_otel and self.tracer:
            span_kind = SpanKind.INTERNAL
            if kind == "CLIENT":
                span_kind = SpanKind.CLIENT
            elif kind == "SERVER":
                span_kind = SpanKind.SERVER

            with self.tracer.start_as_current_span(
                name, kind=span_kind, attributes=attributes
            ) as span:
                try:
                    yield span
                except Exception as e:
                    # Mark span as error
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        else:
            # Fallback implementation
            span_info = self._create_span(name, attributes)
            try:
                yield span_info
            except Exception as e:
                span_info.status = "ERROR"
                span_info.events.append(
                    {
                        "name": "exception",
                        "timestamp": time.time(),
                        "attributes": {"exception": str(e)},
                    }
                )
                raise
            finally:
                self._end_span(span_info)

    def _create_span(self, name: str, attributes: Dict[str, Any]) -> SpanInfo:
        """Create a fallback span."""
        import uuid

        span_id = str(uuid.uuid4())[:16]
        trace_id = str(uuid.uuid4())[:32]

        parent_span = getattr(self._current_span, "span", None)
        parent_span_id = parent_span.span_id if parent_span else None

        span_info = SpanInfo(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            name=name,
            start_time=time.time(),
            attributes=attributes,
        )

        with self._lock:
            self._spans[span_id] = span_info

        # Set as current span
        self._current_span.span = span_info

        return span_info

    def _end_span(self, span_info: SpanInfo) -> None:
        """End a fallback span."""
        span_info.end_time = time.time()

        # Clear current span if it matches
        current = getattr(self._current_span, "span", None)
        if current and current.span_id == span_info.span_id:
            self._current_span.span = None

    def add_span_attribute(self, key: str, value: Any) -> None:
        """Add attribute to current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if self.enable_otel and self.tracer:
            span = trace.get_current_span()
            if span:
                span.set_attribute(key, value)
        else:
            current = getattr(self._current_span, "span", None)
            if current:
                current.attributes[key] = value

    def add_span_event(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add event to current span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if self.enable_otel and self.tracer:
            span = trace.get_current_span()
            if span:
                span.add_event(name, attributes=attributes or {})
        else:
            current = getattr(self._current_span, "span", None)
            if current:
                current.events.append(
                    {
                        "name": name,
                        "timestamp": time.time(),
                        "attributes": attributes or {},
                    }
                )

    def trace_operation(self, operation_name: str, **attributes):
        """Decorator for tracing operations.

        Args:
            operation_name: Name for the traced operation
            **attributes: Additional span attributes

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                with self.start_span(operation_name, attributes=attributes):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_span_info(self, span_id: str) -> Optional[SpanInfo]:
        """Get information about a span (fallback mode only).

        Args:
            span_id: ID of the span

        Returns:
            SpanInfo or None
        """
        with self._lock:
            return self._spans.get(span_id)

    def get_all_spans(self) -> list[SpanInfo]:
        """Get all recorded spans (fallback mode only).

        Returns:
            List of SpanInfo objects
        """
        with self._lock:
            return list(self._spans.values())

    def clear_spans(self) -> None:
        """Clear all recorded spans (fallback mode only)."""
        with self._lock:
            self._spans.clear()


# Global singleton
_tracing_manager: Optional[TracingManager] = None
_tracing_lock = threading.Lock()


def get_tracer(
    service_name: str = "nethical-governance",
    baseline_sample_rate: float = 0.1,
    error_sample_rate: float = 1.0,
) -> TracingManager:
    """Get or create the global tracing manager.

    Args:
        service_name: Service name for traces
        baseline_sample_rate: Baseline sampling rate (10% default)
        error_sample_rate: Error sampling rate (100% default)

    Returns:
        TracingManager instance
    """
    global _tracing_manager

    if _tracing_manager is None:
        with _tracing_lock:
            if _tracing_manager is None:
                _tracing_manager = TracingManager(
                    service_name=service_name,
                    baseline_sample_rate=baseline_sample_rate,
                    error_sample_rate=error_sample_rate,
                )

    return _tracing_manager


# Convenience functions
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Start a trace span (convenience function).

    Args:
        name: Span name
        attributes: Optional attributes

    Returns:
        Context manager for the span
    """
    tracer = get_tracer()
    return tracer.start_span(name, attributes)


def add_span_attribute(key: str, value: Any) -> None:
    """Add attribute to current span (convenience function)."""
    tracer = get_tracer()
    tracer.add_span_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add event to current span (convenience function)."""
    tracer = get_tracer()
    tracer.add_span_event(name, attributes)
