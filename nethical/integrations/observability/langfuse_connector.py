"""Langfuse integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LangfuseConnector(ObservabilityProvider):
    """Langfuse integration for Nethical governance observability."""
    
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com"
    ):
        """Initialize Langfuse connector.
        
        Args:
            public_key: Langfuse public API key
            secret_key: Langfuse secret API key
            host: Langfuse host URL
        """
        try:
            from langfuse import Langfuse
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
            self.available = True
            logger.info(f"Langfuse connector initialized for host: {host}")
        except ImportError:
            logger.warning("Langfuse not available. Install with: pip install langfuse")
            self.langfuse = None
            self.available = False
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            trace = self.langfuse.trace(
                id=span.trace_id,
                name=span.name,
                metadata=span.attributes
            )
            
            trace.span(
                id=span.span_id,
                parent_observation_id=span.parent_span_id,
                name=span.name,
                start_time=span.start_time,
                end_time=span.end_time,
                metadata={
                    **span.attributes,
                    "governance": span.governance_result
                }
            )
        except Exception as e:
            logger.error(f"Failed to log trace to Langfuse: {e}")
    
    def log_governance_event(
        self,
        action: str,
        decision: str,
        risk_score: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Log a governance evaluation event."""
        if not self.available:
            return
            
        try:
            # Truncate action for storage limits
            truncated_action = action[:500] if len(action) > 500 else action
            
            self.langfuse.event(
                name="nethical_governance",
                metadata={
                    "action": truncated_action,
                    "decision": decision,
                    "risk_score": risk_score,
                    **metadata
                },
                level="WARNING" if decision in ["BLOCK", "RESTRICT"] else "DEFAULT"
            )
        except Exception as e:
            logger.error(f"Failed to log governance event to Langfuse: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            # Langfuse doesn't have native metrics, log as event
            self.langfuse.event(
                name="nethical_metrics",
                metadata={
                    "total_evaluations": metrics.total_evaluations,
                    "allowed_count": metrics.allowed_count,
                    "blocked_count": metrics.blocked_count,
                    "restricted_count": metrics.restricted_count,
                    "average_risk_score": metrics.average_risk_score,
                    "pii_detections": metrics.pii_detections,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p99_ms": metrics.latency_p99_ms,
                    "timestamp": metrics.timestamp.isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to log metrics to Langfuse: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        if not self.available:
            return "Langfuse not available"
            
        # Langfuse dashboards are created in UI
        # Return project URL
        try:
            return f"https://cloud.langfuse.com/project/{self.langfuse.project_id}"
        except:
            return "https://cloud.langfuse.com"
    
    def flush(self) -> None:
        """Flush buffered events."""
        if self.available and self.langfuse:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse: {e}")


class NethicalLangfuseCallback:
    """Callback handler for automatic Langfuse logging."""
    
    def __init__(self, connector: LangfuseConnector):
        """Initialize callback handler.
        
        Args:
            connector: LangfuseConnector instance
        """
        self.connector = connector
        self._current_trace_id: Optional[str] = None
    
    def on_governance_start(self, action: str, agent_id: str) -> str:
        """Called when governance evaluation starts.
        
        Args:
            action: The action being evaluated
            agent_id: Agent identifier
            
        Returns:
            Trace ID for this evaluation
        """
        import uuid
        self._current_trace_id = str(uuid.uuid4())
        return self._current_trace_id
    
    def on_governance_end(
        self,
        trace_id: str,
        action: str,
        result: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """Called when governance evaluation completes.
        
        Args:
            trace_id: Trace ID from on_governance_start
            action: The action that was evaluated
            result: Governance evaluation result
            duration_ms: Evaluation duration in milliseconds
        """
        self.connector.log_governance_event(
            action=action,
            decision=result.get('decision', 'UNKNOWN'),
            risk_score=result.get('risk_score', 0.0),
            metadata={
                "trace_id": trace_id,
                "duration_ms": duration_ms,
                "pii_detected": result.get('pii_detected', False)
            }
        )
