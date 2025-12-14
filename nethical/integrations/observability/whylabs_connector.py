"""WhyLabs integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WhyLabsConnector(ObservabilityProvider):
    """WhyLabs integration for Nethical governance observability."""
    
    def __init__(
        self,
        api_key: str,
        org_id: str,
        dataset_id: str = "nethical-governance"
    ):
        """Initialize WhyLabs connector.
        
        Args:
            api_key: WhyLabs API key
            org_id: Organization ID
            dataset_id: Dataset/model ID
        """
        try:
            import whylogs as why
            from whylogs.api.writer.whylabs import WhyLabsWriter
            
            self.writer = WhyLabsWriter(
                api_key=api_key,
                org_id=org_id,
                dataset_id=dataset_id
            )
            self.dataset_id = dataset_id
            self.available = True
            logger.info(f"WhyLabs connector initialized for dataset: {dataset_id}")
        except ImportError:
            logger.warning("WhyLabs not available. Install with: pip install whylogs")
            self.writer = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize WhyLabs: {e}")
            self.writer = None
            self.available = False
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            import whylogs as why
            import pandas as pd
            
            # Create profile from span data
            data = {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "action": span.attributes.get("action", "")[:500],
                "decision": span.governance_result.get("decision", "UNKNOWN") if span.governance_result else "UNKNOWN",
                "risk_score": span.governance_result.get("risk_score", 0.0) if span.governance_result else 0.0,
                "latency_ms": (span.end_time - span.start_time).total_seconds() * 1000 if span.end_time else 0
            }
            
            df = pd.DataFrame([data])
            results = why.log(df)
            results.writer(self.writer).write()
            
        except Exception as e:
            logger.error(f"Failed to log trace to WhyLabs: {e}")
    
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
            import whylogs as why
            import pandas as pd
            
            data = {
                "action": action[:500],
                "decision": decision,
                "risk_score": risk_score,
                "pii_detected": metadata.get("pii_detected", False),
                "duration_ms": metadata.get("duration_ms", 0)
            }
            
            df = pd.DataFrame([data])
            results = why.log(df)
            results.writer(self.writer).write()
            
        except Exception as e:
            logger.error(f"Failed to log governance event to WhyLabs: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            import whylogs as why
            import pandas as pd
            
            data = {
                "total_evaluations": metrics.total_evaluations,
                "allowed_count": metrics.allowed_count,
                "blocked_count": metrics.blocked_count,
                "restricted_count": metrics.restricted_count,
                "average_risk_score": metrics.average_risk_score,
                "pii_detections": metrics.pii_detections,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p99_ms": metrics.latency_p99_ms
            }
            
            df = pd.DataFrame([data])
            results = why.log(df)
            results.writer(self.writer).write()
            
        except Exception as e:
            logger.error(f"Failed to log metrics to WhyLabs: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        if not self.available:
            return "WhyLabs not available"
            
        # WhyLabs dashboards are created in UI
        return f"https://hub.whylabsapp.com/resources/model/{self.dataset_id}"
