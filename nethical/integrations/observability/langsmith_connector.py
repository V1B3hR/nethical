"""LangSmith integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LangSmithConnector(ObservabilityProvider):
    """LangSmith integration for Nethical governance observability."""
    
    def __init__(
        self,
        api_key: str,
        project_name: str = "nethical-governance",
        endpoint: str = "https://api.smith.langchain.com"
    ):
        """Initialize LangSmith connector.
        
        Args:
            api_key: LangSmith API key
            project_name: Project name in LangSmith
            endpoint: LangSmith API endpoint
        """
        try:
            from langsmith import Client
            self.client = Client(api_key=api_key, api_url=endpoint)
            self.project_name = project_name
            self.available = True
            logger.info(f"LangSmith connector initialized for project: {project_name}")
        except ImportError:
            logger.warning("LangSmith not available. Install with: pip install langsmith")
            self.client = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {e}")
            self.client = None
            self.available = False
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            self.client.create_run(
                name=span.name,
                run_type="chain",
                inputs={"action": span.attributes.get("action", "")},
                outputs=span.governance_result or {},
                start_time=span.start_time,
                end_time=span.end_time,
                project_name=self.project_name,
                extra={
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    **span.attributes
                }
            )
        except Exception as e:
            logger.error(f"Failed to log trace to LangSmith: {e}")
    
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
            from datetime import datetime
            
            self.client.create_run(
                name="governance_evaluation",
                run_type="tool",
                inputs={"action": action[:500]},
                outputs={
                    "decision": decision,
                    "risk_score": risk_score
                },
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                project_name=self.project_name,
                extra=metadata,
                tags=["nethical", "governance", decision.lower()]
            )
        except Exception as e:
            logger.error(f"Failed to log governance event to LangSmith: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            from datetime import datetime
            
            self.client.create_run(
                name="governance_metrics",
                run_type="chain",
                inputs={},
                outputs={
                    "total_evaluations": metrics.total_evaluations,
                    "allowed_count": metrics.allowed_count,
                    "blocked_count": metrics.blocked_count,
                    "restricted_count": metrics.restricted_count,
                    "average_risk_score": metrics.average_risk_score,
                    "pii_detections": metrics.pii_detections,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p99_ms": metrics.latency_p99_ms
                },
                start_time=metrics.timestamp,
                end_time=metrics.timestamp,
                project_name=self.project_name,
                tags=["nethical", "metrics"]
            )
        except Exception as e:
            logger.error(f"Failed to log metrics to LangSmith: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        if not self.available:
            return "LangSmith not available"
            
        # LangSmith dashboards are created in UI
        return f"https://smith.langchain.com/o/projects/{self.project_name}"
