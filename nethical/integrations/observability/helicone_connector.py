"""Helicone integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging
import requests

logger = logging.getLogger(__name__)


class HeliconeConnector(ObservabilityProvider):
    """Helicone integration for Nethical governance observability."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.helicone.ai/v1"
    ):
        """Initialize Helicone connector.
        
        Args:
            api_key: Helicone API key
            base_url: Helicone API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.available = True
        logger.info("Helicone connector initialized")
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            payload = {
                "request_id": span.span_id,
                "trace_id": span.trace_id,
                "model": "nethical-governance",
                "prompt": span.attributes.get("action", "")[:1000],
                "response": str(span.governance_result)[:1000] if span.governance_result else "",
                "start_time": span.start_time.isoformat(),
                "end_time": span.end_time.isoformat() if span.end_time else None,
                "properties": {
                    **span.attributes,
                    "governance": span.governance_result
                }
            }
            
            response = requests.post(
                f"{self.base_url}/log",
                json=payload,
                headers=self.headers,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.warning(f"Helicone log failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to log trace to Helicone: {e}")
    
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
            import uuid
            
            payload = {
                "request_id": str(uuid.uuid4()),
                "model": "nethical-governance",
                "prompt": action[:1000],
                "response": decision,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "properties": {
                    "decision": decision,
                    "risk_score": risk_score,
                    **metadata
                },
                "user_properties": {
                    "governance_version": "1.0"
                }
            }
            
            requests.post(
                f"{self.base_url}/log",
                json=payload,
                headers=self.headers,
                timeout=5
            )
            
        except Exception as e:
            logger.error(f"Failed to log governance event to Helicone: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            import uuid
            
            payload = {
                "request_id": str(uuid.uuid4()),
                "model": "nethical-governance-metrics",
                "prompt": "metrics",
                "response": "aggregated",
                "start_time": metrics.timestamp.isoformat(),
                "end_time": metrics.timestamp.isoformat(),
                "properties": {
                    "total_evaluations": metrics.total_evaluations,
                    "allowed_count": metrics.allowed_count,
                    "blocked_count": metrics.blocked_count,
                    "restricted_count": metrics.restricted_count,
                    "average_risk_score": metrics.average_risk_score,
                    "pii_detections": metrics.pii_detections,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p99_ms": metrics.latency_p99_ms
                }
            }
            
            requests.post(
                f"{self.base_url}/log",
                json=payload,
                headers=self.headers,
                timeout=5
            )
            
        except Exception as e:
            logger.error(f"Failed to log metrics to Helicone: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        # Helicone dashboards are created in UI
        return "https://www.helicone.ai/dashboard"
