"""TruLens integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TruLensConnector(ObservabilityProvider):
    """TruLens integration for Nethical governance observability."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        app_id: str = "nethical-governance"
    ):
        """Initialize TruLens connector.
        
        Args:
            database_url: Database URL for TruLens (optional)
            app_id: Application identifier
        """
        try:
            from trulens_eval import Tru, TruSession
            from trulens_eval.feedback import Feedback
            
            if database_url:
                self.session = TruSession(database_url=database_url)
            else:
                self.session = TruSession()
                
            self.tru = Tru(session=self.session)
            self.app_id = app_id
            self.available = True
            logger.info(f"TruLens connector initialized for app: {app_id}")
        except ImportError:
            logger.warning("TruLens not available. Install with: pip install trulens-eval")
            self.tru = None
            self.session = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize TruLens: {e}")
            self.tru = None
            self.session = None
            self.available = False
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            from trulens_eval.schema import Record, RecordAppCall
            
            # Create a record for the span
            record = Record(
                app_id=self.app_id,
                input=span.attributes.get("action", ""),
                output=str(span.governance_result) if span.governance_result else "",
                meta={
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "parent_span_id": span.parent_span_id,
                    **span.attributes
                },
                tags=["nethical", "governance"],
                ts=span.start_time
            )
            
            self.tru.add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to log trace to TruLens: {e}")
    
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
            from trulens_eval.schema import Record
            from datetime import datetime
            
            record = Record(
                app_id=self.app_id,
                input=action[:1000],
                output=decision,
                meta={
                    "decision": decision,
                    "risk_score": risk_score,
                    **metadata
                },
                tags=["nethical", "governance", decision.lower()],
                ts=datetime.utcnow(),
                cost={
                    "risk": risk_score
                }
            )
            
            self.tru.add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to log governance event to TruLens: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            from trulens_eval.schema import Record
            
            record = Record(
                app_id=f"{self.app_id}-metrics",
                input="metrics_aggregation",
                output="completed",
                meta={
                    "total_evaluations": metrics.total_evaluations,
                    "allowed_count": metrics.allowed_count,
                    "blocked_count": metrics.blocked_count,
                    "restricted_count": metrics.restricted_count,
                    "average_risk_score": metrics.average_risk_score,
                    "pii_detections": metrics.pii_detections,
                    "latency_p50_ms": metrics.latency_p50_ms,
                    "latency_p99_ms": metrics.latency_p99_ms
                },
                tags=["nethical", "metrics"],
                ts=metrics.timestamp,
                perf={
                    "start_time": metrics.timestamp,
                    "end_time": metrics.timestamp
                }
            )
            
            self.tru.add_record(record)
            
        except Exception as e:
            logger.error(f"Failed to log metrics to TruLens: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        if not self.available:
            return "TruLens not available"
            
        try:
            # TruLens has a built-in dashboard
            # Start it programmatically (in practice, run separately)
            return "Run: streamlit run $(python -c 'import trulens_eval; print(trulens_eval.__file__.replace(\"__init__.py\", \"pages/Dashboard.py\"))')"
        except:
            return "TruLens dashboard available via: tru.run_dashboard()"
    
    def close(self) -> None:
        """Close TruLens session."""
        if self.available and self.session:
            try:
                self.session.close()
            except Exception as e:
                logger.error(f"Failed to close TruLens session: {e}")
