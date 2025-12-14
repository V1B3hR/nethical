"""Arize AI integration for Nethical governance observability."""

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ArizeConnector(ObservabilityProvider):
    """Arize AI integration for Nethical governance observability."""
    
    def __init__(
        self,
        api_key: str,
        space_key: str,
        model_id: str = "nethical-governance",
        model_version: str = "1.0"
    ):
        """Initialize Arize connector.
        
        Args:
            api_key: Arize API key
            space_key: Arize space key
            model_id: Model identifier
            model_version: Model version
        """
        try:
            from arize.pandas.logger import Client as ArizeClient
            from arize.utils.types import ModelTypes, Environments
            
            self.client = ArizeClient(api_key=api_key, space_key=space_key)
            self.model_id = model_id
            self.model_version = model_version
            self.model_type = ModelTypes.GENERATIVE_LLM
            self.environment = Environments.PRODUCTION
            self.available = True
            logger.info(f"Arize connector initialized for model: {model_id}")
        except ImportError:
            logger.warning("Arize not available. Install with: pip install arize")
            self.client = None
            self.available = False
        except Exception as e:
            logger.error(f"Failed to initialize Arize: {e}")
            self.client = None
            self.available = False
    
    def log_trace(self, span: TraceSpan) -> None:
        """Log a trace span with governance data."""
        if not self.available:
            return
            
        try:
            import pandas as pd
            from arize.utils.types import Schema, EmbeddingColumnNames
            
            # Convert span to dataframe row
            data = {
                "prediction_id": [span.span_id],
                "prediction_timestamp": [span.start_time],
                "prediction_label": [span.governance_result.get("decision", "UNKNOWN") if span.governance_result else "UNKNOWN"],
                "actual_label": [None],  # For later feedback
                "prompt": [span.attributes.get("action", "")[:1000]],
                "response": [str(span.governance_result)[:1000] if span.governance_result else ""],
                "latency_ms": [(span.end_time - span.start_time).total_seconds() * 1000 if span.end_time else 0]
            }
            
            df = pd.DataFrame(data)
            
            schema = Schema(
                prediction_id_column_name="prediction_id",
                timestamp_column_name="prediction_timestamp",
                prediction_label_column_name="prediction_label",
                actual_label_column_name="actual_label",
                prompt_column_names=EmbeddingColumnNames(
                    vector_column_name=None,
                    data_column_name="prompt"
                ),
                response_column_names=EmbeddingColumnNames(
                    vector_column_name=None,
                    data_column_name="response"
                )
            )
            
            response = self.client.log(
                dataframe=df,
                model_id=self.model_id,
                model_version=self.model_version,
                model_type=self.model_type,
                environment=self.environment,
                schema=schema
            )
            
            if response.status_code != 200:
                logger.error(f"Arize log failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to log trace to Arize: {e}")
    
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
            import pandas as pd
            from arize.utils.types import Schema
            from datetime import datetime
            import uuid
            
            data = {
                "prediction_id": [str(uuid.uuid4())],
                "prediction_timestamp": [datetime.utcnow()],
                "prediction_label": [decision],
                "prediction_score": [risk_score],
                "prompt": [action[:1000]],
                "response": [decision],
                "risk_score": [risk_score]
            }
            
            df = pd.DataFrame(data)
            
            schema = Schema(
                prediction_id_column_name="prediction_id",
                timestamp_column_name="prediction_timestamp",
                prediction_label_column_name="prediction_label",
                prediction_score_column_name="prediction_score"
            )
            
            self.client.log(
                dataframe=df,
                model_id=self.model_id,
                model_version=self.model_version,
                model_type=self.model_type,
                environment=self.environment,
                schema=schema
            )
        except Exception as e:
            logger.error(f"Failed to log governance event to Arize: {e}")
    
    def log_metrics(self, metrics: GovernanceMetrics) -> None:
        """Log aggregated governance metrics."""
        if not self.available:
            return
            
        try:
            import pandas as pd
            from arize.utils.types import Schema
            import uuid
            
            # Log as a single prediction with metrics as features
            data = {
                "prediction_id": [str(uuid.uuid4())],
                "prediction_timestamp": [metrics.timestamp],
                "total_evaluations": [metrics.total_evaluations],
                "allowed_count": [metrics.allowed_count],
                "blocked_count": [metrics.blocked_count],
                "restricted_count": [metrics.restricted_count],
                "average_risk_score": [metrics.average_risk_score],
                "pii_detections": [metrics.pii_detections],
                "latency_p50_ms": [metrics.latency_p50_ms],
                "latency_p99_ms": [metrics.latency_p99_ms]
            }
            
            df = pd.DataFrame(data)
            
            schema = Schema(
                prediction_id_column_name="prediction_id",
                timestamp_column_name="prediction_timestamp"
            )
            
            self.client.log(
                dataframe=df,
                model_id=f"{self.model_id}-metrics",
                model_version=self.model_version,
                model_type=self.model_type,
                environment=self.environment,
                schema=schema
            )
        except Exception as e:
            logger.error(f"Failed to log metrics to Arize: {e}")
    
    def create_dashboard(self, name: str) -> str:
        """Create a governance dashboard."""
        if not self.available:
            return "Arize not available"
            
        # Arize dashboards are created in UI
        return f"https://app.arize.com/organizations/{self.model_id}"
