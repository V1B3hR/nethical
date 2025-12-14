"""Google Vertex AI integration with Nethical governance."""

from typing import Dict, Any, Optional, List
import logging

from .base import CloudMLProvider, ExperimentRun, RunStatus

logger = logging.getLogger(__name__)


class VertexAIConnector(CloudMLProvider):
    """Google Vertex AI integration with Nethical governance."""
    
    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        enable_governance: bool = True
    ):
        """Initialize Vertex AI connector.
        
        Args:
            project: GCP project ID
            location: GCP region
            enable_governance: Whether to enable governance checks
        """
        self.project = project
        self.location = location
        self.enable_governance = enable_governance
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.aiplatform = None
        self.available = False
        
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=project, location=location)
            self.aiplatform = aiplatform
            self.available = True
            logger.info(f"Vertex AI connector initialized for project {project} in {location}")
        except ImportError:
            logger.warning("Vertex AI not available. Install with: pip install google-cloud-aiplatform")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
        
        if enable_governance:
            try:
                from nethical.core import IntegratedGovernance
                self.governance = IntegratedGovernance()
                logger.info("Governance enabled for Vertex AI")
            except Exception as e:
                logger.warning(f"Could not initialize governance: {e}")
                self.enable_governance = False
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a Vertex AI experiment run."""
        if not self.available:
            logger.error("Vertex AI not available")
            return "unavailable"
        
        try:
            experiment = self.aiplatform.Experiment.get_or_create(
                experiment_name=experiment_name
            )
            
            run = experiment.start_run(run=run_name)
            run_id = run.name
            
            self.active_runs[run_id] = ExperimentRun(
                run_id=run_id,
                experiment_name=experiment_name,
                parameters={},
                metrics={}
            )
            
            logger.info(f"Started Vertex AI run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start Vertex AI run: {e}")
            return f"error-{e}"
    
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log parameters to Vertex AI."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].parameters.update(parameters)
            
            try:
                self.aiplatform.log_params(parameters)
                logger.debug(f"Logged parameters to Vertex AI run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to Vertex AI."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            
            try:
                self.aiplatform.log_metrics(metrics)
                logger.debug(f"Logged metrics to Vertex AI run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
    
    def predict_with_governance(
        self,
        endpoint_id: str,
        instances: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make predictions with governance checks.
        
        Args:
            endpoint_id: Vertex AI endpoint ID
            instances: Input instances for prediction
            
        Returns:
            Prediction results with governance info
        """
        if not self.available:
            return {"error": "Vertex AI not available"}
        
        try:
            endpoint = self.aiplatform.Endpoint(endpoint_id)
            
            # Pre-check inputs with governance
            if self.enable_governance:
                for i, instance in enumerate(instances):
                    result = self.governance.process_action(
                        action=str(instance),
                        agent_id=f"vertex-ai-{endpoint_id}",
                        action_type="model_input"
                    )
                    if result.get('risk_score', 0) > 0.7:
                        return {
                            "error": f"Input {i} blocked by governance",
                            "reason": result.get('reason'),
                            "governance_result": result
                        }
            
            # Make prediction
            predictions = endpoint.predict(instances=instances)
            
            # Post-check outputs with governance
            if self.enable_governance:
                for i, pred in enumerate(predictions.predictions):
                    result = self.governance.process_action(
                        action=str(pred),
                        agent_id=f"vertex-ai-{endpoint_id}",
                        action_type="model_output"
                    )
                    if result.get('risk_score', 0) > 0.7:
                        predictions.predictions[i] = {
                            "filtered": True,
                            "reason": result.get('reason')
                        }
            
            return {
                "predictions": predictions.predictions,
                "deployed_model_id": predictions.deployed_model_id
            }
            
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            return {"error": str(e)}
    
    def end_run(self, run_id: str, status: str = "completed"):
        """End a Vertex AI experiment run."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].status = RunStatus(status)
            
            try:
                self.aiplatform.end_run()
                logger.info(f"Ended Vertex AI run {run_id} with status {status}")
            except Exception as e:
                logger.error(f"Failed to end run: {e}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run metadata."""
        return self.active_runs.get(run_id)
