"""Databricks integration with Nethical governance."""

from typing import Dict, Any, Optional, List
import logging

from .base import CloudMLProvider, ExperimentRun, RunStatus

logger = logging.getLogger(__name__)


class DatabricksConnector(CloudMLProvider):
    """Databricks integration with Nethical governance."""
    
    def __init__(
        self,
        workspace_url: str,
        token: Optional[str] = None,
        enable_governance: bool = True
    ):
        """Initialize Databricks connector.
        
        Args:
            workspace_url: Databricks workspace URL
            token: Databricks access token (optional, can use env var)
            enable_governance: Whether to enable governance checks
        """
        self.workspace_url = workspace_url
        self.enable_governance = enable_governance
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.workspace = None
        self.available = False
        
        try:
            from databricks.sdk import WorkspaceClient
            
            self.workspace = WorkspaceClient(
                host=workspace_url,
                token=token
            )
            self.available = True
            logger.info(f"Databricks connector initialized for {workspace_url}")
        except ImportError:
            logger.warning("Databricks SDK not available. Install with: pip install databricks-sdk")
        except Exception as e:
            logger.error(f"Failed to initialize Databricks: {e}")
        
        if enable_governance:
            try:
                from nethical.core import IntegratedGovernance
                self.governance = IntegratedGovernance()
                logger.info("Governance enabled for Databricks")
            except Exception as e:
                logger.warning(f"Could not initialize governance: {e}")
                self.enable_governance = False
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a Databricks MLflow run."""
        if not self.available:
            logger.error("Databricks not available")
            return "unavailable"
        
        try:
            import mlflow
            
            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(experiment_name)
            
            run = mlflow.start_run(run_name=run_name)
            run_id = run.info.run_id
            
            self.active_runs[run_id] = ExperimentRun(
                run_id=run_id,
                experiment_name=experiment_name,
                parameters={},
                metrics={}
            )
            
            logger.info(f"Started Databricks run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start Databricks run: {e}")
            return f"error-{e}"
    
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log parameters to Databricks MLflow."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].parameters.update(parameters)
            
            try:
                import mlflow
                mlflow.log_params(parameters)
                logger.debug(f"Logged parameters to Databricks run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to Databricks MLflow."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
                logger.debug(f"Logged metrics to Databricks run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
    
    def query_with_governance(
        self,
        endpoint_name: str,
        query: str
    ) -> Dict[str, Any]:
        """Query a Databricks serving endpoint with governance.
        
        Args:
            endpoint_name: Databricks serving endpoint name
            query: Query string
            
        Returns:
            Query results with governance info
        """
        if not self.available:
            return {"error": "Databricks not available"}
        
        try:
            # Pre-check with governance
            if self.enable_governance:
                result = self.governance.process_action(
                    action=query,
                    agent_id=f"databricks-{endpoint_name}",
                    action_type="model_input"
                )
                if result.get('risk_score', 0) > 0.7:
                    return {
                        "error": "Query blocked by governance",
                        "reason": result.get('reason'),
                        "governance_result": result
                    }
            
            # Query endpoint
            response = self.workspace.serving_endpoints.query(
                name=endpoint_name,
                dataframe_records=[{"query": query}]
            )
            
            # Post-check with governance
            if self.enable_governance:
                result = self.governance.process_action(
                    action=str(response.predictions),
                    agent_id=f"databricks-{endpoint_name}",
                    action_type="model_output"
                )
                if result.get('risk_score', 0) > 0.7:
                    return {
                        "predictions": ["[FILTERED]"],
                        "governance": result
                    }
            
            return {"predictions": response.predictions}
            
        except Exception as e:
            logger.error(f"Failed to query endpoint: {e}")
            return {"error": str(e)}
    
    def register_model_with_governance(
        self,
        model_name: str,
        model_uri: str
    ) -> Dict[str, Any]:
        """Register a model with governance validation.
        
        Args:
            model_name: Name for the model
            model_uri: URI of the model to register
            
        Returns:
            Registration result with governance info
        """
        if not self.available:
            return {"error": "Databricks not available"}
        
        try:
            import mlflow
            
            # Validate model before registration
            if self.enable_governance:
                result = self.governance.process_action(
                    action=f"Register model: {model_name} from {model_uri}",
                    agent_id="databricks-registry",
                    action_type="model_registration"
                )
                if result.get('risk_score', 0) > 0.7:
                    return {
                        "error": "Model registration blocked",
                        "reason": result.get('reason'),
                        "governance_result": result
                    }
            
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            return {
                "name": model_version.name,
                "version": model_version.version,
                "status": model_version.status
            }
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return {"error": str(e)}
    
    def end_run(self, run_id: str, status: str = "completed"):
        """End a Databricks MLflow run."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].status = RunStatus(status)
            
            try:
                import mlflow
                mlflow.end_run()
                logger.info(f"Ended Databricks run {run_id} with status {status}")
            except Exception as e:
                logger.error(f"Failed to end run: {e}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run metadata."""
        return self.active_runs.get(run_id)
