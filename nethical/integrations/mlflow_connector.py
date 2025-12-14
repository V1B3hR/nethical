"""MLflow Production Connector with Real SDK Integration.

This module provides a production-ready MLflow connector that uses the actual
MLflow SDK instead of stub logging. It supports both local file-based tracking
and remote tracking servers.

Requirements:
    pip install mlflow>=2.0.0

Usage:
    from nethical.integrations.mlflow_connector import MLflowConnector
    
    # Local file-based tracking
    connector = MLflowConnector(tracking_uri="file:./mlruns")
    
    # Remote tracking server
    connector = MLflowConnector(tracking_uri="http://localhost:5000")
    
    # Start experiment
    run_id = connector.start_run("my_experiment", "run_001")
    
    # Log parameters and metrics
    connector.log_parameters(run_id, {"lr": 0.001, "batch_size": 32})
    connector.log_metrics(run_id, {"accuracy": 0.95, "loss": 0.05})
    
    # Log artifact
    connector.log_artifact(run_id, "model.pkl")
    
    # End run
    connector.end_run(run_id, "completed")
"""

import logging
import os
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow>=2.0.0")


class MLflowConnector:
    """Production MLflow connector with real SDK integration.
    
    Features:
    - Actual MLflow SDK calls (not stubs)
    - Support for local file tracking
    - Support for remote tracking servers
    - Error handling for connectivity issues
    - Experiment and run lifecycle management
    - Parameters, metrics, and artifact logging
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        registry_uri: Optional[str] = None,
    ):
        """Initialize MLflow connector.
        
        Args:
            tracking_uri: MLflow tracking URI. Can be:
                - File path: "file:./mlruns" (local)
                - HTTP URL: "http://localhost:5000" (remote)
                - None: Uses MLFLOW_TRACKING_URI env var or defaults to local
            experiment_name: Default experiment name
            registry_uri: Optional model registry URI (defaults to tracking_uri)
            
        Raises:
            ImportError: If MLflow is not installed
            ConnectionError: If cannot connect to remote server
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not installed. Install with: pip install mlflow>=2.0.0")
        
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.experiment_name = experiment_name
        self.registry_uri = registry_uri
        self._client = None
        self._active_run = None
        
        # Initialize MLflow
        self._init_mlflow()
    
    def _init_mlflow(self):
        """Initialize MLflow client and set tracking URI."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.tracking_uri}")
            
            # Set registry URI if provided
            if self.registry_uri:
                mlflow.set_registry_uri(self.registry_uri)
                logger.info(f"MLflow registry URI set to: {self.registry_uri}")
            
            # Create client
            self._client = MlflowClient(tracking_uri=self.tracking_uri)
            
            # Verify connection for remote servers
            if self.tracking_uri.startswith("http"):
                try:
                    self._client.list_experiments()
                    logger.info("Successfully connected to MLflow tracking server")
                except Exception as e:
                    logger.error(f"Cannot connect to MLflow server: {e}")
                    raise ConnectionError(f"MLflow server unreachable: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            raise
    
    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Start a new MLflow run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional name for the run
            tags: Optional tags for the run
            description: Optional run description
            
        Returns:
            MLflow run ID
            
        Raises:
            MlflowException: If MLflow operation fails
        """
        try:
            # Set or create experiment
            experiment = self._client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self._client.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(experiment_name)
            
            # Start run
            run = mlflow.start_run(
                run_name=run_name,
                tags=tags,
                description=description,
            )
            
            self._active_run = run
            run_id = run.info.run_id
            
            logger.info(f"Started MLflow run: {run_id} (experiment: {experiment_name})")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            run_id: MLflow run ID
            parameters: Dictionary of parameters to log
            
        Raises:
            MlflowException: If MLflow operation fails
        """
        try:
            # MLflow has restrictions on parameter values
            # Convert non-string values to strings
            params_to_log = {}
            for key, value in parameters.items():
                if not isinstance(value, str):
                    params_to_log[key] = str(value)
                else:
                    params_to_log[key] = value
            
            # Use context manager if this is the active run
            if self._active_run and self._active_run.info.run_id == run_id:
                mlflow.log_params(params_to_log)
            else:
                # Log to specific run
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_params(params_to_log)
            
            logger.info(f"Logged {len(params_to_log)} parameters to run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics to MLflow.
        
        Args:
            run_id: MLflow run ID
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
            
        Raises:
            MlflowException: If MLflow operation fails
        """
        try:
            # Use context manager if this is the active run
            if self._active_run and self._active_run.info.run_id == run_id:
                mlflow.log_metrics(metrics, step=step)
            else:
                # Log to specific run
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics(metrics, step=step)
            
            logger.info(
                f"Logged {len(metrics)} metrics to run {run_id}"
                + (f" at step {step}" if step is not None else "")
            )
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_artifact(self, run_id: str, artifact_path: str, artifact_folder: Optional[str] = None):
        """Log artifact to MLflow.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Local path to the artifact file or directory
            artifact_folder: Optional artifact folder in MLflow
            
        Raises:
            MlflowException: If MLflow operation fails
            FileNotFoundError: If artifact path doesn't exist
        """
        try:
            # Check if artifact exists
            if not Path(artifact_path).exists():
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")
            
            # Use context manager if this is the active run
            if self._active_run and self._active_run.info.run_id == run_id:
                if Path(artifact_path).is_dir():
                    mlflow.log_artifacts(artifact_path, artifact_folder)
                else:
                    mlflow.log_artifact(artifact_path, artifact_folder)
            else:
                # Log to specific run
                with mlflow.start_run(run_id=run_id):
                    if Path(artifact_path).is_dir():
                        mlflow.log_artifacts(artifact_path, artifact_folder)
                    else:
                        mlflow.log_artifact(artifact_path, artifact_folder)
            
            logger.info(f"Logged artifact {artifact_path} to run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise
    
    def set_tags(self, run_id: str, tags: Dict[str, str]):
        """Set tags on a run.
        
        Args:
            run_id: MLflow run ID
            tags: Dictionary of tags to set
            
        Raises:
            MlflowException: If MLflow operation fails
        """
        try:
            # Use context manager if this is the active run
            if self._active_run and self._active_run.info.run_id == run_id:
                mlflow.set_tags(tags)
            else:
                # Set tags on specific run
                with mlflow.start_run(run_id=run_id):
                    mlflow.set_tags(tags)
            
            logger.info(f"Set {len(tags)} tags on run {run_id}")
            
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")
            raise
    
    def end_run(self, run_id: str, status: str = "FINISHED"):
        """End an MLflow run.
        
        Args:
            run_id: MLflow run ID
            status: Run status (FINISHED, FAILED, KILLED)
            
        Raises:
            MlflowException: If MLflow operation fails
        """
        try:
            # Map status strings to MLflow status
            status_map = {
                "completed": "FINISHED",
                "failed": "FAILED",
                "killed": "KILLED",
                "FINISHED": "FINISHED",
                "FAILED": "FAILED",
                "KILLED": "KILLED",
            }
            mlflow_status = status_map.get(status, "FINISHED")
            
            # End run
            if self._active_run and self._active_run.info.run_id == run_id:
                mlflow.end_run(status=mlflow_status)
                self._active_run = None
            else:
                # Update run status
                self._client.set_terminated(run_id, status=mlflow_status)
            
            logger.info(f"Ended MLflow run {run_id} with status {mlflow_status}")
            
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
            raise
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run information.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with run information or None if not found
        """
        try:
            run = self._client.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "parameters": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            return None
    
    def list_experiments(self) -> list:
        """List all experiments.
        
        Returns:
            List of experiment dictionaries
        """
        try:
            experiments = self._client.search_experiments()
            return [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of MLflow connection.
        
        Returns:
            Health check result with status and details
        """
        try:
            # Try to list experiments as health check
            self._client.search_experiments(max_results=1)
            
            return {
                "status": "healthy",
                "tracking_uri": self.tracking_uri,
                "mlflow_version": mlflow.__version__,
                "active_run": self._active_run.info.run_id if self._active_run else None,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "tracking_uri": self.tracking_uri,
                "error": str(e),
            }
