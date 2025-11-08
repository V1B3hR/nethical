"""
ML Platform Integrations

Interfaces for integrating with external ML platforms and services.

Features:
- MLflow integration (stub)
- Weights & Biases (W&B) integration (stub)
- SageMaker integration (stub)
- Azure ML integration (stub)
- In-memory/custom integration
- Generic ML platform interface
- Structured event hooks for auditing/dispatch
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Mapping


# --------- helpers ---------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(ts: Optional[datetime]) -> Optional[str]:
    if not ts:
        return None
    # Format like 2025-01-01T00:00:00.000Z
    return ts.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sanitize_keys(
    d: Mapping[str, Any], sensitive_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Basic sanitization: mask values for keys that look sensitive.
    This is a generic utility; for PHI or domain-specific redaction, compose with upstream detectors.
    """
    sensitive = set(k.lower() for k in (sensitive_keys or [])) | {
        "password",
        "pass",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "credential",
        "key",
    }
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k.lower() in sensitive:
            out[k] = "***"
        else:
            out[k] = v
    return out


# --------- types ---------


class MLPlatform(Enum):
    """Supported ML platforms"""

    MLFLOW = "mlflow"
    WANDB = "wandb"
    SAGEMAKER = "sagemaker"
    AZURE_ML = "azure_ml"
    CUSTOM = "custom"
    IN_MEMORY = "in_memory"


class RunStatus(Enum):
    """Run lifecycle status"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    UNKNOWN = "unknown"


@dataclass
class ExperimentRun:
    """ML experiment run metadata"""

    run_id: str
    experiment_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=_now_utc)
    end_time: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    tags: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (UTC timestamps, enum values)"""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "start_time": _iso_utc(self.start_time),
            "end_time": _iso_utc(self.end_time),
            "status": self.status.value,
            "tags": self.tags,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """JSON string dump of run dict"""
        return json.dumps(self.to_dict())

    def duration_seconds(self) -> Optional[float]:
        if not self.end_time:
            return None
        return (self.end_time - self.start_time).total_seconds()


# --------- interface ---------


class MLPlatformInterface(ABC):
    """Abstract interface for ML platforms"""

    # Optional structured event handler signature: (event_name, payload_dict) -> None
    def __init__(self):
        self.event_handlers: List[Callable[[str, Dict[str, Any]], None]] = []

    def add_event_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback to receive events: run_started, params_logged, metrics_logged, artifact_logged, run_ended"""
        self.event_handlers.append(handler)

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        for h in self.event_handlers:
            try:
                h(event, payload)
            except Exception as e:
                logging.error(f"Event handler failed for {event}: {e}")

    @abstractmethod
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a new experiment run"""
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log experiment parameters"""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, run_id: str, artifact_path: str):
        """Log artifact (model, plot, etc.)"""
        raise NotImplementedError

    @abstractmethod
    def end_run(self, run_id: str, status: str = "completed"):
        """End experiment run"""
        raise NotImplementedError

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        """Optional: set/update tags for a run"""
        # Default no-op for platforms that don't support tags directly

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Optional: retrieve the run metadata if available"""
        return None


# --------- MLflow ---------


class MLflowIntegration(MLPlatformInterface):
    """
    MLflow integration (stub)

    Note: Requires mlflow library
    This is a stub that logs intent to interact with MLflow.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow integration

        Args:
            tracking_uri: MLflow tracking server URI (env MLFLOW_TRACKING_URI is also respected by mlflow itself)
        """
        super().__init__()
        self.tracking_uri = tracking_uri or "http://localhost:5000"
        self.active_runs: Dict[str, ExperimentRun] = {}

        # NOTE: Actual implementation would initialize MLflow client
        # try:
        #     import os, mlflow
        #     if self.tracking_uri:
        #         mlflow.set_tracking_uri(self.tracking_uri)
        #     self.client = mlflow.tracking.MlflowClient()
        # except ImportError:
        #     logging.error("mlflow not installed. Install with: pip install mlflow")
        #     self.client = None

        logging.info(f"MLflow integration initialized (stub) - Tracking URI: {self.tracking_uri}")

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start MLflow run"""
        import uuid

        run_id = str(uuid.uuid4())

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters={},
            metrics={},
            tags={"run_name": run_name} if run_name else {},
        )

        self.active_runs[run_id] = run
        logging.info(
            f"[STUB] MLflow start_run: {run_id} (experiment: {experiment_name}, name: {run_name or 'default'})"
        )
        self._emit("run_started", {"platform": MLPlatform.MLFLOW.value, **run.to_dict()})
        # Actual: mlflow.set_experiment(experiment_name); mlflow.start_run(run_name=run_name)
        return run_id

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log parameters to MLflow"""
        if run_id in self.active_runs:
            sanitized = _sanitize_keys(parameters)
            self.active_runs[run_id].parameters.update(sanitized)
            logging.info(f"[STUB] MLflow log_parameters[{run_id}]: {list(sanitized.keys())}")
            self._emit(
                "params_logged",
                {"platform": MLPlatform.MLFLOW.value, "run_id": run_id, "parameters": sanitized},
            )
            # Actual: mlflow.log_params(parameters)

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow"""
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            logging.info(
                f"[STUB] MLflow log_metrics[{run_id}]: {list(metrics.keys())} (step={step})"
            )
            self._emit(
                "metrics_logged",
                {
                    "platform": MLPlatform.MLFLOW.value,
                    "run_id": run_id,
                    "metrics": metrics,
                    "step": step,
                },
            )
            # Actual: mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, run_id: str, artifact_path: str):
        """Log artifact to MLflow"""
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(artifact_path)
            logging.info(f"[STUB] MLflow log_artifact[{run_id}]: {artifact_path}")
            self._emit(
                "artifact_logged",
                {
                    "platform": MLPlatform.MLFLOW.value,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )
            # Actual: mlflow.log_artifact(artifact_path)

    def end_run(self, run_id: str, status: str = "completed"):
        """End MLflow run"""
        if run_id in self.active_runs:
            st = RunStatus(status) if status in RunStatus._value2member_map_ else RunStatus.UNKNOWN
            run = self.active_runs[run_id]
            run.status = st
            run.end_time = _now_utc()
            logging.info(f"[STUB] MLflow end_run[{run_id}] -> {st.value}")
            self._emit("run_ended", {"platform": MLPlatform.MLFLOW.value, **run.to_dict()})
            # Actual: mlflow.end_run(status=status)

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        if run_id in self.active_runs:
            self.active_runs[run_id].tags.update(tags)
            logging.info(f"[STUB] MLflow set_tags[{run_id}]: {list(tags.keys())}")
            self._emit(
                "tags_set", {"platform": MLPlatform.MLFLOW.value, "run_id": run_id, "tags": tags}
            )

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self.active_runs.get(run_id)


# --------- Weights & Biases ---------


class WandBIntegration(MLPlatformInterface):
    """
    Weights & Biases integration (stub)

    Note: Requires wandb library
    This is a stub that logs intent to interact with W&B.
    """

    def __init__(self, project: str, entity: Optional[str] = None):
        """
        Initialize W&B integration

        Args:
            project: W&B project name
            entity: W&B entity (username or team)
        """
        super().__init__()
        self.project = project
        self.entity = entity
        self.active_runs: Dict[str, ExperimentRun] = {}

        # NOTE: Actual implementation would initialize wandb
        # try:
        #     import wandb
        #     self.wandb = wandb
        # except ImportError:
        #     logging.error("wandb not installed. Install with: pip install wandb")
        #     self.wandb = None

        logging.info(
            f"W&B integration initialized (stub) - Project: {project}, Entity: {entity or 'default'}"
        )

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start W&B run"""
        import uuid

        run_id = str(uuid.uuid4())

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters={},
            metrics={},
            tags={"run_name": run_name} if run_name else {},
        )

        self.active_runs[run_id] = run
        logging.info(f"[STUB] W&B start_run: {run_id} (name: {run_name or 'default'})")
        self._emit("run_started", {"platform": MLPlatform.WANDB.value, **run.to_dict()})
        # Actual: wandb.init(project=self.project, entity=self.entity, name=run_name)
        return run_id

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log config to W&B"""
        if run_id in self.active_runs:
            sanitized = _sanitize_keys(parameters)
            self.active_runs[run_id].parameters.update(sanitized)
            logging.info(f"[STUB] W&B log_parameters[{run_id}]: {list(sanitized.keys())}")
            self._emit(
                "params_logged",
                {"platform": MLPlatform.WANDB.value, "run_id": run_id, "parameters": sanitized},
            )
            # Actual: wandb.config.update(parameters)

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B"""
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            logging.info(f"[STUB] W&B log_metrics[{run_id}]: {list(metrics.keys())} (step={step})")
            self._emit(
                "metrics_logged",
                {
                    "platform": MLPlatform.WANDB.value,
                    "run_id": run_id,
                    "metrics": metrics,
                    "step": step,
                },
            )
            # Actual: wandb.log(metrics, step=step)

    def log_artifact(self, run_id: str, artifact_path: str):
        """Log artifact to W&B"""
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(artifact_path)
            logging.info(f"[STUB] W&B log_artifact[{run_id}]: {artifact_path}")
            self._emit(
                "artifact_logged",
                {
                    "platform": MLPlatform.WANDB.value,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )
            # Actual: wandb.save(artifact_path)

    def end_run(self, run_id: str, status: str = "completed"):
        """End W&B run"""
        if run_id in self.active_runs:
            st = RunStatus(status) if status in RunStatus._value2member_map_ else RunStatus.UNKNOWN
            run = self.active_runs[run_id]
            run.status = st
            run.end_time = _now_utc()
            logging.info(f"[STUB] W&B end_run[{run_id}] -> {st.value}")
            self._emit("run_ended", {"platform": MLPlatform.WANDB.value, **run.to_dict()})
            # Actual: wandb.finish()

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        if run_id in self.active_runs:
            self.active_runs[run_id].tags.update(tags)
            logging.info(f"[STUB] W&B set_tags[{run_id}]: {list(tags.keys())}")
            self._emit(
                "tags_set", {"platform": MLPlatform.WANDB.value, "run_id": run_id, "tags": tags}
            )

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self.active_runs.get(run_id)


# --------- SageMaker ---------


class SageMakerIntegration(MLPlatformInterface):
    """
    AWS SageMaker integration (stub)

    Note: Requires boto3 and sagemaker libraries
    This is a stub that logs intent to interact with SageMaker.
    """

    def __init__(self, region: str = "us-east-1", role: Optional[str] = None):
        """
        Initialize SageMaker integration

        Args:
            region: AWS region
            role: IAM role ARN for SageMaker
        """
        super().__init__()
        self.region = region
        self.role = role
        self.active_runs: Dict[str, ExperimentRun] = {}

        logging.info(f"SageMaker integration initialized (stub) - Region: {region}")

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start SageMaker training job"""
        import uuid

        run_id = f"sm-{uuid.uuid4().hex[:8]}"

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters={},
            metrics={},
            tags={"run_name": run_name} if run_name else {},
        )

        self.active_runs[run_id] = run
        logging.info(f"[STUB] SageMaker start_job: {run_id} (experiment: {experiment_name})")
        self._emit("run_started", {"platform": MLPlatform.SAGEMAKER.value, **run.to_dict()})
        # Actual: Create SageMaker training job
        return run_id

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log hyperparameters to SageMaker"""
        if run_id in self.active_runs:
            sanitized = _sanitize_keys(parameters)
            self.active_runs[run_id].parameters.update(sanitized)
            logging.info(f"[STUB] SageMaker log_hyperparameters[{run_id}]")
            self._emit(
                "params_logged",
                {"platform": MLPlatform.SAGEMAKER.value, "run_id": run_id, "parameters": sanitized},
            )

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to CloudWatch (via SageMaker)"""
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            logging.info(f"[STUB] SageMaker log_metrics[{run_id}] (step={step})")
            self._emit(
                "metrics_logged",
                {
                    "platform": MLPlatform.SAGEMAKER.value,
                    "run_id": run_id,
                    "metrics": metrics,
                    "step": step,
                },
            )

    def log_artifact(self, run_id: str, artifact_path: str):
        """Upload model artifact to S3"""
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(artifact_path)
            logging.info(f"[STUB] SageMaker upload_artifact[{run_id}]: {artifact_path}")
            self._emit(
                "artifact_logged",
                {
                    "platform": MLPlatform.SAGEMAKER.value,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )

    def end_run(self, run_id: str, status: str = "completed"):
        """Complete SageMaker training job"""
        if run_id in self.active_runs:
            st = RunStatus(status) if status in RunStatus._value2member_map_ else RunStatus.UNKNOWN
            run = self.active_runs[run_id]
            run.status = st
            run.end_time = _now_utc()
            logging.info(f"[STUB] SageMaker end_job[{run_id}] -> {st.value}")
            self._emit("run_ended", {"platform": MLPlatform.SAGEMAKER.value, **run.to_dict()})


# --------- Azure ML ---------


class AzureMLIntegration(MLPlatformInterface):
    """
    Azure ML integration (stub)

    Note: Requires azure-ai-ml or azureml-sdk libraries.
    This is a stub that logs intent to interact with Azure ML.
    """

    def __init__(self, workspace: Optional[str] = None, subscription_id: Optional[str] = None):
        """
        Initialize Azure ML integration

        Args:
            workspace: Azure ML workspace name or resource ID
            subscription_id: Azure subscription ID (optional)
        """
        super().__init__()
        self.workspace = workspace or "default-workspace"
        self.subscription_id = subscription_id
        self.active_runs: Dict[str, ExperimentRun] = {}

        logging.info(
            f"Azure ML integration initialized (stub) - Workspace: {self.workspace}, "
            f"Subscription: {self.subscription_id or 'env/default'}"
        )

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        import uuid

        run_id = f"az-{uuid.uuid4().hex[:8]}"
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters={},
            metrics={},
            tags={"run_name": run_name} if run_name else {},
        )
        self.active_runs[run_id] = run
        logging.info(f"[STUB] AzureML start_run: {run_id} (experiment: {experiment_name})")
        self._emit("run_started", {"platform": MLPlatform.AZURE_ML.value, **run.to_dict()})
        return run_id

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        if run_id in self.active_runs:
            sanitized = _sanitize_keys(parameters)
            self.active_runs[run_id].parameters.update(sanitized)
            logging.info(f"[STUB] AzureML log_parameters[{run_id}]")
            self._emit(
                "params_logged",
                {"platform": MLPlatform.AZURE_ML.value, "run_id": run_id, "parameters": sanitized},
            )

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            logging.info(f"[STUB] AzureML log_metrics[{run_id}] (step={step})")
            self._emit(
                "metrics_logged",
                {
                    "platform": MLPlatform.AZURE_ML.value,
                    "run_id": run_id,
                    "metrics": metrics,
                    "step": step,
                },
            )

    def log_artifact(self, run_id: str, artifact_path: str):
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(artifact_path)
            logging.info(f"[STUB] AzureML log_artifact[{run_id}]: {artifact_path}")
            self._emit(
                "artifact_logged",
                {
                    "platform": MLPlatform.AZURE_ML.value,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )

    def end_run(self, run_id: str, status: str = "completed"):
        if run_id in self.active_runs:
            st = RunStatus(status) if status in RunStatus._value2member_map_ else RunStatus.UNKNOWN
            run = self.active_runs[run_id]
            run.status = st
            run.end_time = _now_utc()
            logging.info(f"[STUB] AzureML end_run[{run_id}] -> {st.value}")
            self._emit("run_ended", {"platform": MLPlatform.AZURE_ML.value, **run.to_dict()})

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self.active_runs.get(run_id)


# --------- In-memory / Custom ---------


class InMemoryIntegration(MLPlatformInterface):
    """
    Simple in-memory platform useful for testing or custom pipelines.
    Stores all run metadata locally and emits structured events.
    """

    def __init__(self, name: str = "in_memory"):
        super().__init__()
        self.name = name
        self.active_runs: Dict[str, ExperimentRun] = {}

    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        import uuid

        run_id = f"mem-{uuid.uuid4().hex[:8]}"
        run = ExperimentRun(
            run_id=run_id,
            experiment_name=experiment_name,
            parameters={},
            metrics={},
            tags={"run_name": run_name} if run_name else {},
        )
        self.active_runs[run_id] = run
        logging.info(f"[INMEM] start_run: {run_id} (experiment: {experiment_name})")
        self._emit("run_started", {"platform": MLPlatform.IN_MEMORY.value, **run.to_dict()})
        return run_id

    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        if run_id in self.active_runs:
            sanitized = _sanitize_keys(parameters)
            self.active_runs[run_id].parameters.update(sanitized)
            logging.debug(f"[INMEM] log_parameters[{run_id}]: {sanitized}")
            self._emit(
                "params_logged",
                {"platform": MLPlatform.IN_MEMORY.value, "run_id": run_id, "parameters": sanitized},
            )

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            logging.debug(f"[INMEM] log_metrics[{run_id}]: {metrics} (step={step})")
            self._emit(
                "metrics_logged",
                {
                    "platform": MLPlatform.IN_MEMORY.value,
                    "run_id": run_id,
                    "metrics": metrics,
                    "step": step,
                },
            )

    def log_artifact(self, run_id: str, artifact_path: str):
        if run_id in self.active_runs:
            self.active_runs[run_id].artifacts.append(artifact_path)
            logging.debug(f"[INMEM] log_artifact[{run_id}]: {artifact_path}")
            self._emit(
                "artifact_logged",
                {
                    "platform": MLPlatform.IN_MEMORY.value,
                    "run_id": run_id,
                    "artifact_path": artifact_path,
                },
            )

    def end_run(self, run_id: str, status: str = "completed"):
        if run_id in self.active_runs:
            st = RunStatus(status) if status in RunStatus._value2member_map_ else RunStatus.UNKNOWN
            run = self.active_runs[run_id]
            run.status = st
            run.end_time = _now_utc()
            logging.info(f"[INMEM] end_run[{run_id}] -> {st.value}")
            self._emit("run_ended", {"platform": MLPlatform.IN_MEMORY.value, **run.to_dict()})

    def set_tags(self, run_id: str, tags: Dict[str, str]) -> None:
        if run_id in self.active_runs:
            self.active_runs[run_id].tags.update(tags)
            self._emit(
                "tags_set", {"platform": MLPlatform.IN_MEMORY.value, "run_id": run_id, "tags": tags}
            )

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        return self.active_runs.get(run_id)


# --------- Manager ---------


class MLPlatformManager:
    """
    Manage multiple ML platform integrations

    Example:
        manager = MLPlatformManager()
        manager.add_platform("mlflow", MLflowIntegration("http://localhost:5000"))
        manager.add_platform("wandb", WandBIntegration("my-project"))

        # Start runs on all platforms
        run_ids = manager.start_run_all("my_experiment")

        # Log to all platforms
        manager.log_metrics_all(run_ids, {"accuracy": 0.95})
    """

    def __init__(self):
        self.platforms: Dict[str, MLPlatformInterface] = {}

    def add_platform(self, name: str, platform: MLPlatformInterface):
        """Add ML platform"""
        self.platforms[name] = platform

    def remove_platform(self, name: str) -> None:
        """Remove an ML platform by name"""
        self.platforms.pop(name, None)

    def start_run_all(self, experiment_name: str, run_name: Optional[str] = None) -> Dict[str, str]:
        """Start run on all platforms"""
        run_ids: Dict[str, str] = {}
        for name, platform in self.platforms.items():
            try:
                run_id = platform.start_run(experiment_name, run_name)
                run_ids[name] = run_id
            except Exception as e:
                logging.error(f"Failed to start run on {name}: {e}")
        return run_ids

    def log_parameters_all(self, run_ids: Dict[str, str], parameters: Dict[str, Any]):
        """Log parameters to all platforms"""
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                continue
            try:
                platform.log_parameters(run_id, parameters)
            except Exception as e:
                logging.error(f"Failed to log parameters to {name}: {e}")

    def log_metrics_all(
        self, run_ids: Dict[str, str], metrics: Dict[str, float], step: Optional[int] = None
    ):
        """Log metrics to all platforms"""
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                continue
            try:
                platform.log_metrics(run_id, metrics, step)
            except Exception as e:
                logging.error(f"Failed to log metrics to {name}: {e}")

    def log_artifact_all(self, run_ids: Dict[str, str], artifact_path: str):
        """Log artifact to all platforms"""
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                continue
            try:
                platform.log_artifact(run_id, artifact_path)
            except Exception as e:
                logging.error(f"Failed to log artifact to {name}: {e}")

    def set_tags_all(self, run_ids: Dict[str, str], tags: Dict[str, str]):
        """Set tags on all platforms"""
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                continue
            try:
                platform.set_tags(run_id, tags)
            except Exception as e:
                logging.error(f"Failed to set tags on {name}: {e}")

    def end_run_all(self, run_ids: Dict[str, str], status: str = "completed"):
        """End run on all platforms"""
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                continue
            try:
                platform.end_run(run_id, status)
            except Exception as e:
                logging.error(f"Failed to end run on {name}: {e}")

    def get_summaries(self, run_ids: Dict[str, str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get structured run summaries (if supported by the platform).
        Returns dict of name -> run_dict or None if not available.
        """
        summaries: Dict[str, Optional[Dict[str, Any]]] = {}
        for name, run_id in run_ids.items():
            platform = self.platforms.get(name)
            if not platform:
                summaries[name] = None
                continue
            try:
                run = platform.get_run(run_id)
                summaries[name] = run.to_dict() if run else None
            except Exception as e:
                logging.error(f"Failed to get run summary from {name}: {e}")
                summaries[name] = None
        return summaries

    def run_all_context(self, experiment_name: str, run_name: Optional[str] = None):
        """
        Context manager to ensure runs are closed.
        Usage:
            with manager.run_all_context("exp", "run") as run_ids:
                manager.log_metrics_all(run_ids, {...})
        """
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            run_ids = self.start_run_all(experiment_name, run_name)
            try:
                yield run_ids
                self.end_run_all(run_ids, status=RunStatus.COMPLETED.value)
            except Exception:
                logging.exception("Exception inside run_all_context; marking runs as failed")
                self.end_run_all(run_ids, status=RunStatus.FAILED.value)
                raise

        return _ctx()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    print("ML platform integrations initialized (Nethical-enhanced stubs)")

    # Create manager with multiple platforms
    manager = MLPlatformManager()
    manager.add_platform("mlflow", MLflowIntegration())
    manager.add_platform("wandb", WandBIntegration("demo-project"))
    manager.add_platform("sagemaker", SageMakerIntegration(region="us-east-1"))
    manager.add_platform("azureml", AzureMLIntegration(workspace="demo-ws"))
    manager.add_platform("inmem", InMemoryIntegration())

    # Add a simple event handler to print structured events
    def print_event(evt: str, payload: Dict[str, Any]):
        print(f"[event:{evt}] {json.dumps(payload)}")

    for p in manager.platforms.values():
        p.add_event_handler(print_event)

    # Start experiment
    run_ids = manager.start_run_all("demo_experiment", "demo_run")
    print(f"Started runs: {run_ids}")

    # Log params, metrics, artifact
    manager.log_parameters_all(
        run_ids, {"lr": 0.001, "batch_size": 64, "api_key": "should_be_masked"}
    )
    manager.log_metrics_all(run_ids, {"accuracy": 0.95, "loss": 0.05}, step=1)
    manager.log_artifact_all(run_ids, "artifacts/model.pt")
    manager.set_tags_all(run_ids, {"stage": "demo", "owner": "nethical"})

    # End runs
    manager.end_run_all(run_ids)

    # Print summaries (in-memory only returns non-None for stubs that store state)
    print("Summaries:", json.dumps(manager.get_summaries(run_ids), indent=2))
    print("ML platform integration demo complete")
