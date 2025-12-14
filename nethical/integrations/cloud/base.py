"""Base class for cloud ML platform integrations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RunStatus(Enum):
    """ML run status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    UNKNOWN = "unknown"


@dataclass
class ExperimentRun:
    """ML experiment run metadata."""
    run_id: str
    experiment_name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    tags: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None


class CloudMLProvider(ABC):
    """Base class for cloud ML platform integrations."""
    
    @abstractmethod
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a new experiment run.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Optional run name
            
        Returns:
            Run ID
        """
        pass
    
    @abstractmethod
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Log experiment parameters.
        
        Args:
            run_id: Run identifier
            parameters: Parameters to log
        """
        pass
    
    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for a run.
        
        Args:
            run_id: Run identifier
            metrics: Metrics to log
            step: Optional step/iteration number
        """
        pass
    
    @abstractmethod
    def end_run(self, run_id: str, status: str = "completed") -> None:
        """End an experiment run.
        
        Args:
            run_id: Run identifier
            status: Run status (completed, failed, etc.)
        """
        pass
    
    def log_artifact(self, run_id: str, artifact_path: str) -> None:
        """Log an artifact (optional).
        
        Args:
            run_id: Run identifier
            artifact_path: Path to artifact
        """
        pass
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run metadata (optional).
        
        Args:
            run_id: Run identifier
            
        Returns:
            ExperimentRun or None
        """
        return None
