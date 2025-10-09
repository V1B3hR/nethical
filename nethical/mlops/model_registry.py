"""
MLOps Model Registry Module

This module provides comprehensive model management functionality including
version control, metadata tracking, and model promotion workflows.

Features:
- Model versioning and lineage tracking
- Model metadata and performance metrics storage
- Model promotion workflow (dev -> staging -> production)
- Model comparison and rollback capabilities
- Integration with model serving systems
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'loss': self.loss,
            'custom_metrics': self.custom_metrics
        }


@dataclass
class ModelVersion:
    """Model version metadata"""
    name: str
    version: str
    model_path: str
    stage: ModelStage
    status: ModelStatus = ModelStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_dataset: Optional[str] = None
    framework: Optional[str] = None
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def version_id(self) -> str:
        """Unique version identifier"""
        return f"{self.name}:{self.version}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'version_id': self.version_id,
            'model_path': self.model_path,
            'stage': self.stage.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description,
            'tags': self.tags,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'hyperparameters': self.hyperparameters,
            'training_dataset': self.training_dataset,
            'framework': self.framework,
            'parent_version': self.parent_version,
            'metadata': self.metadata
        }


class ModelRegistry:
    """
    Model registry for managing ML models with versioning and lifecycle management
    """
    
    def __init__(self, registry_dir: Union[str, Path] = "models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage directories by stage
        self.dev_dir = self.registry_dir / "development"
        self.staging_dir = self.registry_dir / "staging"
        self.production_dir = self.registry_dir / "production"
        self.archived_dir = self.registry_dir / "archived"
        self.metadata_dir = self.registry_dir / "metadata"
        
        for d in [self.dev_dir, self.staging_dir, self.production_dir, 
                  self.archived_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.models: Dict[str, ModelVersion] = {}
        self._load_registry()
    
    def register_model(self,
                       name: str,
                       version: str,
                       model_path: Union[str, Path],
                       stage: ModelStage = ModelStage.DEVELOPMENT,
                       metrics: Optional[ModelMetrics] = None,
                       hyperparameters: Optional[Dict[str, Any]] = None,
                       description: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       **kwargs) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            name: Model name
            version: Version identifier
            model_path: Path to model artifacts
            stage: Model stage (default: development)
            metrics: Model performance metrics
            hyperparameters: Training hyperparameters
            description: Model description
            tags: Tags for categorization
            **kwargs: Additional metadata
            
        Returns:
            ModelVersion object
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model version
        model_version = ModelVersion(
            name=name,
            version=version,
            model_path=str(model_path),
            stage=stage,
            metrics=metrics,
            hyperparameters=hyperparameters or {},
            description=description,
            tags=tags or [],
            metadata=kwargs
        )
        
        # Copy model to appropriate stage directory
        stage_dir = self._get_stage_dir(stage)
        dest_path = stage_dir / f"{name}_v{version}{model_path.suffix}"
        shutil.copy2(model_path, dest_path)
        model_version.model_path = str(dest_path)
        
        # Save metadata
        self.models[model_version.version_id] = model_version
        self._save_metadata(model_version)
        
        print(f"[INFO] Registered model {model_version.version_id} in {stage.value}")
        return model_version
    
    def promote_model(self, 
                      version_id: str,
                      to_stage: ModelStage,
                      validate: bool = True) -> bool:
        """
        Promote model to a higher stage
        
        Args:
            version_id: Model version ID (name:version)
            to_stage: Target stage
            validate: Whether to validate before promotion
            
        Returns:
            Success status
        """
        if version_id not in self.models:
            raise ValueError(f"Model version not found: {version_id}")
        
        model = self.models[version_id]
        
        # Validation checks
        if validate:
            if to_stage == ModelStage.PRODUCTION:
                if not model.metrics:
                    print("[WARNING] Promoting to production without metrics")
                if model.stage != ModelStage.STAGING:
                    print("[WARNING] Skipping staging for production deployment")
        
        # Move model file
        old_path = Path(model.model_path)
        new_stage_dir = self._get_stage_dir(to_stage)
        new_path = new_stage_dir / old_path.name
        
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            model.model_path = str(new_path)
        
        # Update stage
        model.stage = to_stage
        self._save_metadata(model)
        
        print(f"[INFO] Promoted {version_id} to {to_stage.value}")
        return True
    
    def get_model(self, 
                  name: Optional[str] = None,
                  version: Optional[str] = None,
                  stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """
        Get model by name, version, or stage
        
        Args:
            name: Model name
            version: Specific version
            stage: Model stage
            
        Returns:
            ModelVersion or None
        """
        if name and version:
            version_id = f"{name}:{version}"
            return self.models.get(version_id)
        
        # Filter by criteria
        candidates = list(self.models.values())
        
        if name:
            candidates = [m for m in candidates if m.name == name]
        
        if stage:
            candidates = [m for m in candidates if m.stage == stage]
        
        if not candidates:
            return None
        
        # Return latest version
        return sorted(candidates, key=lambda m: m.created_at, reverse=True)[0]
    
    def list_models(self, 
                    name: Optional[str] = None,
                    stage: Optional[ModelStage] = None,
                    status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """List models with optional filtering"""
        models = list(self.models.values())
        
        if name:
            models = [m for m in models if m.name == name]
        if stage:
            models = [m for m in models if m.stage == stage]
        if status:
            models = [m for m in models if m.status == status]
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def compare_models(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        if version_id1 not in self.models or version_id2 not in self.models:
            raise ValueError("One or both model versions not found")
        
        model1 = self.models[version_id1]
        model2 = self.models[version_id2]
        
        comparison = {
            'model1': model1.to_dict(),
            'model2': model2.to_dict(),
            'metrics_comparison': {}
        }
        
        if model1.metrics and model2.metrics:
            m1_dict = model1.metrics.to_dict()
            m2_dict = model2.metrics.to_dict()
            
            for key in m1_dict:
                if key != 'custom_metrics' and m1_dict[key] is not None and m2_dict[key] is not None:
                    comparison['metrics_comparison'][key] = {
                        'model1': m1_dict[key],
                        'model2': m2_dict[key],
                        'diff': m2_dict[key] - m1_dict[key]
                    }
        
        return comparison
    
    def archive_model(self, version_id: str) -> bool:
        """Archive a model version"""
        if version_id not in self.models:
            raise ValueError(f"Model version not found: {version_id}")
        
        model = self.models[version_id]
        return self.promote_model(version_id, ModelStage.ARCHIVED, validate=False)
    
    def delete_model(self, version_id: str, force: bool = False) -> bool:
        """
        Delete a model version
        
        Args:
            version_id: Model version ID
            force: Force delete even if in production
            
        Returns:
            Success status
        """
        if version_id not in self.models:
            raise ValueError(f"Model version not found: {version_id}")
        
        model = self.models[version_id]
        
        if model.stage == ModelStage.PRODUCTION and not force:
            raise ValueError("Cannot delete production model without force=True")
        
        # Delete model file
        model_path = Path(model.model_path)
        if model_path.exists():
            model_path.unlink()
        
        # Delete metadata
        metadata_file = self.metadata_dir / f"{version_id.replace(':', '_')}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Remove from registry
        del self.models[version_id]
        
        print(f"[INFO] Deleted model {version_id}")
        return True
    
    def _get_stage_dir(self, stage: ModelStage) -> Path:
        """Get directory for model stage"""
        stage_map = {
            ModelStage.DEVELOPMENT: self.dev_dir,
            ModelStage.STAGING: self.staging_dir,
            ModelStage.PRODUCTION: self.production_dir,
            ModelStage.ARCHIVED: self.archived_dir
        }
        return stage_map[stage]
    
    def _save_metadata(self, model: ModelVersion):
        """Save model metadata"""
        filename = f"{model.version_id.replace(':', '_')}.json"
        filepath = self.metadata_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(model.to_dict(), f, indent=2)
    
    def _load_registry(self):
        """Load existing model registry"""
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    # Note: Full deserialization would reconstruct ModelVersion objects
                    # For now, just log that we found them
                    print(f"[DEBUG] Found model: {data.get('version_id')}")
            except Exception as e:
                print(f"[ERROR] Failed to load {metadata_file}: {e}")


# Legacy functions for backward compatibility
MODEL_DIR = Path("models")
CANDIDATES = MODEL_DIR / "candidates"
CURRENT = MODEL_DIR / "current"


def promote_model(model_filename):
    """Legacy function - use ModelRegistry.promote_model() instead"""
    src = CANDIDATES / model_filename
    dst = CURRENT / model_filename
    CURRENT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[INFO] Promoted model {model_filename} to production.")


def list_models():
    """Legacy function - use ModelRegistry.list_models() instead"""
    print("[INFO] Candidate models:")
    if CANDIDATES.exists():
        for f in CANDIDATES.glob("*.json"):
            print(" -", f.name)
    print("[INFO] Current production models:")
    if CURRENT.exists():
        for f in CURRENT.glob("*.json"):
            print(" -", f.name)


if __name__ == "__main__":
    # Demo usage
    registry = ModelRegistry()
    print("Model registry initialized")
    print(f"Registry directory: {registry.registry_dir}")
