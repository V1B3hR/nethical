"""
MLOps Model Registry Module (Enhanced for nethical system)

This enhanced registry provides comprehensive model lifecycle management aligned
with production-grade MLOps practices used by the nethical system.

Key Features:
- Robust model versioning (supports manual + automatic semantic versioning)
- Complete metadata & lineage tracking (parent/children relationships)
- Promotion workflow enforcement (development -> staging -> production -> archived)
- Stage transition policy & validation hooks
- Rich model metrics management & incremental updates
- Artifact management (single file or directory with recursive hashing)
- Environment + framework + signature capture
- Model comparison and rollback capabilities
- Search & filtering by tag, stage, status, metrics thresholds & metadata keys
- Deprecation workflow & rollback helpers
- Atomic, fault-tolerant metadata persistence + full deserialization on load
- Event callback system (registered, promoted, metrics_updated, deleted, etc.)
- Thread-safety (coarse-grained registry lock)
- Structured logging (replace prints)
- Backward compatible legacy helper functions retained

NOTE:
This module is intentionally dependency-light. If future integration with
remote object stores (S3, GCS, etc.) is needed, abstract the artifact I/O layer.

"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging
import re
import sys
import platform

# -----------------------------------------------------------------------------
# Logging Configuration (can be overridden by application)
# -----------------------------------------------------------------------------
logger = logging.getLogger("nethical.model_registry")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model status."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class ModelRegistryError(Exception):
    """Base exception for registry errors."""


class VersionConflictError(ModelRegistryError):
    """Raised when registering duplicate or incompatible version."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when model/version is not found."""


class StageTransitionError(ModelRegistryError):
    """Raised when an invalid stage transition is attempted."""


class ValidationError(ModelRegistryError):
    """Raised when validation rules fail."""


# -----------------------------------------------------------------------------
# Metrics & Data Classes
# -----------------------------------------------------------------------------
@dataclass
class ModelMetrics:
    """Model performance metrics (extend as needed)."""

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    loss: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "loss": self.loss,
            "custom_metrics": self.custom_metrics,
        }

    def update(self, **metrics: float):
        for k, v in metrics.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.custom_metrics[k] = v


@dataclass
class ArtifactInfo:
    """Captured information about stored artifacts."""

    path: str
    is_dir: bool
    file_count: int
    size_bytes: int
    hash: Optional[str] = None  # Combined hash if directory or file
    file_hashes: Dict[str, str] = field(default_factory=dict)  # relative path -> hash


@dataclass
class ModelVersion:
    """Model version metadata with lineage and environment metadata."""

    name: str
    version: str
    model_path: str
    stage: ModelStage
    status: ModelStatus = ModelStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_dataset: Optional[str] = None
    framework: Optional[str] = None
    framework_version: Optional[str] = None
    parent_version: Optional[str] = None
    children: List[str] = field(default_factory=list)  # list of version_ids referencing descendants
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifact: Optional[ArtifactInfo] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    model_signature: Optional[Dict[str, Any]] = None  # e.g., input/output schema
    deprecated_at: Optional[datetime] = None
    rollback_source: Optional[str] = None  # version_id we rolled back from if applicable

    @property
    def version_id(self) -> str:
        return f"{self.name}:{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "version_id": self.version_id,
            "model_path": self.model_path,
            "stage": self.stage.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "hyperparameters": self.hyperparameters,
            "training_dataset": self.training_dataset,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "parent_version": self.parent_version,
            "children": self.children,
            "metadata": self.metadata,
            "artifact": asdict(self.artifact) if self.artifact else None,
            "environment": self.environment,
            "model_signature": self.model_signature,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "rollback_source": self.rollback_source,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelVersion":
        metrics = data.get("metrics")
        mv_metrics = ModelMetrics(**metrics) if metrics else None
        artifact_data = data.get("artifact")
        artifact = ArtifactInfo(**artifact_data) if artifact_data else None
        return ModelVersion(
            name=data["name"],
            version=data["version"],
            model_path=data["model_path"],
            stage=ModelStage(data["stage"]),
            status=ModelStatus(data.get("status", ModelStatus.ACTIVE.value)),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            description=data.get("description"),
            tags=data.get("tags", []),
            metrics=mv_metrics,
            hyperparameters=data.get("hyperparameters", {}),
            training_dataset=data.get("training_dataset"),
            framework=data.get("framework"),
            framework_version=data.get("framework_version"),
            parent_version=data.get("parent_version"),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
            artifact=artifact,
            environment=data.get("environment", {}),
            model_signature=data.get("model_signature"),
            deprecated_at=(
                datetime.fromisoformat(data["deprecated_at"]) if data.get("deprecated_at") else None
            ),
            rollback_source=data.get("rollback_source"),
        )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+].+)?$")


def is_semver(v: str) -> bool:
    return bool(SEMVER_RE.match(v))


def increment_semver(latest: str) -> str:
    """Increment patch component of semantic version string."""
    if not is_semver(latest):
        raise ValueError(f"Not a semantic version: {latest}")
    major, minor, patch = latest.split(".")[:3]
    return f"{major}.{minor}.{int(patch) + 1}"


def compute_file_hash(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_hash(dir_path: Path) -> Tuple[str, Dict[str, str], int, int]:
    file_hashes: Dict[str, str] = {}
    total_size = 0
    count = 0
    h = hashlib.sha256()
    for file in sorted(dir_path.rglob("*")):
        if file.is_file():
            rel = file.relative_to(dir_path).as_posix()
            fh = compute_file_hash(file)
            file_hashes[rel] = fh
            h.update(rel.encode())
            h.update(fh.encode())
            total_size += file.stat().st_size
            count += 1
    return h.hexdigest(), file_hashes, count, total_size


# -----------------------------------------------------------------------------
# Model Registry
# -----------------------------------------------------------------------------
class ModelRegistry:
    """
    Model registry for managing ML models with versioning and lifecycle management.

    Thread-safe for basic operations (metadata + file moves). Not safe for
    concurrent writes to same model version (callers must coordinate).
    """

    DEFAULT_STAGE_TRANSITIONS: Dict[ModelStage, Tuple[ModelStage, ...]] = {
        ModelStage.DEVELOPMENT: (ModelStage.STAGING, ModelStage.ARCHIVED),
        ModelStage.STAGING: (ModelStage.PRODUCTION, ModelStage.ARCHIVED),
        ModelStage.PRODUCTION: (ModelStage.ARCHIVED,),
        ModelStage.ARCHIVED: (),
    }

    def __init__(
        self,
        registry_dir: Union[str, Path] = "models",
        enforce_transitions: bool = True,
        auto_semver: bool = True,
        capture_environment: bool = True,
    ):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.dev_dir = self.registry_dir / "development"
        self.staging_dir = self.registry_dir / "staging"
        self.production_dir = self.registry_dir / "production"
        self.archived_dir = self.registry_dir / "archived"
        self.metadata_dir = self.registry_dir / "metadata"

        for d in [
            self.dev_dir,
            self.staging_dir,
            self.production_dir,
            self.archived_dir,
            self.metadata_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, ModelVersion] = {}
        self._lock = threading.RLock()
        self._callbacks: Dict[str, List[Callable[[ModelVersion], None]]] = {}
        self.enforce_transitions = enforce_transitions
        self.auto_semver = auto_semver
        self.capture_environment = capture_environment

        self._load_registry()
        logger.info(
            "ModelRegistry initialized at %s (loaded=%d)", self.registry_dir, len(self.models)
        )

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------
    def register_callback(self, event: str, fn: Callable[[ModelVersion], None]):
        with self._lock:
            self._callbacks.setdefault(event, []).append(fn)

    def _emit(self, event: str, model: ModelVersion):
        for fn in self._callbacks.get(event, []):
            try:
                fn(model)
            except Exception as e:
                logger.exception("Callback error for event '%s': %s", event, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_model(
        self,
        name: str,
        version: Optional[str],
        model_path: Union[str, Path],
        stage: ModelStage = ModelStage.DEVELOPMENT,
        metrics: Optional[ModelMetrics] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        parent_version: Optional[str] = None,
        model_signature: Optional[Dict[str, Any]] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        training_dataset: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        allow_overwrite: bool = False,
        **extra,
    ) -> ModelVersion:
        """
        Register a new model version.

        If version is None and auto_semver is True:
            - Determine latest semantic version for this model
            - Increment patch

        If version provided but exists, raises VersionConflictError unless allow_overwrite=True.

        Args:
            name: Model name
            version: Version string (semantic x.y.z recommended)
            model_path: File or directory of model artifact(s)
            stage: Initial stage (default: development)
            metrics: Performance metrics
            hyperparameters: Hyperparams used in training
            description: Human readable description
            tags: List of tags
            created_by: Creator identifier
            parent_version: Parent version_id for lineage tracking
            model_signature: I/O schema or contract
            framework / framework_version
            training_dataset: Identifier for dataset used
            metadata: Additional arbitrary metadata
            allow_overwrite: Allow replacing existing version metadata & artifact
            **extra: Additional metadata appended under 'extra'

        Returns:
            ModelVersion
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        with self._lock:
            if version is None and self.auto_semver:
                version = self._next_semver(name)
            elif version is None:
                raise ValidationError("Version must be supplied when auto_semver=False.")

            version_id = f"{name}:{version}"
            if version_id in self.models and not allow_overwrite:
                raise VersionConflictError(f"Model version already exists: {version_id}")

            # Copy artifact(s)
            stage_dir = self._get_stage_dir(stage)
            dest_dir = stage_dir / name
            dest_dir.mkdir(parents=True, exist_ok=True)

            if model_path.is_dir():
                # Directory copy
                dest_path = dest_dir / f"v{version}"
                if dest_path.exists():
                    if allow_overwrite:
                        shutil.rmtree(dest_path)
                    else:
                        raise VersionConflictError(f"Artifact directory exists: {dest_path}")
                shutil.copytree(model_path, dest_path)
                directory_hash, file_hashes, file_count, size_bytes = compute_directory_hash(
                    dest_path
                )
                artifact_info = ArtifactInfo(
                    path=str(dest_path),
                    is_dir=True,
                    file_count=file_count,
                    size_bytes=size_bytes,
                    hash=directory_hash,
                    file_hashes=file_hashes,
                )
                final_artifact_path = dest_path
            else:
                # Single file copy
                ext = "".join(model_path.suffixes)
                dest_file = dest_dir / f"{name}_v{version}{ext}"
                shutil.copy2(model_path, dest_file)
                file_hash = compute_file_hash(dest_file)
                artifact_info = ArtifactInfo(
                    path=str(dest_file),
                    is_dir=False,
                    file_count=1,
                    size_bytes=dest_file.stat().st_size,
                    hash=file_hash,
                    file_hashes={Path(dest_file).name: file_hash},
                )
                final_artifact_path = dest_file

            env_meta = self._capture_environment() if self.capture_environment else {}

            model_version = ModelVersion(
                name=name,
                version=version,
                model_path=str(final_artifact_path),
                stage=stage,
                metrics=metrics,
                hyperparameters=hyperparameters or {},
                description=description,
                tags=sorted(set(tags or [])),
                created_by=created_by,
                parent_version=parent_version,
                model_signature=model_signature,
                framework=framework,
                framework_version=framework_version,
                training_dataset=training_dataset,
                metadata={**(metadata or {}), "extra": extra} if extra else (metadata or {}),
                artifact=artifact_info,
                environment=env_meta,
            )

            # Link lineage
            if parent_version:
                parent = self.models.get(parent_version)
                if parent:
                    if model_version.version_id not in parent.children:
                        parent.children.append(model_version.version_id)
                        self._save_metadata(parent)

            self.models[model_version.version_id] = model_version
            self._save_metadata(model_version)
            logger.info("Registered model %s in stage '%s'", model_version.version_id, stage.value)
            self._emit("registered", model_version)
            return model_version

    def promote_model(
        self,
        version_id: str,
        to_stage: ModelStage,
        validate: bool = True,
        force: bool = False,
    ) -> bool:
        """
        Promote a model to a new stage with optional validation and policy enforcement.

        Args:
            version_id: name:version
            to_stage: destination stage
            validate: run built-in validations
            force: bypass transition policy (still moves artifacts)

        Returns:
            True on success
        """
        with self._lock:
            model = self._require_model(version_id)
            from_stage = model.stage

            if from_stage == to_stage:
                logger.warning("Model %s already in stage %s", version_id, to_stage.value)
                return True

            if self.enforce_transitions and not force:
                allowed = self.DEFAULT_STAGE_TRANSITIONS[from_stage]
                if to_stage not in allowed:
                    raise StageTransitionError(
                        f"Invalid transition {from_stage.value} -> {to_stage.value}. "
                        f"Allowed: {[s.value for s in allowed]}"
                    )

            if validate:
                self._validate_promotion(model, to_stage)

            # Move artifact path into new stage dir (preserve sub-structure)
            old_path = Path(model.model_path)
            new_stage_dir = self._get_stage_dir(to_stage) / model.name
            new_stage_dir.mkdir(parents=True, exist_ok=True)
            if old_path.is_dir():
                new_path = new_stage_dir / old_path.name
                if new_path.exists():
                    shutil.rmtree(new_path)
                shutil.move(str(old_path), str(new_path))
            else:
                new_path = new_stage_dir / old_path.name
                shutil.move(str(old_path), str(new_path))

            model.model_path = str(new_path)
            model.stage = to_stage
            self._save_metadata(model)
            logger.info("Promoted %s: %s -> %s", version_id, from_stage.value, to_stage.value)
            self._emit("promoted", model)
            return True

    def update_metrics(self, version_id: str, **metrics: float) -> ModelVersion:
        with self._lock:
            model = self._require_model(version_id)
            if not model.metrics:
                model.metrics = ModelMetrics()
            model.metrics.update(**metrics)
            self._save_metadata(model)
            logger.info("Updated metrics for %s: %s", version_id, metrics)
            self._emit("metrics_updated", model)
            return model

    def add_tags(self, version_id: str, *tags: str) -> ModelVersion:
        with self._lock:
            model = self._require_model(version_id)
            model.tags = sorted(set(model.tags).union(tags))
            self._save_metadata(model)
            logger.info("Added tags to %s: %s", version_id, tags)
            self._emit("tags_added", model)
            return model

    def deprecate_model(self, version_id: str, reason: Optional[str] = None) -> ModelVersion:
        with self._lock:
            model = self._require_model(version_id)
            model.status = ModelStatus.DEPRECATED
            model.deprecated_at = datetime.utcnow()
            if reason:
                model.metadata.setdefault("deprecation", {})["reason"] = reason
            self._save_metadata(model)
            logger.info("Deprecated model %s (%s)", version_id, reason or "no reason")
            self._emit("deprecated", model)
            return model

    def rollback_to(
        self, target_version_id: str, new_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Create a new version based on an existing production (or other) version's artifact.
        Does NOT modify the source version.
        """
        with self._lock:
            source = self._require_model(target_version_id)
            if not Path(source.model_path).exists():
                raise FileNotFoundError(f"Source artifact missing: {source.model_path}")

            # register new model using source artifact
            name = source.name
            v = new_version
            registered = self.register_model(
                name=name,
                version=v,
                model_path=source.model_path,
                stage=ModelStage.DEVELOPMENT,
                description=f"Rollback clone of {target_version_id}",
                tags=list(set(source.tags + ["rollback"])),
                created_by="rollback",
                parent_version=source.version_id,
                model_signature=source.model_signature,
                framework=source.framework,
                framework_version=source.framework_version,
                training_dataset=source.training_dataset,
                metadata={"rollback_source": source.version_id},
                allow_overwrite=False,
            )
            registered.rollback_source = source.version_id
            self._save_metadata(registered)
            logger.info(
                "Created rollback version %s from %s", registered.version_id, target_version_id
            )
            self._emit("rollback", registered)
            return registered

    def get_model(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        latest: bool = True,
    ) -> Optional[ModelVersion]:
        with self._lock:
            if name and version:
                return self.models.get(f"{name}:{version}")
            candidates = list(self.models.values())
            if name:
                candidates = [m for m in candidates if m.name == name]
            if stage:
                candidates = [m for m in candidates if m.stage == stage]
            if not candidates:
                return None
            candidates.sort(
                key=lambda m: (m.created_at, self._version_sort_key(m.version)), reverse=True
            )
            return candidates[0] if latest else candidates

    def list_models(
        self,
        name: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[Iterable[str]] = None,
        include_archived: bool = True,
    ) -> List[ModelVersion]:
        with self._lock:
            models = list(self.models.values())
            if name:
                models = [m for m in models if m.name == name]
            if stage:
                models = [m for m in models if m.stage == stage]
            if status:
                models = [m for m in models if m.status == status]
            if tags:
                tag_set = set(tags)
                models = [m for m in models if tag_set.issubset(m.tags)]
            if not include_archived:
                models = [m for m in models if m.stage != ModelStage.ARCHIVED]
            models.sort(
                key=lambda m: (m.created_at, self._version_sort_key(m.version)), reverse=True
            )
            return models

    def search(
        self,
        metric_gt: Optional[Dict[str, float]] = None,
        metadata_contains: Optional[Dict[str, Any]] = None,
        name_pattern: Optional[str] = None,
    ) -> List[ModelVersion]:
        """
        Flexible search:
            metric_gt: dict of metric -> minimum threshold
            metadata_contains: metadata key/value must match (simple equality)
            name_pattern: regex on model name
        """
        with self._lock:
            results = list(self.models.values())
            if metric_gt:

                def metric_pass(m: ModelVersion) -> bool:
                    if not m.metrics:
                        return False
                    md = m.metrics.to_dict()
                    for k, v in metric_gt.items():
                        val = md.get(k) or (md.get("custom_metrics", {}) or {}).get(k)
                        if val is None or val <= v:
                            return False
                    return True

                results = [m for m in results if metric_pass(m)]
            if metadata_contains:

                def meta_pass(m: ModelVersion) -> bool:
                    for k, v in metadata_contains.items():
                        if m.metadata.get(k) != v:
                            return False
                    return True

                results = [m for m in results if meta_pass(m)]
            if name_pattern:
                regex = re.compile(name_pattern)
                results = [m for m in results if regex.search(m.name)]
            results.sort(
                key=lambda m: (m.created_at, self._version_sort_key(m.version)), reverse=True
            )
            return results

    def compare_models(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        with self._lock:
            m1 = self._require_model(version_id1)
            m2 = self._require_model(version_id2)
            comparison = {
                "model1": m1.to_dict(),
                "model2": m2.to_dict(),
                "metrics_comparison": {},
            }

            if m1.metrics and m2.metrics:
                d1 = m1.metrics.to_dict()
                d2 = m2.metrics.to_dict()
                keys = set(d1.keys()) | set(d2.keys())
                for k in keys:
                    if k == "custom_metrics":
                        continue
                    v1 = d1.get(k)
                    v2 = d2.get(k)
                    if v1 is not None and v2 is not None:
                        comparison["metrics_comparison"][k] = {
                            "model1": v1,
                            "model2": v2,
                            "diff": v2 - v1,
                        }
                # custom metrics
                cm1 = d1.get("custom_metrics") or {}
                cm2 = d2.get("custom_metrics") or {}
                for ck in set(cm1.keys()) | set(cm2.keys()):
                    v1 = cm1.get(ck)
                    v2 = cm2.get(ck)
                    if v1 is not None and v2 is not None:
                        comparison["metrics_comparison"][f"custom:{ck}"] = {
                            "model1": v1,
                            "model2": v2,
                            "diff": v2 - v1,
                        }
            return comparison

    def archive_model(self, version_id: str) -> bool:
        return self.promote_model(version_id, ModelStage.ARCHIVED, validate=False)

    def delete_model(self, version_id: str, force: bool = False) -> bool:
        with self._lock:
            model = self._require_model(version_id)
            if model.stage == ModelStage.PRODUCTION and not force:
                raise ValidationError("Cannot delete production model without force=True")

            # Remove artifact
            artifact_path = Path(model.model_path)
            if artifact_path.exists():
                if artifact_path.is_dir():
                    shutil.rmtree(artifact_path)
                else:
                    artifact_path.unlink()

            # Remove metadata
            metadata_file = self.metadata_dir / f"{version_id.replace(':', '_')}.json"
            if metadata_file.exists():
                metadata_file.unlink()

            # Remove lineage references
            if model.parent_version:
                parent = self.models.get(model.parent_version)
                if parent and model.version_id in parent.children:
                    parent.children.remove(model.version_id)
                    self._save_metadata(parent)

            for child_id in model.children:
                child = self.models.get(child_id)
                if child and child.parent_version == version_id:
                    child.parent_version = None
                    self._save_metadata(child)

            del self.models[version_id]
            logger.info("Deleted model %s", version_id)
            self._emit("deleted", model)
            return True

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _capture_environment(self) -> Dict[str, Any]:
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "timestamp": datetime.utcnow().isoformat(),
            # Avoid heavy pip freeze by default (toggle externally if needed)
        }

    def _require_model(self, version_id: str) -> ModelVersion:
        model = self.models.get(version_id)
        if not model:
            raise ModelNotFoundError(f"Model version not found: {version_id}")
        return model

    def _get_stage_dir(self, stage: ModelStage) -> Path:
        mapping = {
            ModelStage.DEVELOPMENT: self.dev_dir,
            ModelStage.STAGING: self.staging_dir,
            ModelStage.PRODUCTION: self.production_dir,
            ModelStage.ARCHIVED: self.archived_dir,
        }
        return mapping[stage]

    def _validate_promotion(self, model: ModelVersion, to_stage: ModelStage):
        # Example validation rules (extend as necessary).
        if to_stage == ModelStage.STAGING:
            if not model.metrics:
                logger.warning("Promoting to staging without metrics: %s", model.version_id)
        if to_stage == ModelStage.PRODUCTION:
            if not model.metrics:
                logger.warning("Promoting to production without metrics: %s", model.version_id)
            # Example threshold enforcement (customize)
            if model.metrics and model.metrics.accuracy is not None:
                if model.metrics.accuracy < 0.5:
                    raise ValidationError("Accuracy below minimal threshold for production (0.5)")

    def _save_metadata(self, model: ModelVersion):
        filename = f"{model.version_id.replace(':', '_')}.json"
        filepath = self.metadata_dir / filename
        temp_fd, temp_path = tempfile.mkstemp(dir=str(self.metadata_dir), prefix=".tmp_meta_")
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(model.to_dict(), f, indent=2, sort_keys=True)
            os.replace(temp_path, filepath)
        except Exception:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            finally:
                raise

    def _load_registry(self):
        """Load existing model metadata files into memory."""
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                model = ModelVersion.from_dict(data)
                self.models[model.version_id] = model
            except Exception as e:
                logger.error("Failed to load %s: %s", metadata_file, e)

    def export_registry(self, export_path: Union[str, Path]) -> Path:
        export_path = Path(export_path)
        payload = {
            "exported_at": datetime.utcnow().isoformat(),
            "models": {vid: mv.to_dict() for vid, mv in self.models.items()},
        }
        with open(export_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Exported registry metadata to %s", export_path)
        return export_path

    def import_registry(
        self, import_path: Union[str, Path], merge: bool = True, overwrite: bool = False
    ):
        with self._lock:
            with open(import_path) as f:
                payload = json.load(f)
            models_data = payload.get("models", {})
            for vid, data in models_data.items():
                if vid in self.models and not (merge and overwrite):
                    logger.warning("Skipping existing model during import: %s", vid)
                    continue
                model = ModelVersion.from_dict(data)
                self.models[vid] = model
                self._save_metadata(model)
            logger.info(
                "Imported registry: %d models (merge=%s overwrite=%s)",
                len(models_data),
                merge,
                overwrite,
            )

    def _next_semver(self, name: str) -> str:
        existing = [
            m.version for m in self.models.values() if m.name == name and is_semver(m.version)
        ]
        if not existing:
            return "0.1.0"
        existing.sort(key=self._version_sort_key)
        return increment_semver(existing[-1])

    @staticmethod
    def _version_sort_key(version: str):
        if is_semver(version):
            parts = version.split(".")
            return tuple(int(p) for p in parts[:3])
        return (0, 0, 0)

    # ------------------------------------------------------------------
    # Backward compatibility wrappers
    # ------------------------------------------------------------------
    def legacy_promote(self, model_filename: str):
        """Legacy simplistic promote; replaced by register/promote workflow."""
        logger.warning("legacy_promote is deprecated; use promote_model with version IDs.")
        src = CANDIDATES / model_filename
        dst = CURRENT / model_filename
        CURRENT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Legacy promoted %s", model_filename)


# -----------------------------------------------------------------------------
# Legacy constants & functions (retained for backward compatibility)
# -----------------------------------------------------------------------------
MODEL_DIR = Path("models")
CANDIDATES = MODEL_DIR / "candidates"
CURRENT = MODEL_DIR / "current"


def promote_model(model_filename):
    """Legacy function - use ModelRegistry.promote_model() instead"""
    src = CANDIDATES / model_filename
    dst = CURRENT / model_filename
    CURRENT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Promoted model %s to production (legacy)", model_filename)


def list_models():
    """Legacy function - use ModelRegistry.list_models() instead"""
    logger.info("Candidate models:")
    if CANDIDATES.exists():
        for f in CANDIDATES.glob("*.json"):
            logger.info(" - %s", f.name)
    logger.info("Current production models:")
    if CURRENT.exists():
        for f in CURRENT.glob("*.json"):
            logger.info(" - %s", f.name)


# -----------------------------------------------------------------------------
# Demo usage (can be removed or guarded)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    registry = ModelRegistry()
    logger.info("Enhanced Model registry initialized at %s", registry.registry_dir)
    # Example registration (commented out to avoid accidental execution)
    # mv = registry.register_model(
    #     name="example_model",
    #     version=None,
    #     model_path="path/to/model.pkl",
    #     description="Example registration",
    #     metrics=ModelMetrics(accuracy=0.92),
    #     tags=["baseline", "demo"]
    # )
    # registry.promote_model(mv.version_id, ModelStage.STAGING)
