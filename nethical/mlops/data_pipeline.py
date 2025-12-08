"""
MLOps Data Pipeline Module (Enhanced)

This enhanced module provides comprehensive, extensible, and production-oriented
data ingestion, validation, preprocessing, and versioning capabilities for the
Nethical system.

Key Enhancements:
- Pluggable & extensible architecture
- Improved schema handling & full schema serialization
- Robust custom exceptions with context
- Version deduplication based on checksum (idempotent ingestion)
- Lightweight lineage & manifest tracking
- Automatic schema inference with optional compatibility enforcement
- Transformation registry & composable preprocessing pipeline
- Built-in profiling & quality metrics (null counts, basic stats)
- Safe HTTP ingestion using urllib (no new dependencies)
- Support for compressed CSV (.csv.gz, .csv.zip)
- Reconstructable DataVersion objects from metadata
- Search & filtering utilities for versions
- Deterministic checksum (sorted columns + stable serialization)
- Workspace-level manifest management
- Graceful fallbacks & logging improvements
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
from urllib.request import urlopen

import pandas as pd

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("NETHICAL_PIPELINE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s"
)
logger = logging.getLogger("nethical.data_pipeline")


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class DataPipelineError(Exception):
    """Base class for pipeline exceptions."""


class DataIngestionError(DataPipelineError):
    """Raised when ingestion fails."""


class DataValidationError(DataPipelineError):
    """Raised when validation fails."""

    def __init__(self, errors: List[str], message: str = "Data validation failed"):
        self.errors = errors
        super().__init__(f"{message}: {errors}")


class DataVersionNotFoundError(DataPipelineError):
    """Raised when a requested data version does not exist."""


class SchemaCompatibilityError(DataPipelineError):
    """Raised when schema compatibility check fails."""


# -----------------------------------------------------------------------------
# Enums & Core Data Objects
# -----------------------------------------------------------------------------
class DataSource(Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    S3 = "s3"
    HTTP = "http"
    DATABASE = "database"


class DataStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"
    PROCESSED = "processed"


@dataclass
class DataSchema:
    """
    Data schema definition for validation & compatibility.

    columns: dict[column_name -> expected logical type (string form)]
    required_columns: subset of columns required to exist
    constraints: optional mapping of column -> {min, max, allowed_values, regex, nullable}
    """

    name: str
    version: str
    columns: Dict[str, str]
    required_columns: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def validate(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        errors: List[str] = []

        # Required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(missing_cols)}")

        # Column types (soft check, flexible mapping)
        for col, expected_type in self.columns.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    errors.append(
                        f"Column '{col}': expected {expected_type}, got {actual_type}"
                    )

        # Constraints
        for col, cdict in self.constraints.items():
            if col not in df.columns:
                continue
            series = df[col]
            if cdict.get("nullable") is False and series.isna().any():
                errors.append(f"Column '{col}': contains nulls but nullable=False")
            if "min" in cdict:
                try:
                    if series.min() < cdict["min"]:
                        errors.append(
                            f"Column '{col}': value below minimum {cdict['min']}"
                        )
                except Exception:
                    errors.append(
                        f"Column '{col}': failed min comparison (non-numeric?)"
                    )
            if "max" in cdict:
                try:
                    if series.max() > cdict["max"]:
                        errors.append(
                            f"Column '{col}': value above maximum {cdict['max']}"
                        )
                except Exception:
                    errors.append(
                        f"Column '{col}': failed max comparison (non-numeric?)"
                    )
            if "allowed_values" in cdict:
                invalid = set(series.dropna().unique()) - set(cdict["allowed_values"])
                if invalid:
                    errors.append(
                        f"Column '{col}': has disallowed values {sorted(list(invalid))}"
                    )

        return (len(errors) == 0, errors)

    def ensure_compatible(self, other: DataSchema, strict: bool = False) -> None:
        """
        Ensure that another schema is compatible (e.g., new ingestion matches existing schema).
        If strict: all columns & types must match exactly.
        If not strict: required columns must exist and share compatible types.
        """
        if strict:
            if set(self.columns.keys()) != set(other.columns.keys()):
                raise SchemaCompatibilityError("Strict schema mismatch: columns differ")
            for col, expected_type in self.columns.items():
                other_type = other.columns.get(col)
                if other_type and not self._type_compatible(other_type, expected_type):
                    raise SchemaCompatibilityError(
                        f"Strict mismatch on column '{col}' ({expected_type} vs {other_type})"
                    )
        else:
            missing_required = set(self.required_columns) - set(other.columns.keys())
            if missing_required:
                raise SchemaCompatibilityError(
                    f"Missing required columns: {sorted(missing_required)}"
                )
            for col in self.required_columns:
                if not self._type_compatible(other.columns[col], self.columns[col]):
                    raise SchemaCompatibilityError(
                        f"Type mismatch for required column '{col}'"
                    )

    def _type_compatible(self, actual: str, expected: str) -> bool:
        type_map = {
            "int64": {"int", "integer", "int64"},
            "float64": {"float", "float64", "double"},
            "object": {"str", "string", "object"},
            "bool": {"bool", "boolean"},
            "datetime64": {"datetime", "datetime64[ns]", "timestamp"},
        }
        # Normalize
        actual_norm = actual.lower()
        expected_norm = expected.lower()
        for canonical, aliases in type_map.items():
            if (
                canonical in actual_norm or actual_norm in aliases
            ) and expected_norm in aliases:
                return True
        # fallback partial
        return expected_norm in actual_norm or actual_norm in expected_norm

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "columns": self.columns,
            "required_columns": self.required_columns,
            "constraints": self.constraints,
            "description": self.description,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DataSchema":
        return DataSchema(
            name=d["name"],
            version=d.get("version", "1.0"),
            columns=d["columns"],
            required_columns=d.get("required_columns", []),
            constraints=d.get("constraints", {}),
            description=d.get("description"),
        )


@dataclass
class DataVersion:
    """Data version metadata with lineage & transformation tracking."""

    version_id: str
    source: DataSource
    path: str
    schema: DataSchema
    checksum: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DataStatus = DataStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    parent_version: Optional[str] = None
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    profile: Optional[Dict[str, Any]] = None  # optional data profiling snapshot

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "source": self.source.value,
            "path": self.path,
            "schema": self.schema.to_dict(),
            "checksum": self.checksum,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "status": self.status.value,
            "validation_errors": self.validation_errors,
            "parent_version": self.parent_version,
            "transformations": self.transformations,
            "profile": self.profile,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DataVersion":
        return DataVersion(
            version_id=d["version_id"],
            source=DataSource(d["source"]),
            path=d["path"],
            schema=DataSchema.from_dict(d["schema"]),
            checksum=d["checksum"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=d.get("metadata", {}),
            status=DataStatus(d.get("status", DataStatus.PENDING.value)),
            validation_errors=d.get("validation_errors", []),
            parent_version=d.get("parent_version"),
            transformations=d.get("transformations", []),
            profile=d.get("profile"),
        )


# -----------------------------------------------------------------------------
# DataPipeline
# -----------------------------------------------------------------------------
class DataPipeline:
    """
    Comprehensive data pipeline for ML operations in the Nethical system.

    Responsibilities:
      - Ingestion from supported sources
      - Validation & schema management
      - Versioning, deduplication & lineage
      - Transformation management & preprocessing
      - Metadata & profiling
    """

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        workspace_dir: Union[str, Path] = "data",
        auto_manifest: bool = True,
        strict_schema: bool = False,
        profile_on_ingest: bool = True,
    ):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.workspace / "raw"
        self.processed_dir = self.workspace / "processed"
        self.versions_dir = self.workspace / "versions"

        for d in (self.raw_dir, self.processed_dir, self.versions_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.strict_schema = strict_schema
        self.profile_on_ingest = profile_on_ingest

        self.versions: Dict[str, DataVersion] = {}
        self._transform_registry: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}

        self._load_versions()
        if auto_manifest:
            self._write_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_transformation(
        self, name: str, func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> None:
        if name in self._transform_registry:
            logger.warning(f"Overwriting transformation '{name}'")
        self._transform_registry[name] = func

    def list_transformations(self) -> List[str]:
        return sorted(self._transform_registry.keys())

    def ingest(
        self,
        source: Union[str, Path],
        source_type: DataSource = DataSource.LOCAL,
        schema: Optional[DataSchema] = None,
        validate: bool = True,
        enforce_compatibility_with: Optional[str] = None,
        allow_duplicate: bool = False,
        tag: Optional[str] = None,
    ) -> DataVersion:
        """
        Ingest data from a source with optional schema validation and compatibility enforcement.

        enforce_compatibility_with: version_id whose schema acts as contract (useful for incremental loads)
        allow_duplicate: if False, reusing identical checksum returns existing version (idempotent)
        tag: optional label stored in metadata
        """
        start = datetime.now(timezone.utc)
        logger.info(f"Ingesting data from {source} (type={source_type.value})")

        # Read data
        try:
            df = self._dispatch_read(source, source_type)
        except Exception as e:
            raise DataIngestionError(f"Failed to read source {source}: {e}") from e

        # Deterministic checksum
        checksum = self._compute_checksum(df)
        if not allow_duplicate:
            existing = self._find_version_by_checksum(checksum)
            if existing:
                logger.info(
                    f"Duplicate ingestion detected. Reusing version_id={existing.version_id}"
                )
                return existing

        # Infer schema if absent
        if schema is None:
            schema = self._infer_schema(df)

        # Enforce compatibility if requested
        if enforce_compatibility_with:
            base_version = self.get_version(enforce_compatibility_with)
            if base_version is None:
                raise DataVersionNotFoundError(
                    f"Base version '{enforce_compatibility_with}' not found for compatibility check"
                )
            try:
                base_version.schema.ensure_compatible(schema, strict=self.strict_schema)
            except SchemaCompatibilityError as e:
                raise SchemaCompatibilityError(
                    f"Schema incompatible with base version {enforce_compatibility_with}: {e}"
                ) from e

        # Compose version_id
        version_id = self._generate_version_id(checksum)

        # Save raw data
        raw_path = self.raw_dir / f"{version_id}.parquet"
        self._save_parquet(df, raw_path)

        # Prepare version object
        version = DataVersion(
            version_id=version_id,
            source=source_type,
            path=str(raw_path),
            schema=schema,
            checksum=checksum,
            metadata={
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "ingest_duration_seconds": (
                    datetime.now(timezone.utc) - start
                ).total_seconds(),
                "tag": tag,
                "source_reference": str(source),
            },
        )

        # Validate
        if validate and schema:
            is_valid, errors = schema.validate(df)
            version.status = DataStatus.VALID if is_valid else DataStatus.INVALID
            version.validation_errors = errors
            if not is_valid:
                logger.warning(f"Validation failed for {version_id}: {errors}")
        else:
            version.status = DataStatus.WARNING

        # Profile
        if self.profile_on_ingest:
            version.profile = self._basic_profile(df)

        # Persist
        self._persist_version(version)
        logger.info(
            f"Ingestion complete. version_id={version_id} status={version.status.value}"
        )
        return version

    def validate_data(
        self,
        version_id: str,
        schema: Optional[DataSchema] = None,
        raise_on_fail: bool = False,
    ) -> tuple[bool, List[str]]:
        version = self._require_version(version_id)
        df = self._read_parquet(version.path)

        target_schema = schema or version.schema
        is_valid, errors = target_schema.validate(df)
        version.status = DataStatus.VALID if is_valid else DataStatus.INVALID
        version.validation_errors = errors
        self._save_version_metadata(version)

        if not is_valid and raise_on_fail:
            raise DataValidationError(errors)
        return is_valid, errors

    def preprocess(
        self,
        version_id: str,
        transformations: Optional[
            Iterable[Union[Callable[[pd.DataFrame], pd.DataFrame], str]]
        ] = None,
        save: bool = True,
        materialize: bool = True,
        tag: Optional[str] = None,
    ) -> DataVersion:
        """
        Apply transformations and optionally produce a new processed version.

        transformations: iterable of callables OR names of registered transformations.
        materialize: if True, creates a new DataVersion (processed). If False, updates existing version's metadata only.
        """
        base_version = self._require_version(version_id)
        df = self._read_parquet(base_version.path)
        logger.info(f"Preprocessing version_id={version_id}")

        applied = []
        if transformations:
            for t in transformations:
                if isinstance(t, str):
                    if t not in self._transform_registry:
                        raise DataPipelineError(f"Transformation '{t}' not registered")
                    func = self._transform_registry[t]
                    name = t
                else:
                    func = t
                    name = getattr(func, "__name__", "anonymous_transform")
                before_cols = set(df.columns)
                df = func(df)
                after_cols = set(df.columns)
                applied.append(
                    {
                        "name": name,
                        "added_columns": sorted(list(after_cols - before_cols)),
                        "removed_columns": sorted(list(before_cols - after_cols)),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                logger.info(f"Applied transformation: {name}")

        # Save processed artifact
        processed_path = self.processed_dir / f"{version_id}_processed.parquet"
        if save:
            self._save_parquet(df, processed_path)

        if materialize:
            checksum = self._compute_checksum(df)
            new_version_id = self._generate_version_id(checksum, suffix="proc")
            new_path = self.processed_dir / f"{new_version_id}.parquet"
            if save:
                self._save_parquet(df, new_path)
            new_version = DataVersion(
                version_id=new_version_id,
                source=base_version.source,
                path=str(new_path),
                schema=self._infer_schema(df),  # processed schema may differ
                checksum=checksum,
                parent_version=version_id,
                status=DataStatus.PROCESSED,
                transformations=applied,
                metadata={
                    "rows": len(df),
                    "columns": df.shape[1],
                    "column_names": list(df.columns),
                    "tag": tag,
                    "derived_from": version_id,
                },
                profile=self._basic_profile(df),
            )
            self._persist_version(new_version)
            logger.info(
                f"Created processed version {new_version_id} (parent={version_id})"
            )
            return new_version
        else:
            # Mutating metadata only on original version
            base_version.transformations.extend(applied)
            base_version.metadata["last_preprocess_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            base_version.profile = self._basic_profile(df)
            self._save_version_metadata(base_version)
            logger.info(
                f"Updated version metadata without materialization: {version_id}"
            )
            return base_version

    def get_version(self, version_id: str) -> Optional[DataVersion]:
        return self.versions.get(version_id)

    def list_versions(
        self, status: Optional[DataStatus] = None, limit: Optional[int] = None
    ) -> List[DataVersion]:
        versions = list(self.versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        if limit:
            versions = versions[:limit]
        return versions

    def search_versions_by_column(self, column_name: str) -> List[DataVersion]:
        return [v for v in self.versions.values() if column_name in v.schema.columns]

    def export_lineage(self) -> List[Dict[str, str]]:
        """Return simple parent-child lineage edges."""
        edges = []
        for v in self.versions.values():
            if v.parent_version:
                edges.append({"parent": v.parent_version, "child": v.version_id})
        return edges

    def reload(self) -> None:
        """Reload versions from disk (useful if external changes occurred)."""
        self.versions.clear()
        self._load_versions()
        logger.info("Reloaded versions from disk.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _dispatch_read(
        self, source: Union[str, Path], source_type: DataSource
    ) -> pd.DataFrame:
        if source_type == DataSource.LOCAL:
            return self._read_local(source)
        elif source_type == DataSource.HTTP:
            return self._read_http(str(source))
        elif source_type == DataSource.KAGGLE:
            logger.error("Kaggle integration not implemented yet.")
            raise NotImplementedError("Kaggle API integration pending.")
        else:
            raise DataIngestionError(f"Unsupported source type: {source_type.value}")

    def _read_local(self, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        if suffix == ".json":
            return pd.read_json(path)
        if suffix == ".gz" and path.name.endswith(".csv.gz"):
            return pd.read_csv(path, compression="gzip")
        if suffix == ".zip" and path.name.endswith(".csv.zip"):
            with zipfile.ZipFile(path) as z:
                # naive: take first CSV
                for n in z.namelist():
                    if n.lower().endswith(".csv"):
                        with z.open(n) as f:
                            return pd.read_csv(f)
            raise DataIngestionError("No CSV file found inside zip archive")
        if suffix in (".tar", ".tgz", ".bz2"):
            # Basic support: first CSV
            with tarfile.open(path) as t:
                for m in t.getmembers():
                    if m.name.lower().endswith(".csv"):
                        f = t.extractfile(m)
                        if f:
                            return pd.read_csv(f)
            raise DataIngestionError("No CSV file found inside tar archive")
        raise DataIngestionError(f"Unsupported file format: {suffix}")

    def _read_http(self, url: str) -> pd.DataFrame:
        logger.info(f"Fetching remote CSV: {url}")
        with urlopen(
            url
        ) as resp:  # nosec - controlled usage for open HTTP CSV retrieval
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type.lower():
                return pd.read_json(resp)
            # fallback treat as CSV
            return pd.read_csv(resp)

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        # Stable ordering: columns sorted
        sorted_cols = sorted(df.columns)
        normalized = df[sorted_cols]
        # Convert to canonical JSON lines
        data_str = normalized.to_json(
            orient="records", date_format="iso", date_unit="s"
        )
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def _infer_schema(self, df: pd.DataFrame) -> DataSchema:
        columns = {col: str(df[col].dtype) for col in df.columns}
        return DataSchema(
            name="auto_inferred",
            version="1.0",
            columns=columns,
            required_columns=list(df.columns),
            constraints={},
        )

    def _generate_version_id(self, checksum: str, suffix: Optional[str] = None) -> str:
        base = (
            f"v_{checksum[:10]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        if suffix:
            return f"{base}_{suffix}"
        return base

    def _basic_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        null_counts = df.isna().sum().to_dict()
        dtypes = {c: str(dt) for c, dt in df.dtypes.items()}
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        numeric_summary = {}
        for c in numeric_cols:
            s = df[c]
            numeric_summary[c] = {
                "min": float(s.min()) if not s.empty else None,
                "max": float(s.max()) if not s.empty else None,
                "mean": float(s.mean()) if not s.empty else None,
                "std": float(s.std()) if not s.empty else None,
            }
        return {
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "null_counts": null_counts,
            "dtypes": dtypes,
            "numeric_summary": numeric_summary,
        }

    def _persist_version(self, version: DataVersion) -> None:
        self.versions[version.version_id] = version
        self._save_version_metadata(version)
        self._write_manifest()

    def _save_version_metadata(self, version: DataVersion) -> None:
        version_file = self.versions_dir / f"{version.version_id}.json"
        with open(version_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def _load_versions(self) -> None:
        for jf in self.versions_dir.glob("*.json"):
            if jf.name == self.MANIFEST_FILENAME:
                continue
            try:
                with open(jf) as f:
                    data = json.load(f)
                    version = DataVersion.from_dict(data)
                    self.versions[version.version_id] = version
            except Exception as e:
                logger.error(f"Failed to load version metadata {jf}: {e}")

    def _require_version(self, version_id: str) -> DataVersion:
        version = self.get_version(version_id)
        if version is None:
            raise DataVersionNotFoundError(f"Version '{version_id}' not found")
        return version

    def _find_version_by_checksum(self, checksum: str) -> Optional[DataVersion]:
        for v in self.versions.values():
            if v.checksum == checksum:
                return v
        return None

    def _read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _save_parquet(self, df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Parquet save failed ({e}); attempting fallback to CSV")
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Fallback CSV saved to {csv_path}")

    def _write_manifest(self) -> None:
        manifest_path = self.versions_dir / self.MANIFEST_FILENAME
        manifest = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "version_count": len(self.versions),
            "versions": [
                {
                    "version_id": v.version_id,
                    "status": v.status.value,
                    "rows": v.metadata.get("rows"),
                    "columns": v.metadata.get("columns"),
                    "parent": v.parent_version,
                    "source": v.source.value,
                    "timestamp": v.timestamp.isoformat(),
                }
                for v in sorted(self.versions.values(), key=lambda x: x.timestamp)
            ],
            "lineage_edges": self.export_lineage(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Context Manager Support
    # ------------------------------------------------------------------
    def __enter__(self) -> "DataPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always refresh manifest on exit
        try:
            self._write_manifest()
        except Exception as e:
            logger.error(f"Failed to write manifest on exit: {e}")


# -----------------------------------------------------------------------------
# Legacy Compatibility Functions
# -----------------------------------------------------------------------------
def fetch_kaggle_dataset(url):
    """Legacy placeholder - prefer DataPipeline.ingest()"""
    logger.info(
        f"Manual download required (Kaggle not integrated yet). Download from: {url}"
    )


def ingest_all(
    dataset_list_path=Path("datasets/datasets"), download_dir=Path("data/external")
):
    """Legacy function; retained for backward compatibility."""
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(dataset_list_path) as f:
            for line in f:
                url = line.strip()
                if url and url.startswith("http"):
                    fetch_kaggle_dataset(url)
        logger.info(f"Dataset ingestion completed. Check directory: {download_dir}/")
    except FileNotFoundError:
        logger.error(f"Dataset list file not found: {dataset_list_path}")
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")


# -----------------------------------------------------------------------------
# Demo / Self-Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = DataPipeline(workspace_dir="data_workspace_demo", strict_schema=False)
    print(f"Pipeline initialized. Workspace={pipeline.workspace}")

    # Example transformation
    def drop_null_id(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            return df.dropna(subset=["id"])
        return df

    pipeline.register_transformation("drop_null_id", drop_null_id)
    # Additional demo usage could be scripted here.
