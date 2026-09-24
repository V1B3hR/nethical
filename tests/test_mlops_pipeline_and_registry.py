# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit test suite for MLOps components: ModelRegistry, DataPipeline, and Monitoring.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from nethical.mlops.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelStage,
    ModelStatus,
    ModelMetrics,
)
from nethical.mlops.data_pipeline import (
    DataPipeline,
    DataSource,
    DataSchema,
    DataVersion,
)
from nethical.mlops.monitoring import (
    Alert,
    AlertSeverity,
    MetricPoint,
    MetricType,
)


class TestMLOpsModelRegistry:
    """Test model registry versioning, promotion, and persistence."""

    def test_model_registration_and_stage_transitions(self, tmp_path):
        registry_dir = tmp_path / "registry"
        registry = ModelRegistry(registry_dir=registry_dir)

        # Create dummy artifact
        artifact_path = tmp_path / "model.pt"
        artifact_path.write_text("dummy model weights", encoding="utf-8")

        # Register model version
        version = registry.register_model(
            name="governance_classifier",
            version="1.0.0",
            model_path=artifact_path,
            description="Initial baseline model",
            tags=["nlp", "governance"],
            metrics={"accuracy": 0.95, "f1_score": 0.92},
        )

        assert version.name == "governance_classifier"
        assert version.version == "1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT
        assert version.status == ModelStatus.ACTIVE
        assert version.created_at.tzinfo == timezone.utc

        # Promote to Staging
        success_staging = registry.promote_model(
            version_id=version.version_id,
            target_stage=ModelStage.STAGING,
        )
        assert success_staging is True
        assert registry.get_model(version.version_id).stage == ModelStage.STAGING

        # Promote to Production
        success_prod = registry.promote_model(
            version_id=version.version_id,
            target_stage=ModelStage.PRODUCTION,
        )
        assert success_prod is True
        assert registry.get_model(version.version_id).stage == ModelStage.PRODUCTION

        # Reload registry from disk and verify persistence
        new_registry = ModelRegistry(registry_dir=registry_dir)
        loaded_model = new_registry.get_model(version.version_id)
        assert loaded_model is not None
        assert loaded_model.stage == ModelStage.PRODUCTION
        assert loaded_model.metrics.accuracy == 0.95

    def test_registry_export_and_import(self, tmp_path):
        registry = ModelRegistry(registry_dir=tmp_path / "reg1")
        artifact = tmp_path / "weights.bin"
        artifact.write_text("weights", encoding="utf-8")

        v1 = registry.register_model(
            name="detector_v1",
            version="1.0.0",
            model_path=artifact,
        )

        export_path = tmp_path / "export.json"
        registry.export_registry(export_path)

        assert export_path.exists()
        imported_registry = ModelRegistry(registry_dir=tmp_path / "reg2")
        imported_registry.import_registry(export_path)

        assert v1.version_id in imported_registry.models
        assert imported_registry.models[v1.version_id].name == "detector_v1"


class TestMLOpsDataPipeline:
    """Test data pipeline schema definition and version persistence."""

    def test_data_pipeline_ingest_and_persistence(self, tmp_path):
        pipeline = DataPipeline(workspace_dir=tmp_path / "data_pipeline")

        schema = DataSchema(
            name="training_text_schema",
            version="1.0.0",
            columns={"text": "object", "label": "int64"},
            required_columns=["text", "label"],
        )

        data_file = tmp_path / "dataset.csv"
        data_file.write_text("text,label\nhello,0\nworld,1\n", encoding="utf-8")

        version = pipeline.ingest(
            source=data_file,
            source_type=DataSource.LOCAL,
            schema=schema,
            tag="unit_test_v1",
        )

        assert version.version_id.startswith("v_")
        assert version.timestamp.tzinfo == timezone.utc
        assert "text" in version.schema.columns
        assert "label" in version.schema.columns

        # Verify version loaded from disk
        reloaded_pipeline = DataPipeline(workspace_dir=tmp_path / "data_pipeline")
        fetched = reloaded_pipeline.get_version(version.version_id)
        assert fetched is not None
        assert fetched.checksum == version.checksum


class TestMLOpsMonitoringModels:
    """Test MLOps monitoring dataclasses and UTC handling."""

    def test_alert_utc_timestamp_and_dict(self):
        alert = Alert(
            alert_id="alert_001",
            severity=AlertSeverity.ERROR,
            title="Drift Detected",
            message="Feature drift exceeds threshold 0.15",
        )

        assert alert.timestamp.tzinfo == timezone.utc
        alert_dict = alert.to_dict()
        assert alert_dict["severity"] == "error"
        assert alert_dict["alert_id"] == "alert_001"

    def test_metric_point_utc_timestamp_and_dict(self):
        metric = MetricPoint(
            name="inference_latency_p99",
            value=12.4,
            metric_type=MetricType.GAUGE,
        )

        assert metric.timestamp.tzinfo == timezone.utc
        m_dict = metric.to_dict()
        assert m_dict["name"] == "inference_latency_p99"
        assert m_dict["value"] == 12.4
