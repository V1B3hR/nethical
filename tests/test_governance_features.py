"""Tests for Governance & Ethics features."""

import pytest
from pathlib import Path
import tempfile
import json

from nethical.governance.ethics_benchmark import (
    EthicsBenchmark,
    BenchmarkCase,
    DetectionResult,
    ViolationType,
)
from nethical.governance.threshold_config import (
    ThresholdVersionManager,
    Threshold,
    ThresholdType,
    DEFAULT_THRESHOLDS,
)


class TestEthicsBenchmark:
    """Test ethics benchmark system."""

    def test_benchmark_creation(self):
        """Test creating a benchmark."""
        benchmark = EthicsBenchmark()
        assert len(benchmark.test_cases) == 0

    def test_add_case(self):
        """Test adding benchmark cases."""
        benchmark = EthicsBenchmark()

        case = BenchmarkCase(
            id="test001",
            description="Test case",
            input_data={"text": "test"},
            ground_truth=ViolationType.NONE,
        )

        benchmark.add_case(case)
        assert len(benchmark.test_cases) == 1

    def test_benchmark_evaluation(self):
        """Test benchmark evaluation."""
        benchmark = EthicsBenchmark()

        # Add test cases
        benchmark.add_case(
            BenchmarkCase(
                id="pos001",
                description="Positive case",
                input_data={"text": "malicious content"},
                ground_truth=ViolationType.MANIPULATION,
            )
        )

        benchmark.add_case(
            BenchmarkCase(
                id="neg001",
                description="Negative case",
                input_data={"text": "normal content"},
                ground_truth=ViolationType.NONE,
            )
        )

        # Simple detector
        def detector(input_data):
            if "malicious" in input_data.get("text", ""):
                return DetectionResult(ViolationType.MANIPULATION, 0.9)
            return DetectionResult(ViolationType.NONE, 0.95)

        # Evaluate
        metrics = benchmark.evaluate(detector)

        assert metrics.total_cases == 2
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_metrics_targets(self):
        """Test metrics target checking."""
        benchmark = EthicsBenchmark()

        # Add both positive and negative cases for proper metrics
        benchmark.add_case(
            BenchmarkCase(
                id="test001",
                description="Positive test",
                input_data={"text": "malicious"},
                ground_truth=ViolationType.MANIPULATION,
            )
        )

        benchmark.add_case(
            BenchmarkCase(
                id="test002",
                description="Negative test",
                input_data={"text": "normal"},
                ground_truth=ViolationType.NONE,
            )
        )

        def perfect_detector(input_data):
            if "malicious" in input_data.get("text", ""):
                return DetectionResult(ViolationType.MANIPULATION, 1.0)
            return DetectionResult(ViolationType.NONE, 1.0)

        metrics = benchmark.evaluate(perfect_detector)
        passed, reasons = metrics.meets_targets()

        assert passed
        assert len(reasons) == 0

    def test_save_load_cases(self):
        """Test saving and loading benchmark cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = EthicsBenchmark()

            benchmark.add_case(
                BenchmarkCase(
                    id="test001",
                    description="Test case",
                    input_data={"text": "test"},
                    ground_truth=ViolationType.MANIPULATION,
                )
            )

            file_path = Path(tmpdir) / "cases.json"
            benchmark.save_cases(str(file_path))

            # Load in new benchmark
            benchmark2 = EthicsBenchmark()
            benchmark2.load_cases(str(file_path))

            assert len(benchmark2.test_cases) == 1
            assert benchmark2.test_cases[0].id == "test001"


class TestThresholdConfig:
    """Test threshold configuration versioning."""

    def test_create_version(self):
        """Test creating a threshold version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThresholdVersionManager(tmpdir)

            config = manager.create_version(
                version="1.0.0",
                author="test",
                description="Test version",
                thresholds=DEFAULT_THRESHOLDS,
            )

            assert config.version == "1.0.0"
            assert len(config.thresholds) == len(DEFAULT_THRESHOLDS)
            assert manager.current_version == "1.0.0"

    def test_get_version(self):
        """Test retrieving versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThresholdVersionManager(tmpdir)

            manager.create_version(
                version="1.0.0",
                author="test",
                description="Test",
                thresholds=DEFAULT_THRESHOLDS,
            )

            # Get current version
            config = manager.get_version()
            assert config is not None
            assert config.version == "1.0.0"

            # Get specific version
            config2 = manager.get_version("1.0.0")
            assert config2.version == "1.0.0"

    def test_threshold_evaluation(self):
        """Test threshold evaluation."""
        threshold = Threshold(
            name="test",
            threshold_type=ThresholdType.CONFIDENCE,
            value=0.8,
            operator=">=",
            description="Test threshold",
        )

        assert threshold.evaluate(0.9) is True
        assert threshold.evaluate(0.8) is True
        assert threshold.evaluate(0.7) is False

    def test_compare_versions(self):
        """Test version comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThresholdVersionManager(tmpdir)

            # Create v1
            thresholds_v1 = DEFAULT_THRESHOLDS.copy()
            manager.create_version(
                version="1.0.0",
                author="test",
                description="V1",
                thresholds=thresholds_v1,
                set_current=False,
            )

            # Create v2 with changes
            thresholds_v2 = thresholds_v1.copy()
            thresholds_v2["manipulation_detection"] = Threshold(
                name="manipulation_detection",
                threshold_type=ThresholdType.CONFIDENCE,
                value=0.90,  # Changed from 0.85
                operator=">=",
                description="Updated threshold",
                unit="probability",
            )

            manager.create_version(
                version="2.0.0",
                author="test",
                description="V2",
                thresholds=thresholds_v2,
            )

            # Compare
            diff = manager.compare_versions("1.0.0", "2.0.0")

            assert diff["total_changes"] > 0
            assert len(diff["changed"]) > 0

    def test_evaluate_thresholds(self):
        """Test evaluating multiple thresholds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ThresholdVersionManager(tmpdir)

            manager.create_version(
                version="1.0.0",
                author="test",
                description="Test",
                thresholds=DEFAULT_THRESHOLDS,
            )

            test_values = {
                "manipulation_detection": 0.90,  # Above threshold (0.85)
                "privacy_risk": 0.65,  # Below threshold (0.7)
            }

            results = manager.evaluate_thresholds(test_values)

            assert results["manipulation_detection"] is True
            assert results["privacy_risk"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
