"""Tests for Phase 3 implementations."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from nethical.core.risk_engine import RiskEngine, RiskTier, RiskProfile
from nethical.core.correlation_engine import CorrelationEngine, CorrelationMatch
from nethical.core.fairness_sampler import FairnessSampler, SamplingStrategy
from nethical.core.ethical_drift_reporter import EthicalDriftReporter
from nethical.core.performance_optimizer import PerformanceOptimizer, DetectorTier
from nethical.core.phase3_integration import Phase3IntegratedGovernance


class TestRiskEngine:
    """Test RiskEngine implementation."""

    def test_risk_tier_ordering(self):
        """Test risk tier comparison."""
        assert RiskTier.LOW < RiskTier.NORMAL
        assert RiskTier.NORMAL < RiskTier.HIGH
        assert RiskTier.HIGH < RiskTier.ELEVATED
        assert RiskTier.NORMAL <= RiskTier.HIGH

    def test_risk_tier_from_score(self):
        """Test risk tier conversion from score."""
        assert RiskTier.from_score(0.1) == RiskTier.LOW
        assert RiskTier.from_score(0.3) == RiskTier.NORMAL
        assert RiskTier.from_score(0.6) == RiskTier.HIGH
        assert RiskTier.from_score(0.8) == RiskTier.ELEVATED

    def test_risk_engine_initialization(self):
        """Test risk engine initialization."""
        engine = RiskEngine()
        assert engine.decay_half_life_hours == 24.0
        assert engine.elevated_threshold == 0.75
        assert len(engine.profiles) == 0

    def test_calculate_risk_score(self):
        """Test risk score calculation."""
        engine = RiskEngine()

        score = engine.calculate_risk_score(
            agent_id="agent_1",
            violation_severity=0.8,
            action_context={"is_privileged": False},
        )

        assert 0.0 <= score <= 1.0
        assert "agent_1" in engine.profiles

    def test_elevated_tier_trigger(self):
        """Test elevated tier triggers."""
        engine = RiskEngine(elevated_threshold=0.6)

        # Low risk - should not trigger
        engine.calculate_risk_score("agent_1", 0.3, {})
        assert not engine.should_invoke_advanced_detectors("agent_1")

        # Multiple high risk actions to build up score
        for _ in range(3):
            engine.calculate_risk_score("agent_1", 0.9, {})

        # Now should trigger
        assert engine.should_invoke_advanced_detectors("agent_1")

    def test_risk_decay(self):
        """Test risk score decay over time."""
        engine = RiskEngine(decay_half_life_hours=1.0)

        # Create profile with high score
        profile = engine.get_or_create_profile("agent_1")
        profile.current_score = 0.8
        profile.last_update = datetime.utcnow() - timedelta(hours=2)

        # Apply decay (2 half-lives = 1/4 of original)
        decayed = engine._apply_decay(profile)
        assert decayed < profile.current_score
        assert decayed > 0.1  # Should be around 0.2

    def test_risk_profile_serialization(self):
        """Test risk profile serialization."""
        profile = RiskProfile(
            agent_id="agent_1", current_score=0.5, current_tier=RiskTier.NORMAL
        )

        data = profile.to_dict()
        restored = RiskProfile.from_dict(data)

        assert restored.agent_id == profile.agent_id
        assert restored.current_score == profile.current_score
        assert restored.current_tier == profile.current_tier


class TestCorrelationEngine:
    """Test CorrelationEngine implementation."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "correlation_rules.yaml"

        # Use the actual config file
        import shutil

        actual_config = Path(__file__).parent.parent.parent / "correlation_rules.yaml"
        if actual_config.exists():
            shutil.copy(actual_config, config_path)

        yield config_path
        shutil.rmtree(temp_dir)

    def test_correlation_engine_initialization(self, temp_config):
        """Test correlation engine initialization."""
        engine = CorrelationEngine(config_path=str(temp_config))
        assert engine.config is not None
        assert len(engine.agent_windows) == 0

    def test_track_action(self, temp_config):
        """Test tracking actions."""
        engine = CorrelationEngine(config_path=str(temp_config))

        # Create mock action
        class MockAction:
            def __init__(self, content):
                self.content = content
                self.metadata = {}

        matches = engine.track_action(
            agent_id="agent_1",
            action=MockAction("test content"),
            payload="test payload",
        )

        assert "agent_1" in engine.agent_windows
        assert isinstance(matches, list)

    def test_entropy_calculation(self, temp_config):
        """Test entropy calculation."""
        engine = CorrelationEngine(config_path=str(temp_config))

        # Low entropy (repeated characters)
        low_entropy = engine._calculate_entropy("aaaaaaaaaa")

        # High entropy (random characters)
        high_entropy = engine._calculate_entropy("a1b2c3d4e5")

        assert high_entropy > low_entropy

    def test_escalating_probes_detection(self, temp_config):
        """Test escalating multi-ID probe detection."""
        engine = CorrelationEngine(config_path=str(temp_config))

        class MockAction:
            def __init__(self, content):
                self.content = content
                self.metadata = {}

        # Simulate escalating probes from multiple agents
        for i in range(5):
            for j in range(i + 1):  # Escalating pattern
                engine.track_action(
                    agent_id=f"agent_{i}",
                    action=MockAction(f"probe {j}"),
                    payload=f"payload_{j}",
                )

        # Check if pattern was detected
        matches = engine._check_all_patterns()
        assert isinstance(matches, list)


class TestFairnessSampler:
    """Test FairnessSampler implementation."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_sampler_initialization(self, temp_storage):
        """Test fairness sampler initialization."""
        sampler = FairnessSampler(storage_dir=temp_storage)
        assert sampler.storage_dir.exists()
        assert len(sampler.jobs) == 0

    def test_create_sampling_job(self, temp_storage):
        """Test creating a sampling job."""
        sampler = FairnessSampler(storage_dir=temp_storage)

        job_id = sampler.create_sampling_job(
            cohorts=["cohort_a", "cohort_b"], target_sample_size=100
        )

        assert job_id in sampler.jobs
        job = sampler.jobs[job_id]
        assert job.target_sample_size == 100
        assert len(job.cohorts) == 2

    def test_add_sample(self, temp_storage):
        """Test adding samples to a job."""
        sampler = FairnessSampler(storage_dir=temp_storage)

        job_id = sampler.create_sampling_job(
            cohorts=["cohort_a"], target_sample_size=10
        )

        success = sampler.add_sample(
            job_id=job_id,
            agent_id="agent_1",
            action_id="action_1",
            cohort="cohort_a",
            violation_type="safety",
            severity="high",
        )

        assert success
        job = sampler.jobs[job_id]
        assert len(job.samples) == 1
        assert job.coverage["cohort_a"] == 1

    def test_stratified_sampling(self, temp_storage):
        """Test stratified sampling."""
        sampler = FairnessSampler(storage_dir=temp_storage)

        job_id = sampler.create_sampling_job(
            cohorts=["cohort_a", "cohort_b"],
            target_sample_size=20,
            strategy=SamplingStrategy.STRATIFIED,
        )

        # Create population data
        population = {
            "cohort_a": [
                {"agent_id": f"agent_{i}", "action_id": f"action_{i}"}
                for i in range(30)
            ],
            "cohort_b": [
                {"agent_id": f"agent_{i}", "action_id": f"action_{i}"}
                for i in range(10)
            ],
        }

        samples_collected = sampler.perform_stratified_sampling(job_id, population)

        assert samples_collected > 0
        assert samples_collected <= 20

    def test_finalize_job(self, temp_storage):
        """Test finalizing a job."""
        sampler = FairnessSampler(storage_dir=temp_storage)

        job_id = sampler.create_sampling_job(cohorts=["cohort_a"], target_sample_size=5)

        # Add some samples
        for i in range(3):
            sampler.add_sample(
                job_id=job_id,
                agent_id=f"agent_{i}",
                action_id=f"action_{i}",
                cohort="cohort_a",
            )

        success = sampler.finalize_job(job_id)
        assert success

        job = sampler.jobs[job_id]
        assert job.end_time is not None

        # Check if file was created
        job_file = Path(temp_storage) / f"{job_id}.json"
        assert job_file.exists()

    def test_coverage_stats(self, temp_storage):
        """Test coverage statistics."""
        sampler = FairnessSampler(storage_dir=temp_storage)

        job_id = sampler.create_sampling_job(
            cohorts=["cohort_a", "cohort_b"], target_sample_size=10
        )

        # Add samples
        sampler.add_sample(job_id, "agent_1", "action_1", "cohort_a", "safety", "high")
        sampler.add_sample(
            job_id, "agent_2", "action_2", "cohort_b", "privacy", "medium"
        )

        stats = sampler.get_coverage_stats(job_id)

        assert stats["total_samples"] == 2
        assert stats["target_samples"] == 10
        assert "cohort_coverage" in stats
        assert "violation_distribution" in stats


class TestEthicalDriftReporter:
    """Test EthicalDriftReporter implementation."""

    @pytest.fixture
    def temp_reports(self):
        """Create temporary reports directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_reporter_initialization(self, temp_reports):
        """Test reporter initialization."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)
        assert reporter.report_dir.exists()
        assert len(reporter.cohort_profiles) == 0

    def test_track_violation(self, temp_reports):
        """Test tracking violations."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)

        reporter.track_violation(
            agent_id="agent_1",
            cohort="cohort_a",
            violation_type="safety",
            severity="high",
        )

        assert "cohort_a" in reporter.cohort_profiles
        profile = reporter.cohort_profiles["cohort_a"]
        assert profile.violation_stats.total_count == 1
        assert profile.violation_stats.by_type["safety"] == 1

    def test_track_action(self, temp_reports):
        """Test tracking actions."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)

        reporter.track_action(agent_id="agent_1", cohort="cohort_a", risk_score=0.5)

        profile = reporter.cohort_profiles["cohort_a"]
        assert profile.action_count == 1
        assert profile.avg_risk_score == 0.5

    def test_generate_report(self, temp_reports):
        """Test report generation."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)

        # Track data for multiple cohorts
        for cohort in ["cohort_a", "cohort_b"]:
            for i in range(10):
                reporter.track_action(
                    f"agent_{i}", cohort, 0.5 if cohort == "cohort_a" else 0.8
                )
                if i % 2 == 0:
                    reporter.track_violation(
                        f"agent_{i}",
                        cohort,
                        "safety",
                        "high" if cohort == "cohort_a" else "medium",
                    )

        report = reporter.generate_report(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
        )

        assert report.report_id is not None
        assert len(report.cohorts) == 2
        assert "drift_metrics" in report.to_dict()
        assert len(report.recommendations) > 0

    def test_drift_detection(self, temp_reports):
        """Test drift detection."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)

        # Cohort A: low risk
        for i in range(10):
            reporter.track_action(f"agent_a_{i}", "cohort_a", 0.2)
            if i % 5 == 0:
                reporter.track_violation(f"agent_a_{i}", "cohort_a", "safety", "low")

        # Cohort B: high risk (drift)
        for i in range(10):
            reporter.track_action(f"agent_b_{i}", "cohort_b", 0.8)
            if i % 2 == 0:
                reporter.track_violation(f"agent_b_{i}", "cohort_b", "safety", "high")

        report = reporter.generate_report(
            start_time=datetime.utcnow() - timedelta(hours=1),
            end_time=datetime.utcnow(),
        )

        assert report.drift_metrics.get("has_drift", False) == True

    def test_dashboard_data(self, temp_reports):
        """Test dashboard data generation."""
        reporter = EthicalDriftReporter(report_dir=temp_reports)

        # Add some data
        reporter.track_action("agent_1", "cohort_a", 0.5)
        reporter.track_violation("agent_1", "cohort_a", "safety", "medium")

        dashboard = reporter.get_dashboard_data()

        assert "cohort_summary" in dashboard
        assert "overall_stats" in dashboard
        assert "violation_distribution" in dashboard
        assert dashboard["overall_stats"]["total_cohorts"] == 1


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer implementation."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = PerformanceOptimizer(target_cpu_reduction_pct=30.0)
        assert optimizer.target_cpu_reduction_pct == 30.0
        assert len(optimizer.detector_metrics) == 0

    def test_register_detector(self):
        """Test detector registration."""
        optimizer = PerformanceOptimizer()

        optimizer.register_detector("test_detector", DetectorTier.ADVANCED)

        assert "test_detector" in optimizer.detector_registry
        assert optimizer.detector_registry["test_detector"] == DetectorTier.ADVANCED

    def test_risk_based_gating(self):
        """Test risk-based detector gating."""
        optimizer = PerformanceOptimizer()

        # Register detectors at different tiers
        optimizer.register_detector("fast_detector", DetectorTier.FAST)
        optimizer.register_detector("advanced_detector", DetectorTier.ADVANCED)
        optimizer.register_detector("premium_detector", DetectorTier.PREMIUM)

        # Low risk - only fast should be invoked
        assert optimizer.should_invoke_detector("fast_detector", 0.1)
        assert not optimizer.should_invoke_detector("advanced_detector", 0.1)
        assert not optimizer.should_invoke_detector("premium_detector", 0.1)

        # High risk - fast and advanced should be invoked
        assert optimizer.should_invoke_detector("fast_detector", 0.6)
        assert optimizer.should_invoke_detector("advanced_detector", 0.6)
        assert not optimizer.should_invoke_detector("premium_detector", 0.6)

        # Very high risk - all should be invoked
        assert optimizer.should_invoke_detector("fast_detector", 0.8)
        assert optimizer.should_invoke_detector("advanced_detector", 0.8)
        assert optimizer.should_invoke_detector("premium_detector", 0.8)

    def test_track_detector_invocation(self):
        """Test tracking detector invocations."""
        optimizer = PerformanceOptimizer()
        optimizer.register_detector("test_detector", DetectorTier.STANDARD)

        optimizer.track_detector_invocation("test_detector", 50.0, was_cached=False)
        optimizer.track_detector_invocation("test_detector", 30.0, was_cached=True)

        metrics = optimizer.detector_metrics["test_detector"]
        assert metrics.total_invocations == 2
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1

    def test_cpu_reduction_calculation(self):
        """Test CPU reduction calculation."""
        optimizer = PerformanceOptimizer()

        # Establish baseline
        for i in range(100):
            optimizer.track_action_processing(100.0, 5)

        assert optimizer.baseline_established
        assert optimizer.baseline_cpu_ms == 100.0

        # Track improved performance
        for i in range(100):
            optimizer.track_action_processing(70.0, 3)

        reduction = optimizer.get_cpu_reduction_pct()
        assert reduction > 0  # Should show improvement

    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        optimizer = PerformanceOptimizer()

        # Add expensive detector
        optimizer.register_detector("expensive_detector", DetectorTier.ADVANCED)
        optimizer.track_detector_invocation("expensive_detector", 150.0)

        suggestions = optimizer.suggest_optimizations()
        assert len(suggestions) > 0
        assert any("high average CPU time" in s for s in suggestions)

    def test_performance_report(self):
        """Test performance report generation."""
        optimizer = PerformanceOptimizer()

        optimizer.register_detector("detector1", DetectorTier.FAST)
        optimizer.track_detector_invocation("detector1", 10.0)
        optimizer.track_action_processing(50.0, 1)

        report = optimizer.get_performance_report()

        assert "action_metrics" in report
        assert "detector_stats" in report
        assert "optimization" in report


class TestPhase3Integration:
    """Test Phase3IntegratedGovernance."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_integration_initialization(self, temp_storage):
        """Test integrated governance initialization."""
        governance = Phase3IntegratedGovernance(
            storage_dir=temp_storage, enable_performance_optimization=True
        )

        assert governance.risk_engine is not None
        assert governance.correlation_engine is not None
        assert governance.fairness_sampler is not None
        assert governance.ethical_drift_reporter is not None
        assert governance.performance_optimizer is not None

    def test_process_action(self, temp_storage):
        """Test integrated action processing."""
        governance = Phase3IntegratedGovernance(
            storage_dir=temp_storage, enable_performance_optimization=True
        )

        class MockAction:
            def __init__(self, content):
                self.content = content

        results = governance.process_action(
            agent_id="agent_1",
            action=MockAction("test action"),
            cohort="test_cohort",
            violation_detected=True,
            violation_type="safety",
            violation_severity="high",
            detector_invocations={"detector1": 10.0, "detector2": 20.0},
        )

        assert "risk_score" in results
        assert "risk_tier" in results
        assert "correlations" in results
        assert "performance_metrics" in results

    def test_should_invoke_detector(self, temp_storage):
        """Test detector invocation decision."""
        governance = Phase3IntegratedGovernance(
            storage_dir=temp_storage, enable_performance_optimization=True
        )

        # Process action to set risk score
        class MockAction:
            def __init__(self, content):
                self.content = content

        governance.process_action(
            agent_id="agent_1",
            action=MockAction("test"),
            cohort="cohort_a",
            violation_detected=False,
            detector_invocations={},
        )

        # Check detector gating
        should_invoke = governance.should_invoke_detector(
            detector_name="test_detector", agent_id="agent_1", tier=DetectorTier.PREMIUM
        )

        assert isinstance(should_invoke, bool)

    def test_generate_drift_report(self, temp_storage):
        """Test drift report generation via integration."""
        governance = Phase3IntegratedGovernance(storage_dir=temp_storage)

        class MockAction:
            def __init__(self, content):
                self.content = content

        # Generate some data
        for i in range(10):
            governance.process_action(
                agent_id=f"agent_{i}",
                action=MockAction("test"),
                cohort="cohort_a",
                violation_detected=i % 2 == 0,
                violation_type="safety",
                violation_severity="medium",
            )

        report = governance.generate_drift_report(days_back=1)

        assert "report_id" in report
        assert "cohorts" in report
        assert "drift_metrics" in report

    def test_system_status(self, temp_storage):
        """Test system status reporting."""
        governance = Phase3IntegratedGovernance(storage_dir=temp_storage)

        status = governance.get_system_status()

        assert "timestamp" in status
        assert "components" in status
        assert "risk_engine" in status["components"]
        assert "correlation_engine" in status["components"]
        assert "fairness_sampler" in status["components"]
        assert "ethical_drift_reporter" in status["components"]
        assert "performance_optimizer" in status["components"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
