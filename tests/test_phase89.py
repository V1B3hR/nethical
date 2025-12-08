"""Tests for Phase 8-9: Human-in-the-Loop & Optimization"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from nethical.core import (
    # Phase 8
    EscalationQueue,
    FeedbackTag,
    ReviewPriority,
    ReviewStatus,
    # Phase 9
    MultiObjectiveOptimizer,
    OptimizationTechnique,
    ConfigStatus,
    # Integration
    Phase89IntegratedGovernance,
)


class TestPhase8HumanFeedback:
    """Tests for Phase 8 human-in-the-loop components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def escalation_queue(self, temp_dir):
        """Create escalation queue for testing."""
        return EscalationQueue(
            storage_path=str(Path(temp_dir) / "escalations.db"),
            triage_sla_seconds=3600,
            resolution_sla_seconds=86400,
        )

    def test_initialization(self, escalation_queue):
        """Test escalation queue initialization."""
        assert escalation_queue is not None
        assert escalation_queue.triage_sla_seconds == 3600
        assert escalation_queue.resolution_sla_seconds == 86400
        assert len(escalation_queue.pending_cases) == 0

    def test_add_case(self, escalation_queue):
        """Test adding case to escalation queue."""
        case = escalation_queue.add_case(
            judgment_id="judg_123",
            action_id="act_456",
            agent_id="agent_789",
            decision="block",
            confidence=0.65,
            violations=[
                {"type": "safety", "severity": 4, "description": "Critical violation"}
            ],
            priority=ReviewPriority.HIGH,
        )

        assert case is not None
        assert case.judgment_id == "judg_123"
        assert case.action_id == "act_456"
        assert case.agent_id == "agent_789"
        assert case.decision == "block"
        assert case.confidence == 0.65
        assert case.priority == ReviewPriority.HIGH
        assert case.status == ReviewStatus.PENDING
        assert len(escalation_queue.pending_cases) == 1

    def test_get_next_case(self, escalation_queue):
        """Test getting next case from queue."""
        # Add multiple cases
        case1 = escalation_queue.add_case(
            judgment_id="judg_1",
            action_id="act_1",
            agent_id="agent_1",
            decision="block",
            confidence=0.6,
            violations=[{"type": "safety", "severity": 3}],
            priority=ReviewPriority.MEDIUM,
        )

        case2 = escalation_queue.add_case(
            judgment_id="judg_2",
            action_id="act_2",
            agent_id="agent_2",
            decision="terminate",
            confidence=0.5,
            violations=[{"type": "safety", "severity": 5}],
            priority=ReviewPriority.EMERGENCY,
        )

        # Get next case (should be highest priority)
        next_case = escalation_queue.get_next_case(reviewer_id="reviewer_1")

        assert next_case is not None
        assert next_case.case_id == case1.case_id or next_case.case_id == case2.case_id
        assert next_case.status == ReviewStatus.IN_REVIEW
        assert next_case.assigned_to == "reviewer_1"
        assert next_case.started_review_at is not None

    def test_submit_feedback(self, escalation_queue):
        """Test submitting feedback for a case."""
        # Add and get case
        case = escalation_queue.add_case(
            judgment_id="judg_123",
            action_id="act_456",
            agent_id="agent_789",
            decision="block",
            confidence=0.65,
            violations=[{"type": "safety", "severity": 4}],
            priority=ReviewPriority.HIGH,
        )

        # Start review
        case_to_review = escalation_queue.get_next_case(reviewer_id="reviewer_1")

        # Submit feedback
        feedback = escalation_queue.submit_feedback(
            case_id=case_to_review.case_id,
            reviewer_id="reviewer_1",
            feedback_tags=[FeedbackTag.FALSE_POSITIVE],
            rationale="This was actually safe content, detector was too aggressive",
            corrected_decision="allow",
            confidence=0.9,
        )

        assert feedback is not None
        assert feedback.judgment_id == case.judgment_id
        assert feedback.reviewer_id == "reviewer_1"
        assert FeedbackTag.FALSE_POSITIVE in feedback.feedback_tags
        assert feedback.corrected_decision == "allow"
        assert feedback.confidence == 0.9

        # Check case is completed
        updated_case = escalation_queue.get_case(case.case_id)
        assert updated_case.status == ReviewStatus.COMPLETED
        assert updated_case.completed_at is not None

    def test_sla_metrics(self, escalation_queue):
        """Test SLA metrics calculation."""
        # Add and process some cases
        case = escalation_queue.add_case(
            judgment_id="judg_1",
            action_id="act_1",
            agent_id="agent_1",
            decision="block",
            confidence=0.6,
            violations=[{"type": "safety", "severity": 4}],
            priority=ReviewPriority.HIGH,
        )

        case_to_review = escalation_queue.get_next_case(reviewer_id="reviewer_1")
        escalation_queue.submit_feedback(
            case_id=case_to_review.case_id,
            reviewer_id="reviewer_1",
            feedback_tags=[FeedbackTag.CORRECT_DECISION],
            rationale="Decision was correct",
        )

        metrics = escalation_queue.get_sla_metrics()

        assert metrics is not None
        assert metrics.total_cases == 1
        assert metrics.completed_cases == 1
        assert metrics.pending_cases == 0

    def test_feedback_summary(self, escalation_queue):
        """Test feedback summary for continuous improvement."""
        # Add and process cases with different feedback
        for i in range(3):
            case = escalation_queue.add_case(
                judgment_id=f"judg_{i}",
                action_id=f"act_{i}",
                agent_id=f"agent_{i}",
                decision="block",
                confidence=0.6,
                violations=[{"type": "safety", "severity": 3}],
                priority=ReviewPriority.MEDIUM,
            )

            case_to_review = escalation_queue.get_next_case(reviewer_id="reviewer_1")

            # Vary feedback tags
            tags = (
                [FeedbackTag.CORRECT_DECISION]
                if i == 0
                else [FeedbackTag.FALSE_POSITIVE]
            )

            escalation_queue.submit_feedback(
                case_id=case_to_review.case_id,
                reviewer_id="reviewer_1",
                feedback_tags=tags,
                rationale=f"Feedback for case {i}",
            )

        summary = escalation_queue.get_feedback_summary()

        assert summary is not None
        assert summary["total_feedback"] == 3
        assert "tag_counts" in summary
        assert "false_positive_rate" in summary


class TestPhase9Optimization:
    """Tests for Phase 9 optimization components."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create optimizer for testing."""
        return MultiObjectiveOptimizer(
            storage_path=str(Path(temp_dir) / "optimization.db")
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer is not None
        assert len(optimizer.objectives) == 4
        assert optimizer.promotion_gate is not None

    def test_create_configuration(self, optimizer):
        """Test creating configuration."""
        config = optimizer.create_configuration(
            config_version="test_v1",
            classifier_threshold=0.6,
            gray_zone_lower=0.3,
            gray_zone_upper=0.7,
        )

        assert config is not None
        assert config.config_version == "test_v1"
        assert config.classifier_threshold == 0.6
        assert config.gray_zone_lower == 0.3
        assert config.gray_zone_upper == 0.7
        assert config.status == ConfigStatus.CANDIDATE

    def test_record_metrics(self, optimizer):
        """Test recording metrics."""
        config = optimizer.create_configuration(config_version="test_v1")

        metrics = optimizer.record_metrics(
            config_id=config.config_id,
            detection_recall=0.85,
            detection_precision=0.90,
            false_positive_rate=0.05,
            decision_latency_ms=10.0,
            human_agreement=0.88,
            total_cases=100,
        )

        assert metrics is not None
        assert metrics.detection_recall == 0.85
        assert metrics.detection_precision == 0.90
        assert metrics.false_positive_rate == 0.05
        assert metrics.decision_latency_ms == 10.0
        assert metrics.human_agreement == 0.88
        assert metrics.total_cases == 100
        assert metrics.fitness_score > 0  # Should calculate fitness

    def test_calculate_fitness(self, optimizer):
        """Test fitness calculation."""
        fitness = optimizer.calculate_fitness(
            recall=0.9, fp_rate=0.05, latency_ms=10.0, agreement=0.85
        )

        # fitness = 0.4*recall - 0.25*fp_rate - 0.15*latency + 0.2*agreement
        # fitness = 0.4*0.9 - 0.25*0.05 - 0.15*0.1 + 0.2*0.85
        # fitness = 0.36 - 0.0125 - 0.015 + 0.17 = 0.5025
        assert fitness > 0
        assert abs(fitness - 0.5025) < 0.01  # Allow small floating point error

    def test_promotion_gate(self, optimizer):
        """Test promotion gate validation."""
        # Create baseline config
        baseline_config = optimizer.create_configuration(config_version="baseline_v1")
        baseline_metrics = optimizer.record_metrics(
            config_id=baseline_config.config_id,
            detection_recall=0.80,
            detection_precision=0.85,
            false_positive_rate=0.08,
            decision_latency_ms=12.0,
            human_agreement=0.86,
            total_cases=100,
        )

        # Create candidate config with better metrics
        candidate_config = optimizer.create_configuration(config_version="candidate_v1")
        candidate_metrics = optimizer.record_metrics(
            config_id=candidate_config.config_id,
            detection_recall=0.84,  # +4% recall gain
            detection_precision=0.87,
            false_positive_rate=0.09,  # +1% FP increase (within limit)
            decision_latency_ms=14.0,  # +2ms latency (within limit)
            human_agreement=0.88,
            total_cases=150,
        )

        # Check promotion gate
        passed, reasons = optimizer.check_promotion_gate(
            candidate_id=candidate_config.config_id,
            baseline_id=baseline_config.config_id,
        )

        assert passed is True
        assert len(reasons) > 0

    def test_random_search(self, optimizer):
        """Test random search optimization."""

        def evaluate(config):
            """Dummy evaluation function."""
            import random

            return optimizer.record_metrics(
                config_id=config.config_id,
                detection_recall=random.uniform(0.75, 0.95),
                detection_precision=random.uniform(0.75, 0.95),
                false_positive_rate=random.uniform(0.02, 0.1),
                decision_latency_ms=random.uniform(8, 15),
                human_agreement=random.uniform(0.82, 0.95),
                total_cases=100,
            )

        results = optimizer.random_search(
            param_ranges={
                "classifier_threshold": (0.4, 0.7),
                "gray_zone_lower": (0.3, 0.5),
                "gray_zone_upper": (0.5, 0.7),
            },
            evaluate_fn=evaluate,
            n_iterations=10,
        )

        assert len(results) == 10
        assert all(
            isinstance(r[0], type(optimizer.create_configuration("dummy")))
            for r in results
        )
        # Results should be sorted by fitness (descending)
        assert results[0][1].fitness_score >= results[-1][1].fitness_score


class TestPhase89Integration:
    """Tests for Phase 8-9 integrated governance."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def governance(self, temp_dir):
        """Create integrated governance for testing."""
        return Phase89IntegratedGovernance(
            storage_dir=temp_dir, triage_sla_seconds=3600, resolution_sla_seconds=86400
        )

    def test_initialization(self, governance):
        """Test integrated governance initialization."""
        assert governance is not None
        assert governance.escalation_queue is not None
        assert governance.optimizer is not None

    def test_should_escalate(self, governance):
        """Test escalation decision logic."""
        # Test critical violation escalation
        should_escalate, priority = governance.should_escalate(
            decision="allow",
            confidence=0.9,
            violations=[{"severity": 5, "type": "critical"}],
        )
        assert should_escalate is True
        assert priority == ReviewPriority.EMERGENCY

        # Test block decision escalation
        should_escalate, priority = governance.should_escalate(
            decision="block",
            confidence=0.85,
            violations=[{"severity": 3, "type": "safety"}],
        )
        assert should_escalate is True
        assert priority == ReviewPriority.HIGH

        # Test low confidence escalation
        should_escalate, priority = governance.should_escalate(
            decision="allow",
            confidence=0.65,
            violations=[{"severity": 2, "type": "safety"}],
        )
        assert should_escalate is True
        assert priority == ReviewPriority.MEDIUM

    def test_process_with_escalation(self, governance):
        """Test processing action with escalation."""
        result = governance.process_with_escalation(
            judgment_id="judg_123",
            action_id="act_456",
            agent_id="agent_789",
            decision="block",
            confidence=0.65,
            violations=[{"severity": 4, "type": "safety"}],
        )

        assert result is not None
        assert result["escalated"] is True
        assert "escalation_case_id" in result
        assert result["decision"] == "block"

    def test_continuous_improvement_cycle(self, governance):
        """Test continuous improvement cycle."""
        # Add some feedback first
        result = governance.process_with_escalation(
            judgment_id="judg_1",
            action_id="act_1",
            agent_id="agent_1",
            decision="block",
            confidence=0.6,
            violations=[{"severity": 3, "type": "safety"}],
        )

        case = governance.get_next_case(reviewer_id="reviewer_1")
        if case:
            governance.submit_feedback(
                case_id=case.case_id,
                reviewer_id="reviewer_1",
                feedback_tags=[FeedbackTag.CORRECT_DECISION],
                rationale="Decision was appropriate",
            )

        # Run improvement cycle
        cycle_result = governance.continuous_improvement_cycle()

        assert cycle_result is not None
        assert "feedback_summary" in cycle_result
        assert "sla_metrics" in cycle_result
        assert "human_agreement" in cycle_result
        assert "needs_optimization" in cycle_result
        assert "recommendations" in cycle_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
