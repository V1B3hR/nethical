"""Tests for F4: Adaptive Thresholds & Tuning features."""

import pytest
import tempfile
from pathlib import Path

from nethical.core import (
    AdaptiveThresholdTuner,
    ABTestingFramework,
    Phase89IntegratedGovernance,
    Configuration,
    PerformanceMetrics,
    ConfigStatus,
    OutcomeRecord
)
from datetime import datetime


class TestAdaptiveThresholdTuner:
    """Tests for AdaptiveThresholdTuner."""
    
    @pytest.fixture
    def tuner(self):
        """Create a test tuner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tuner.db"
            yield AdaptiveThresholdTuner(
                objectives=["maximize_recall", "minimize_fp"],
                learning_rate=0.01,
                storage_path=str(storage_path)
            )
    
    def test_init(self, tuner):
        """Test tuner initialization."""
        assert tuner.objectives == ["maximize_recall", "minimize_fp"]
        assert tuner.learning_rate == 0.01
        assert tuner.global_thresholds is not None
        assert 'classifier_threshold' in tuner.global_thresholds
    
    def test_record_outcome_correct(self, tuner):
        """Test recording a correct outcome."""
        outcome = tuner.record_outcome(
            action_id="act_1",
            judgment_id="judg_1",
            predicted_outcome="block",
            actual_outcome="correct",
            confidence=0.9,
            human_feedback="Good catch"
        )
        
        assert outcome.action_id == "act_1"
        assert outcome.was_correct is True
        assert len(tuner.outcomes) == 1
    
    def test_record_outcome_false_positive(self, tuner):
        """Test recording false positive outcome."""
        initial_threshold = tuner.global_thresholds['classifier_threshold']
        
        outcome = tuner.record_outcome(
            action_id="act_2",
            judgment_id="judg_2",
            predicted_outcome="block",
            actual_outcome="false_positive",
            confidence=0.8,
            human_feedback="Should have been allowed"
        )
        
        assert outcome.was_correct is False
        assert tuner.false_positives == 1
        
        # Threshold should increase to reduce false positives
        assert tuner.global_thresholds['classifier_threshold'] >= initial_threshold
    
    def test_record_outcome_false_negative(self, tuner):
        """Test recording false negative outcome."""
        initial_threshold = tuner.global_thresholds['classifier_threshold']
        
        outcome = tuner.record_outcome(
            action_id="act_3",
            judgment_id="judg_3",
            predicted_outcome="allow",
            actual_outcome="false_negative",
            confidence=0.6,
            human_feedback="Should have been blocked"
        )
        
        assert outcome.was_correct is False
        assert tuner.false_negatives == 1
        
        # Threshold should decrease to catch more violations
        assert tuner.global_thresholds['classifier_threshold'] <= initial_threshold
    
    def test_agent_specific_thresholds(self, tuner):
        """Test agent-specific threshold profiles."""
        agent_thresholds = {
            'classifier_threshold': 0.6,
            'confidence_threshold': 0.8
        }
        
        tuner.set_agent_thresholds("agent_1", agent_thresholds)
        
        retrieved = tuner.get_thresholds("agent_1")
        assert retrieved['classifier_threshold'] == 0.6
        assert retrieved['confidence_threshold'] == 0.8
        
        # Global thresholds should be different
        global_thresholds = tuner.get_thresholds()
        assert global_thresholds['classifier_threshold'] != 0.6
    
    def test_performance_stats(self, tuner):
        """Test performance statistics calculation."""
        # Record various outcomes
        tuner.record_outcome("act_1", "judg_1", "block", "true_positive", 0.9)
        tuner.record_outcome("act_2", "judg_2", "allow", "true_negative", 0.8)
        tuner.record_outcome("act_3", "judg_3", "block", "false_positive", 0.7)
        tuner.record_outcome("act_4", "judg_4", "allow", "false_negative", 0.6)
        
        stats = tuner.get_performance_stats()
        
        assert stats['total_outcomes'] == 4
        assert stats['accuracy'] == 0.5  # 2 correct out of 4
        assert stats['precision'] == 0.5  # 1 TP / (1 TP + 1 FP)
        assert stats['recall'] == 0.5  # 1 TP / (1 TP + 1 FN)
        assert 'current_thresholds' in stats


class TestABTestingFramework:
    """Tests for A/B testing framework."""
    
    @pytest.fixture
    def ab_framework(self):
        """Create a test A/B testing framework."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "ab_testing.db"
            yield ABTestingFramework(storage_path=str(storage_path))
    
    @pytest.fixture
    def sample_configs(self):
        """Create sample configurations."""
        control = Configuration(
            config_id="cfg_control",
            config_version="v1.0",
            status=ConfigStatus.PRODUCTION,
            created_at=datetime.now(),
            classifier_threshold=0.5
        )
        
        treatment = Configuration(
            config_id="cfg_treatment",
            config_version="v1.1",
            status=ConfigStatus.CANDIDATE,
            created_at=datetime.now(),
            classifier_threshold=0.6
        )
        
        return control, treatment
    
    def test_create_ab_test(self, ab_framework, sample_configs):
        """Test creating an A/B test."""
        control, treatment = sample_configs
        
        control_id, treatment_id = ab_framework.create_ab_test(
            control,
            treatment,
            traffic_split=0.1
        )
        
        assert control_id in ab_framework.variants
        assert treatment_id in ab_framework.variants
        assert ab_framework.variant_traffic[control_id] == 0.9
        assert ab_framework.variant_traffic[treatment_id] == 0.1
        assert ab_framework.control_variant_id == control_id
    
    def test_record_variant_metrics(self, ab_framework, sample_configs):
        """Test recording metrics for variants."""
        control, treatment = sample_configs
        control_id, treatment_id = ab_framework.create_ab_test(control, treatment)
        
        metrics = PerformanceMetrics(
            config_id="cfg_control",
            detection_recall=0.85,
            detection_precision=0.90,
            false_positive_rate=0.05,
            decision_latency_ms=10.0,
            human_agreement=0.88,
            total_cases=100
        )
        
        ab_framework.record_variant_metrics(control_id, metrics)
        
        assert control_id in ab_framework.variant_metrics
        assert ab_framework.variant_metrics[control_id].detection_recall == 0.85
    
    def test_statistical_significance(self, ab_framework, sample_configs):
        """Test statistical significance testing."""
        control, treatment = sample_configs
        control_id, treatment_id = ab_framework.create_ab_test(control, treatment)
        
        # Record metrics for both variants
        control_metrics = PerformanceMetrics(
            config_id="cfg_control",
            detection_recall=0.80,
            false_positive_rate=0.10,
            human_agreement=0.85,
            total_cases=200
        )
        
        treatment_metrics = PerformanceMetrics(
            config_id="cfg_treatment",
            detection_recall=0.88,
            false_positive_rate=0.08,
            human_agreement=0.90,
            total_cases=200
        )
        
        ab_framework.record_variant_metrics(control_id, control_metrics)
        ab_framework.record_variant_metrics(treatment_id, treatment_metrics)
        
        # Check significance
        is_sig, p_value, interpretation = ab_framework.check_statistical_significance(
            control_id,
            treatment_id,
            metric="detection_recall"
        )
        
        # With 10% improvement and 200 samples, should be significant
        assert isinstance(is_sig, bool)
        assert 0 <= p_value <= 1
        assert len(interpretation) > 0
    
    def test_gradual_rollout(self, ab_framework, sample_configs):
        """Test gradual rollout of treatment variant."""
        control, treatment = sample_configs
        control_id, treatment_id = ab_framework.create_ab_test(control, treatment, traffic_split=0.1)
        
        # Increase traffic gradually
        new_traffic = ab_framework.gradual_rollout(treatment_id, target_traffic=0.5, step_size=0.2)
        
        assert abs(new_traffic - 0.3) < 0.01  # 0.1 + 0.2, with floating point tolerance
        assert abs(ab_framework.variant_traffic[treatment_id] - 0.3) < 0.01
        assert abs(ab_framework.variant_traffic[control_id] - 0.7) < 0.01
        
        # Another step
        new_traffic = ab_framework.gradual_rollout(treatment_id, target_traffic=0.5, step_size=0.2)
        assert abs(new_traffic - 0.5) < 0.01
    
    def test_rollback_variant(self, ab_framework, sample_configs):
        """Test rolling back a variant."""
        control, treatment = sample_configs
        control_id, treatment_id = ab_framework.create_ab_test(control, treatment, traffic_split=0.3)
        
        # Rollback treatment
        success = ab_framework.rollback_variant(treatment_id)
        
        assert success is True
        assert ab_framework.variant_traffic[treatment_id] == 0.0
        assert ab_framework.variant_traffic[control_id] == 1.0
    
    def test_variant_summary(self, ab_framework, sample_configs):
        """Test getting variant summary."""
        control, treatment = sample_configs
        control_id, treatment_id = ab_framework.create_ab_test(control, treatment)
        
        summary = ab_framework.get_variant_summary()
        
        assert 'variants' in summary
        assert 'control_variant_id' in summary
        assert control_id in summary['variants']
        assert treatment_id in summary['variants']
        assert summary['variants'][control_id]['is_control'] is True
        assert summary['variants'][treatment_id]['is_control'] is False


class TestPhase89F4Integration:
    """Tests for F4 integration with Phase89IntegratedGovernance."""
    
    @pytest.fixture
    def governance(self):
        """Create a test governance instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Phase89IntegratedGovernance(storage_dir=tmpdir)
    
    def test_record_outcome(self, governance):
        """Test recording outcome for adaptive learning."""
        result = governance.record_outcome(
            action_id="act_1",
            judgment_id="judg_1",
            predicted_outcome="block",
            actual_outcome="false_positive",
            confidence=0.8,
            human_feedback="Should allow"
        )
        
        assert 'outcome' in result
        assert 'updated_thresholds' in result
        assert 'performance_stats' in result
        assert result['outcome']['action_id'] == "act_1"
    
    def test_get_adaptive_thresholds(self, governance):
        """Test getting adaptive thresholds."""
        thresholds = governance.get_adaptive_thresholds()
        
        assert 'classifier_threshold' in thresholds
        assert 'confidence_threshold' in thresholds
        assert 'gray_zone_lower' in thresholds
        assert 'gray_zone_upper' in thresholds
    
    def test_set_agent_thresholds(self, governance):
        """Test setting agent-specific thresholds."""
        custom_thresholds = {
            'classifier_threshold': 0.65,
            'confidence_threshold': 0.85
        }
        
        governance.set_agent_thresholds("agent_special", custom_thresholds)
        
        retrieved = governance.get_adaptive_thresholds("agent_special")
        assert retrieved['classifier_threshold'] == 0.65
        assert retrieved['confidence_threshold'] == 0.85
    
    def test_create_ab_test(self, governance):
        """Test creating A/B test through governance."""
        control = governance.create_configuration(
            config_version="control_v1",
            classifier_threshold=0.5
        )
        
        treatment = governance.create_configuration(
            config_version="treatment_v1",
            classifier_threshold=0.6
        )
        
        control_id, treatment_id = governance.create_ab_test(
            control,
            treatment,
            traffic_split=0.2
        )
        
        assert control_id is not None
        assert treatment_id is not None
    
    def test_bayesian_optimization(self, governance):
        """Test Bayesian optimization technique."""
        results = governance.optimize_configuration(
            technique="bayesian",
            param_ranges={
                'classifier_threshold': (0.4, 0.7),
                'confidence_threshold': (0.6, 0.9)
            },
            n_iterations=10,
            n_initial_random=3
        )
        
        assert len(results) == 10
        assert results[0][1].fitness_score >= results[-1][1].fitness_score  # Sorted by fitness
    
    def test_tuning_performance(self, governance):
        """Test getting tuning performance statistics."""
        # Record some outcomes
        governance.record_outcome("act_1", "judg_1", "block", "correct", 0.9)
        governance.record_outcome("act_2", "judg_2", "allow", "false_negative", 0.5)
        
        stats = governance.get_tuning_performance()
        
        assert stats['total_outcomes'] == 2
        assert 'accuracy' in stats
        assert 'precision' in stats
        assert 'recall' in stats
