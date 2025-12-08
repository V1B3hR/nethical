"""
Unit tests for Phase 6.1: AI/ML Security Framework
"""

import pytest
import numpy as np
from datetime import datetime

from nethical.security.ai_ml_security import (
    AdversarialDefenseSystem,
    ModelPoisoningDetector,
    DifferentialPrivacyManager,
    FederatedLearningCoordinator,
    ExplainableAISystem,
    AIMLSecurityManager,
    AdversarialAttackType,
    PoisoningType,
    PrivacyMechanism,
    PrivacyBudget,
)


class TestAdversarialDefenseSystem:
    """Test adversarial example detection."""

    def test_initialization(self):
        """Test adversarial defense initialization."""
        defense = AdversarialDefenseSystem(
            perturbation_threshold=0.15, confidence_threshold=0.85
        )

        assert defense.perturbation_threshold == 0.15
        assert defense.confidence_threshold == 0.85
        assert defense.enable_input_smoothing is True
        assert len(defense.detection_history) == 0

    def test_detect_clean_input(self):
        """Test detection with clean (non-adversarial) input."""
        defense = AdversarialDefenseSystem()

        def model_func(x):
            return "class_A"

        result = defense.detect_adversarial_example(
            input_data=[1.0, 2.0, 3.0],
            model_prediction_func=model_func,
            baseline_input=[1.0, 2.0, 3.0],
        )

        assert result.is_adversarial is False
        assert result.perturbation_magnitude < 0.01
        assert result.attack_type is None

    def test_detect_adversarial_input(self):
        """Test detection with adversarial input."""
        defense = AdversarialDefenseSystem(perturbation_threshold=0.05)

        def model_func(x):
            return "class_B"

        result = defense.detect_adversarial_example(
            input_data=[1.5, 2.5, 3.5],
            model_prediction_func=model_func,
            baseline_input=[1.0, 2.0, 3.0],
        )

        assert result.is_adversarial is True
        assert result.confidence > 0.0
        assert result.attack_type is not None

    def test_perturbation_calculation_numeric(self):
        """Test perturbation calculation for numeric inputs."""
        defense = AdversarialDefenseSystem()

        input_data = [1.0, 2.0, 3.0]
        baseline = [1.0, 2.0, 3.0]

        perturbation = defense._calculate_perturbation(input_data, baseline)
        assert perturbation == 0.0

        perturbed_input = [1.5, 2.5, 3.5]
        perturbation = defense._calculate_perturbation(perturbed_input, baseline)
        assert perturbation > 0.0

    def test_perturbation_calculation_text(self):
        """Test perturbation calculation for text inputs."""
        defense = AdversarialDefenseSystem()

        input_data = "hello world"
        baseline = "hello world"

        perturbation = defense._calculate_perturbation(input_data, baseline)
        assert perturbation == 0.0

        perturbed_input = "hello world!!!"
        perturbation = defense._calculate_perturbation(perturbed_input, baseline)
        assert perturbation > 0.0

    def test_attack_type_identification(self):
        """Test attack type identification."""
        defense = AdversarialDefenseSystem()

        # High perturbation suggests PGD
        attack_type = defense._identify_attack_type(
            perturbation_mag=0.6, consistency_score=0.5
        )
        assert attack_type == AdversarialAttackType.PGD

        # Low consistency suggests DeepFool
        attack_type = defense._identify_attack_type(
            perturbation_mag=0.2, consistency_score=0.2
        )
        assert attack_type == AdversarialAttackType.DEEPFOOL

    def test_detection_statistics(self):
        """Test detection statistics tracking."""
        defense = AdversarialDefenseSystem()

        def model_func(x):
            return "class"

        # Detect several inputs
        for i in range(5):
            defense.detect_adversarial_example(
                input_data=[float(i)] * 3,
                model_prediction_func=model_func,
                baseline_input=[0.0, 0.0, 0.0],
            )

        stats = defense.get_detection_statistics()

        assert stats["total_detections"] == 5
        assert "adversarial_count" in stats
        assert "clean_count" in stats
        assert "adversarial_rate" in stats


class TestModelPoisoningDetector:
    """Test model poisoning detection."""

    def test_initialization(self):
        """Test poisoning detector initialization."""
        detector = ModelPoisoningDetector(
            gradient_threshold=2.5, loss_anomaly_threshold=3.5
        )

        assert detector.gradient_threshold == 2.5
        assert detector.loss_anomaly_threshold == 3.5
        assert detector.enable_activation_analysis is True

    def test_detect_clean_training(self):
        """Test detection with clean training data."""
        detector = ModelPoisoningDetector()

        result = detector.detect_poisoning(
            training_batch=[1, 2, 3, 4, 5],
            gradients=[0.1, 0.2, 0.15, 0.18],
            loss_values=[0.5, 0.48, 0.46, 0.45],
        )

        assert result.is_poisoned is False
        assert result.gradient_anomaly_score >= 0.0

    def test_detect_gradient_poisoning(self):
        """Test detection of gradient manipulation."""
        detector = ModelPoisoningDetector(gradient_threshold=1.0)

        # Build up baseline gradients
        for _ in range(10):
            detector.detect_poisoning(
                training_batch=[1, 2, 3], gradients=[0.1, 0.1, 0.1]
            )

        # Inject anomalous gradients
        result = detector.detect_poisoning(
            training_batch=[1, 2, 3], gradients=[10.0, 10.0, 10.0]
        )

        assert result.gradient_anomaly_score > 0.0

    def test_detect_loss_anomalies(self):
        """Test detection of loss anomalies."""
        detector = ModelPoisoningDetector()

        result = detector.detect_poisoning(
            training_batch=[1, 2, 3],
            loss_values=[0.1, 0.1, 5.0, 0.1],  # Anomalous spike
        )

        # Should detect high variance
        assert result.gradient_anomaly_score >= 0.0

    def test_poisoning_type_identification(self):
        """Test poisoning type identification."""
        detector = ModelPoisoningDetector()

        # Gradient manipulation
        poison_type = detector._identify_poisoning_type(
            gradient_score=5.0, loss_score=1.0
        )
        assert poison_type == PoisoningType.GRADIENT_MANIPULATION

        # Label flipping
        poison_type = detector._identify_poisoning_type(
            gradient_score=1.0, loss_score=10.0
        )
        assert poison_type == PoisoningType.LABEL_FLIPPING

    def test_affected_samples_estimation(self):
        """Test estimation of affected samples."""
        detector = ModelPoisoningDetector()

        batch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        affected = detector._estimate_affected_samples(batch, is_poisoned=True)
        assert affected > 0

        affected = detector._estimate_affected_samples(batch, is_poisoned=False)
        assert affected == 0

    def test_detection_statistics(self):
        """Test poisoning detection statistics."""
        detector = ModelPoisoningDetector()

        for i in range(10):
            detector.detect_poisoning(training_batch=[1, 2, 3], gradients=[0.1 * i] * 3)

        stats = detector.get_detection_statistics()

        assert stats["total_detections"] == 10
        assert "poisoned_count" in stats
        assert "clean_count" in stats


class TestDifferentialPrivacyManager:
    """Test differential privacy."""

    def test_initialization(self):
        """Test privacy manager initialization."""
        manager = DifferentialPrivacyManager(epsilon=2.0, delta=1e-6)

        assert manager.budget.epsilon == 2.0
        assert manager.budget.delta == 1e-6
        assert manager.mechanism == PrivacyMechanism.LAPLACE

    def test_invalid_epsilon(self):
        """Test validation of epsilon parameter."""
        with pytest.raises(ValueError):
            DifferentialPrivacyManager(epsilon=0.0)

        with pytest.raises(ValueError):
            DifferentialPrivacyManager(epsilon=-1.0)

    def test_invalid_delta(self):
        """Test validation of delta parameter."""
        with pytest.raises(ValueError):
            DifferentialPrivacyManager(delta=-0.1)

        with pytest.raises(ValueError):
            DifferentialPrivacyManager(delta=1.0)

    def test_add_noise_laplace(self):
        """Test Laplace noise addition."""
        manager = DifferentialPrivacyManager(
            epsilon=1.0, mechanism=PrivacyMechanism.LAPLACE
        )

        original_value = 100.0
        noised_value, success = manager.add_noise(original_value)

        assert success is True
        assert noised_value != original_value  # Should have noise
        assert abs(noised_value - original_value) < 100  # Reasonable noise

    def test_add_noise_gaussian(self):
        """Test Gaussian noise addition."""
        manager = DifferentialPrivacyManager(
            epsilon=1.0, mechanism=PrivacyMechanism.GAUSSIAN
        )

        original_value = 50.0
        noised_value, success = manager.add_noise(original_value)

        assert success is True
        assert noised_value != original_value

    def test_budget_tracking(self):
        """Test privacy budget tracking."""
        manager = DifferentialPrivacyManager(epsilon=1.0)

        initial_budget = manager.budget.remaining_epsilon

        manager.add_noise(100.0, epsilon_cost=0.2)

        assert manager.budget.spent_epsilon > 0
        assert manager.budget.remaining_epsilon < initial_budget
        assert manager.budget.query_count == 1

    def test_budget_depletion(self):
        """Test privacy budget depletion."""
        manager = DifferentialPrivacyManager(epsilon=0.5)

        # Deplete budget
        manager.add_noise(100.0, epsilon_cost=0.6)

        assert manager.budget.is_depleted is True

        # Should fail after depletion
        _, success = manager.add_noise(100.0)
        assert success is False

    def test_privacy_loss_tracking(self):
        """Test privacy loss calculation."""
        manager = DifferentialPrivacyManager(epsilon=2.0)

        manager.add_noise(100.0, epsilon_cost=0.5)

        loss = manager.get_privacy_loss()

        assert loss["epsilon_loss"] == 0.5
        assert loss["epsilon_remaining"] == 1.5

    def test_budget_reset(self):
        """Test privacy budget reset."""
        manager = DifferentialPrivacyManager(epsilon=1.0)

        manager.add_noise(100.0)
        assert manager.budget.query_count > 0

        manager.reset_budget()

        assert manager.budget.spent_epsilon == 0.0
        assert manager.budget.query_count == 0
        assert len(manager.query_log) == 0

    def test_audit_log_export(self):
        """Test privacy query audit log."""
        manager = DifferentialPrivacyManager(epsilon=1.0)

        for i in range(5):
            manager.add_noise(float(i))

        audit_log = manager.export_audit_log()

        assert len(audit_log) == 5
        assert all("query_id" in entry for entry in audit_log)
        assert all("epsilon_cost" in entry for entry in audit_log)


class TestPrivacyBudget:
    """Test privacy budget calculations."""

    def test_remaining_epsilon(self):
        """Test remaining epsilon calculation."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)

        budget.spent_epsilon = 0.5
        assert budget.remaining_epsilon == 1.5

        budget.spent_epsilon = 2.5  # Overspent
        assert budget.remaining_epsilon == 0.0

    def test_is_depleted(self):
        """Test budget depletion check."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)

        assert budget.is_depleted is False

        budget.spent_epsilon = 1.0
        assert budget.is_depleted is True

    def test_to_dict(self):
        """Test budget dictionary conversion."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        budget.spent_epsilon = 0.3
        budget.query_count = 5

        budget_dict = budget.to_dict()

        assert budget_dict["epsilon"] == 1.0
        assert budget_dict["spent_epsilon"] == 0.3
        assert budget_dict["remaining_epsilon"] == 0.7
        assert budget_dict["query_count"] == 5


class TestFederatedLearningCoordinator:
    """Test federated learning."""

    def test_initialization(self):
        """Test federated coordinator initialization."""
        coordinator = FederatedLearningCoordinator(
            min_participants=5, enable_secure_aggregation=True
        )

        assert coordinator.min_participants == 5
        assert coordinator.enable_secure_aggregation is True
        assert len(coordinator.rounds) == 0

    def test_insufficient_participants(self):
        """Test error with insufficient participants."""
        coordinator = FederatedLearningCoordinator(min_participants=5)

        with pytest.raises(ValueError):
            coordinator.aggregate_updates([{"id": 1}, {"id": 2}])

    def test_aggregate_updates(self):
        """Test aggregation of participant updates."""
        coordinator = FederatedLearningCoordinator(min_participants=3)

        updates = [
            {"id": 1, "gradients": [0.1, 0.2, 0.3]},
            {"id": 2, "gradients": [0.15, 0.25, 0.35]},
            {"id": 3, "gradients": [0.12, 0.22, 0.32]},
        ]

        round_result = coordinator.aggregate_updates(updates)

        assert round_result.participant_count == 3
        assert round_result.validation_accuracy > 0.0
        assert round_result.aggregated_weights is not None

    def test_poisoning_detection_in_aggregation(self):
        """Test poisoning detection during aggregation."""
        coordinator = FederatedLearningCoordinator(
            min_participants=3, enable_poisoning_detection=True
        )

        updates = [
            {"id": 1, "gradients": [0.1, 0.2]},
            {"id": 2, "gradients": [0.15, 0.25]},
            {"id": 3, "gradients": [100.0, 200.0]},  # Malicious
        ]

        round_result = coordinator.aggregate_updates(updates)

        # Should complete despite malicious update
        assert round_result.participant_count >= 3

    def test_secure_aggregation(self):
        """Test secure aggregation mode."""
        coordinator = FederatedLearningCoordinator(
            min_participants=2, enable_secure_aggregation=True
        )

        updates = [
            {"id": 1, "weights": {"layer1": [1.0, 2.0]}},
            {"id": 2, "weights": {"layer1": [1.5, 2.5]}},
        ]

        round_result = coordinator.aggregate_updates(updates)

        assert round_result.aggregated_weights is not None
        assert "update_count" in round_result.aggregated_weights

    def test_training_statistics(self):
        """Test federated training statistics."""
        coordinator = FederatedLearningCoordinator(min_participants=2)

        for i in range(3):
            updates = [{"id": 1, "gradients": [0.1]}, {"id": 2, "gradients": [0.2]}]
            coordinator.aggregate_updates(updates)

        stats = coordinator.get_training_statistics()

        assert stats["total_rounds"] == 3
        assert stats["average_accuracy"] > 0.0
        assert "total_participants" in stats


class TestExplainableAISystem:
    """Test explainable AI."""

    def test_initialization(self):
        """Test explainable AI initialization."""
        explainer = ExplainableAISystem(
            compliance_frameworks=["GDPR", "HIPAA"], explanation_method="shap"
        )

        assert "GDPR" in explainer.compliance_frameworks
        assert explainer.explanation_method == "shap"

    def test_generate_explanation(self):
        """Test explanation generation."""
        explainer = ExplainableAISystem()

        features = {"age": 45, "income": 75000, "credit_score": 720}

        report = explainer.generate_explanation(
            model_id="credit_model_v1", input_features=features, prediction="approved"
        )

        assert report.model_id == "credit_model_v1"
        assert report.prediction == "approved"
        assert len(report.feature_importance) > 0
        assert report.confidence_score >= 0.0
        assert len(report.human_readable_explanation) > 0

    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        explainer = ExplainableAISystem()

        features = {"feature1": 100, "feature2": 50, "feature3": 25}

        importance = explainer._calculate_feature_importance(features, None)

        assert len(importance) == 3
        assert all(0 <= v <= 1 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 0.01  # Should sum to 1

    def test_human_explanation_generation(self):
        """Test human-readable explanation."""
        explainer = ExplainableAISystem(compliance_frameworks=["GDPR"])

        features = {"age": 30, "score": 85}
        importance = {"age": 0.4, "score": 0.6}

        explanation = explainer._generate_human_explanation(
            features, importance, "accepted"
        )

        assert "accepted" in explanation
        assert "GDPR" in explanation
        assert len(explanation) > 0

    def test_compliance_report_export(self):
        """Test compliance report export."""
        explainer = ExplainableAISystem()

        # Generate some explanations
        for i in range(5):
            explainer.generate_explanation(
                model_id=f"model_{i}",
                input_features={"feature": i},
                prediction=f"class_{i % 2}",
            )

        report = explainer.export_compliance_report()

        assert report["total_explanations"] == 5
        assert "compliance_frameworks" in report
        assert "average_confidence" in report
        assert len(report["explanations"]) <= 100


class TestAIMLSecurityManager:
    """Test comprehensive AI/ML security manager."""

    def test_initialization_all_enabled(self):
        """Test manager with all features enabled."""
        manager = AIMLSecurityManager(
            enable_adversarial_defense=True,
            enable_poisoning_detection=True,
            enable_differential_privacy=True,
            enable_federated_learning=True,
            enable_explainable_ai=True,
        )

        assert manager.adversarial_defense is not None
        assert manager.poisoning_detector is not None
        assert manager.privacy_manager is not None
        assert manager.federated_coordinator is not None
        assert manager.explainable_ai is not None

    def test_initialization_selective(self):
        """Test manager with selective features."""
        manager = AIMLSecurityManager(
            enable_adversarial_defense=True,
            enable_poisoning_detection=False,
            enable_differential_privacy=True,
            enable_federated_learning=False,
            enable_explainable_ai=True,
        )

        assert manager.adversarial_defense is not None
        assert manager.poisoning_detector is None
        assert manager.privacy_manager is not None
        assert manager.federated_coordinator is None
        assert manager.explainable_ai is not None

    def test_security_status(self):
        """Test comprehensive security status."""
        manager = AIMLSecurityManager()

        status = manager.get_security_status()

        assert "adversarial_defense" in status
        assert "poisoning_detection" in status
        assert "differential_privacy" in status
        assert "federated_learning" in status
        assert "explainable_ai" in status

        # Check enabled flags
        assert status["adversarial_defense"]["enabled"] is True
        assert status["differential_privacy"]["enabled"] is True

    def test_security_report_export(self):
        """Test security report export."""
        manager = AIMLSecurityManager()

        report = manager.export_security_report()

        assert "timestamp" in report
        assert "security_status" in report
        assert "compliance" in report

    def test_privacy_epsilon_configuration(self):
        """Test privacy epsilon configuration."""
        manager = AIMLSecurityManager(privacy_epsilon=2.0, privacy_delta=1e-6)

        assert manager.privacy_manager.budget.epsilon == 2.0
        assert manager.privacy_manager.budget.delta == 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
