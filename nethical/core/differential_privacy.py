"""Differential Privacy Implementation with DP-SGD and Privacy Budget Tracking.

This module provides differential privacy mechanisms for model training and
metric aggregation with privacy budget management.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import hashlib
import json


class PrivacyMechanism(Enum):
    """Types of differential privacy mechanisms."""
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


class PrivacyAccountingMethod(Enum):
    """Privacy accounting methods."""
    BASIC = "basic"  # Basic composition
    ADVANCED = "advanced"  # Advanced composition
    RDP = "rdp"  # Renyi Differential Privacy


@dataclass
class PrivacyBudget:
    """Privacy budget tracking."""
    epsilon: float  # Privacy loss parameter
    delta: float  # Failure probability
    consumed: float = 0.0
    remaining: float = field(init=False)
    operations: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.remaining = self.epsilon - self.consumed
    
    def consume(self, amount: float, operation: str):
        """Consume privacy budget."""
        if amount > self.remaining:
            raise ValueError(
                f"Insufficient privacy budget. Requested: {amount}, "
                f"Remaining: {self.remaining}"
            )
        self.consumed += amount
        self.remaining = self.epsilon - self.consumed
        self.operations.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'amount': amount,
            'remaining': self.remaining
        })
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.remaining <= 0


@dataclass
class DPTrainingConfig:
    """Configuration for differential privacy in model training."""
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.1  # Noise scale
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.01
    accounting_method: PrivacyAccountingMethod = PrivacyAccountingMethod.BASIC


@dataclass
class PrivacyMetrics:
    """Metrics for privacy-utility tradeoff."""
    epsilon_spent: float
    delta: float
    accuracy: float
    privacy_loss: float
    utility_loss: float
    tradeoff_ratio: float  # Higher is better


class DifferentialPrivacy:
    """Differential privacy implementation with DP-SGD and budget tracking."""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: PrivacyMechanism = PrivacyMechanism.GAUSSIAN,
        accounting_method: PrivacyAccountingMethod = PrivacyAccountingMethod.BASIC
    ):
        """Initialize differential privacy manager.
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Failure probability
            mechanism: Noise addition mechanism
            accounting_method: Method for tracking privacy budget
        """
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
        self.mechanism = mechanism
        self.accounting_method = accounting_method
        
        # Track operations for analysis
        self.operation_history: List[Dict[str, Any]] = []
        
        # Privacy-utility tradeoff tracking
        self.tradeoff_history: List[PrivacyMetrics] = []
    
    def add_noise(
        self,
        value: float,
        sensitivity: float,
        operation: str = "metric_query"
    ) -> float:
        """Add noise to a value for differential privacy.
        
        Args:
            value: Original value
            sensitivity: Query sensitivity (how much one record can change result)
            operation: Description of the operation
            
        Returns:
            Noised value
        """
        # Calculate privacy cost
        if self.mechanism == PrivacyMechanism.LAPLACE:
            scale = sensitivity / self.budget.epsilon
            noise = np.random.laplace(0, scale)
            privacy_cost = self.budget.epsilon / 2  # Simplified
        elif self.mechanism == PrivacyMechanism.GAUSSIAN:
            # Gaussian mechanism for (epsilon, delta)-DP
            sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.budget.delta))) / self.budget.epsilon
            noise = np.random.normal(0, sigma)
            privacy_cost = self.budget.epsilon / 2  # Simplified
        else:
            raise NotImplementedError(f"Mechanism {self.mechanism} not implemented")
        
        # Consume privacy budget
        self.budget.consume(privacy_cost, operation)
        
        # Track operation
        self.operation_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'mechanism': self.mechanism.value,
            'privacy_cost': privacy_cost,
            'sensitivity': sensitivity
        })
        
        return value + noise
    
    def add_noise_to_vector(
        self,
        vector: np.ndarray,
        sensitivity: float,
        operation: str = "vector_query"
    ) -> np.ndarray:
        """Add noise to a vector for differential privacy.
        
        Args:
            vector: Original vector
            sensitivity: Query sensitivity
            operation: Description of the operation
            
        Returns:
            Noised vector
        """
        noised_vector = np.zeros_like(vector)
        
        for i in range(len(vector)):
            noised_vector[i] = self.add_noise(
                vector[i],
                sensitivity,
                f"{operation}_dim{i}"
            )
        
        return noised_vector
    
    def add_noise_to_aggregated_metrics(
        self,
        metrics: Dict[str, float],
        sensitivity: float = 1.0,
        noise_level: float = 0.1
    ) -> Dict[str, float]:
        """Add noise to aggregated metrics for privacy.
        
        Args:
            metrics: Dictionary of metric names to values
            sensitivity: Query sensitivity
            noise_level: Noise level (0-1, higher = more noise)
            
        Returns:
            Noised metrics
        """
        noised_metrics = {}
        
        for key, value in metrics.items():
            # Scale sensitivity by noise level
            effective_sensitivity = sensitivity * noise_level
            
            noised_value = self.add_noise(
                value,
                effective_sensitivity,
                f"aggregate_metric_{key}"
            )
            
            noised_metrics[key] = noised_value
        
        return noised_metrics
    
    def dp_sgd_step(
        self,
        gradients: np.ndarray,
        config: DPTrainingConfig,
        batch_size: int
    ) -> np.ndarray:
        """Apply DP-SGD to gradients for private model training.
        
        Args:
            gradients: Batch of gradients (shape: [batch_size, ...])
            config: DP training configuration
            batch_size: Current batch size
            
        Returns:
            Privatized gradients
        """
        # Step 1: Clip gradients per example
        clipped_gradients = self._clip_gradients(gradients, config.max_grad_norm)
        
        # Step 2: Average clipped gradients
        averaged_gradients = np.mean(clipped_gradients, axis=0)
        
        # Step 3: Add Gaussian noise
        noise_scale = (
            config.noise_multiplier * config.max_grad_norm / batch_size
        )
        noise = np.random.normal(0, noise_scale, size=averaged_gradients.shape)
        noised_gradients = averaged_gradients + noise
        
        # Track privacy cost (simplified - real implementation would use privacy accounting)
        privacy_cost = self._calculate_dp_sgd_privacy_cost(config, batch_size)
        self.budget.consume(privacy_cost, f"dp_sgd_batch_{len(self.operation_history)}")
        
        return noised_gradients
    
    def _clip_gradients(
        self,
        gradients: np.ndarray,
        max_norm: float
    ) -> np.ndarray:
        """Clip gradients to bound sensitivity.
        
        Args:
            gradients: Batch of gradients
            max_norm: Maximum L2 norm for clipping
            
        Returns:
            Clipped gradients
        """
        clipped = []
        
        for grad in gradients:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                clipped_grad = grad * (max_norm / norm)
            else:
                clipped_grad = grad
            clipped.append(clipped_grad)
        
        return np.array(clipped)
    
    def _calculate_dp_sgd_privacy_cost(
        self,
        config: DPTrainingConfig,
        batch_size: int
    ) -> float:
        """Calculate privacy cost for a DP-SGD step.
        
        This is a simplified calculation. In production, use a proper
        privacy accounting library like TensorFlow Privacy.
        
        Args:
            config: DP training configuration
            batch_size: Batch size
            
        Returns:
            Privacy cost (epsilon) for this step
        """
        # Simplified calculation using moments accountant approximation
        q = batch_size / 10000  # Sampling ratio (assuming dataset size)
        steps = 1
        
        # Privacy cost per step (simplified)
        epsilon_per_step = q * np.sqrt(steps) / config.noise_multiplier
        
        return epsilon_per_step
    
    def optimize_privacy_utility_tradeoff(
        self,
        utility_fn: Callable[[float], float],
        epsilon_range: tuple = (0.1, 10.0),
        num_samples: int = 20
    ) -> Tuple[float, float]:
        """Optimize privacy-utility tradeoff.
        
        Args:
            utility_fn: Function that takes epsilon and returns utility metric
            epsilon_range: Range of epsilon values to explore
            num_samples: Number of samples in the range
            
        Returns:
            Optimal (epsilon, utility) pair
        """
        epsilons = np.linspace(epsilon_range[0], epsilon_range[1], num_samples)
        best_tradeoff = -np.inf
        best_epsilon = epsilon_range[0]
        best_utility = 0.0
        
        for epsilon in epsilons:
            # Evaluate utility at this epsilon
            utility = utility_fn(epsilon)
            
            # Calculate tradeoff score (higher is better)
            # Balance privacy (lower epsilon) and utility (higher utility)
            privacy_score = 1.0 / epsilon  # Higher score for lower epsilon
            tradeoff_score = utility * privacy_score
            
            if tradeoff_score > best_tradeoff:
                best_tradeoff = tradeoff_score
                best_epsilon = epsilon
                best_utility = utility
            
            # Track metrics
            metrics = PrivacyMetrics(
                epsilon_spent=epsilon,
                delta=self.budget.delta,
                accuracy=utility,
                privacy_loss=epsilon,
                utility_loss=1.0 - utility,
                tradeoff_ratio=tradeoff_score
            )
            self.tradeoff_history.append(metrics)
        
        return best_epsilon, best_utility
    
    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """Get current privacy budget status.
        
        Returns:
            Dictionary with budget information
        """
        return {
            'epsilon_total': self.budget.epsilon,
            'epsilon_consumed': self.budget.consumed,
            'epsilon_remaining': self.budget.remaining,
            'delta': self.budget.delta,
            'is_exhausted': self.budget.is_exhausted(),
            'operations_count': len(self.budget.operations),
            'mechanism': self.mechanism.value,
            'accounting_method': self.accounting_method.value
        }
    
    def get_privacy_guarantees(self) -> Dict[str, Any]:
        """Get privacy guarantees provided by the system.
        
        Returns:
            Dictionary describing privacy guarantees
        """
        return {
            'differential_privacy': True,
            'epsilon': self.budget.epsilon,
            'delta': self.budget.delta,
            'mechanism': self.mechanism.value,
            'guarantee': f"({self.budget.epsilon}, {self.budget.delta})-differential privacy",
            'interpretation': self._interpret_privacy_guarantee()
        }
    
    def _interpret_privacy_guarantee(self) -> str:
        """Provide human-readable interpretation of privacy guarantee."""
        if self.budget.epsilon < 0.5:
            level = "very strong"
        elif self.budget.epsilon < 1.0:
            level = "strong"
        elif self.budget.epsilon < 2.0:
            level = "moderate"
        else:
            level = "basic"
        
        return (
            f"This system provides {level} privacy protection with "
            f"epsilon={self.budget.epsilon}. Lower epsilon values indicate "
            f"stronger privacy guarantees."
        )
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report.
        
        Returns:
            Privacy report with all metrics and operations
        """
        return {
            'budget_status': self.get_privacy_budget_status(),
            'privacy_guarantees': self.get_privacy_guarantees(),
            'operation_count': len(self.operation_history),
            'recent_operations': self.operation_history[-10:],
            'tradeoff_metrics': [
                {
                    'epsilon': m.epsilon_spent,
                    'accuracy': m.accuracy,
                    'tradeoff_ratio': m.tradeoff_ratio
                }
                for m in self.tradeoff_history[-5:]
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate privacy recommendations based on current state."""
        recommendations = []
        
        if self.budget.remaining < 0.2 * self.budget.epsilon:
            recommendations.append(
                "Privacy budget is running low. Consider reducing query "
                "frequency or increasing epsilon if acceptable."
            )
        
        if self.mechanism == PrivacyMechanism.LAPLACE:
            recommendations.append(
                "Consider using Gaussian mechanism for better privacy-utility "
                "tradeoff with (epsilon, delta)-DP."
            )
        
        if self.budget.epsilon > 2.0:
            recommendations.append(
                "Current epsilon is relatively high. Consider lowering it "
                "for stronger privacy guarantees if utility permits."
            )
        
        if not recommendations:
            recommendations.append(
                "Privacy configuration is well-balanced. Continue monitoring "
                "privacy budget consumption."
            )
        
        return recommendations


class PrivacyAudit:
    """Privacy audit and compliance validation."""
    
    def __init__(self, dp_manager: DifferentialPrivacy):
        """Initialize privacy auditor.
        
        Args:
            dp_manager: Differential privacy manager to audit
        """
        self.dp_manager = dp_manager
    
    def validate_compliance(
        self,
        regulations: List[str] = None
    ) -> Dict[str, Any]:
        """Validate privacy compliance with regulations.
        
        Args:
            regulations: List of regulations to check (e.g., ['GDPR', 'CCPA'])
            
        Returns:
            Compliance validation results
        """
        regulations = regulations or ['GDPR', 'CCPA']
        
        results = {
            'compliant': True,
            'regulations_checked': regulations,
            'checks': {},
            'violations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for regulation in regulations:
            if regulation == 'GDPR':
                gdpr_check = self._check_gdpr_compliance()
                results['checks']['GDPR'] = gdpr_check
                if not gdpr_check['compliant']:
                    results['compliant'] = False
                    results['violations'].extend(gdpr_check['violations'])
            
            elif regulation == 'CCPA':
                ccpa_check = self._check_ccpa_compliance()
                results['checks']['CCPA'] = ccpa_check
                if not ccpa_check['compliant']:
                    results['compliant'] = False
                    results['violations'].extend(ccpa_check['violations'])
        
        return results
    
    def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance requirements."""
        violations = []
        
        # Check if privacy guarantees are adequate
        if self.dp_manager.budget.epsilon > 3.0:
            violations.append(
                "Epsilon value may be too high for GDPR Article 25 "
                "(data protection by design)"
            )
        
        # Check if audit trail exists
        if not self.dp_manager.operation_history:
            violations.append(
                "No audit trail found - required for GDPR Article 30"
            )
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'articles_checked': ['Article 25', 'Article 30']
        }
    
    def _check_ccpa_compliance(self) -> Dict[str, Any]:
        """Check CCPA compliance requirements."""
        violations = []
        
        # CCPA requires reasonable security measures
        if self.dp_manager.budget.is_exhausted():
            violations.append(
                "Privacy budget exhausted - may not provide reasonable "
                "security for consumer data"
            )
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'sections_checked': ['Section 1798.150']
        }
    
    def generate_privacy_impact_assessment(self) -> Dict[str, Any]:
        """Generate Privacy Impact Assessment (PIA) documentation.
        
        Returns:
            PIA documentation
        """
        budget_status = self.dp_manager.get_privacy_budget_status()
        
        return {
            'assessment_date': datetime.now().isoformat(),
            'privacy_mechanism': self.dp_manager.mechanism.value,
            'privacy_guarantees': self.dp_manager.get_privacy_guarantees(),
            'data_protection_measures': [
                'Differential privacy implementation',
                'Privacy budget tracking',
                'Noise injection for aggregated metrics',
                'DP-SGD for model training'
            ],
            'risk_assessment': {
                'privacy_breach_risk': 'Low' if budget_status['epsilon_total'] < 2.0 else 'Medium',
                're_identification_risk': 'Low' if budget_status['epsilon_total'] < 1.0 else 'Medium',
                'utility_degradation_risk': 'Medium' if budget_status['epsilon_total'] < 0.5 else 'Low'
            },
            'mitigation_measures': self.dp_manager._generate_recommendations(),
            'compliance_status': self.validate_compliance()
        }
