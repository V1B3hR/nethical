"""Phase 8-9 Integration: Human-in-the-Loop & Continuous Optimization

.. deprecated::
   This module is deprecated and maintained only for backward compatibility.
   Please use :class:`nethical.core.integrated_governance.IntegratedGovernance` instead,
   which provides a unified interface for all phases (3, 4, 5-7, 8-9).

This module provides integrated access to:
- Phase 8: Escalation queue, human feedback, SLA tracking
- Phase 9: Multi-objective optimization, configuration management, promotion gates

Migration Guide:
    Old::
        from nethical.core.phase89_integration import Phase89IntegratedGovernance
        governance = Phase89IntegratedGovernance(
            storage_dir="./data",
            triage_sla_seconds=3600
        )
    
    New::
        from nethical.core.integrated_governance import IntegratedGovernance
        governance = IntegratedGovernance(
            storage_dir="./data",
            enable_escalation=True,
            enable_optimization=True,
            triage_sla_seconds=3600
        )

Usage:
    from nethical.core import Phase89IntegratedGovernance
    
    governance = Phase89IntegratedGovernance(
        storage_dir="./data",
        triage_sla_seconds=3600,
        resolution_sla_seconds=86400
    )
    
    # Process action and escalate if needed
    result = governance.process_with_escalation(action, violations)
    
    # Human review
    case = governance.get_next_case(reviewer_id="reviewer_1")
    governance.submit_feedback(case.case_id, ...)
    
    # Optimization
    best_config = governance.optimize_configuration(technique="random_search")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

from .human_feedback import (
    EscalationQueue,
    FeedbackTag,
    ReviewPriority,
    HumanFeedback,
    EscalationCase,
    SLAMetrics
)
from .optimization import (
    MultiObjectiveOptimizer,
    Configuration,
    PerformanceMetrics,
    OptimizationTechnique,
    ConfigStatus,
    AdaptiveThresholdTuner,
    ABTestingFramework
)


class Phase89IntegratedGovernance:
    """Integrated governance with human-in-the-loop and optimization.
    
    .. deprecated::
       This class is deprecated. Use :class:`~nethical.core.integrated_governance.IntegratedGovernance` instead.
       Phase89IntegratedGovernance is maintained for backward compatibility only.
    
    Features:
    - Automated escalation based on decision criteria
    - Human review workflow with SLA tracking
    - Continuous optimization of system parameters
    - Promotion gate validation
    - A/B testing support
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        triage_sla_seconds: float = 3600,
        resolution_sla_seconds: float = 86400,
        auto_escalate_on_block: bool = True,
        auto_escalate_on_low_confidence: bool = True,
        low_confidence_threshold: float = 0.7
    ):
        """Initialize integrated governance.
        
        .. deprecated::
           Use IntegratedGovernance instead for unified access to all phases.
        
        Args:
            storage_dir: Directory for data storage
            triage_sla_seconds: SLA for starting review (default 1 hour)
            resolution_sla_seconds: SLA for completing review (default 24 hours)
            auto_escalate_on_block: Auto-escalate BLOCK/TERMINATE decisions
            auto_escalate_on_low_confidence: Auto-escalate low confidence decisions
            low_confidence_threshold: Threshold for low confidence escalation
        """
        warnings.warn(
            "Phase89IntegratedGovernance is deprecated and will be removed in a future version. "
            "Use IntegratedGovernance from nethical.core.integrated_governance instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        # Phase 8: Human-in-the-Loop
        self.escalation_queue = EscalationQueue(
            storage_path=str(storage_path / "escalations.db"),
            triage_sla_seconds=triage_sla_seconds,
            resolution_sla_seconds=resolution_sla_seconds
        )
        
        # Phase 9: Optimization
        self.optimizer = MultiObjectiveOptimizer(
            storage_path=str(storage_path / "optimization.db")
        )
        
        # F4: Adaptive Thresholds & Tuning
        self.adaptive_tuner = AdaptiveThresholdTuner(
            objectives=["maximize_recall", "minimize_fp"],
            learning_rate=0.01,
            storage_path=str(storage_path / "adaptive_tuning.db")
        )
        
        self.ab_testing = ABTestingFramework(
            storage_path=str(storage_path / "ab_testing.db")
        )
        
        # Configuration
        self.auto_escalate_on_block = auto_escalate_on_block
        self.auto_escalate_on_low_confidence = auto_escalate_on_low_confidence
        self.low_confidence_threshold = low_confidence_threshold
        
        # Current active configuration
        self.active_config: Optional[Configuration] = None
    
    # ==================== Phase 8: Escalation & Review ====================
    
    def should_escalate(
        self,
        decision: str,
        confidence: float,
        violations: List[Dict[str, Any]]
    ) -> Tuple[bool, ReviewPriority]:
        """Determine if case should be escalated.
        
        Args:
            decision: Decision made (allow, block, etc.)
            confidence: Confidence in decision
            violations: List of violations
            
        Returns:
            Tuple of (should_escalate, priority)
        """
        # Emergency/critical violations
        critical_violations = [
            v for v in violations 
            if v.get('severity', 0) >= 4  # CRITICAL or EMERGENCY
        ]
        if critical_violations:
            return True, ReviewPriority.EMERGENCY
        
        # Blocking decisions
        if self.auto_escalate_on_block and decision in ['block', 'terminate']:
            return True, ReviewPriority.HIGH
        
        # Low confidence decisions
        if self.auto_escalate_on_low_confidence and confidence < self.low_confidence_threshold:
            # More violations = higher priority
            if len(violations) >= 3:
                return True, ReviewPriority.HIGH
            elif len(violations) >= 1:
                return True, ReviewPriority.MEDIUM
        
        return False, ReviewPriority.LOW
    
    def process_with_escalation(
        self,
        judgment_id: str,
        action_id: str,
        agent_id: str,
        decision: str,
        confidence: float,
        violations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process action and escalate if needed.
        
        Args:
            judgment_id: Judgment ID
            action_id: Action ID
            agent_id: Agent ID
            decision: Decision made
            confidence: Decision confidence
            violations: List of violations
            context: Additional context
            
        Returns:
            Processing result with escalation info
        """
        should_escalate, priority = self.should_escalate(decision, confidence, violations)
        
        result = {
            'judgment_id': judgment_id,
            'decision': decision,
            'confidence': confidence,
            'escalated': should_escalate
        }
        
        if should_escalate:
            case = self.escalation_queue.add_case(
                judgment_id=judgment_id,
                action_id=action_id,
                agent_id=agent_id,
                decision=decision,
                confidence=confidence,
                violations=violations,
                priority=priority,
                context=context
            )
            result['escalation_case_id'] = case.case_id
            result['priority'] = priority.value
        
        return result
    
    def get_next_case(self, reviewer_id: str) -> Optional[EscalationCase]:
        """Get next case for review.
        
        Args:
            reviewer_id: ID of reviewer
            
        Returns:
            Next case or None
        """
        return self.escalation_queue.get_next_case(reviewer_id)
    
    def submit_feedback(
        self,
        case_id: str,
        reviewer_id: str,
        feedback_tags: List[FeedbackTag],
        rationale: str,
        corrected_decision: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HumanFeedback:
        """Submit feedback for a case.
        
        Args:
            case_id: Case ID
            reviewer_id: Reviewer ID
            feedback_tags: Feedback tags
            rationale: Human rationale
            corrected_decision: Corrected decision if applicable
            confidence: Confidence in feedback
            metadata: Additional metadata
            
        Returns:
            Created feedback
        """
        return self.escalation_queue.submit_feedback(
            case_id=case_id,
            reviewer_id=reviewer_id,
            feedback_tags=feedback_tags,
            rationale=rationale,
            corrected_decision=corrected_decision,
            confidence=confidence,
            metadata=metadata
        )
    
    def get_sla_metrics(self) -> SLAMetrics:
        """Get SLA metrics.
        
        Returns:
            SLA metrics
        """
        return self.escalation_queue.get_sla_metrics()
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback summary for continuous improvement.
        
        Returns:
            Feedback summary
        """
        return self.escalation_queue.get_feedback_summary()
    
    # ==================== Phase 9: Optimization ====================
    
    def create_configuration(
        self,
        config_version: str,
        **kwargs
    ) -> Configuration:
        """Create new configuration.
        
        Args:
            config_version: Version identifier
            **kwargs: Configuration parameters
            
        Returns:
            Created configuration
        """
        return self.optimizer.create_configuration(config_version, **kwargs)
    
    def record_metrics(
        self,
        config_id: str,
        detection_recall: float,
        detection_precision: float,
        false_positive_rate: float,
        decision_latency_ms: float,
        human_agreement: float,
        total_cases: int
    ) -> PerformanceMetrics:
        """Record metrics for configuration.
        
        Args:
            config_id: Configuration ID
            detection_recall: Recall metric
            detection_precision: Precision metric
            false_positive_rate: FP rate
            decision_latency_ms: Latency in ms
            human_agreement: Agreement rate
            total_cases: Total cases
            
        Returns:
            Performance metrics
        """
        return self.optimizer.record_metrics(
            config_id=config_id,
            detection_recall=detection_recall,
            detection_precision=detection_precision,
            false_positive_rate=false_positive_rate,
            decision_latency_ms=decision_latency_ms,
            human_agreement=human_agreement,
            total_cases=total_cases
        )
    
    def optimize_configuration(
        self,
        technique: str = "random_search",
        base_config: Optional[Configuration] = None,
        **kwargs
    ) -> List[Tuple[Configuration, PerformanceMetrics]]:
        """Optimize configuration using specified technique.
        
        Args:
            technique: Optimization technique (grid_search, random_search, evolutionary)
            base_config: Base configuration (for evolutionary)
            **kwargs: Technique-specific parameters
            
        Returns:
            List of (config, metrics) sorted by fitness
        """
        # Dummy evaluation function (should be replaced with real evaluation)
        def dummy_evaluate(config: Configuration) -> PerformanceMetrics:
            import random
            return self.optimizer.record_metrics(
                config_id=config.config_id,
                detection_recall=random.uniform(0.7, 0.95),
                detection_precision=random.uniform(0.7, 0.95),
                false_positive_rate=random.uniform(0.01, 0.1),
                decision_latency_ms=random.uniform(5, 20),
                human_agreement=random.uniform(0.8, 0.95),
                total_cases=100
            )
        
        if technique == "grid_search":
            param_grid = kwargs.get('param_grid', {
                'classifier_threshold': [0.4, 0.5, 0.6],
                'gray_zone_lower': [0.3, 0.4, 0.5],
                'gray_zone_upper': [0.5, 0.6, 0.7]
            })
            return self.optimizer.grid_search(
                param_grid=param_grid,
                evaluate_fn=dummy_evaluate,
                max_iterations=kwargs.get('max_iterations', 27)
            )
        
        elif technique == "random_search":
            param_ranges = kwargs.get('param_ranges', {
                'classifier_threshold': (0.3, 0.7),
                'confidence_threshold': (0.5, 0.9),
                'gray_zone_lower': (0.2, 0.5),
                'gray_zone_upper': (0.5, 0.8)
            })
            return self.optimizer.random_search(
                param_ranges=param_ranges,
                evaluate_fn=dummy_evaluate,
                n_iterations=kwargs.get('n_iterations', 50)
            )
        
        elif technique == "evolutionary":
            if not base_config:
                base_config = self.optimizer.create_configuration("base_v1")
            return self.optimizer.evolutionary_search(
                base_config=base_config,
                evaluate_fn=dummy_evaluate,
                population_size=kwargs.get('population_size', 20),
                n_generations=kwargs.get('n_generations', 10),
                mutation_rate=kwargs.get('mutation_rate', 0.2)
            )
        
        elif technique == "bayesian":
            param_ranges = kwargs.get('param_ranges', {
                'classifier_threshold': (0.3, 0.7),
                'confidence_threshold': (0.5, 0.9),
                'gray_zone_lower': (0.2, 0.5),
                'gray_zone_upper': (0.5, 0.8)
            })
            return self.optimizer.bayesian_optimization(
                param_ranges=param_ranges,
                evaluate_fn=dummy_evaluate,
                n_iterations=kwargs.get('n_iterations', 30),
                n_initial_random=kwargs.get('n_initial_random', 5)
            )
        
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def check_promotion_gate(
        self,
        candidate_id: str,
        baseline_id: str
    ) -> Tuple[bool, List[str]]:
        """Check if candidate passes promotion gate.
        
        Args:
            candidate_id: Candidate config ID
            baseline_id: Baseline config ID
            
        Returns:
            Tuple of (passed, reasons)
        """
        return self.optimizer.check_promotion_gate(candidate_id, baseline_id)
    
    def promote_configuration(self, config_id: str) -> bool:
        """Promote configuration to production.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            True if successful
        """
        promoted = self.optimizer.promote_configuration(config_id)
        if promoted:
            self.active_config = self.optimizer.configurations[config_id]
        return promoted
    
    def get_best_configuration(self) -> Optional[Tuple[Configuration, PerformanceMetrics]]:
        """Get best configuration by fitness.
        
        Returns:
            Tuple of (config, metrics) or None
        """
        return self.optimizer.get_best_configuration()
    
    # ==================== Continuous Improvement Loop ====================
    
    def continuous_improvement_cycle(self) -> Dict[str, Any]:
        """Run one cycle of continuous improvement.
        
        Process:
        1. Collect human feedback
        2. Calculate human agreement metrics
        3. Trigger optimization if needed
        4. Check promotion gate
        5. Return recommendations
        
        Returns:
            Cycle results and recommendations
        """
        feedback_summary = self.get_feedback_summary()
        sla_metrics = self.get_sla_metrics()
        
        # Calculate human agreement from feedback
        total_feedback = feedback_summary.get('total_feedback', 0)
        correction_rate = feedback_summary.get('correction_rate', 0.0)
        human_agreement = 1.0 - correction_rate if total_feedback > 0 else 1.0
        
        # Determine if optimization is needed
        needs_optimization = (
            feedback_summary.get('false_positive_rate', 0.0) > 0.1 or
            feedback_summary.get('missed_violation_rate', 0.0) > 0.05 or
            human_agreement < 0.85
        )
        
        result = {
            'feedback_summary': feedback_summary,
            'sla_metrics': sla_metrics.to_dict(),
            'human_agreement': human_agreement,
            'needs_optimization': needs_optimization,
            'recommendations': []
        }
        
        if needs_optimization:
            result['recommendations'].append(
                "High FP rate or missed violations detected - trigger optimization"
            )
        
        if sla_metrics.pending_cases > 50:
            result['recommendations'].append(
                f"High pending case count ({sla_metrics.pending_cases}) - scale review capacity"
            )
        
        if sla_metrics.sla_breaches > 10:
            result['recommendations'].append(
                f"SLA breaches detected ({sla_metrics.sla_breaches}) - review triage process"
            )
        
        return result
    
    # ==================== F4: Adaptive Thresholds & Tuning ====================
    
    def record_outcome(
        self,
        action_id: str,
        judgment_id: str,
        predicted_outcome: str,
        actual_outcome: str,
        confidence: float,
        human_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Record outcome for adaptive learning.
        
        This method feeds outcome data into the adaptive threshold tuner,
        which automatically adjusts thresholds to improve performance.
        
        Args:
            action_id: Action ID
            judgment_id: Judgment ID
            predicted_outcome: Predicted outcome (e.g., "allow", "block")
            actual_outcome: Actual outcome (e.g., "false_positive", "correct")
            confidence: Prediction confidence (0-1)
            human_feedback: Optional human feedback
            
        Returns:
            Dictionary with outcome record and updated thresholds
        """
        outcome = self.adaptive_tuner.record_outcome(
            action_id=action_id,
            judgment_id=judgment_id,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            confidence=confidence,
            human_feedback=human_feedback
        )
        
        # Get updated thresholds
        thresholds = self.adaptive_tuner.get_thresholds()
        stats = self.adaptive_tuner.get_performance_stats()
        
        return {
            'outcome': outcome.to_dict(),
            'updated_thresholds': thresholds,
            'performance_stats': stats
        }
    
    def get_adaptive_thresholds(self, agent_id: Optional[str] = None) -> Dict[str, float]:
        """Get current adaptive thresholds.
        
        Args:
            agent_id: Optional agent ID for agent-specific thresholds
            
        Returns:
            Dictionary of threshold values
        """
        return self.adaptive_tuner.get_thresholds(agent_id)
    
    def set_agent_thresholds(
        self,
        agent_id: str,
        thresholds: Dict[str, float]
    ) -> None:
        """Set agent-specific thresholds.
        
        Args:
            agent_id: Agent ID
            thresholds: Threshold values
        """
        self.adaptive_tuner.set_agent_thresholds(agent_id, thresholds)
    
    def get_tuning_performance(self) -> Dict[str, Any]:
        """Get performance statistics from adaptive tuner.
        
        Returns:
            Performance statistics
        """
        return self.adaptive_tuner.get_performance_stats()
    
    def create_ab_test(
        self,
        control_config: Configuration,
        treatment_config: Configuration,
        traffic_split: float = 0.1
    ) -> Tuple[str, str]:
        """Create A/B test for threshold variants.
        
        Args:
            control_config: Control configuration
            treatment_config: Treatment configuration
            traffic_split: Traffic percentage to treatment (0-1)
            
        Returns:
            Tuple of (control_variant_id, treatment_variant_id)
        """
        return self.ab_testing.create_ab_test(
            control_config,
            treatment_config,
            traffic_split
        )
    
    def record_ab_metrics(
        self,
        variant_id: str,
        metrics: PerformanceMetrics
    ) -> None:
        """Record metrics for A/B test variant.
        
        Args:
            variant_id: Variant ID
            metrics: Performance metrics
        """
        self.ab_testing.record_variant_metrics(variant_id, metrics)
    
    def check_ab_significance(
        self,
        control_variant_id: str,
        treatment_variant_id: str,
        metric: str = "detection_recall"
    ) -> Tuple[bool, float, str]:
        """Check statistical significance of A/B test.
        
        Args:
            control_variant_id: Control variant ID
            treatment_variant_id: Treatment variant ID
            metric: Metric to test
            
        Returns:
            Tuple of (is_significant, p_value, interpretation)
        """
        return self.ab_testing.check_statistical_significance(
            control_variant_id,
            treatment_variant_id,
            metric
        )
    
    def gradual_rollout(
        self,
        treatment_variant_id: str,
        target_traffic: float,
        step_size: float = 0.1
    ) -> float:
        """Gradually increase traffic to treatment variant.
        
        Args:
            treatment_variant_id: Treatment variant ID
            target_traffic: Target traffic percentage (0-1)
            step_size: Traffic increase per step
            
        Returns:
            New traffic percentage
        """
        return self.ab_testing.gradual_rollout(
            treatment_variant_id,
            target_traffic,
            step_size
        )
    
    def rollback_variant(self, variant_id: str) -> bool:
        """Rollback A/B test variant.
        
        Args:
            variant_id: Variant ID
            
        Returns:
            True if rolled back successfully
        """
        return self.ab_testing.rollback_variant(variant_id)
    
    def get_ab_summary(self) -> Dict[str, Any]:
        """Get A/B test summary.
        
        Returns:
            Dictionary with variant information
        """
        return self.ab_testing.get_variant_summary()
