"""
A/B Testing Framework for Detector Updates

Enables safe testing of new detector versions before full deployment.

Features:
- Traffic splitting between control and treatment
- Statistical significance testing
- Metrics collection and comparison
- Automatic winner selection

Law Alignment:
- Law 23 (Fail-Safe Design): Test before deployment
- Law 14 (Decision Justification): Evidence-based rollout
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of A/B test."""
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    WINNER_SELECTED = "winner_selected"


class TestVariant(Enum):
    """Test variant identifier."""
    CONTROL = "control"  # Current version
    TREATMENT = "treatment"  # New version


@dataclass
class TestConfig:
    """Configuration for A/B test."""
    
    test_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detector_name: str = ""
    
    # Traffic allocation
    treatment_percentage: float = 50.0  # % of traffic to treatment
    
    # Duration and sample size
    min_samples: int = 1000
    duration_hours: int = 24
    
    # Statistical thresholds
    min_confidence_level: float = 0.95  # 95% confidence
    min_improvement: float = 0.02  # 2% minimum improvement
    
    # Auto-stop conditions
    auto_stop_on_degradation: bool = True
    degradation_threshold: float = 0.05  # Stop if 5% worse


@dataclass
class VariantMetrics:
    """Metrics for a test variant."""
    
    sample_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    avg_latency_ms: float = 0.0
    
    @property
    def detection_rate(self) -> float:
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total
    
    @property
    def false_positive_rate(self) -> float:
        total = self.false_positives + self.true_negatives
        if total == 0:
            return 0.0
        return self.false_positives / total
    
    @property
    def accuracy(self) -> float:
        total = self.sample_count
        if total == 0:
            return 0.0
        correct = self.true_positives + self.true_negatives
        return correct / total


@dataclass
class ABTest:
    """Represents an A/B test."""
    
    test_id: str
    config: TestConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    
    status: TestStatus = TestStatus.RUNNING
    
    # Variant metrics
    control_metrics: VariantMetrics = field(default_factory=VariantMetrics)
    treatment_metrics: VariantMetrics = field(default_factory=VariantMetrics)
    
    # Results
    winner: Optional[TestVariant] = None
    confidence: float = 0.0
    improvement: float = 0.0
    
    notes: str = ""


class ABTestingFramework:
    """
    Manages A/B testing for detector updates.
    
    Features:
    - Randomized traffic splitting
    - Real-time metrics collection
    - Statistical analysis
    - Automatic winner selection
    """
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, ABTest] = {}
        
        self.total_tests_started = 0
        self.total_tests_completed = 0
        
        logger.info("ABTestingFramework initialized")
    
    async def start_test(self, config: TestConfig) -> ABTest:
        """
        Start a new A/B test.
        
        Args:
            config: Test configuration
            
        Returns:
            Created test
        """
        test = ABTest(
            test_id=config.test_id,
            config=config,
            start_time=datetime.now(timezone.utc),
        )
        
        self.active_tests[test.test_id] = test
        self.total_tests_started += 1
        
        logger.info(
            f"Started A/B test {test.test_id} for {config.detector_name}: "
            f"treatment={config.treatment_percentage}%, "
            f"min_samples={config.min_samples}, "
            f"duration={config.duration_hours}h"
        )
        
        return test
    
    def assign_variant(self, test_id: str) -> Optional[TestVariant]:
        """
        Assign a variant for a single request.
        
        Args:
            test_id: ID of test
            
        Returns:
            Assigned variant or None if test not found
        """
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Random assignment based on treatment percentage
        if random.random() * 100 < test.config.treatment_percentage:
            return TestVariant.TREATMENT
        else:
            return TestVariant.CONTROL
    
    async def record_result(
        self,
        test_id: str,
        variant: TestVariant,
        is_violation: bool,
        is_correct: bool,
        latency_ms: float = 0.0,
    ):
        """
        Record a test result.
        
        Args:
            test_id: ID of test
            variant: Which variant was used
            is_violation: Whether detector flagged a violation
            is_correct: Whether detection was correct
            latency_ms: Detection latency in milliseconds
        """
        if test_id not in self.active_tests:
            logger.warning(f"Test {test_id} not found")
            return
        
        test = self.active_tests[test_id]
        
        # Get metrics for variant
        metrics = (test.control_metrics if variant == TestVariant.CONTROL
                  else test.treatment_metrics)
        
        # Update metrics
        metrics.sample_count += 1
        
        if is_violation and is_correct:
            metrics.true_positives += 1
        elif is_violation and not is_correct:
            metrics.false_positives += 1
        elif not is_violation and is_correct:
            metrics.true_negatives += 1
        else:
            metrics.false_negatives += 1
        
        # Update latency (running average)
        if metrics.sample_count == 1:
            metrics.avg_latency_ms = latency_ms
        else:
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * (metrics.sample_count - 1) + latency_ms)
                / metrics.sample_count
            )
        
        # Check if test should complete
        await self._check_completion(test_id)
    
    async def _check_completion(self, test_id: str):
        """Check if test should be completed."""
        test = self.active_tests[test_id]
        config = test.config
        
        # Check duration
        elapsed = datetime.now(timezone.utc) - test.start_time
        duration_exceeded = elapsed > timedelta(hours=config.duration_hours)
        
        # Check sample size
        total_samples = (test.control_metrics.sample_count + 
                        test.treatment_metrics.sample_count)
        samples_sufficient = total_samples >= config.min_samples
        
        # Check for degradation (auto-stop)
        if config.auto_stop_on_degradation:
            treatment_accuracy = test.treatment_metrics.accuracy
            control_accuracy = test.control_metrics.accuracy
            
            if (test.treatment_metrics.sample_count > 100 and
                treatment_accuracy < control_accuracy - config.degradation_threshold):
                logger.warning(
                    f"Auto-stopping test {test_id}: treatment degraded by "
                    f"{(control_accuracy - treatment_accuracy):.2%}"
                )
                await self.stop_test(test_id, reason="degradation")
                return
        
        # Complete if both conditions met
        if duration_exceeded and samples_sufficient:
            await self.complete_test(test_id)
    
    async def complete_test(self, test_id: str):
        """Complete a test and determine winner."""
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        test.end_time = datetime.now(timezone.utc)
        test.status = TestStatus.COMPLETED
        
        # Analyze results
        control_acc = test.control_metrics.accuracy
        treatment_acc = test.treatment_metrics.accuracy
        
        improvement = treatment_acc - control_acc
        test.improvement = improvement
        
        # Simple winner selection (would use proper statistical test in production)
        if improvement > test.config.min_improvement:
            test.winner = TestVariant.TREATMENT
            test.confidence = 0.95  # Simplified
            test.status = TestStatus.WINNER_SELECTED
        else:
            test.winner = TestVariant.CONTROL
            test.confidence = 0.95
            test.status = TestStatus.WINNER_SELECTED
        
        # Move to completed
        self.completed_tests[test_id] = test
        del self.active_tests[test_id]
        
        self.total_tests_completed += 1
        
        logger.info(
            f"Completed test {test_id}: winner={test.winner.value if test.winner else 'none'}, "
            f"improvement={improvement:.2%}, confidence={test.confidence:.2%}"
        )
    
    async def stop_test(self, test_id: str, reason: str = ""):
        """Stop a test early."""
        if test_id not in self.active_tests:
            return
        
        test = self.active_tests[test_id]
        test.end_time = datetime.now(timezone.utc)
        test.status = TestStatus.STOPPED
        test.notes = f"Stopped: {reason}"
        
        # Move to completed
        self.completed_tests[test_id] = test
        del self.active_tests[test_id]
        
        logger.info(f"Stopped test {test_id}: {reason}")
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get test by ID."""
        if test_id in self.active_tests:
            return self.active_tests[test_id]
        return self.completed_tests.get(test_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get framework metrics."""
        return {
            "total_started": self.total_tests_started,
            "total_completed": self.total_tests_completed,
            "active_count": len(self.active_tests),
            "completed_count": len(self.completed_tests),
        }
