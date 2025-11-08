"""Phase 9: Continuous Optimization

This module implements:
- Automated tuning of rule weights, classifier thresholds, and escalation boundaries
- Multi-objective optimization (max recall, min FP rate, min latency)
- Techniques: grid/random search, evolutionary strategies, Bayesian optimization
- Continuous feedback loop from human labels to retrain models
- Promotion gate for configuration changes

Design:
- Multi-objective fitness scoring
- Configuration versioning and A/B testing support
- Automated promotion gate validation
- Integration with human feedback for continuous improvement
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import math


class OptimizationTechnique(Enum):
    """Optimization techniques."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"


class ConfigStatus(Enum):
    """Configuration status."""

    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class OptimizationObjective:
    """Single optimization objective."""

    name: str
    weight: float
    maximize: bool  # True to maximize, False to minimize
    current_value: float = 0.0
    target_value: Optional[float] = None

    def score(self, value: float) -> float:
        """Calculate weighted score for this objective.

        Args:
            value: Current value

        Returns:
            Weighted score (higher is better)
        """
        if self.maximize:
            return self.weight * value
        else:
            return -self.weight * value


@dataclass
class Configuration:
    """System configuration for optimization."""

    config_id: str
    config_version: str
    status: ConfigStatus
    created_at: datetime

    # Rule parameters
    rule_weights: Dict[str, float] = field(default_factory=dict)

    # ML parameters
    classifier_threshold: float = 0.5
    confidence_threshold: float = 0.7

    # Risk parameters
    gray_zone_lower: float = 0.4
    gray_zone_upper: float = 0.6

    # Escalation parameters
    escalation_confidence_threshold: float = 0.9
    escalation_violation_count: int = 3

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_id": self.config_id,
            "config_version": self.config_version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "rule_weights": self.rule_weights,
            "classifier_threshold": self.classifier_threshold,
            "confidence_threshold": self.confidence_threshold,
            "gray_zone_lower": self.gray_zone_lower,
            "gray_zone_upper": self.gray_zone_upper,
            "escalation_confidence_threshold": self.escalation_confidence_threshold,
            "escalation_violation_count": self.escalation_violation_count,
            "metadata": self.metadata,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a configuration."""

    config_id: str

    # Detection metrics
    detection_recall: float = 0.0
    detection_precision: float = 0.0
    false_positive_rate: float = 0.0

    # Performance metrics
    decision_latency_ms: float = 0.0

    # Human agreement
    human_agreement: float = 0.0

    # Composite score
    fitness_score: float = 0.0

    # Sample size
    total_cases: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_id": self.config_id,
            "detection_recall": self.detection_recall,
            "detection_precision": self.detection_precision,
            "false_positive_rate": self.false_positive_rate,
            "decision_latency_ms": self.decision_latency_ms,
            "human_agreement": self.human_agreement,
            "fitness_score": self.fitness_score,
            "total_cases": self.total_cases,
        }


@dataclass
class PromotionGate:
    """Promotion gate criteria for new configurations."""

    min_recall_gain: float = 0.03  # +3% absolute
    max_fp_increase: float = 0.02  # +2% absolute
    max_latency_increase_ms: float = 5.0
    min_human_agreement: float = 0.85
    min_sample_size: int = 100

    def evaluate(
        self, candidate_metrics: PerformanceMetrics, baseline_metrics: PerformanceMetrics
    ) -> Tuple[bool, List[str]]:
        """Evaluate if candidate passes promotion gate.

        Args:
            candidate_metrics: Metrics for candidate configuration
            baseline_metrics: Metrics for current baseline

        Returns:
            Tuple of (passed, reasons)
        """
        passed = True
        reasons = []

        # Check sample size
        if candidate_metrics.total_cases < self.min_sample_size:
            passed = False
            reasons.append(
                f"Insufficient sample size: {candidate_metrics.total_cases} < {self.min_sample_size}"
            )

        # Check recall gain
        recall_gain = candidate_metrics.detection_recall - baseline_metrics.detection_recall
        if recall_gain < self.min_recall_gain:
            passed = False
            reasons.append(f"Recall gain insufficient: {recall_gain:.3f} < {self.min_recall_gain}")
        else:
            reasons.append(f"✓ Recall gain: {recall_gain:+.3f} >= {self.min_recall_gain}")

        # Check FP increase
        fp_increase = candidate_metrics.false_positive_rate - baseline_metrics.false_positive_rate
        if fp_increase > self.max_fp_increase:
            passed = False
            reasons.append(
                f"FP rate increase too high: {fp_increase:+.3f} > {self.max_fp_increase}"
            )
        else:
            reasons.append(f"✓ FP increase: {fp_increase:+.3f} <= {self.max_fp_increase}")

        # Check latency increase
        latency_increase = (
            candidate_metrics.decision_latency_ms - baseline_metrics.decision_latency_ms
        )
        if latency_increase > self.max_latency_increase_ms:
            passed = False
            reasons.append(
                f"Latency increase too high: {latency_increase:+.1f}ms > {self.max_latency_increase_ms}ms"
            )
        else:
            reasons.append(
                f"✓ Latency increase: {latency_increase:+.1f}ms <= {self.max_latency_increase_ms}ms"
            )

        # Check human agreement
        if candidate_metrics.human_agreement < self.min_human_agreement:
            passed = False
            reasons.append(
                f"Human agreement too low: {candidate_metrics.human_agreement:.3f} < {self.min_human_agreement}"
            )
        else:
            reasons.append(
                f"✓ Human agreement: {candidate_metrics.human_agreement:.3f} >= {self.min_human_agreement}"
            )

        return passed, reasons


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for continuous improvement.

    Optimizes composite fitness function:
    fitness = w1*recall - w2*fp_rate - w3*latency + w4*agreement

    Supports:
    - Grid search
    - Random search
    - Evolutionary strategies
    - Bayesian optimization (simplified)
    """

    def __init__(
        self,
        storage_path: str = "./data/optimization.db",
        objectives: Optional[List[OptimizationObjective]] = None,
    ):
        """Initialize optimizer.

        Args:
            storage_path: Path to SQLite database
            objectives: List of optimization objectives
        """
        self.storage_path = Path(storage_path)

        # Default objectives (from TrainTestPipeline.md)
        self.objectives = objectives or [
            OptimizationObjective("detection_recall", 0.4, maximize=True),
            OptimizationObjective("false_positive_rate", 0.25, maximize=False),
            OptimizationObjective("decision_latency", 0.15, maximize=False),
            OptimizationObjective("human_agreement", 0.2, maximize=True),
        ]

        self.promotion_gate = PromotionGate()

        # Configuration history
        self.configurations: Dict[str, Configuration] = {}
        self.metrics_history: Dict[str, PerformanceMetrics] = {}

        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize SQLite storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Configurations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS configurations (
                config_id TEXT PRIMARY KEY,
                config_version TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                rule_weights TEXT,
                classifier_threshold REAL,
                confidence_threshold REAL,
                gray_zone_lower REAL,
                gray_zone_upper REAL,
                escalation_confidence_threshold REAL,
                escalation_violation_count INTEGER,
                metadata TEXT
            )
        """
        )

        # Metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                config_id TEXT PRIMARY KEY,
                detection_recall REAL,
                detection_precision REAL,
                false_positive_rate REAL,
                decision_latency_ms REAL,
                human_agreement REAL,
                fitness_score REAL,
                total_cases INTEGER,
                measured_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (config_id) REFERENCES configurations(config_id)
            )
        """
        )

        conn.commit()
        conn.close()

    def create_configuration(
        self,
        config_version: str,
        rule_weights: Optional[Dict[str, float]] = None,
        classifier_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
        gray_zone_lower: float = 0.4,
        gray_zone_upper: float = 0.6,
        escalation_confidence_threshold: float = 0.9,
        escalation_violation_count: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Configuration:
        """Create a new configuration.

        Args:
            config_version: Version identifier
            rule_weights: Rule weight overrides
            classifier_threshold: ML classifier threshold
            confidence_threshold: Confidence threshold
            gray_zone_lower: Lower bound of gray zone
            gray_zone_upper: Upper bound of gray zone
            escalation_confidence_threshold: Threshold for escalation
            escalation_violation_count: Violation count for escalation
            metadata: Additional metadata

        Returns:
            Created configuration
        """
        import uuid

        config = Configuration(
            config_id=f"cfg_{uuid.uuid4().hex[:12]}",
            config_version=config_version,
            status=ConfigStatus.CANDIDATE,
            created_at=datetime.now(),
            rule_weights=rule_weights or {},
            classifier_threshold=classifier_threshold,
            confidence_threshold=confidence_threshold,
            gray_zone_lower=gray_zone_lower,
            gray_zone_upper=gray_zone_upper,
            escalation_confidence_threshold=escalation_confidence_threshold,
            escalation_violation_count=escalation_violation_count,
            metadata=metadata or {},
        )

        # Store in database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO configurations
            (config_id, config_version, status, created_at, rule_weights,
             classifier_threshold, confidence_threshold, gray_zone_lower,
             gray_zone_upper, escalation_confidence_threshold,
             escalation_violation_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                config.config_id,
                config.config_version,
                config.status.value,
                config.created_at.isoformat(),
                json.dumps(config.rule_weights),
                config.classifier_threshold,
                config.confidence_threshold,
                config.gray_zone_lower,
                config.gray_zone_upper,
                config.escalation_confidence_threshold,
                config.escalation_violation_count,
                json.dumps(config.metadata),
            ),
        )

        conn.commit()
        conn.close()

        self.configurations[config.config_id] = config

        return config

    def record_metrics(
        self,
        config_id: str,
        detection_recall: float,
        detection_precision: float,
        false_positive_rate: float,
        decision_latency_ms: float,
        human_agreement: float,
        total_cases: int,
    ) -> PerformanceMetrics:
        """Record performance metrics for a configuration.

        Args:
            config_id: Configuration ID
            detection_recall: Recall metric
            detection_precision: Precision metric
            false_positive_rate: False positive rate
            decision_latency_ms: Decision latency in milliseconds
            human_agreement: Human agreement rate
            total_cases: Total number of cases evaluated

        Returns:
            Performance metrics object
        """
        # Calculate fitness score
        fitness_score = self.calculate_fitness(
            detection_recall, false_positive_rate, decision_latency_ms, human_agreement
        )

        metrics = PerformanceMetrics(
            config_id=config_id,
            detection_recall=detection_recall,
            detection_precision=detection_precision,
            false_positive_rate=false_positive_rate,
            decision_latency_ms=decision_latency_ms,
            human_agreement=human_agreement,
            fitness_score=fitness_score,
            total_cases=total_cases,
        )

        # Store in database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO performance_metrics
            (config_id, detection_recall, detection_precision, false_positive_rate,
             decision_latency_ms, human_agreement, fitness_score, total_cases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.config_id,
                metrics.detection_recall,
                metrics.detection_precision,
                metrics.false_positive_rate,
                metrics.decision_latency_ms,
                metrics.human_agreement,
                metrics.fitness_score,
                metrics.total_cases,
            ),
        )

        conn.commit()
        conn.close()

        self.metrics_history[config_id] = metrics

        return metrics

    def calculate_fitness(
        self, recall: float, fp_rate: float, latency_ms: float, agreement: float
    ) -> float:
        """Calculate composite fitness score.

        Uses default weights from TrainTestPipeline.md:
        fitness = 0.4*recall - 0.25*fp_rate - 0.15*latency + 0.2*agreement

        Note: latency is normalized to 0-1 scale assuming max 100ms

        Args:
            recall: Detection recall (0-1)
            fp_rate: False positive rate (0-1)
            latency_ms: Decision latency in milliseconds
            agreement: Human agreement rate (0-1)

        Returns:
            Fitness score (higher is better)
        """
        # Normalize latency to 0-1 scale (assuming max 100ms)
        normalized_latency = min(latency_ms / 100.0, 1.0)

        fitness = 0.0
        for obj in self.objectives:
            if obj.name == "detection_recall":
                fitness += obj.score(recall)
            elif obj.name == "false_positive_rate":
                fitness += obj.score(fp_rate)
            elif obj.name == "decision_latency":
                fitness += obj.score(normalized_latency)
            elif obj.name == "human_agreement":
                fitness += obj.score(agreement)

        return fitness

    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        evaluate_fn: Callable[[Configuration], PerformanceMetrics],
        max_iterations: int = 100,
    ) -> List[Tuple[Configuration, PerformanceMetrics]]:
        """Perform grid search over parameter space.

        Args:
            param_grid: Dictionary of parameter names to lists of values
            evaluate_fn: Function to evaluate configuration
            max_iterations: Maximum iterations

        Returns:
            List of (configuration, metrics) tuples sorted by fitness
        """
        results = []
        iterations = 0

        # Generate all combinations (simplified - not full cartesian product)
        for threshold in param_grid.get("classifier_threshold", [0.5]):
            for gz_lower in param_grid.get("gray_zone_lower", [0.4]):
                for gz_upper in param_grid.get("gray_zone_upper", [0.6]):
                    if iterations >= max_iterations:
                        break

                    config = self.create_configuration(
                        config_version=f"grid_v{iterations}",
                        classifier_threshold=threshold,
                        gray_zone_lower=gz_lower,
                        gray_zone_upper=gz_upper,
                    )

                    metrics = evaluate_fn(config)
                    results.append((config, metrics))
                    iterations += 1

        # Sort by fitness
        results.sort(key=lambda x: x[1].fitness_score, reverse=True)

        return results

    def random_search(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        evaluate_fn: Callable[[Configuration], PerformanceMetrics],
        n_iterations: int = 50,
    ) -> List[Tuple[Configuration, PerformanceMetrics]]:
        """Perform random search over parameter space.

        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            evaluate_fn: Function to evaluate configuration
            n_iterations: Number of random samples

        Returns:
            List of (configuration, metrics) tuples sorted by fitness
        """
        results = []

        for i in range(n_iterations):
            # Sample random parameters
            params = {}
            for param, (min_val, max_val) in param_ranges.items():
                params[param] = random.uniform(min_val, max_val)

            config = self.create_configuration(
                config_version=f"random_v{i}",
                classifier_threshold=params.get("classifier_threshold", 0.5),
                confidence_threshold=params.get("confidence_threshold", 0.7),
                gray_zone_lower=params.get("gray_zone_lower", 0.4),
                gray_zone_upper=params.get("gray_zone_upper", 0.6),
            )

            metrics = evaluate_fn(config)
            results.append((config, metrics))

        # Sort by fitness
        results.sort(key=lambda x: x[1].fitness_score, reverse=True)

        return results

    def evolutionary_search(
        self,
        base_config: Configuration,
        evaluate_fn: Callable[[Configuration], PerformanceMetrics],
        population_size: int = 20,
        n_generations: int = 10,
        mutation_rate: float = 0.2,
    ) -> List[Tuple[Configuration, PerformanceMetrics]]:
        """Perform evolutionary search.

        Simple evolutionary strategy:
        1. Start with base configuration
        2. Generate population through mutation
        3. Select top performers
        4. Generate next generation

        Args:
            base_config: Base configuration
            evaluate_fn: Function to evaluate configuration
            population_size: Size of population
            n_generations: Number of generations
            mutation_rate: Mutation rate (0-1)

        Returns:
            List of (configuration, metrics) tuples sorted by fitness
        """

        def mutate(config: Configuration, generation: int) -> Configuration:
            """Mutate configuration."""
            return self.create_configuration(
                config_version=f"evo_gen{generation}_v{random.randint(0, 9999)}",
                classifier_threshold=max(
                    0.0, min(1.0, config.classifier_threshold + random.gauss(0, mutation_rate))
                ),
                confidence_threshold=max(
                    0.0, min(1.0, config.confidence_threshold + random.gauss(0, mutation_rate))
                ),
                gray_zone_lower=max(
                    0.0, min(0.5, config.gray_zone_lower + random.gauss(0, mutation_rate * 0.5))
                ),
                gray_zone_upper=max(
                    0.5, min(1.0, config.gray_zone_upper + random.gauss(0, mutation_rate * 0.5))
                ),
            )

        # Initialize population
        population = [(base_config, evaluate_fn(base_config))]

        for gen in range(n_generations):
            # Generate offspring through mutation
            while len(population) < population_size:
                # Select random parent from top 50%
                parent = random.choice(population[: max(1, len(population) // 2)])[0]
                offspring = mutate(parent, gen)
                metrics = evaluate_fn(offspring)
                population.append((offspring, metrics))

            # Select top performers for next generation
            population.sort(key=lambda x: x[1].fitness_score, reverse=True)
            population = population[: population_size // 2]

        # Final evaluation - keep all evaluated configs
        population.sort(key=lambda x: x[1].fitness_score, reverse=True)

        return population

    def check_promotion_gate(self, candidate_id: str, baseline_id: str) -> Tuple[bool, List[str]]:
        """Check if candidate passes promotion gate.

        Args:
            candidate_id: Candidate configuration ID
            baseline_id: Baseline configuration ID

        Returns:
            Tuple of (passed, reasons)
        """
        candidate_metrics = self.metrics_history.get(candidate_id)
        baseline_metrics = self.metrics_history.get(baseline_id)

        if not candidate_metrics:
            return False, [f"No metrics found for candidate {candidate_id}"]
        if not baseline_metrics:
            return False, [f"No metrics found for baseline {baseline_id}"]

        return self.promotion_gate.evaluate(candidate_metrics, baseline_metrics)

    def promote_configuration(self, config_id: str) -> bool:
        """Promote configuration to production.

        Args:
            config_id: Configuration ID

        Returns:
            True if promoted successfully
        """
        config = self.configurations.get(config_id)
        if not config:
            return False

        # Demote current production configs
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE configurations
            SET status = ?
            WHERE status = ?
        """,
            (ConfigStatus.DEPRECATED.value, ConfigStatus.PRODUCTION.value),
        )

        # Promote new config
        cursor.execute(
            """
            UPDATE configurations
            SET status = ?
            WHERE config_id = ?
        """,
            (ConfigStatus.PRODUCTION.value, config_id),
        )

        conn.commit()
        conn.close()

        config.status = ConfigStatus.PRODUCTION

        return True

    def bayesian_optimization(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        evaluate_fn: Callable[[Configuration], PerformanceMetrics],
        n_iterations: int = 30,
        n_initial_random: int = 5,
    ) -> List[Tuple[Configuration, PerformanceMetrics]]:
        """Perform Bayesian optimization over parameter space.

        Uses a simplified Gaussian Process approach to intelligently explore
        the parameter space by balancing exploration and exploitation.

        Args:
            param_ranges: Dictionary of parameter names to (min, max) tuples
            evaluate_fn: Function to evaluate configuration
            n_iterations: Total number of iterations
            n_initial_random: Number of initial random samples

        Returns:
            List of (configuration, metrics) tuples sorted by fitness
        """
        results = []

        # Phase 1: Initial random exploration
        for i in range(n_initial_random):
            params = {}
            for param, (min_val, max_val) in param_ranges.items():
                params[param] = random.uniform(min_val, max_val)

            config = self.create_configuration(
                config_version=f"bayes_init_v{i}",
                classifier_threshold=params.get("classifier_threshold", 0.5),
                confidence_threshold=params.get("confidence_threshold", 0.7),
                gray_zone_lower=params.get("gray_zone_lower", 0.4),
                gray_zone_upper=params.get("gray_zone_upper", 0.6),
            )

            metrics = evaluate_fn(config)
            results.append((config, metrics))

        # Phase 2: Bayesian optimization using acquisition function
        # Simplified approach: use expected improvement around best point
        for i in range(n_initial_random, n_iterations):
            # Get best configuration so far
            best_metrics = max(results, key=lambda x: x[1].fitness_score)[1]
            best_metrics.fitness_score

            # Sample around best configuration with adaptive exploration
            # Higher variance early on, lower variance later
            exploration_factor = 1.0 - (i - n_initial_random) / (n_iterations - n_initial_random)
            variance = 0.1 + 0.2 * exploration_factor  # Variance from 0.1 to 0.3

            # Get best config parameters
            best_config = max(results, key=lambda x: x[1].fitness_score)[0]

            # Sample with Gaussian perturbation around best point
            params = {
                "classifier_threshold": self._clip_value(
                    best_config.classifier_threshold + random.gauss(0, variance),
                    *param_ranges.get("classifier_threshold", (0.3, 0.7)),
                ),
                "confidence_threshold": self._clip_value(
                    best_config.confidence_threshold + random.gauss(0, variance),
                    *param_ranges.get("confidence_threshold", (0.5, 0.9)),
                ),
                "gray_zone_lower": self._clip_value(
                    best_config.gray_zone_lower + random.gauss(0, variance * 0.5),
                    *param_ranges.get("gray_zone_lower", (0.2, 0.5)),
                ),
                "gray_zone_upper": self._clip_value(
                    best_config.gray_zone_upper + random.gauss(0, variance * 0.5),
                    *param_ranges.get("gray_zone_upper", (0.5, 0.8)),
                ),
            }

            config = self.create_configuration(
                config_version=f"bayes_v{i}",
                classifier_threshold=params["classifier_threshold"],
                confidence_threshold=params["confidence_threshold"],
                gray_zone_lower=params["gray_zone_lower"],
                gray_zone_upper=params["gray_zone_upper"],
            )

            metrics = evaluate_fn(config)
            results.append((config, metrics))

        # Sort by fitness
        results.sort(key=lambda x: x[1].fitness_score, reverse=True)

        return results

    def _clip_value(self, value: float, min_val: float, max_val: float) -> float:
        """Clip value to range."""
        return max(min_val, min(max_val, value))

    def get_best_configuration(self) -> Optional[Tuple[Configuration, PerformanceMetrics]]:
        """Get best configuration by fitness score.

        Returns:
            Tuple of (configuration, metrics) or None
        """
        if not self.metrics_history:
            return None

        best_config_id = max(
            self.metrics_history.keys(), key=lambda cid: self.metrics_history[cid].fitness_score
        )

        return (self.configurations[best_config_id], self.metrics_history[best_config_id])


@dataclass
class OutcomeRecord:
    """Record of an action outcome for learning."""

    action_id: str
    judgment_id: str
    predicted_outcome: str
    actual_outcome: str
    confidence: float
    timestamp: datetime
    human_feedback: Optional[str] = None
    was_correct: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "judgment_id": self.judgment_id,
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "human_feedback": self.human_feedback,
            "was_correct": self.was_correct,
        }


class AdaptiveThresholdTuner:
    """Adaptive threshold tuner using outcome-based learning.

    Features:
    - Learns from outcome feedback (false positives, false negatives)
    - Automatically adjusts thresholds based on objectives
    - Supports agent-specific threshold profiles
    - Tracks performance over time
    """

    def __init__(
        self, objectives: List[str], learning_rate: float = 0.01, storage_path: Optional[str] = None
    ):
        """Initialize adaptive threshold tuner.

        Args:
            objectives: List of objectives (e.g., ["maximize_recall", "minimize_fp"])
            learning_rate: Learning rate for threshold adjustment
            storage_path: Optional path to SQLite database for persistence
        """
        self.objectives = objectives
        self.learning_rate = learning_rate
        self.storage_path = Path(storage_path) if storage_path else None

        # Outcome history
        self.outcomes: List[OutcomeRecord] = []

        # Agent-specific thresholds
        self.agent_thresholds: Dict[str, Dict[str, float]] = {}

        # Global thresholds
        self.global_thresholds: Dict[str, float] = {
            "classifier_threshold": 0.5,
            "confidence_threshold": 0.7,
            "gray_zone_lower": 0.4,
            "gray_zone_upper": 0.6,
        }

        # Performance tracking
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.true_negatives = 0

        if self.storage_path:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize SQLite storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Outcomes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                action_id TEXT PRIMARY KEY,
                judgment_id TEXT,
                predicted_outcome TEXT,
                actual_outcome TEXT,
                confidence REAL,
                timestamp TEXT,
                human_feedback TEXT,
                was_correct INTEGER
            )
        """
        )

        # Agent thresholds table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_thresholds (
                agent_id TEXT PRIMARY KEY,
                classifier_threshold REAL,
                confidence_threshold REAL,
                gray_zone_lower REAL,
                gray_zone_upper REAL,
                updated_at TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def record_outcome(
        self,
        action_id: str,
        judgment_id: str,
        predicted_outcome: str,
        actual_outcome: str,
        confidence: float,
        human_feedback: Optional[str] = None,
    ) -> OutcomeRecord:
        """Record an outcome for learning.

        Args:
            action_id: Action ID
            judgment_id: Judgment ID
            predicted_outcome: Predicted outcome (e.g., "allow", "block")
            actual_outcome: Actual outcome (e.g., "false_positive", "correct")
            confidence: Prediction confidence (0-1)
            human_feedback: Optional human feedback

        Returns:
            Outcome record
        """
        was_correct = actual_outcome in ["correct", "true_positive", "true_negative"]

        outcome = OutcomeRecord(
            action_id=action_id,
            judgment_id=judgment_id,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            confidence=confidence,
            timestamp=datetime.now(),
            human_feedback=human_feedback,
            was_correct=was_correct,
        )

        self.outcomes.append(outcome)

        # Update statistics
        if actual_outcome == "false_positive":
            self.false_positives += 1
        elif actual_outcome == "false_negative":
            self.false_negatives += 1
        elif actual_outcome == "true_positive":
            self.true_positives += 1
        elif actual_outcome == "true_negative":
            self.true_negatives += 1
        elif actual_outcome == "correct":
            # "correct" is a generic positive outcome, treat as true positive
            if predicted_outcome in ["block", "flag", "review"]:
                self.true_positives += 1
            else:
                self.true_negatives += 1

        # Store in database if available
        if self.storage_path:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO outcomes
                (action_id, judgment_id, predicted_outcome, actual_outcome,
                 confidence, timestamp, human_feedback, was_correct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    outcome.action_id,
                    outcome.judgment_id,
                    outcome.predicted_outcome,
                    outcome.actual_outcome,
                    outcome.confidence,
                    outcome.timestamp.isoformat(),
                    outcome.human_feedback,
                    1 if outcome.was_correct else 0,
                ),
            )

            conn.commit()
            conn.close()

        # Adapt thresholds based on outcome
        self._adapt_thresholds(outcome)

        return outcome

    def _adapt_thresholds(self, outcome: OutcomeRecord) -> None:
        """Adapt thresholds based on outcome.

        Args:
            outcome: Outcome record
        """
        # Adjust based on objectives and outcome
        if "minimize_fp" in self.objectives or "minimize_false_positives" in self.objectives:
            if outcome.actual_outcome == "false_positive":
                # Increase threshold to reduce false positives
                self.global_thresholds["classifier_threshold"] = min(
                    0.9, self.global_thresholds["classifier_threshold"] + self.learning_rate
                )

        if "maximize_recall" in self.objectives:
            if outcome.actual_outcome == "false_negative":
                # Decrease threshold to catch more violations
                self.global_thresholds["classifier_threshold"] = max(
                    0.1, self.global_thresholds["classifier_threshold"] - self.learning_rate
                )

        # Adjust confidence threshold based on accuracy
        if not outcome.was_correct and outcome.confidence > 0.8:
            # High confidence but wrong - increase confidence threshold
            self.global_thresholds["confidence_threshold"] = min(
                0.95, self.global_thresholds["confidence_threshold"] + self.learning_rate * 0.5
            )

    def get_thresholds(self, agent_id: Optional[str] = None) -> Dict[str, float]:
        """Get thresholds for agent or global.

        Args:
            agent_id: Optional agent ID for agent-specific thresholds

        Returns:
            Dictionary of threshold values
        """
        if agent_id and agent_id in self.agent_thresholds:
            return self.agent_thresholds[agent_id].copy()
        return self.global_thresholds.copy()

    def set_agent_thresholds(self, agent_id: str, thresholds: Dict[str, float]) -> None:
        """Set agent-specific thresholds.

        Args:
            agent_id: Agent ID
            thresholds: Threshold values
        """
        self.agent_thresholds[agent_id] = thresholds.copy()

        # Store in database if available
        if self.storage_path:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_thresholds
                (agent_id, classifier_threshold, confidence_threshold,
                 gray_zone_lower, gray_zone_upper, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    agent_id,
                    thresholds.get("classifier_threshold", 0.5),
                    thresholds.get("confidence_threshold", 0.7),
                    thresholds.get("gray_zone_lower", 0.4),
                    thresholds.get("gray_zone_upper", 0.6),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.

        Returns:
            Dictionary of statistics
        """
        total = (
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        )

        if total == 0:
            return {
                "total_outcomes": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
            }

        accuracy = (self.true_positives + self.true_negatives) / total

        # Precision: TP / (TP + FP)
        precision = 0.0
        if (self.true_positives + self.false_positives) > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives)

        # Recall: TP / (TP + FN)
        recall = 0.0
        if (self.true_positives + self.false_negatives) > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)

        fp_rate = self.false_positives / total
        fn_rate = self.false_negatives / total

        return {
            "total_outcomes": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "current_thresholds": self.global_thresholds.copy(),
        }


class ABTestingFramework:
    """A/B testing framework for threshold variants.

    Features:
    - Variant management and traffic splitting
    - Statistical significance testing
    - Gradual rollout controls
    - Automatic rollback on performance degradation
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize A/B testing framework.

        Args:
            storage_path: Optional path to SQLite database for persistence
        """
        self.storage_path = Path(storage_path) if storage_path else None

        # Active variants
        self.variants: Dict[str, Configuration] = {}
        self.variant_metrics: Dict[str, PerformanceMetrics] = {}
        self.variant_traffic: Dict[str, float] = {}  # Variant ID -> traffic percentage

        # Test configuration
        self.control_variant_id: Optional[str] = None
        self.min_sample_size: int = 100
        self.significance_level: float = 0.05  # 95% confidence

        if self.storage_path:
            self._init_storage()

    def _init_storage(self) -> None:
        """Initialize SQLite storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Variants table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ab_variants (
                variant_id TEXT PRIMARY KEY,
                config_id TEXT,
                traffic_percentage REAL,
                is_control INTEGER,
                created_at TEXT,
                status TEXT
            )
        """
        )

        # Variant metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ab_variant_metrics (
                variant_id TEXT PRIMARY KEY,
                sample_size INTEGER,
                detection_recall REAL,
                false_positive_rate REAL,
                decision_latency_ms REAL,
                human_agreement REAL,
                last_updated TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def create_ab_test(
        self,
        control_config: Configuration,
        treatment_config: Configuration,
        traffic_split: float = 0.1,
    ) -> Tuple[str, str]:
        """Create an A/B test with control and treatment variants.

        Args:
            control_config: Control configuration (current production)
            treatment_config: Treatment configuration (new variant)
            traffic_split: Percentage of traffic to treatment (0-1)

        Returns:
            Tuple of (control_variant_id, treatment_variant_id)
        """
        import uuid

        control_variant_id = f"ab_control_{uuid.uuid4().hex[:8]}"
        treatment_variant_id = f"ab_treatment_{uuid.uuid4().hex[:8]}"

        self.variants[control_variant_id] = control_config
        self.variants[treatment_variant_id] = treatment_config

        self.variant_traffic[control_variant_id] = 1.0 - traffic_split
        self.variant_traffic[treatment_variant_id] = traffic_split

        self.control_variant_id = control_variant_id

        # Store in database if available
        if self.storage_path:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO ab_variants
                (variant_id, config_id, traffic_percentage, is_control, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    control_variant_id,
                    control_config.config_id,
                    1.0 - traffic_split,
                    1,
                    datetime.now().isoformat(),
                    "active",
                ),
            )

            cursor.execute(
                """
                INSERT INTO ab_variants
                (variant_id, config_id, traffic_percentage, is_control, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    treatment_variant_id,
                    treatment_config.config_id,
                    traffic_split,
                    0,
                    datetime.now().isoformat(),
                    "active",
                ),
            )

            conn.commit()
            conn.close()

        return control_variant_id, treatment_variant_id

    def record_variant_metrics(self, variant_id: str, metrics: PerformanceMetrics) -> None:
        """Record metrics for a variant.

        Args:
            variant_id: Variant ID
            metrics: Performance metrics
        """
        self.variant_metrics[variant_id] = metrics

        # Store in database if available
        if self.storage_path:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO ab_variant_metrics
                (variant_id, sample_size, detection_recall, false_positive_rate,
                 decision_latency_ms, human_agreement, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    variant_id,
                    metrics.total_cases,
                    metrics.detection_recall,
                    metrics.false_positive_rate,
                    metrics.decision_latency_ms,
                    metrics.human_agreement,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()

    def check_statistical_significance(
        self, control_variant_id: str, treatment_variant_id: str, metric: str = "detection_recall"
    ) -> Tuple[bool, float, str]:
        """Check if difference between variants is statistically significant.

        Uses a simplified z-test for proportions.

        Args:
            control_variant_id: Control variant ID
            treatment_variant_id: Treatment variant ID
            metric: Metric to test (detection_recall, false_positive_rate, etc.)

        Returns:
            Tuple of (is_significant, p_value, interpretation)
        """
        control_metrics = self.variant_metrics.get(control_variant_id)
        treatment_metrics = self.variant_metrics.get(treatment_variant_id)

        if not control_metrics or not treatment_metrics:
            return False, 1.0, "Insufficient data"

        if (
            control_metrics.total_cases < self.min_sample_size
            or treatment_metrics.total_cases < self.min_sample_size
        ):
            return False, 1.0, f"Need at least {self.min_sample_size} samples per variant"

        # Get metric values
        if metric == "detection_recall":
            p1 = control_metrics.detection_recall
            p2 = treatment_metrics.detection_recall
        elif metric == "false_positive_rate":
            p1 = control_metrics.false_positive_rate
            p2 = treatment_metrics.false_positive_rate
        elif metric == "human_agreement":
            p1 = control_metrics.human_agreement
            p2 = treatment_metrics.human_agreement
        else:
            return False, 1.0, f"Unknown metric: {metric}"

        # Simplified z-test for proportions
        n1 = control_metrics.total_cases
        n2 = treatment_metrics.total_cases

        # Pooled proportion
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

        if se == 0:
            return False, 1.0, "Zero standard error"

        # Z-score
        z = abs(p1 - p2) / se

        # Approximate p-value (two-tailed)
        # Using normal approximation: p ≈ 2 * (1 - Φ(|z|))
        # For |z| > 1.96, p < 0.05 (significant at 95% confidence)
        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        is_significant = p_value < self.significance_level

        improvement = ((p2 - p1) / p1 * 100) if p1 > 0 else 0

        interpretation = (
            f"Treatment is {improvement:+.1f}% vs control. "
            f"{'Significant' if is_significant else 'Not significant'} "
            f"(p={p_value:.4f})"
        )

        return is_significant, p_value, interpretation

    def _normal_cdf(self, z: float) -> float:
        """Approximate normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def gradual_rollout(
        self, treatment_variant_id: str, target_traffic: float, step_size: float = 0.1
    ) -> float:
        """Gradually increase traffic to treatment variant.

        Args:
            treatment_variant_id: Treatment variant ID
            target_traffic: Target traffic percentage (0-1)
            step_size: Traffic increase per step (0-1)

        Returns:
            New traffic percentage
        """
        if treatment_variant_id not in self.variant_traffic:
            return 0.0

        current_traffic = self.variant_traffic[treatment_variant_id]
        new_traffic = min(target_traffic, current_traffic + step_size)

        self.variant_traffic[treatment_variant_id] = new_traffic

        # Update control traffic
        if self.control_variant_id:
            self.variant_traffic[self.control_variant_id] = 1.0 - new_traffic

        return new_traffic

    def rollback_variant(self, variant_id: str) -> bool:
        """Rollback variant and route all traffic to control.

        Args:
            variant_id: Variant ID to rollback

        Returns:
            True if rolled back successfully
        """
        if variant_id not in self.variants:
            return False

        # Set traffic to 0
        self.variant_traffic[variant_id] = 0.0

        # Route all traffic to control
        if self.control_variant_id:
            self.variant_traffic[self.control_variant_id] = 1.0

        # Mark as rolled back in database
        if self.storage_path:
            conn = sqlite3.connect(str(self.storage_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE ab_variants
                SET status = 'rolled_back', traffic_percentage = 0
                WHERE variant_id = ?
            """,
                (variant_id,),
            )

            conn.commit()
            conn.close()

        return True

    def get_variant_summary(self) -> Dict[str, Any]:
        """Get summary of all variants.

        Returns:
            Dictionary with variant information
        """
        summary = {"variants": {}, "control_variant_id": self.control_variant_id}

        for variant_id, config in self.variants.items():
            metrics = self.variant_metrics.get(variant_id)
            summary["variants"][variant_id] = {
                "config_version": config.config_version,
                "traffic": self.variant_traffic.get(variant_id, 0.0),
                "is_control": variant_id == self.control_variant_id,
                "metrics": metrics.to_dict() if metrics else None,
            }

        return summary
