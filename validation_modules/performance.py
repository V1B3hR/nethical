"""
Performance Validation Module

Implements:
- Stratified train/test splitting
- Performance metrics calculation
- Confidence intervals via bootstrapping
- ROC/PR curves
- Per-class performance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceValidator:
    """Validate model/system performance with robust evaluation"""

    def __init__(self, random_seed: int = 42):
        """
        Initialize performance validator

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        stratified: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets

        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            stratified: Whether to use stratified split

        Returns:
            X_train, X_test, y_train, y_test
        """
        if stratified:
            return train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_seed
            )
        else:
            return train_test_split(
                X, y, test_size=test_size, random_state=self.random_seed
            )

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional, for ROC AUC)

        Returns:
            Dictionary of metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

        # Precision, recall, F1 for binary classification
        # Note: Using 'binary' average assumes binary classification task (classes 0 and 1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # ROC AUC if probabilities provided
        if y_prob is not None:
            try:
                # For binary classification: if 2D array with 2 columns, use positive class probs
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    # Binary classification with probs for both classes
                    y_prob_positive = y_prob[:, 1]
                else:
                    # Single column of probabilities (already for positive class)
                    y_prob_positive = y_prob.flatten()
                metrics["roc_auc"] = roc_auc_score(y_true, y_prob_positive)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                metrics["roc_auc"] = None

        metrics["total_samples"] = len(y_true)

        return metrics

    def calculate_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str = "accuracy",
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate confidence intervals via bootstrapping

        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_name: Metric to calculate CI for
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dictionary with point estimate and confidence interval
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        n_samples = len(y_true)

        # Calculate metric function
        def calc_metric(yt, yp):
            if metric_name == "accuracy":
                return accuracy_score(yt, yp)
            elif metric_name == "precision":
                p, _, _, _ = precision_recall_fscore_support(
                    yt, yp, average="binary", zero_division=0
                )
                return p
            elif metric_name == "recall":
                _, r, _, _ = precision_recall_fscore_support(
                    yt, yp, average="binary", zero_division=0
                )
                return r
            elif metric_name == "f1_score":
                _, _, f1, _ = precision_recall_fscore_support(
                    yt, yp, average="binary", zero_division=0
                )
                return f1
            else:
                return accuracy_score(yt, yp)

        # Bootstrap
        bootstrap_scores = []
        rng = np.random.RandomState(self.random_seed)

        for i in range(n_iterations):
            # Sample with replacement
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate metric
            score = calc_metric(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)

        # Calculate confidence interval
        bootstrap_scores = np.array(bootstrap_scores)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        point_estimate = calc_metric(y_true, y_pred)

        return {
            "metric": metric_name,
            "point_estimate": point_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "confidence_level": confidence_level,
            "n_iterations": n_iterations,
            "std_error": np.std(bootstrap_scores),
        }

    def calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class performance metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names

        Returns:
            Dictionary of per-class metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))

        if class_names is None:
            class_names = [f"class_{c}" for c in classes]

        # Calculate metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0
        )

        per_class = {}
        for i, cls in enumerate(classes):
            cls_name = class_names[i] if i < len(class_names) else f"class_{cls}"
            per_class[cls_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }

        return per_class

    def validate_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive performance validation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            config: Validation configuration

        Returns:
            Comprehensive validation results
        """
        config = config or {}

        results = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed,
        }

        # Basic metrics
        logger.info("Calculating performance metrics...")
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        results["metrics"] = metrics

        # Confidence intervals
        if config.get("bootstrap", {}).get("enabled", True):
            logger.info("Calculating confidence intervals...")
            ci_config = config.get("bootstrap", {})
            n_iterations = ci_config.get("n_iterations", 1000)
            confidence_level = ci_config.get("confidence_level", 0.95)

            confidence_intervals = {}
            for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
                ci = self.calculate_confidence_intervals(
                    y_true,
                    y_pred,
                    metric_name,
                    n_iterations=n_iterations,
                    confidence_level=confidence_level,
                )
                confidence_intervals[metric_name] = ci

            results["confidence_intervals"] = confidence_intervals

        # Per-class metrics
        logger.info("Calculating per-class metrics...")
        per_class = self.calculate_per_class_metrics(y_true, y_pred)
        results["per_class_metrics"] = per_class

        # ROC curve data (if probabilities provided)
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = y_prob.flatten()

                fpr, tpr, thresholds = roc_curve(y_true, y_prob_positive)
                results["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }

                precision, recall, pr_thresholds = precision_recall_curve(
                    y_true, y_prob_positive
                )
                results["precision_recall_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                }
            except Exception as e:
                logger.warning(f"Could not calculate curves: {e}")

        return results

    def check_thresholds(
        self, metrics: Dict, thresholds: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet thresholds

        Args:
            metrics: Calculated metrics
            thresholds: Threshold values

        Returns:
            Tuple of (all_passed, failed_checks)
        """
        failed_checks = []

        for metric, threshold in thresholds.items():
            if metric in metrics:
                value = metrics[metric]
                if value is not None:
                    if value < threshold:
                        failed_checks.append(
                            f"{metric}: {value:.4f} < {threshold:.4f} (threshold)"
                        )
                        logger.warning(
                            f"Threshold violation: {metric}={value:.4f} < {threshold:.4f}"
                        )

        return len(failed_checks) == 0, failed_checks

    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save validation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
