"""
Fairness and Ethics Validation Module

Computes fairness metrics including:
- Demographic parity difference
- Equalized odds difference
- TPR/FPR parity
- Precision, recall, F1 scores
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class FairnessMetrics:
    """Calculate fairness and performance metrics"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize fairness metrics calculator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def calculate_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate standard classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names
            
        Returns:
            Dictionary of metrics
        """
        # Ensure arrays are 1D
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate true positive rate and false positive rate
        tpr = recall  # TPR is same as recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_samples": len(y_true)
        }
        
        return metrics
    
    def calculate_demographic_parity_difference(
        self,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate demographic parity difference
        
        Demographic parity requires that P(Y_pred=1|A=0) ≈ P(Y_pred=1|A=1)
        
        Args:
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute (e.g., protected class)
            
        Returns:
            Dictionary with demographic parity metrics
        """
        y_pred = np.asarray(y_pred).flatten()
        sensitive_feature = np.asarray(sensitive_feature).flatten()
        
        # Get unique groups
        groups = np.unique(sensitive_feature)
        
        if len(groups) < 2:
            logger.warning("Less than 2 groups found for demographic parity calculation")
            return {"demographic_parity_difference": 0.0, "group_rates": {}}
        
        # Calculate positive rate for each group
        group_rates = {}
        for group in groups:
            mask = sensitive_feature == group
            group_pred = y_pred[mask]
            positive_rate = np.mean(group_pred) if len(group_pred) > 0 else 0.0
            group_rates[str(group)] = positive_rate
        
        # Calculate max difference
        rates = list(group_rates.values())
        dpd = max(rates) - min(rates)
        
        return {
            "demographic_parity_difference": dpd,
            "group_rates": group_rates,
            "max_rate": max(rates),
            "min_rate": min(rates)
        }
    
    def calculate_equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate equalized odds difference
        
        Equalized odds requires:
        - P(Y_pred=1|Y=1,A=0) ≈ P(Y_pred=1|Y=1,A=1) (TPR parity)
        - P(Y_pred=1|Y=0,A=0) ≈ P(Y_pred=1|Y=0,A=1) (FPR parity)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute
            
        Returns:
            Dictionary with equalized odds metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        sensitive_feature = np.asarray(sensitive_feature).flatten()
        
        groups = np.unique(sensitive_feature)
        
        if len(groups) < 2:
            logger.warning("Less than 2 groups found for equalized odds calculation")
            return {
                "equalized_odds_difference": 0.0,
                "tpr_difference": 0.0,
                "fpr_difference": 0.0
            }
        
        # Calculate TPR and FPR for each group
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in groups:
            mask = sensitive_feature == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # Calculate confusion matrix for this group
            if len(y_true_group) > 0:
                cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                else:
                    tpr, fpr = 0.0, 0.0
            else:
                tpr, fpr = 0.0, 0.0
                
            tpr_by_group[str(group)] = tpr
            fpr_by_group[str(group)] = fpr
        
        # Calculate differences
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values)
        fpr_diff = max(fpr_values) - min(fpr_values)
        
        # Equalized odds difference is max of TPR and FPR differences
        eod = max(tpr_diff, fpr_diff)
        
        return {
            "equalized_odds_difference": eod,
            "tpr_difference": tpr_diff,
            "fpr_difference": fpr_diff,
            "tpr_by_group": tpr_by_group,
            "fpr_by_group": fpr_by_group
        }
    
    def calculate_all_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all fairness and performance metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: Optional sensitive attribute for fairness metrics
            
        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed
        }
        
        # Performance metrics
        performance = self.calculate_performance_metrics(y_true, y_pred)
        metrics["performance"] = performance
        
        # Fairness metrics (if sensitive feature provided)
        if sensitive_feature is not None:
            try:
                dpd = self.calculate_demographic_parity_difference(y_pred, sensitive_feature)
                metrics["demographic_parity"] = dpd
                
                eod = self.calculate_equalized_odds_difference(y_true, y_pred, sensitive_feature)
                metrics["equalized_odds"] = eod
                
                logger.info(f"Calculated fairness metrics: DPD={dpd['demographic_parity_difference']:.4f}, "
                          f"EOD={eod['equalized_odds_difference']:.4f}")
            except Exception as e:
                logger.error(f"Error calculating fairness metrics: {e}")
                metrics["fairness_error"] = str(e)
        
        return metrics
    
    def save_metrics(self, metrics: Dict, output_path: Path) -> None:
        """
        Save metrics to JSON file
        
        Args:
            metrics: Metrics dictionary
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_path}")
    
    def check_thresholds(
        self,
        metrics: Dict,
        thresholds: Dict[str, float]
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
        
        # Check performance thresholds
        if "performance" in metrics:
            perf = metrics["performance"]
            for metric, threshold in thresholds.items():
                if metric in perf:
                    value = perf[metric]
                    # Metrics like FPR should be below threshold, others above
                    if "fpr" in metric.lower() or "false" in metric.lower():
                        if value > threshold:
                            failed_checks.append(
                                f"{metric}: {value:.4f} > {threshold:.4f} (should be <=)"
                            )
                    else:
                        if value < threshold:
                            failed_checks.append(
                                f"{metric}: {value:.4f} < {threshold:.4f} (should be >=)"
                            )
        
        # Check fairness thresholds
        if "demographic_parity" in metrics and "demographic_parity_diff" in thresholds:
            dpd = metrics["demographic_parity"]["demographic_parity_difference"]
            threshold = thresholds["demographic_parity_diff"]
            if dpd > threshold:
                failed_checks.append(
                    f"demographic_parity_difference: {dpd:.4f} > {threshold:.4f}"
                )
        
        if "equalized_odds" in metrics and "equalized_odds_diff" in thresholds:
            eod = metrics["equalized_odds"]["equalized_odds_difference"]
            threshold = thresholds["equalized_odds_diff"]
            if eod > threshold:
                failed_checks.append(
                    f"equalized_odds_difference: {eod:.4f} > {threshold:.4f}"
                )
        
        return len(failed_checks) == 0, failed_checks
