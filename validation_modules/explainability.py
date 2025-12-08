"""
Explainability Validation Module

Implements:
- Feature importance validation
- SHAP value analysis (optional)
- Stability checks
- Attribution distribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.inspection import permutation_importance
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import SHAP (optional dependency)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class ExplainabilityValidator:
    """Validate model explainability and interpretability"""

    def __init__(self, random_seed: int = 42):
        """
        Initialize explainability validator

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def calculate_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate permutation-based feature importance

        Args:
            model: Trained model with predict method
            X: Features
            y: Labels
            n_repeats: Number of times to permute each feature
            feature_names: Optional feature names

        Returns:
            Feature importance results
        """
        logger.info("Calculating permutation importance...")

        try:
            result = permutation_importance(
                model,
                X,
                y,
                n_repeats=n_repeats,
                random_state=self.random_seed,
                n_jobs=-1,
            )

            n_features = X.shape[1]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]

            # Sort by importance
            sorted_idx = result.importances_mean.argsort()[::-1]

            importance_data = {}
            for idx in sorted_idx:
                feat_name = feature_names[idx]
                importance_data[feat_name] = {
                    "mean": float(result.importances_mean[idx]),
                    "std": float(result.importances_std[idx]),
                }

            return {
                "method": "permutation",
                "n_repeats": n_repeats,
                "importance": importance_data,
            }

        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {"error": str(e)}

    def calculate_shap_values(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for feature importance

        Args:
            model: Trained model
            X: Features (limited to max_samples)
            feature_names: Optional feature names
            max_samples: Maximum samples for SHAP calculation

        Returns:
            SHAP values and summary
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}

        logger.info("Calculating SHAP values...")

        try:
            # Limit samples for performance
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Create explainer (try TreeExplainer first, fall back to KernelExplainer)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except (AttributeError, TypeError, ValueError) as e:
                # Fall back to KernelExplainer (slower but works with any model)
                logger.info(
                    f"TreeExplainer failed ({e}), falling back to KernelExplainer"
                )
                explainer = shap.KernelExplainer(model.predict, X_sample[:100])
                shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Binary classification: SHAP returns list [shap_for_class_0, shap_for_class_1]
                # We use index 1 for positive class (convention: positive class is at index 1)
                shap_values = shap_values[1]  # Use positive class

            n_features = X.shape[1]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]

            # Calculate mean absolute SHAP values
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            # Sort by importance
            sorted_idx = mean_abs_shap.argsort()[::-1]

            importance_data = {}
            for idx in sorted_idx:
                feat_name = feature_names[idx]
                importance_data[feat_name] = {
                    "mean_abs_shap": float(mean_abs_shap[idx]),
                    "mean_shap": float(shap_values[:, idx].mean()),
                    "std_shap": float(shap_values[:, idx].std()),
                }

            return {
                "method": "shap",
                "n_samples": len(X_sample),
                "importance": importance_data,
            }

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {"error": str(e)}

    def check_stability(
        self, importance_runs: List[Dict[str, float]], threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Check stability of feature importance across multiple runs

        Args:
            importance_runs: List of importance dictionaries from multiple runs
            threshold: Minimum correlation threshold for stability

        Returns:
            Stability metrics
        """
        if len(importance_runs) < 2:
            return {
                "stable": True,
                "message": "Only one run provided, cannot assess stability",
            }

        # Get common features
        common_features = set(importance_runs[0].keys())
        for run in importance_runs[1:]:
            common_features &= set(run.keys())

        common_features = sorted(common_features)

        # Create matrix of importance values
        importance_matrix = np.array(
            [[run[feat] for feat in common_features] for run in importance_runs]
        )

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(importance_runs)):
            for j in range(i + 1, len(importance_runs)):
                corr = np.corrcoef(importance_matrix[i], importance_matrix[j])[0, 1]
                correlations.append(corr)

        mean_correlation = np.mean(correlations)
        min_correlation = np.min(correlations)

        stable = mean_correlation >= threshold

        return {
            "stable": stable,
            "mean_correlation": float(mean_correlation),
            "min_correlation": float(min_correlation),
            "threshold": threshold,
            "n_runs": len(importance_runs),
            "n_features": len(common_features),
        }

    def analyze_attribution_distribution(
        self, importance_dict: Dict[str, float], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze distribution of feature attributions

        Args:
            importance_dict: Feature importance dictionary
            top_k: Number of top features to analyze

        Returns:
            Attribution analysis
        """
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: (
                abs(x[1])
                if isinstance(x[1], (int, float))
                else abs(x[1].get("mean", 0))
            ),
            reverse=True,
        )

        # Extract importance values
        importance_values = []
        for feat, val in sorted_features:
            if isinstance(val, dict):
                importance_values.append(val.get("mean", 0))
            else:
                importance_values.append(val)

        importance_values = np.array(importance_values)

        # Calculate distribution statistics
        total_importance = np.sum(np.abs(importance_values))
        top_k_importance = np.sum(np.abs(importance_values[:top_k]))

        return {
            "total_features": len(importance_dict),
            "top_k": top_k,
            "top_k_features": [feat for feat, _ in sorted_features[:top_k]],
            "top_k_concentration": (
                float(top_k_importance / total_importance)
                if total_importance > 0
                else 0.0
            ),
            "mean_importance": float(np.mean(np.abs(importance_values))),
            "std_importance": float(np.std(importance_values)),
            "max_importance": float(np.max(np.abs(importance_values))),
            "min_importance": float(np.min(np.abs(importance_values))),
        }

    def validate_explainability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        config: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive explainability validation

        Args:
            model: Trained model
            X: Features
            y: Labels
            config: Validation configuration
            feature_names: Optional feature names

        Returns:
            Validation results
        """
        config = config or {}

        results = {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed,
        }

        # Feature importance analysis
        method = config.get("feature_importance", {}).get("method", "permutation")

        if method == "shap" and SHAP_AVAILABLE:
            n_samples = config.get("feature_importance", {}).get("n_samples", 1000)
            importance_result = self.calculate_shap_values(
                model, X, feature_names, max_samples=n_samples
            )
        else:
            n_repeats = config.get("feature_importance", {}).get("n_repeats", 10)
            importance_result = self.calculate_permutation_importance(
                model, X, y, n_repeats=n_repeats, feature_names=feature_names
            )

        results["feature_importance"] = importance_result

        # Stability check (if configured)
        stability_runs = config.get("feature_importance", {}).get("stability_runs", 0)
        if stability_runs > 1 and "error" not in importance_result:
            logger.info(f"Running stability check with {stability_runs} runs...")
            importance_runs = [importance_result["importance"]]

            for i in range(stability_runs - 1):
                if method == "shap" and SHAP_AVAILABLE:
                    run_result = self.calculate_shap_values(model, X, feature_names)
                else:
                    run_result = self.calculate_permutation_importance(
                        model, X, y, feature_names=feature_names
                    )
                if "error" not in run_result:
                    importance_runs.append(run_result["importance"])

            # Extract just the mean values for stability check
            importance_values_list = []
            for run in importance_runs:
                values = {}
                for feat, val in run.items():
                    if isinstance(val, dict):
                        values[feat] = val.get("mean", 0)
                    else:
                        values[feat] = val
                importance_values_list.append(values)

            stability = self.check_stability(
                importance_values_list,
                threshold=config.get("thresholds", {}).get("min_stability", 0.8),
            )
            results["stability"] = stability

        # Attribution distribution
        if "error" not in importance_result:
            distribution = self.analyze_attribution_distribution(
                importance_result["importance"], top_k=10
            )
            results["attribution_distribution"] = distribution

        return results

    def check_thresholds(
        self, results: Dict, thresholds: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check if explainability meets thresholds

        Args:
            results: Validation results
            thresholds: Threshold values

        Returns:
            Tuple of (all_passed, failed_checks)
        """
        failed_checks = []

        # Check stability
        if "stability" in results and "min_stability" in thresholds:
            stability = results["stability"]
            if not stability.get("stable", False):
                failed_checks.append(
                    f"Feature importance unstable: correlation={stability.get('mean_correlation', 0):.4f} "
                    f"< {thresholds['min_stability']:.4f}"
                )

        return len(failed_checks) == 0, failed_checks

    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save validation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
