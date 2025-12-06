"""
Data Integrity Validation Module

Implements:
- Schema validation (nulls, ranges, categorical domains)
- Data drift detection (KS test, PSI)
- Duplicate detection
- Data leakage detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataIntegrityValidator:
    """Validate data integrity, schema, and drift"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data integrity validator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        schema_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Validate dataframe schema
        
        Args:
            df: Input dataframe
            schema_config: Schema configuration
            
        Returns:
            Schema validation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "checks": {}
        }
        
        # Null value checks
        null_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
        results["null_percentages"] = null_percentages
        
        max_null_pct = schema_config.get("max_null_percentage", 0.05) * 100 if schema_config else 5.0
        columns_with_high_nulls = {
            col: pct for col, pct in null_percentages.items() if pct > max_null_pct
        }
        results["checks"]["high_nulls"] = {
            "passed": len(columns_with_high_nulls) == 0,
            "columns": columns_with_high_nulls
        }
        
        # Data types check
        dtypes = df.dtypes.astype(str).to_dict()
        results["dtypes"] = dtypes
        
        # Numeric range checks (if schema provided)
        if schema_config and "ranges" in schema_config:
            range_violations = {}
            for col, (min_val, max_val) in schema_config["ranges"].items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if violations > 0:
                        range_violations[col] = {
                            "expected_range": [min_val, max_val],
                            "actual_range": [float(df[col].min()), float(df[col].max())],
                            "violations": int(violations)
                        }
            results["checks"]["range_violations"] = {
                "passed": len(range_violations) == 0,
                "violations": range_violations
            }
        
        # Categorical domain checks
        if schema_config and "categorical_domains" in schema_config:
            domain_violations = {}
            for col, expected_values in schema_config["categorical_domains"].items():
                if col in df.columns:
                    actual_values = set(df[col].dropna().unique())
                    unexpected = actual_values - set(expected_values)
                    if unexpected:
                        domain_violations[col] = {
                            "unexpected_values": list(unexpected),
                            "count": len(unexpected)
                        }
            results["checks"]["categorical_violations"] = {
                "passed": len(domain_violations) == 0,
                "violations": domain_violations
            }
        
        # Overall schema validation
        results["schema_valid"] = all(
            check.get("passed", True) for check in results["checks"].values()
        )
        
        logger.info(f"Schema validation: {'PASSED' if results['schema_valid'] else 'FAILED'}")
        return results
    
    # Small constant to avoid log(0) in PSI calculation
    PSI_EPSILON = 1e-10
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            expected: Baseline/expected distribution
            actual: Current/actual distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Handle edge cases
        if len(expected) == 0 or len(actual) == 0:
            logger.warning("Empty array provided for PSI calculation")
            return 0.0
        
        # Create bins based on expected distribution
        expected = np.asarray(expected).flatten()
        actual = np.asarray(actual).flatten()
        
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Create bins
        bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        # Calculate distributions
        expected_percents, _ = np.histogram(expected, bins=bin_edges)
        actual_percents, _ = np.histogram(actual, bins=bin_edges)
        
        # Normalize to percentages and add epsilon to avoid log(0)
        expected_percents = expected_percents / len(expected) + self.PSI_EPSILON
        actual_percents = actual_percents / len(actual) + self.PSI_EPSILON
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def detect_drift_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Args:
            reference: Reference/baseline distribution
            current: Current distribution
            alpha: Significance level
            
        Returns:
            Drift detection results
        """
        reference = np.asarray(reference).flatten()
        current = np.asarray(current).flatten()
        
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]
        
        if len(reference) == 0 or len(current) == 0:
            return {
                "drift_detected": False,
                "statistic": 0.0,
                "pvalue": 1.0,
                "error": "Empty arrays"
            }
        
        # Perform KS test
        statistic, pvalue = stats.ks_2samp(reference, current)
        
        return {
            "drift_detected": pvalue < alpha,
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "alpha": alpha
        }
    
    def analyze_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze drift across all features
        
        Args:
            reference_df: Reference/training dataset
            current_df: Current/validation dataset
            config: Drift configuration
            
        Returns:
            Comprehensive drift analysis
        """
        config = config or {}
        psi_threshold = config.get("psi_threshold", 0.2)
        ks_alpha = config.get("ks_test_alpha", 0.05)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "reference_samples": len(reference_df),
            "current_samples": len(current_df),
            "features_analyzed": [],
            "drift_detected": {}
        }
        
        # Analyze each numeric column
        for col in reference_df.columns:
            if col not in current_df.columns:
                continue
                
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                try:
                    # Calculate PSI
                    psi_value = self.calculate_psi(
                        reference_df[col].dropna().values,
                        current_df[col].dropna().values
                    )
                    
                    # Perform KS test
                    ks_result = self.detect_drift_ks_test(
                        reference_df[col].dropna().values,
                        current_df[col].dropna().values,
                        alpha=ks_alpha
                    )
                    
                    results["features_analyzed"].append(col)
                    results["drift_detected"][col] = {
                        "psi": psi_value,
                        "psi_drift": psi_value > psi_threshold,
                        "ks_statistic": ks_result["statistic"],
                        "ks_pvalue": ks_result["pvalue"],
                        "ks_drift": ks_result["drift_detected"]
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing drift for {col}: {e}")
                    results["drift_detected"][col] = {"error": str(e)}
        
        # Summary statistics
        total_features = len(results["features_analyzed"])
        psi_drifted = sum(1 for v in results["drift_detected"].values() 
                         if v.get("psi_drift", False))
        ks_drifted = sum(1 for v in results["drift_detected"].values() 
                        if v.get("ks_drift", False))
        
        results["summary"] = {
            "total_features": total_features,
            "psi_drifted_count": psi_drifted,
            "ks_drifted_count": ks_drifted,
            "psi_drift_percentage": psi_drifted / total_features if total_features > 0 else 0.0,
            "ks_drift_percentage": ks_drifted / total_features if total_features > 0 else 0.0
        }
        
        logger.info(f"Drift analysis: {psi_drifted}/{total_features} features with PSI drift, "
                   f"{ks_drifted}/{total_features} with KS drift")
        
        return results
    
    def detect_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect duplicate rows
        
        Args:
            df: Input dataframe
            subset: Columns to consider for duplicates
            
        Returns:
            Duplicate detection results
        """
        duplicates = df.duplicated(subset=subset)
        duplicate_count = duplicates.sum()
        duplicate_percentage = duplicate_count / len(df) * 100 if len(df) > 0 else 0.0
        
        results = {
            "total_rows": len(df),
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": duplicate_percentage,
            "subset_columns": subset or "all"
        }
        
        logger.info(f"Found {duplicate_count} duplicates ({duplicate_percentage:.2f}%)")
        return results
    
    def detect_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        id_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect data leakage between train and test sets
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
            id_columns: Columns to use as identifiers
            
        Returns:
            Leakage detection results
        """
        results = {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "leakage_detected": False,
            "leaked_samples": 0
        }
        
        if id_columns:
            # Check for ID overlap
            train_ids = set(train_df[id_columns].apply(tuple, axis=1))
            test_ids = set(test_df[id_columns].apply(tuple, axis=1))
            overlap = train_ids & test_ids
            
            results["leaked_samples"] = len(overlap)
            results["leakage_detected"] = len(overlap) > 0
            results["leakage_percentage"] = len(overlap) / len(test_df) * 100 if len(test_df) > 0 else 0.0
        else:
            # Check for exact row overlap (slower)
            train_rows = set(train_df.apply(tuple, axis=1))
            test_rows = set(test_df.apply(tuple, axis=1))
            overlap = train_rows & test_rows
            
            results["leaked_samples"] = len(overlap)
            results["leakage_detected"] = len(overlap) > 0
            results["leakage_percentage"] = len(overlap) / len(test_df) * 100 if len(test_df) > 0 else 0.0
        
        if results["leakage_detected"]:
            logger.warning(f"Data leakage detected: {results['leaked_samples']} samples "
                         f"({results['leakage_percentage']:.2f}%)")
        else:
            logger.info("No data leakage detected")
        
        return results
    
    def save_results(self, results: Dict, output_path: Path) -> None:
        """Save validation results to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")
