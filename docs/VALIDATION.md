# Validation Suite Documentation

## Overview

The Nethical validation suite provides comprehensive testing and validation across multiple dimensions:

- **Ethics/Fairness Benchmark**: Tests ethical violation detection and fairness metrics
- **Performance Validation**: Evaluates model performance with confidence intervals
- **Data Integrity**: Validates data quality, schema, and drift detection
- **Explainability**: Ensures model interpretability and feature importance stability
- **Drift Detection**: Monitors data and model drift over time

## Quick Start

### Running Locally

```bash
# Run all validation suites
python run_validation.py

# Run specific suites
python run_validation.py --suites ethics_benchmark performance

# Use custom config
python run_validation.py --config my_validation.yaml

# Specify output location
python run_validation.py --output validation_results.json
```

### Using pytest

```bash
# Run all validation tests
pytest tests/validation/ -v

# Run specific test file
pytest tests/validation/test_ethics_benchmark.py -v

# Generate JSON report
pytest tests/validation/ -v --json-report --json-report-file=report.json
```

## Configuration

### Configuration File

The validation suite uses `validation.yaml` for configuration. The file supports:

- Global settings (random seed, artifacts directory, log level)
- Per-suite thresholds
- Feature-specific configuration
- Environment variable overrides

Example `validation.yaml`:

```yaml
# Global settings
global:
  random_seed: 42
  artifacts_dir: artifacts/validation
  log_level: INFO

# Ethics benchmark configuration
ethics_benchmark:
  enabled: true
  thresholds:
    precision: 0.92    # Minimum precision
    recall: 0.88       # Minimum recall
    f1_score: 0.90     # Minimum F1 score
  
  fairness:
    demographic_parity_diff: 0.1
    equalized_odds_diff: 0.1

# Performance validation
performance:
  enabled: true
  thresholds:
    accuracy: 0.85
    precision: 0.85
    recall: 0.80
    f1_score: 0.82
    roc_auc: 0.90
  
  bootstrap:
    enabled: true
    n_iterations: 1000
    confidence_level: 0.95

# Data integrity
data_integrity:
  enabled: true
  drift:
    psi_threshold: 0.2
    ks_test_alpha: 0.05
  
  schema:
    max_null_percentage: 0.05

# Explainability
explainability:
  enabled: true
  thresholds:
    min_stability: 0.8
    min_coverage: 0.95
```

### Environment Variable Overrides

Override any configuration value using environment variables:

```bash
# Override precision threshold
export VALIDATION_ETHICS_BENCHMARK_THRESHOLDS_PRECISION=0.95

# Override PSI threshold
export VALIDATION_DATA_INTEGRITY_DRIFT_PSI_THRESHOLD=0.15

# Run validation with overrides
python run_validation.py
```

Format: `VALIDATION_<SUITE>_<SECTION>_<KEY>=value`

## Validation Suites

### Ethics Benchmark

**Purpose**: Tests ethical violation detection across multiple categories.

**Metrics Calculated**:
- Precision, Recall, F1 Score
- True Positives, False Positives, False Negatives, True Negatives
- Demographic Parity Difference
- Equalized Odds Difference
- TPR/FPR Parity

**Violation Categories**:
- Harmful content
- Deception
- Privacy violation
- Discrimination
- Manipulation
- Unauthorized access

**Thresholds** (configurable):
- Precision ≥ 92%
- Recall ≥ 88%
- F1 Score ≥ 90%
- Demographic Parity Difference ≤ 0.1
- Equalized Odds Difference ≤ 0.1

**Artifacts Generated**:
- `metrics.json`: Comprehensive metrics
- `summary.json`: Test summary with threshold checks
- `confusion_matrix.png`: Visualization (if matplotlib available)
- `fairness_metrics.json`: Detailed fairness analysis

### Performance Validation

**Purpose**: Validates model/system performance with robust statistical methods.

**Features**:
- Stratified train/test splitting
- Confidence intervals via bootstrapping
- Per-class performance metrics
- ROC and Precision-Recall curves

**Metrics Calculated**:
- Accuracy, Precision, Recall, F1 Score
- ROC AUC Score
- Specificity
- Confidence Intervals (95% by default)
- Per-class metrics

**Thresholds** (configurable):
- Accuracy ≥ 85%
- Precision ≥ 85%
- Recall ≥ 80%
- F1 Score ≥ 82%
- ROC AUC ≥ 90%

**Artifacts Generated**:
- `performance_metrics.csv`: All metrics in CSV format
- `confidence_intervals.json`: Bootstrap CI results
- `roc_curve.png`: ROC curve visualization
- `pr_curve.png`: Precision-Recall curve
- `per_class_performance.json`: Per-class breakdown

### Data Integrity

**Purpose**: Validates data quality, schema compliance, and detects drift.

**Features**:
- Schema validation (nulls, ranges, categorical domains)
- Population Stability Index (PSI) calculation
- Kolmogorov-Smirnov test for drift
- Duplicate detection
- Data leakage detection

**Checks Performed**:
- Null value percentages per column
- Numeric range violations
- Categorical domain violations
- Statistical drift detection (PSI, KS test)
- Duplicate row detection
- Train/test leakage

**Thresholds** (configurable):
- PSI ≤ 0.2 (drift threshold)
- KS test α = 0.05 (significance level)
- Max nulls ≤ 5% per column
- Max duplicates ≤ 1%

**Artifacts Generated**:
- `schema_validation.json`: Schema check results
- `drift_analysis.csv`: Drift metrics per feature
- `quality_report.json`: Data quality summary
- `drift_visualizations.png`: Drift plots

### Explainability

**Purpose**: Validates model interpretability and feature importance.

**Features**:
- Permutation importance calculation
- SHAP value analysis (optional, requires `shap` package)
- Feature importance stability checks
- Attribution distribution analysis

**Methods Supported**:
- Permutation importance (always available)
- SHAP TreeExplainer (for tree-based models)
- SHAP KernelExplainer (for any model, slower)

**Metrics Calculated**:
- Feature importance rankings
- Importance stability across runs
- Attribution distributions
- Top-K feature concentration

**Thresholds** (configurable):
- Minimum stability ≥ 0.8 (correlation across runs)
- Minimum coverage ≥ 95% (decision coverage)
- Max explanation latency ≤ 500ms

**Artifacts Generated**:
- `feature_importance.png`: Importance plot
- `shap_summary.png`: SHAP summary plot (if available)
- `stability_metrics.json`: Stability analysis
- `attribution_dist.csv`: Attribution distributions

### Drift Detection

**Purpose**: Monitors data and model drift over time.

**Features**:
- Daily and weekly PSI monitoring
- KS test for distribution changes
- Feature-level drift tracking
- Baseline comparison

**Metrics Calculated**:
- PSI (Population Stability Index) per feature
- KS test statistic and p-value
- Drift percentage across features
- Temporal drift trends

**Thresholds** (configurable):
- Daily PSI ≤ 0.2
- Weekly PSI ≤ 0.3
- KS test p-value ≤ 0.05

**Artifacts Generated**:
- `drift_trends.png`: Temporal drift visualization
- `psi_metrics.csv`: PSI values per feature
- `drift_alerts.json`: Drift warnings

## Reproducibility

All validation suites use deterministic seeding for reproducibility:

```python
# Global seed
RANDOM_SEED = 42

# NumPy
np.random.seed(RANDOM_SEED)

# Scikit-learn
train_test_split(..., random_state=RANDOM_SEED)
StratifiedKFold(..., random_state=RANDOM_SEED)

# Bootstrap
np.random.RandomState(RANDOM_SEED)
```

## Artifacts

### Directory Structure

```
artifacts/validation/
├── ethics_benchmark/
│   ├── metrics.json
│   ├── summary.json
│   ├── confusion_matrix.png
│   └── fairness_metrics.json
├── performance/
│   ├── performance_metrics.csv
│   ├── confidence_intervals.json
│   ├── roc_curve.png
│   └── pr_curve.png
├── data_integrity/
│   ├── schema_validation.json
│   ├── drift_analysis.csv
│   └── quality_report.json
├── explainability/
│   ├── feature_importance.png
│   ├── stability_metrics.json
│   └── attribution_dist.csv
└── validation.json  # Overall summary
```

### Artifact Retention

In CI/CD:
- Artifacts are uploaded using `actions/upload-artifact@v4`
- Default retention: 30 days (configurable)
- Accessible via GitHub Actions UI

## CI/CD Integration

### GitHub Actions Workflow

The validation suite runs automatically on:
- Push to main/develop branches
- Pull requests
- Scheduled runs (daily at 6 AM UTC)
- Manual workflow dispatch

### Workflow Features

- Parallel suite execution
- Artifact upload on completion
- PR comments with validation results
- Issue creation on failure
- Metric annotations

### Example Workflow Usage

```yaml
- name: Run Validation Suite
  run: python run_validation.py --output validation_reports/validation.json
  
- name: Upload Artifacts
  uses: actions/upload-artifact@v4
  with:
    name: validation-results
    path: validation_reports/
    retention-days: 30
  if: always()
```

## Troubleshooting

### Common Issues

**Issue**: "Config file not found"
```bash
# Solution: Specify config path explicitly
python run_validation.py --config validation.yaml
```

**Issue**: "ModuleNotFoundError: No module named 'validation_modules'"
```bash
# Solution: Run from project root or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_validation.py
```

**Issue**: "SHAP not available"
```bash
# Solution: Install optional dependency
pip install shap

# Or skip SHAP-based explainability
# (permutation importance will be used instead)
```

**Issue**: "Threshold violations"
```bash
# Solution 1: Adjust thresholds in config
# Solution 2: Override via environment variable
export VALIDATION_ETHICS_BENCHMARK_THRESHOLDS_PRECISION=0.85
```

### Debugging

Enable verbose logging:

```python
# In code
logging.basicConfig(level=logging.DEBUG)

# Via config
global:
  log_level: DEBUG
```

View detailed test output:

```bash
pytest tests/validation/ -v -s  # -s shows print statements
```

## Best Practices

1. **Deterministic Seeds**: Always use consistent random seeds for reproducibility
2. **Version Control Config**: Track `validation.yaml` changes in git
3. **Document Threshold Changes**: Add comments explaining why thresholds changed
4. **Review Artifacts**: Regularly review generated artifacts for insights
5. **Monitor Trends**: Track metrics over time to detect degradation
6. **Stratified Splits**: Use stratified sampling for imbalanced datasets
7. **Confidence Intervals**: Report metrics with confidence intervals
8. **Feature Importance**: Validate model explanations match domain knowledge

## Dependencies

### Required
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- pyyaml >= 6.0

### Optional
- shap >= 0.40.0 (for SHAP-based explainability)
- matplotlib >= 3.5.0 (for visualizations)
- seaborn >= 0.12.0 (for enhanced plots)

### Install

```bash
# Core dependencies
pip install -r requirements.txt

# With optional dependencies
pip install shap matplotlib seaborn
```

## API Reference

### ValidationConfig

```python
from validation_modules.config_loader import ValidationConfig

config = ValidationConfig("validation.yaml")

# Get value with dot notation
precision_threshold = config.get("ethics_benchmark.thresholds.precision", 0.92)

# Get suite config
ethics_config = config.get_suite_config("ethics_benchmark")

# Check if suite enabled
if config.is_enabled("performance"):
    # Run performance validation
    pass
```

### FairnessMetrics

```python
from validation_modules.fairness import FairnessMetrics

fairness = FairnessMetrics(random_seed=42)

# Calculate metrics
metrics = fairness.calculate_all_fairness_metrics(y_true, y_pred, sensitive_feature)

# Check thresholds
passed, failures = fairness.check_thresholds(metrics, thresholds)
```

### PerformanceValidator

```python
from validation_modules.performance import PerformanceValidator

validator = PerformanceValidator(random_seed=42)

# Validate performance
results = validator.validate_performance(y_true, y_pred, y_prob, config)

# Calculate confidence intervals
ci = validator.calculate_confidence_intervals(y_true, y_pred, "accuracy")
```

### DataIntegrityValidator

```python
from validation_modules.data_integrity import DataIntegrityValidator

validator = DataIntegrityValidator(random_seed=42)

# Validate schema
schema_results = validator.validate_schema(df, schema_config)

# Analyze drift
drift_results = validator.analyze_drift(reference_df, current_df, config)
```

### ExplainabilityValidator

```python
from validation_modules.explainability import ExplainabilityValidator

validator = ExplainabilityValidator(random_seed=42)

# Validate explainability
results = validator.validate_explainability(model, X, y, config, feature_names)
```

## Support

For issues or questions:
1. Check this documentation
2. Review existing GitHub issues
3. Create a new issue with:
   - Error messages
   - Configuration used
   - Steps to reproduce

## Changelog

### Version 2.0.0 (Current)
- Complete validation module restructuring
- Added FairnessMetrics module
- Added PerformanceValidator with bootstrapping
- Added DataIntegrityValidator with drift detection
- Added ExplainabilityValidator
- Configurable thresholds via YAML
- Environment variable overrides
- Comprehensive artifact generation
- Deterministic seeding for reproducibility

### Version 1.0.0
- Initial validation suite
- Basic pytest tests
- Simple metrics calculation
