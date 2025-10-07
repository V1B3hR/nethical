# Ethical Taxonomy Integration in Model Training

This document describes the integration of the Ethical Taxonomy system into the `train_any_model.py` training pipeline.

## Overview

The Ethical Taxonomy system tags violations detected during model training with multi-dimensional ethical impact scores across four key dimensions:

- **Privacy**: Data privacy, confidentiality, and personal information protection
- **Manipulation**: Deceptive practices, coercion, and psychological exploitation  
- **Fairness**: Equitable treatment, non-discrimination, and bias mitigation
- **Safety**: Physical and psychological safety, harm prevention

## Usage

Enable ethical taxonomy tagging during training by adding the `--enable-ethical-taxonomy` flag:

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-ethical-taxonomy \
    --taxonomy-path ethics_taxonomy.json
```

### Command-Line Options

- `--enable-ethical-taxonomy`: Enable ethical taxonomy violation tagging
- `--taxonomy-path`: Path to the ethics taxonomy configuration file (default: `ethics_taxonomy.json`)

## What Gets Tagged

During training evaluation, the following violations are automatically tagged with ethical dimensions:

### 1. Calibration Error (`calibration_error`)
Triggered when the Expected Calibration Error (ECE) exceeds 0.08.

**Ethical Impact:**
- **Fairness** (0.7): Miscalibrated predictions can lead to unfair treatment
- **Safety** (0.5): Overconfident predictions may cause safety issues
- **Manipulation** (0.3): Poor calibration can mislead users

**Severity:**
- Medium: 0.08 < ECE ≤ 0.15
- High: ECE > 0.15

### 2. Low Accuracy (`low_accuracy`)
Triggered when model accuracy falls below 0.85.

**Ethical Impact:**
- **Fairness** (0.6): Inaccurate models may treat groups unfairly
- **Safety** (0.4): Poor predictions can impact user safety
- **Manipulation** (0.2): Unreliable outputs may mislead users

**Severity:**
- Medium: 0.70 ≤ accuracy < 0.85
- High: accuracy < 0.70

## Output

When violations are detected, the system generates:

### 1. Console Output

```
[INFO] Analyzing violations with ethical taxonomy...
  - Calibration Error: Primary dimension = fairness
    Dimension scores: fairness=0.80, safety=0.50, manipulation=0.30
  - Low Accuracy: Primary dimension = fairness
    Dimension scores: fairness=0.70, safety=0.40, manipulation=0.20

[INFO] Ethical Taxonomy Coverage: 100.0%
[INFO] Coverage target: 90.0%
[INFO] Meets target: True
```

### 2. JSON Report

A detailed JSON report is saved alongside model metrics:

**Location:** `models/candidates/{model_type}_taxonomy_{timestamp}.json`

**Structure:**
```json
{
  "model_type": "heuristic",
  "timestamp": "2025-10-07T11:51:38.696372+00:00",
  "promoted": false,
  "violation_tags": [
    {
      "violation_type": "calibration_error",
      "primary_dimension": "fairness",
      "dimensions": {
        "fairness": 0.80,
        "safety": 0.50,
        "manipulation": 0.30
      },
      "severity": "medium"
    }
  ],
  "coverage_stats": {
    "total_violation_types": 2,
    "tagged_types": 2,
    "coverage_percentage": 100.0,
    "meets_target": true
  },
  "dimensions": {
    "privacy": {
      "description": "Data privacy, confidentiality, and personal information protection",
      "weight": 1.0,
      "severity_multiplier": 1.2
    }
    // ... other dimensions
  }
}
```

## Coverage Tracking

The system tracks coverage of violation types against the ethical taxonomy:

- **Coverage Percentage**: Percentage of violations that have ethical mappings
- **Target**: 90% coverage by default (configurable in `ethics_taxonomy.json`)
- **Meets Target**: Boolean indicating if coverage target is met

## Integration with Other Features

Ethical Taxonomy works seamlessly with other training features:

### With Drift Tracking
```bash
python training/train_any_model.py \
    --model-type logistic \
    --enable-drift-tracking \
    --enable-ethical-taxonomy
```

### With Audit Logging
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --enable-audit \
    --enable-ethical-taxonomy
```

### With All Features
```bash
python training/train_any_model.py \
    --model-type correlation \
    --enable-audit \
    --enable-drift-tracking \
    --enable-ethical-taxonomy \
    --cohort-id "production_v1"
```

## Configuration

### Adding New Violation Types

To add new violation types to the taxonomy, edit `ethics_taxonomy.json`:

```json
{
  "mapping": {
    "your_violation_type": {
      "privacy": 0.8,
      "fairness": 0.6,
      "safety": 0.4,
      "manipulation": 0.3,
      "description": "Description of the violation"
    }
  }
}
```

### Adjusting Dimension Weights

Modify dimension weights in `ethics_taxonomy.json`:

```json
{
  "dimensions": {
    "privacy": {
      "weight": 1.0,
      "severity_multiplier": 1.2
    }
  }
}
```

## Example

See `examples/train_with_ethical_taxonomy.py` for a complete working example:

```bash
python examples/train_with_ethical_taxonomy.py
```

This example demonstrates:
- Training with ethical taxonomy enabled
- Reading and parsing taxonomy reports
- Understanding dimension scores
- Coverage statistics interpretation

## Benefits

1. **Transparency**: Clear visibility into ethical implications of model violations
2. **Prioritization**: Primary dimension helps prioritize remediation efforts
3. **Compliance**: Documentation for regulatory and ethical audits
4. **Tracking**: Monitor ethical coverage over time
5. **Integration**: Works with existing training infrastructure

## API Reference

### EthicalTaxonomy Class

```python
from nethical.core.ethical_taxonomy import EthicalTaxonomy

# Initialize
taxonomy = EthicalTaxonomy(
    taxonomy_path="ethics_taxonomy.json",
    coverage_target=0.9
)

# Tag a violation
scores = taxonomy.tag_violation(
    violation_type="calibration_error",
    context={'ece': 0.12, 'severity': 'high'}
)

# Create complete tagging
tagging = taxonomy.create_tagging(
    violation_type="low_accuracy",
    context={'accuracy': 0.75}
)

# Get coverage statistics
stats = taxonomy.get_coverage_stats()
```

## Limitations

1. Only training-time violations are currently tagged (calibration error, low accuracy)
2. Dimension scores are static mappings (not dynamically computed from model behavior)
3. Context adjustments are limited to predefined factors

## Future Enhancements

- Tag individual prediction violations during inference
- Dynamic dimension scoring based on data characteristics
- Integration with Phase 8-9 human feedback loops
- Automated recommendations based on ethical dimension scores
- Historical trending of ethical impacts across model versions
