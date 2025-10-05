# Training & Testing Scripts

This directory contains scripts for training and testing Nethical ML models according to the specifications in `TrainTestPipeline.md`.

## Directory Structure

The scripts create and use the following directory structure:

```
nethical/
  data/
    labeled_events/         # Training data and evaluation reports
      training_data.json    # Generated training dataset
      evaluation_report_*.json  # Test evaluation reports
  models/
    candidates/             # Candidate models (not yet promoted)
      model_*.json         # Model files with metadata
    current/                # Production-ready models
      model_*.json         # Promoted models
  scripts/
    train_model.py          # Training pipeline script
    test_model.py           # Testing/evaluation script
```

## Scripts

### train_model.py

Implements the complete training pipeline:

- **Data Generation**: Creates synthetic labeled training data
- **Temporal Split**: Splits data chronologically (80% train, 20% validation)
- **Model Training**: Trains a baseline heuristic classifier
- **Evaluation**: Computes precision, recall, F1, accuracy, and ECE
- **Promotion Gate**: Validates against promotion criteria:
  - Maximum ECE ≤ 0.08
  - Minimum accuracy ≥ 0.85
  - (With baseline: min recall gain ≥ 0.03, max FP increase ≤ 0.02)
- **Model Saving**: Saves to `models/current/` if passing gate, else `models/candidates/`

**Usage:**

```bash
# Train only
python scripts/train_model.py

# Train and test (complete workflow)
python scripts/train_model.py --run-all
```

**Options:**
- `--run-all`: Automatically run the testing pipeline after training completes

**Output:**
- Training data: `data/labeled_events/training_data.json`
- Model file: `models/candidates/model_YYYYMMDD_HHMMSS.json` (or `models/current/` if promoted)
- Console output with detailed metrics and promotion gate results
- (With `--run-all`) Evaluation report: `data/labeled_events/evaluation_report_YYYYMMDD_HHMMSS.json`

### test_model.py

Implements comprehensive model testing:

- **Model Loading**: Loads the latest trained model
- **Test Dataset**: Uses the last 20% of training data as test set
- **Evaluation**: Computes full metrics on test data
- **Reporting**: Generates detailed evaluation report with predictions
- **Baseline Comparison**: Compares ML model with rule-based baseline

**Usage:**

```bash
python scripts/test_model.py
```

**Output:**
- Evaluation report: `data/labeled_events/evaluation_report_YYYYMMDD_HHMMSS.json`
- Console output with test metrics and key findings

## Workflow

### Complete Training & Testing Workflow

#### Option 1: Single Command (Recommended)

Run the complete workflow with one command:
```bash
python scripts/train_model.py --run-all
```

This will:
- Generate synthetic training data
- Train a baseline model
- Validate against promotion criteria
- Save the model
- Automatically run comprehensive testing
- Generate an evaluation report

#### Option 2: Step-by-Step

1. **Train a model:**
   ```bash
   python scripts/train_model.py
   ```
   This will:
   - Generate synthetic training data
   - Train a baseline model
   - Validate against promotion criteria
   - Save the model

2. **Test the model:**
   ```bash
   python scripts/test_model.py
   ```
   This will:
   - Load the trained model
   - Run comprehensive evaluation
   - Generate an evaluation report

3. **Review results:**
   - Check console output for metrics
   - Review `data/labeled_events/evaluation_report_*.json` for detailed results
   - Check model files in `models/candidates/` or `models/current/`

## Metrics Explained

### Classification Metrics

- **Precision**: Of all predicted violations, what percentage were actual violations?
- **Recall**: Of all actual violations, what percentage did we detect?
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness of predictions

### Calibration Metrics

- **Expected Calibration Error (ECE)**: Measures how well the model's confidence scores match actual accuracy
  - Lower is better (target: ≤ 0.08)
  - Indicates if the model's confidence is trustworthy

### Promotion Gate Criteria

From `TrainTestPipeline.md` Section 14:

```yaml
promotion_gate:
  min_recall_gain: 0.03          # +3% absolute improvement
  max_fp_increase: 0.02          # +2% absolute increase allowed
  max_latency_increase_ms: 5
  max_ece: 0.08                  # Maximum calibration error
  min_human_agreement: 0.85      # Minimum accuracy
```

A model must meet all criteria to be promoted to production (`models/current/`).

## Next Steps

After running training and testing:

1. Review the metrics to ensure model quality
2. If promotion gate is passed, model is ready for shadow mode deployment
3. If not passed, adjust training parameters and retrain
4. Compare multiple models using the evaluation reports
5. Deploy to Phase 6 (ML Assisted Enforcement) once validated

## Integration with Existing Code

These scripts integrate with:

- `nethical/core/ml_shadow.py`: Uses MLShadowClassifier for predictions
- Phase 5 shadow mode: Can use saved models for passive inference
- Phase 6 blended enforcement: Promoted models can be used for assisted decisions

## References

- `TrainTestPipeline.md`: Full specification of training/testing pipeline
- `PHASE5-7_GUIDE.md`: ML and anomaly detection phases
- `examples/phase5_demo.py`: Shadow mode demonstration
