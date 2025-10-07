# Risk Tracking Integration - Quick Reference

## What is Risk Tracking?

The Risk Engine tracks risk scores during model training based on performance metrics, calibration errors, and promotion gate results. It provides real-time risk assessment and tier classification (LOW, NORMAL, HIGH, ELEVATED).

## Quick Start

### Basic Usage
```bash
python training/train_any_model.py --model-type heuristic --enable-risk-tracking
```

### With Custom Decay
```bash
python training/train_any_model.py --model-type logistic --enable-risk-tracking --risk-decay-hours 12.0
```

### With All Features (Risk + Audit + Drift)
```bash
python training/train_any_model.py --model-type logistic \
  --enable-risk-tracking \
  --enable-audit \
  --enable-drift-tracking
```

## What Gets Tracked?

### Validation Metrics Risk
✅ Calculated from calibration error (ECE) and accuracy  
✅ High ECE or low accuracy increases risk score  
✅ Risk tier assigned automatically (LOW, NORMAL, HIGH, ELEVATED)  

### Promotion Gate Failures
✅ Promotion gate failures treated as high-severity events  
✅ Risk score increases significantly on promotion failure  
✅ Helps identify problematic models early  

### Risk Components
✅ **Behavior Score**: Violation rate over total actions  
✅ **Severity Score**: Severity of violations  
✅ **Frequency Score**: Recent violation frequency  
✅ **Recency Score**: Time since last violation  

## Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-risk-tracking` | Enable risk tracking | Disabled |
| `--risk-decay-hours` | Risk score decay half-life in hours | 24.0 |

## Risk Tiers

| Tier | Score Range | Description |
|------|-------------|-------------|
| LOW | 0.0 - 0.25 | Normal operation, good metrics |
| NORMAL | 0.25 - 0.50 | Acceptable but monitor |
| HIGH | 0.50 - 0.75 | Concerning, needs attention |
| ELEVATED | 0.75+ | Critical, trigger advanced detectors |

## Use Cases

✅ **Quality Control**: Early detection of poor-performing models  
✅ **Risk Management**: Track training risk over time  
✅ **Compliance**: Document risk profiles for audits  
✅ **Optimization**: Identify models needing retraining  

## Integration with Other Features

| Feature | Combined Usage |
|---------|----------------|
| Audit Logging | `--enable-risk-tracking --enable-audit` |
| Drift Tracking | `--enable-risk-tracking --enable-drift-tracking` |
| Full Pipeline | `--enable-risk-tracking --enable-audit --enable-drift-tracking` |

## Output

### Risk Tracking Summary (during training)
```
[INFO] Risk Tracking:
  Agent ID: model_logistic
  Risk Score: 0.2240
  Risk Tier: LOW
  Violation Severity: 0.4000
```

### Promotion Gate Risk Update (if promotion fails)
```
[INFO] Promotion Gate Risk Update:
  Updated Risk Score: 0.5852
  Updated Risk Tier: HIGH
```

### Final Risk Profile Summary
```
======================================================================
RISK PROFILE SUMMARY
======================================================================
Agent ID: model_heuristic
Final Risk Score: 0.5852
Final Risk Tier: HIGH
Total Actions: 2
Violation Count: 2
Last Update: 2025-10-07T12:18:12.618245

Risk Components:
  Behavior Score: 1.0000
  Severity Score: 0.8000
  Frequency Score: 0.0000
  Recency Score: 1.0000

Tier Transition History:
  2025-10-07T12:18:12.618249: HIGH
```

## Performance Impact

- ⚡ Minimal - only tracks metrics already computed
- 🔄 Non-blocking - doesn't affect training
- 📊 Efficient - in-memory tracking with optional Redis persistence

## Risk Score Calculation

Risk score combines multiple factors:

1. **Calibration Error (ECE)**: Normalized to [0,1], severe if > 0.25
2. **Accuracy**: Low accuracy (< 0.85) increases risk
3. **Promotion Failure**: High severity (0.8) for gate failures
4. **Multi-factor Fusion**: 
   - 30% Behavior Score
   - 30% Severity Score
   - 20% Frequency Score
   - 20% Recency Score

## Advanced Usage

### Risk Decay
Risk scores decay exponentially over time using the formula:
```
decayed_score = current_score * e^(-λt)
where λ = ln(2) / half_life
```

Configure decay with `--risk-decay-hours`:
- Shorter half-life (e.g., 6 hours): Faster decay, more responsive
- Longer half-life (e.g., 48 hours): Slower decay, more persistent

### Elevated Tier Detection
When risk tier reaches ELEVATED (≥0.75), the system signals that advanced detectors should be invoked for enhanced scrutiny:
```
⚠️  ELEVATED risk tier - advanced detectors should be invoked
```

## Best Practices

1. **Monitor Trends**: Track risk scores across multiple training runs
2. **Set Alerts**: Watch for ELEVATED tier transitions
3. **Combine Features**: Use with audit and drift tracking for complete visibility
4. **Adjust Decay**: Tune decay half-life based on your training frequency
5. **Review Profiles**: Examine risk profiles for models with high violations

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Risk tracking not enabled | Check that `--enable-risk-tracking` flag is used |
| No risk profile shown | Verify RiskEngine module is available |
| Unexpected risk tier | Review violation severity calculation in metrics |
| Risk score too sensitive | Increase `--risk-decay-hours` for slower decay |

## Integration with Audit Trail

When both risk tracking and audit logging are enabled, the final risk profile is logged to the Merkle audit trail as a `risk_profile_final` event, providing a permanent, tamper-evident record of the training risk assessment.
