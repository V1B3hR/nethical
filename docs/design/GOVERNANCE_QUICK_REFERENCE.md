# Governance Integration - Quick Reference

## What is Governance Integration?

The governance system validates training data and model predictions for safety and ethical violations during model training.

## Quick Start

### Basic Usage
```bash
python training/train_any_model.py --model-type heuristic --enable-governance
```

### With Audit Logging
```bash
python training/train_any_model.py --model-type logistic --enable-governance --enable-audit
```

## What Gets Validated?

### Training Data (First 100 Samples)
✅ Checked for safety violations before training  
✅ Harmful content, toxic language, malicious patterns  
✅ Problematic samples are flagged but training continues  

### Model Predictions (First 50 Predictions)
✅ Checked during validation phase  
✅ Ensures outputs meet safety standards  
✅ Flags predictions that would be blocked in production  

## Detected Violations (15+ Types)

| Category | Examples |
|----------|----------|
| **Ethical** | Harmful content, bias, discrimination |
| **Safety** | Dangerous commands, unsafe domains |
| **Manipulation** | Social engineering, phishing |
| **Privacy** | PII exposure (emails, SSN, credit cards) |
| **Security** | Prompt injection, adversarial attacks |
| **Content** | Toxic language, hate speech, misinformation |

## Output Example

```
[INFO] Governance validation enabled
[INFO] Running governance validation on training data samples...
[WARN] Governance found 5 problematic data samples
[INFO] Running governance validation on model predictions...
[INFO] Governance validation passed for 50 predictions

[INFO] Governance Validation Summary:
  Data samples validated: 100
  Data violations found: 5
  Predictions validated: 50
  Prediction violations found: 0
```

## Command-Line Options

| Flag | Description |
|------|-------------|
| `--enable-governance` | Enable governance validation |
| `--enable-audit` | Enable Merkle audit logging (optional) |
| `--enable-drift-tracking` | Enable drift tracking (optional) |

## Use Cases

✅ **Compliance**: Meet regulatory requirements for AI safety  
✅ **Quality Assurance**: Validate training data quality  
✅ **Risk Management**: Early detection of problematic data  
✅ **Audit Trail**: Complete documentation of safety checks  

## Integration with Other Features

| Feature | Combined Usage |
|---------|----------------|
| Audit Logging | `--enable-governance --enable-audit` |
| Drift Tracking | `--enable-governance --enable-drift-tracking` |
| Full Pipeline | `--enable-governance --enable-audit --enable-drift-tracking` |

## Performance Impact

- ⚡ Minimal - only validates sample of data
- 🔄 Non-blocking - errors don't stop training
- 📊 Efficient - async validation where possible

## Testing

```bash
# Run governance tests
python tests/test_train_governance.py

# Run demo
python examples/training/demo_governance_training.py
```

## Documentation

- 📖 `GOVERNANCE_TRAINING_IMPLEMENTATION.md` - Complete implementation details
- 📖 `docs/TRAINING_GUIDE.md` - Training guide with governance
- 📖 `training/README.md` - Full training documentation

## Key Benefits

🔒 **Safety-First**: Real-time validation during training  
📝 **Audit-Ready**: Complete governance metrics  
🎯 **Quality**: Ensures data and predictions meet standards  
💡 **Easy**: Just add `--enable-governance` flag  

## Support

For issues or questions:
1. Check `GOVERNANCE_TRAINING_IMPLEMENTATION.md` for details
2. Review test examples in `tests/test_train_governance.py`
3. Run demo: `python examples/training/demo_governance_training.py`

---

**TL;DR**: Add `--enable-governance` to your training command for real-time safety validation!
