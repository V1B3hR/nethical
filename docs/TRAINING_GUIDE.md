# End-to-End Real Data Training - Quick Reference

## Overview

The end-to-end training pipeline downloads, processes, and trains the BaselineMLClassifier using real-world security datasets from Kaggle.

## Quick Start

```bash
# Full pipeline
python scripts/baseline_orchestrator.py

# Or step-by-step
python scripts/baseline_orchestrator.py --download      # Download datasets
python scripts/baseline_orchestrator.py --process-only  # Process CSV files
python scripts/baseline_orchestrator.py --train-only    # Train model
```

## Dataset Sources

All datasets are listed in `datasets/datasets`. See README.md for the complete list.

## Next Steps

1. Review validation metrics
2. Test model with scripts/test_model.py
3. Deploy to Phase 5 shadow mode
4. Monitor performance with Phase 7 anomaly detection

See scripts/README.md and README.md for detailed documentation.
