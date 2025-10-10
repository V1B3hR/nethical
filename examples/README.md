# Nethical Examples

This directory contains comprehensive examples demonstrating Nethical's features and capabilities. Examples are organized by category for easy navigation.

## Quick Start

For first-time users, start with the basic examples:

```bash
# Run the basic usage demo
python basic/basic_usage.py

# Run the unified governance demo
python basic/unified_governance_demo.py
```

## Directory Structure

### 📚 basic/
Fundamental examples for getting started with Nethical.

- **basic_usage.py** - Introduction to core Nethical features and API
- **unified_governance_demo.py** - Demonstrates the IntegratedGovernance interface
- **custom_detectors.py** - How to create and use custom safety detectors

**Start here if you're new to Nethical!**

### 🛡️ governance/
Examples showcasing governance features across different phases.

Phase-specific demonstrations:
- **phase3_demo.py** - Risk scoring, correlation detection, fairness sampling, drift reporting, performance optimization
- **phase4_demo.py** - Merkle anchoring, policy diff auditing, quarantine mode, ethical taxonomy, SLA monitoring
- **phase5_demo.py** - ML shadow mode and baseline classifier
- **phase6_demo.py** - ML assisted enforcement and blended risk engine
- **phase7_demo.py** - Anomaly detection and drift tracking
- **phase567_demo.py** - Integrated ML and anomaly detection features
- **phase89_demo.py** - Human-in-the-loop, escalation queue, continuous optimization

**Use these to understand specific governance capabilities.**

### 🎓 training/
Examples for training and managing ML models with governance.

- **demo_governance_training.py** - Training models with safety validation
- **train_anomaly_detector.py** - Training anomaly detection models
- **train_with_drift_tracking.py** - Training with drift monitoring
- **real_data_training_demo.py** - End-to-end training with real datasets
- **correlation_model_demo.py** - Training correlation detection models

**Use these to learn about ML integration and model training.**

### 🚀 advanced/
Advanced features for enterprise deployments.

Future Track demonstrations:
- **regional_deployment_demo.py** - F1: Multi-region deployment and sharding
- **f2_extensibility_demo.py** - F2: Custom plugins and extensions
- **f3_privacy_demo.py** - F3: Differential privacy and data protection
- **f4_adaptive_tuning_demo.py** - F4: Adaptive threshold tuning
- **f5_simulation_replay_demo.py** - F5: Action replay and simulation
- **f6_marketplace_demo.py** - F6: Plugin marketplace and ecosystem

**Use these for enterprise-scale deployments and advanced features.**

## Running Examples

### Prerequisites

Ensure you have installed Nethical and its dependencies:

```bash
pip install -r requirements.txt
```

### Running a Specific Example

```bash
# From repository root
python examples/basic/basic_usage.py

# Or navigate to examples directory
cd examples
python basic/basic_usage.py
```

### Running with Different Configurations

Many examples accept command-line arguments:

```bash
# Training examples often support model type selection
python training/train_anomaly_detector.py --model-type isolation_forest

# Some demos support verbose mode
python governance/phase3_demo.py --verbose
```

## Example Categories Explained

### Basic Examples
Perfect for beginners. These demonstrate:
- Core API usage
- Basic configuration
- Simple governance workflows
- Custom detector creation

### Governance Examples
Phase-by-phase feature demonstrations:
- **Phase 3**: Risk assessment and optimization
- **Phase 4**: Integrity and audit trails
- **Phase 5-7**: ML integration and anomaly detection
- **Phase 8-9**: Human feedback and optimization

### Training Examples
End-to-end ML workflows including:
- Model training with governance validation
- Real-world dataset processing
- Drift tracking during training
- Anomaly detector training
- Model evaluation and metrics

### Advanced Examples
Enterprise features for scale:
- Multi-region deployment
- Plugin development and marketplace
- Privacy-preserving ML
- Adaptive tuning systems
- Action replay and testing

## Best Practices

1. **Start Simple**: Begin with `basic/basic_usage.py` to understand core concepts
2. **Explore Phases**: Work through phase demos sequentially to understand feature progression
3. **Try Training**: Use training examples to integrate ML models
4. **Scale Up**: Explore advanced examples for production deployments

## Example Output

Most examples include:
- ✅ Success indicators
- 📊 Metrics and statistics
- 📝 Detailed logging
- 🎯 Demonstration results
- ⚠️ Warning messages when applicable

## Getting Help

- **Documentation**: See [docs/](../docs/) for comprehensive guides
- **API Reference**: Check module docstrings for detailed API documentation
- **Issues**: Report problems at https://github.com/V1B3hR/nethical/issues
- **Roadmap**: See [roadmap.md](../roadmap.md) for planned features

## Contributing Examples

We welcome example contributions! If you have a use case or feature demonstration:

1. Follow the existing structure and style
2. Include clear docstrings and comments
3. Add README updates describing your example
4. Test thoroughly before submitting
5. Submit a pull request

## Related Documentation

- [README.md](../README.md) - Project overview
- [TRAINING_GUIDE.md](../docs/TRAINING_GUIDE.md) - ML training documentation
- [AUDIT_LOGGING_GUIDE.md](../docs/AUDIT_LOGGING_GUIDE.md) - Audit trail guide
- [roadmap.md](../roadmap.md) - Development roadmap

## License

All examples are part of the Nethical project and are licensed under GNU General Public License v3.0.
