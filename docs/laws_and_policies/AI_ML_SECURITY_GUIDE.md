# AI/ML Security Guide

## Overview

The AI/ML Security Framework provides comprehensive protection for machine learning systems against adversarial attacks, data poisoning, and privacy violations. This guide covers implementation, best practices, and compliance requirements for military, government, and healthcare deployments.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Quick Start](#quick-start)
4. [Adversarial Defense](#adversarial-defense)
5. [Model Poisoning Detection](#model-poisoning-detection)
6. [Differential Privacy](#differential-privacy)
7. [Federated Learning](#federated-learning)
8. [Explainable AI](#explainable-ai)
9. [Best Practices](#best-practices)
10. [Compliance](#compliance)

## Architecture

The AI/ML Security Framework consists of five integrated components:

```
┌─────────────────────────────────────────────────────────┐
│              AIMLSecurityManager                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────┐      │
│  │  Adversarial     │  │  Model Poisoning     │      │
│  │  Defense         │  │  Detector            │      │
│  └──────────────────┘  └──────────────────────┘      │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────┐      │
│  │  Differential    │  │  Federated Learning  │      │
│  │  Privacy         │  │  Coordinator         │      │
│  └──────────────────┘  └──────────────────────┘      │
│                                                         │
│  ┌──────────────────────────────────────────┐         │
│  │  Explainable AI System                   │         │
│  └──────────────────────────────────────────┘         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Adversarial Defense System

Detects and mitigates adversarial examples using:
- Input perturbation analysis
- Prediction consistency checking
- Feature space anomaly detection
- Ensemble disagreement detection

**Supported Attack Types:**
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- DeepFool
- Carlini-Wagner
- Membership Inference
- Model Inversion
- Backdoor attacks

### 2. Model Poisoning Detector

Identifies poisoned training data through:
- Gradient analysis and anomaly detection
- Loss pattern monitoring
- Activation clustering
- Federated learning validation

**Detected Poisoning Types:**
- Data poisoning
- Label flipping
- Backdoor injection
- Gradient manipulation
- Federated poisoning

### 3. Differential Privacy Manager

Provides privacy-preserving data analysis with:
- Epsilon-delta guarantees
- Laplace and Gaussian mechanisms
- Privacy budget tracking
- Query auditing

### 4. Federated Learning Coordinator

Enables secure distributed training with:
- Secure multi-party computation
- Byzantine-robust aggregation
- Privacy-preserving aggregation
- Participant validation

### 5. Explainable AI System

Generates compliance-ready explanations with:
- Feature importance analysis
- Human-readable explanations
- GDPR/HIPAA/DoD compliance
- Model transparency reports

## Quick Start

### Basic Setup

```python
from nethical.security.ai_ml_security import AIMLSecurityManager

# Initialize with all features enabled
manager = AIMLSecurityManager(
    enable_adversarial_defense=True,
    enable_poisoning_detection=True,
    enable_differential_privacy=True,
    enable_federated_learning=True,
    enable_explainable_ai=True,
    privacy_epsilon=1.0,
    privacy_delta=1e-5
)
```

### Selective Features

```python
# Enable only required features
manager = AIMLSecurityManager(
    enable_adversarial_defense=True,
    enable_differential_privacy=True,
    enable_explainable_ai=True
)
```

## Adversarial Defense

### Detecting Adversarial Examples

```python
from nethical.security.ai_ml_security import AdversarialDefenseSystem

# Initialize defense system
defense = AdversarialDefenseSystem(
    perturbation_threshold=0.1,
    confidence_threshold=0.8,
    enable_input_smoothing=True
)

# Define your model prediction function
def model_predict(input_data):
    # Your model prediction logic
    return model.predict(input_data)

# Detect adversarial examples
result = defense.detect_adversarial_example(
    input_data=suspicious_input,
    model_prediction_func=model_predict,
    baseline_input=original_input
)

if result.is_adversarial:
    print(f"⚠️  Adversarial attack detected!")
    print(f"Attack Type: {result.attack_type.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Perturbation Magnitude: {result.perturbation_magnitude:.4f}")
```

### Detection Statistics

```python
# Get detection statistics
stats = defense.get_detection_statistics()
print(f"Total detections: {stats['total_detections']}")
print(f"Adversarial rate: {stats['adversarial_rate']:.2%}")
```

### Best Practices

1. **Set Appropriate Thresholds**: Adjust `perturbation_threshold` based on your model's sensitivity
2. **Enable Input Smoothing**: Helps detect sophisticated attacks
3. **Monitor Detection Rates**: Track false positives and adjust thresholds
4. **Integrate with Monitoring**: Alert on detected adversarial inputs

## Model Poisoning Detection

### Detecting Training Data Poisoning

```python
from nethical.security.ai_ml_security import ModelPoisoningDetector

# Initialize detector
detector = ModelPoisoningDetector(
    gradient_threshold=2.0,
    loss_anomaly_threshold=3.0,
    enable_activation_analysis=True
)

# During training, check for poisoning
result = detector.detect_poisoning(
    training_batch=batch_data,
    gradients=computed_gradients,
    loss_values=batch_losses
)

if result.is_poisoned:
    print(f"⚠️  Model poisoning detected!")
    print(f"Poisoning Type: {result.poisoning_type.value}")
    print(f"Affected Samples: {result.affected_samples}")
    print(f"Gradient Anomaly Score: {result.gradient_anomaly_score:.2f}")
    
    # Take action: reject batch, alert, etc.
    reject_training_batch(batch_data)
```

### Continuous Monitoring

```python
# Monitor throughout training
for epoch in range(num_epochs):
    for batch in training_data:
        gradients = compute_gradients(batch)
        losses = compute_losses(batch)
        
        # Check for poisoning
        result = detector.detect_poisoning(
            training_batch=batch,
            gradients=gradients,
            loss_values=losses
        )
        
        if not result.is_poisoned:
            apply_gradients(gradients)
```

## Differential Privacy

### Privacy-Preserving Queries

```python
from nethical.security.ai_ml_security import (
    DifferentialPrivacyManager,
    PrivacyMechanism
)

# Initialize privacy manager
privacy_manager = DifferentialPrivacyManager(
    epsilon=1.0,        # Privacy budget
    delta=1e-5,         # Failure probability
    mechanism=PrivacyMechanism.LAPLACE
)

# Add noise to sensitive data
original_count = 1000
noised_count, success = privacy_manager.add_noise(
    data=original_count,
    sensitivity=1.0,
    epsilon_cost=0.1
)

if success:
    print(f"Original: {original_count}")
    print(f"Noised: {noised_count:.0f}")
    print(f"Privacy loss: {privacy_manager.get_privacy_loss()}")
```

### Privacy Budget Management

```python
# Check remaining budget
budget = privacy_manager.budget
print(f"Remaining epsilon: {budget.remaining_epsilon:.2f}")
print(f"Queries made: {budget.query_count}")

# Check if budget depleted
if budget.is_depleted:
    print("⚠️  Privacy budget depleted - no more queries allowed")
```

### Audit Trail

```python
# Export audit log for compliance
audit_log = privacy_manager.export_audit_log()
for entry in audit_log:
    print(f"Query {entry['query_id']}: ε={entry['epsilon_cost']}")
```

## Federated Learning

### Secure Model Aggregation

```python
from nethical.security.ai_ml_security import FederatedLearningCoordinator

# Initialize coordinator
coordinator = FederatedLearningCoordinator(
    min_participants=3,
    enable_secure_aggregation=True,
    enable_poisoning_detection=True
)

# Collect participant updates
participant_updates = [
    {'id': 1, 'gradients': participant1_grads},
    {'id': 2, 'gradients': participant2_grads},
    {'id': 3, 'gradients': participant3_grads}
]

# Aggregate with poisoning detection
round_result = coordinator.aggregate_updates(
    participant_updates=participant_updates,
    validation_data=validation_set
)

if round_result.poisoning_detected:
    print("⚠️  Malicious participant detected")
else:
    # Apply aggregated weights
    model.set_weights(round_result.aggregated_weights)
    print(f"Validation accuracy: {round_result.validation_accuracy:.2%}")
```

### Training Statistics

```python
# Monitor training progress
stats = coordinator.get_training_statistics()
print(f"Total rounds: {stats['total_rounds']}")
print(f"Average accuracy: {stats['average_accuracy']:.2%}")
print(f"Poisoning incidents: {stats['poisoning_incidents']}")
```

## Explainable AI

### Generating Compliance Explanations

```python
from nethical.security.ai_ml_security import ExplainableAISystem

# Initialize explainable AI system
explainer = ExplainableAISystem(
    compliance_frameworks=['GDPR', 'HIPAA', 'DoD_AI_Ethics'],
    explanation_method='feature_importance'
)

# Generate explanation for model prediction
input_features = {
    'age': 45,
    'medical_history_score': 8.5,
    'risk_factors': 3
}

report = explainer.generate_explanation(
    model_id='medical_risk_model_v2',
    input_features=input_features,
    prediction='high_risk',
    model_func=model.predict
)

# Display explanation
print(report.human_readable_explanation)
print(f"\nConfidence: {report.confidence_score:.2%}")
print("\nFeature Importance:")
for feature, importance in report.feature_importance.items():
    print(f"  {feature}: {importance:.2%}")
```

### Compliance Report

```python
# Export compliance report for audit
compliance_report = explainer.export_compliance_report()
print(f"Total explanations: {compliance_report['total_explanations']}")
print(f"Frameworks: {', '.join(compliance_report['compliance_frameworks'])}")
```

## Best Practices

### 1. Defense in Depth

Use multiple security layers:

```python
manager = AIMLSecurityManager(
    enable_adversarial_defense=True,
    enable_poisoning_detection=True,
    enable_differential_privacy=True
)

# Check for adversarial input
adv_result = manager.adversarial_defense.detect_adversarial_example(...)

# Add privacy noise
priv_data, _ = manager.privacy_manager.add_noise(...)

# Generate explanation for compliance
explanation = manager.explainable_ai.generate_explanation(...)
```

### 2. Monitoring and Alerting

```python
# Regular security status checks
status = manager.get_security_status()

if status['adversarial_defense']['statistics']['adversarial_rate'] > 0.05:
    alert_security_team("High adversarial attack rate detected")

if status['differential_privacy']['budget']['is_depleted']:
    alert_privacy_officer("Privacy budget depleted")
```

### 3. Regular Audits

```python
# Export comprehensive security report
report = manager.export_security_report()

# Save for compliance audit
with open('security_audit_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

### 4. Threshold Tuning

Start conservative and tune based on false positive rates:

```python
# Start with strict thresholds
defense = AdversarialDefenseSystem(
    perturbation_threshold=0.05,  # Strict
    confidence_threshold=0.9       # High confidence required
)

# Monitor and adjust
stats = defense.get_detection_statistics()
if stats['adversarial_rate'] > expected_rate:
    # Loosen thresholds if too many false positives
    defense.perturbation_threshold = 0.1
```

## Compliance

### GDPR (Article 22)

```python
# Generate GDPR-compliant explanations
explainer = ExplainableAISystem(
    compliance_frameworks=['GDPR']
)

# For automated decision-making
report = explainer.generate_explanation(
    model_id='credit_scoring_model',
    input_features=applicant_data,
    prediction=decision
)

# Report includes:
# - Feature importance (Article 22)
# - Human-readable explanation
# - Model transparency details
```

### HIPAA

```python
# Privacy-preserving medical data analysis
privacy_manager = DifferentialPrivacyManager(
    epsilon=0.5,  # Stricter for medical data
    delta=1e-6
)

# All queries add privacy noise
patient_count, _ = privacy_manager.add_noise(
    data=sensitive_count,
    sensitivity=1.0
)

# Audit trail for HIPAA compliance
audit_log = privacy_manager.export_audit_log()
```

### DoD AI Ethics Principles

```python
# Explainability for military AI systems
explainer = ExplainableAISystem(
    compliance_frameworks=['DoD_AI_Ethics']
)

# Ensure:
# 1. Responsible AI use
# 2. Equitable treatment
# 3. Traceable decisions
# 4. Reliable performance
# 5. Governable systems
```

### NIST AI RMF

```python
# Comprehensive AI security management
manager = AIMLSecurityManager(
    enable_adversarial_defense=True,  # Trustworthy
    enable_poisoning_detection=True,   # Secure
    enable_differential_privacy=True,  # Privacy-enhanced
    enable_explainable_ai=True         # Accountable
)

# Regular risk assessments
status = manager.get_security_status()
```

## Performance Considerations

### Adversarial Defense

- **Input Smoothing**: Adds ~10-20ms overhead per prediction
- **Perturbation Analysis**: Negligible for most inputs
- **Recommendation**: Use for high-stakes decisions only

### Differential Privacy

- **Noise Addition**: <1ms per query
- **Budget Tracking**: Negligible overhead
- **Recommendation**: Always enable for sensitive data

### Explainable AI

- **Feature Importance**: 50-200ms per explanation
- **SHAP/LIME Integration**: 200ms-2s depending on model
- **Recommendation**: Generate on-demand or cache

## Troubleshooting

### High False Positive Rate

```python
# Increase thresholds
defense.perturbation_threshold = 0.15  # Was 0.1
defense.confidence_threshold = 0.75     # Was 0.85
```

### Privacy Budget Depleted

```python
# Reset budget (use carefully)
privacy_manager.reset_budget()

# Or increase initial budget
privacy_manager = DifferentialPrivacyManager(epsilon=2.0)
```

### Poisoning Detection Too Sensitive

```python
# Increase thresholds
detector.gradient_threshold = 3.0    # Was 2.0
detector.loss_anomaly_threshold = 4.0  # Was 3.0
```

## References

1. **Adversarial ML**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
2. **Differential Privacy**: Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
3. **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks"
4. **Explainable AI**: Ribeiro et al., "Why Should I Trust You?"

## Support

For implementation questions or security concerns:
- Review [SECURITY.md](../../SECURITY.md)
- Open an issue on GitHub
- Contact security team for classified environments
