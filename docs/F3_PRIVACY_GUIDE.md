# F3: Privacy & Data Handling Guide

## Overview

The F3 Privacy & Data Handling track provides comprehensive privacy protection features including:
- Enhanced PII detection and redaction (>95% accuracy)
- Differential privacy for model training and metric aggregation
- Federated analytics for cross-region data sharing
- Data minimization and GDPR/CCPA compliance

## Features

### 1. Enhanced Redaction Pipeline

The redaction pipeline provides automatic PII detection and context-aware redaction.

#### Basic Usage

```python
from nethical.core.redaction_pipeline import EnhancedRedactionPipeline, RedactionPolicy

# Initialize pipeline with aggressive policy
pipeline = EnhancedRedactionPipeline(
    policy=RedactionPolicy.AGGRESSIVE,
    enable_audit=True,
    enable_reversible=True
)

# Detect PII in text
text = "Contact me at john@example.com or (555) 123-4567"
pii_matches = pipeline.detect_pii(text)

# Redact PII
result = pipeline.redact(text, user_id="user123", preserve_utility=True)
print(result.redacted_text)
# Output: "Contact me at [REDACTED]@example.com or (555) XXX-XXXX"
```

#### Redaction Policies

- **MINIMAL**: Only redact critical PII (SSN, credit cards, passports)
- **STANDARD**: Redact common PII types (emails, phones, addresses)
- **AGGRESSIVE**: Redact all detected PII including names and dates

#### Reversible Redaction

For authorized users, redaction can be reversed:

```python
# Enable reversible redaction
pipeline = EnhancedRedactionPipeline(enable_reversible=True)

result = pipeline.redact(text)

# Restore for authorized user
restored = pipeline.restore(result, user_id="admin", authorized=True)
```

#### Audit Trail

All redaction operations are logged:

```python
# View audit trail
for entry in pipeline.audit_trail:
    print(f"{entry.timestamp}: {entry.action} by {entry.user_id}")
```

### 2. Differential Privacy

Implements (ε, δ)-differential privacy with privacy budget tracking.

#### Basic Usage

```python
from nethical.core.differential_privacy import DifferentialPrivacy

# Initialize with privacy budget
dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

# Add noise to a metric
original_value = 0.95
noised_value = dp.add_noise(original_value, sensitivity=0.1)

# Check privacy budget
budget_status = dp.get_privacy_budget_status()
print(f"Remaining budget: {budget_status['epsilon_remaining']}")
```

#### Aggregated Metrics

Add privacy-preserving noise to multiple metrics:

```python
metrics = {
    'accuracy': 0.94,
    'precision': 0.92,
    'recall': 0.90
}

noised_metrics = dp.add_noise_to_aggregated_metrics(
    metrics,
    sensitivity=0.1,
    noise_level=0.1
)
```

#### DP-SGD for Model Training

```python
from nethical.core.differential_privacy import DPTrainingConfig

config = DPTrainingConfig(
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=1.1,
    batch_size=32
)

# Apply DP-SGD to gradients
privatized_gradients = dp.dp_sgd_step(gradients, config, batch_size=32)
```

#### Privacy Compliance

```python
from nethical.core.differential_privacy import PrivacyAudit

audit = PrivacyAudit(dp)
compliance = audit.validate_compliance(['GDPR', 'CCPA'])

print(f"GDPR compliant: {compliance['checks']['GDPR']['compliant']}")
```

### 3. Federated Analytics

Enable cross-region metric aggregation without sharing raw data.

#### Setup

```python
from nethical.core.federated_analytics import FederatedAnalytics

# Initialize for multiple regions
regions = ["us-east-1", "eu-west-1", "ap-south-1"]
fa = FederatedAnalytics(
    regions=regions,
    privacy_preserving=True,
    enable_encryption=True,
    noise_level=0.1
)
```

#### Register Regional Metrics

```python
# Each region registers its own metrics
fa.register_regional_metrics(
    region_id="us-east-1",
    metrics={'accuracy': 0.95, 'latency': 120},
    sample_size=1000
)

fa.register_regional_metrics(
    region_id="eu-west-1",
    metrics={'accuracy': 0.93, 'latency': 140},
    sample_size=800
)
```

#### Compute Global Aggregation

```python
# Privacy-preserving aggregation
aggregated = fa.compute_metrics(privacy_preserving=True)

print(f"Global metrics: {aggregated.aggregated_values}")
print(f"Total samples: {aggregated.total_samples}")
```

#### Privacy-Preserving Correlation

```python
# Detect correlations across regions
correlation = fa.privacy_preserving_correlation(
    'accuracy',
    'latency',
    noise_level=0.1
)

print(f"Correlation: {correlation.correlation:.3f}")
print(f"P-value: {correlation.p_value:.4f}")
```

### 4. Data Minimization

Implement automatic data retention and right-to-be-forgotten.

#### Basic Usage

```python
from nethical.core.data_minimization import (
    DataMinimization,
    DataCategory
)

# Initialize
dm = DataMinimization(
    storage_dir="./data",
    enable_auto_deletion=True,
    anonymization_enabled=True
)
```

#### Store Minimal Data

```python
# Store only necessary fields
record = dm.store_data(
    data={'email': 'user@example.com', 'name': 'John'},
    category=DataCategory.PERSONAL_IDENTIFIABLE,
    user_id='user123',
    minimal_fields_only=True
)

print(f"Expires: {record.expires_at}")
```

#### Anonymization

```python
# Anonymize sensitive data
anonymized = dm.anonymize_data(
    record.record_id,
    anonymization_level="standard"
)
```

#### Right-to-be-Forgotten

```python
# Process deletion request
deletion_request = dm.request_data_deletion(
    user_id='user123',
    categories=[DataCategory.PERSONAL_IDENTIFIABLE]
)

print(f"Status: {deletion_request.status}")
print(f"Deleted: {deletion_request.records_deleted} records")
```

#### Retention Policies

Default retention periods by category:
- **PERSONAL_IDENTIFIABLE**: 30 days
- **SENSITIVE**: 30 days
- **OPERATIONAL**: 90 days
- **AUDIT**: 365 days

```python
# Process expired records
result = dm.process_retention_policy()
print(f"Deleted: {result['deleted']}, Anonymized: {result['anonymized']}")
```

### 5. Integrated Governance

Use all F3 features through the unified governance interface.

#### Initialize with Privacy Features

```python
from nethical.core.integrated_governance import IntegratedGovernance

# Enable privacy features
gov = IntegratedGovernance(
    storage_dir="./governance",
    region_id="us-east-1",
    privacy_mode="differential",
    epsilon=1.0,
    redaction_policy="aggressive"
)
```

#### Access Privacy Components

```python
# Use redaction pipeline
result = gov.redaction_pipeline.redact("Email: test@example.com")

# Check differential privacy budget
if gov.differential_privacy:
    budget = gov.differential_privacy.get_privacy_budget_status()
    print(f"Privacy budget remaining: {budget['epsilon_remaining']}")

# Use federated analytics
if gov.federated_analytics:
    fa_stats = gov.federated_analytics.get_statistics()
    print(f"Regions: {fa_stats['regions']}")

# Access data minimization
dm_stats = gov.data_minimization.get_statistics()
print(f"Active records: {dm_stats['active_records']}")
```

## Best Practices

### 1. Privacy Budget Management

- Start with lower epsilon values (0.5-1.0) for stronger privacy
- Monitor budget consumption regularly
- Reserve budget for critical operations
- Use privacy-utility tradeoff optimization

### 2. Redaction Guidelines

- Use AGGRESSIVE policy for highly sensitive data
- Enable audit trails for compliance
- Preserve utility when possible for analytics
- Use reversible redaction only when necessary

### 3. Federated Analytics

- Ensure minimum sample sizes per region (≥10)
- Use privacy-preserving mode for sensitive metrics
- Validate privacy guarantees regularly
- Encrypt metrics in transit

### 4. Data Minimization

- Store only essential data fields
- Set appropriate retention periods
- Process deletion requests promptly
- Anonymize before deletion when possible

## Compliance

### GDPR Compliance

F3 features support GDPR requirements:
- **Article 25**: Data protection by design (differential privacy)
- **Article 30**: Records of processing (audit trails)
- **Article 17**: Right to erasure (right-to-be-forgotten)
- **Article 32**: Security of processing (encryption, anonymization)

### CCPA Compliance

F3 features support CCPA requirements:
- **Section 1798.100**: Consumer rights (data access)
- **Section 1798.105**: Right to deletion
- **Section 1798.150**: Security requirements

## Performance Considerations

- PII detection: ~100-500 ms per page of text
- Differential privacy noise addition: ~1-5 ms per metric
- Federated aggregation: ~10-50 ms per region
- Data minimization operations: ~1-10 ms per record

## Examples

See `examples/advanced/f3_privacy_demo.py` for comprehensive demonstrations of all features.

## Testing

Run F3 tests:

```bash
pytest tests/test_privacy_features.py -v
```

## References

- [Differential Privacy](https://en.wikipedia.org/wiki/Differential_privacy)
- [GDPR](https://gdpr.eu/)
- [CCPA](https://oag.ca.gov/privacy/ccpa)
- [PII Detection Best Practices](https://www.nist.gov/privacy-framework)
