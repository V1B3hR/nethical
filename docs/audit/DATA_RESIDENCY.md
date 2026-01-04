# Data Residency Management Guide

**Version**: 1.0.0  
**Last Updated**: 2025-12-02  
**Status**: Active

---

## Overview

Nethical's Data Residency Management system ensures that data stays in required jurisdictions according to regulatory requirements. This is critical for compliance with:

- **GDPR** - EU data must remain within the European Economic Area (EEA)
- **CCPA** - California consumer data protection
- **Data Sovereignty Laws** - Various national data localization requirements

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Data Residency Manager                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ EU Region │   │ US Region │   │AP Region  │
            │           │   │           │   │           │
            │ • eu-west │   │ • us-east │   │ • ap-south│
            │ • eu-cent │   │ • us-west │   │ • ap-ne   │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────────────────────────────────────┐
            │         Data Classification Engine         │
            │   PII │ Decisions │ Audit │ Policies       │
            └───────────────────────────────────────────┘
```

## Supported Regions

| Region Code | Location | Jurisdiction |
|-------------|----------|--------------|
| `eu-west-1` | Ireland | EU |
| `eu-west-2` | London | UK |
| `eu-central-1` | Frankfurt | EU |
| `us-east-1` | N. Virginia | US |
| `us-west-1` | N. California | CA (CCPA) |
| `us-west-2` | Oregon | US |
| `ap-south-1` | Mumbai | APAC |
| `ap-northeast-1` | Tokyo | APAC |
| `ap-southeast-1` | Singapore | APAC |
| `global` | Replicated | Global |

## Data Classifications

| Classification | Description | Default Policy |
|----------------|-------------|----------------|
| `public` | No restrictions | Any region |
| `internal` | Internal use only | Any region |
| `confidential` | Restricted access | Regional only |
| `pii` | Personal Identifiable Information | Jurisdiction-specific |
| `sensitive_pii` | Special category data | Strict jurisdiction |
| `restricted` | Highest classification | Origin region only |

## Default Policies

### PII Data

Personal data is subject to strict residency controls:

```yaml
pii:
  allowed_regions:
    - eu-west-1
    - eu-west-2  
    - eu-central-1
  required_jurisdiction: eu
  processing_restriction: region-only
  cross_region_transfer_allowed: false
  requires_encryption: true
  retention_days: 2555  # 7 years
```

### Decision Data

AI decisions are stored in the region where they were made:

```yaml
decisions:
  allowed_regions:
    - eu-west-1
    - eu-central-1
    - us-east-1
  required_jurisdiction: null  # Based on origin
  processing_restriction: region-only
  cross_region_transfer_allowed: false
  requires_encryption: true
  retention_days: 365
```

### Audit Logs

Audit logs remain in their originating region:

```yaml
audit_logs:
  allowed_regions:
    - eu-west-1
    - eu-central-1
    - us-east-1
  required_jurisdiction: null
  processing_restriction: region-only
  cross_region_transfer_allowed: false
  requires_encryption: true
  retention_days: 2555  # 7 years
```

### Policies and Models

Non-personal data can be replicated globally:

```yaml
policies:
  allowed_regions:
    - global
  required_jurisdiction: null
  processing_restriction: any
  cross_region_transfer_allowed: true
  requires_encryption: false
  retention_days: -1  # No expiration
```

## Usage

### Basic Usage

```python
from nethical.compliance import (
    DataResidencyManager,
    DataType,
    DataRegion,
    DataClassification,
)

# Initialize manager
manager = DataResidencyManager()

# Classify data
data_type, classification = manager.classify_data(
    content={"email": "user@example.com"},
    metadata={"source": "user_registration"}
)
# Returns: (DataType.PII, DataClassification.PII)

# Validate storage location
is_valid, violation = manager.validate_storage_location(
    data_type=DataType.PII,
    target_region=DataRegion.EU_WEST_1,
    data_classification=DataClassification.PII,
)
# Returns: (True, None) for EU region
```

### Cross-Region Transfer Validation

```python
# Check if transfer is allowed
is_valid, violation = manager.validate_cross_region_transfer(
    data_type=DataType.PII,
    source_region=DataRegion.EU_WEST_1,
    target_region=DataRegion.US_EAST_1,
)

if not is_valid:
    print(f"Transfer blocked: {violation.description}")
    # Transfer blocked: Cross-region transfer of 'pii' from 'eu-west-1' 
    # to 'us-east-1' is not allowed
```

### Recording Data Movement

```python
# Record data movement for audit trail
record = manager.record_data_movement(
    data_id="user-123",
    data_type=DataType.PII,
    classification=DataClassification.PII,
    source_region=DataRegion.EU_WEST_1,
    target_region=DataRegion.EU_CENTRAL_1,
    movement_type="copy",
    reason="Disaster recovery backup",
    authorized_by="admin@company.com",
)
```

### Custom Policies

```python
from nethical.compliance import ResidencyPolicy

# Create custom policy
custom_policy = ResidencyPolicy(
    data_type=DataType.GENERAL,
    allowed_regions={DataRegion.US_EAST_1, DataRegion.US_WEST_2},
    required_jurisdiction=None,
    processing_restriction="region-only",
    cross_region_transfer_allowed=False,
    requires_encryption=True,
    retention_days=180,
)

# Update manager with custom policy
manager.update_policy(custom_policy)
```

## API Endpoints

### Data Classification

```http
POST /v2/data/classify
Content-Type: application/json

{
  "content": {"field": "value"},
  "metadata": {"source": "api"}
}
```

### Residency Validation

```http
POST /v2/data/validate-residency
Content-Type: application/json

{
  "data_type": "pii",
  "target_region": "eu-west-1"
}
```

### Transfer Validation

```http
POST /v2/data/validate-transfer
Content-Type: application/json

{
  "data_type": "decisions",
  "source_region": "eu-west-1",
  "target_region": "us-east-1"
}
```

## Enforcement

### Automatic Blocking

The Data Residency Manager automatically blocks non-compliant operations:

1. **Storage Validation** - Data cannot be stored in unauthorized regions
2. **Transfer Blocking** - Cross-region transfers are blocked for restricted data
3. **Violation Logging** - All violations are logged for audit purposes

### Audit Trail

All data movements are recorded with:

- Record ID
- Data ID
- Data type and classification
- Source and target regions
- Movement type (copy, move, process)
- Authorization status
- Timestamp
- Authorizer (if applicable)

## Monitoring

### Violations Summary

```python
summary = manager.get_violations_summary()
# {
#     "total_violations": 5,
#     "blocked": 4,
#     "by_type": {"cross_region_blocked": 3, "invalid_region": 2},
#     "by_severity": {"high": 3, "critical": 2},
#     "last_violation": "2025-12-02T10:30:00Z"
# }
```

### Audit Trail Query

```python
records = manager.get_movement_audit_trail(
    data_type=DataType.PII,
    start_time=datetime(2025, 12, 1),
    end_time=datetime.now(),
)
```

## Integration with Fundamental Laws

The Data Residency Management system adheres to:

| Law | Implementation |
|-----|----------------|
| **Law 15: Audit Compliance** | Full audit trail of all data movements |
| **Law 22: Digital Security** | Encryption requirements and access controls |
| **Law 23: Fail-Safe Design** | Automatic blocking of non-compliant operations |

## Best Practices

1. **Default Deny** - Restrict by default, allow explicitly
2. **Classify Early** - Classify data at ingestion time
3. **Encrypt Sensitive Data** - Always encrypt PII and sensitive data
4. **Regular Audits** - Review movement logs regularly
5. **Minimal Transfer** - Only transfer data when necessary

## Troubleshooting

### Common Issues

**Issue**: Transfer blocked unexpectedly

```python
# Check the policy
policy = manager.get_policy(DataType.PII)
print(f"Allowed regions: {policy.allowed_regions}")
print(f"Cross-region allowed: {policy.cross_region_transfer_allowed}")
```

**Issue**: Classification incorrect

```python
# Provide explicit metadata
data_type, classification = manager.classify_data(
    content=data,
    metadata={
        "data_type": "general",  # Override classification
        "classification": "internal"
    }
)
```

## Related Documentation

- [GDPR Compliance Guide](./EU_AI_ACT_COMPLIANCE.md)
- [Security Hardening Guide](../SECURITY_HARDENING_GUIDE.md)
- [Audit Logging Guide](../AUDIT_LOGGING_GUIDE.md)

---

**Document Maintainer**: Nethical Core Team  
**Review Cycle**: Quarterly
