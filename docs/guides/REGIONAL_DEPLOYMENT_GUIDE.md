# Regional Deployment Guide

This guide demonstrates how to deploy and use Nethical's regionalization and sharding features (F1) for multi-region, geographically-distributed AI governance deployments.

## Overview

The regionalization feature enables geographic distribution and hierarchical data organization for multi-region deployments, providing:

- **Geographic Distribution**: Region-aware data storage and processing
- **Logical Sharding**: Hierarchical aggregation by department/team/project
- **Regional Policy Variation**: Region-specific compliance requirements (GDPR, CCPA, AI_ACT)
- **Data Residency**: Regional data storage compliance and cross-border transfer controls

## Key Features

- **Region-aware Processing**: All entities (actions, violations, judgments) support `region_id` field
- **Domain-based Organization**: `logical_domain` field for hierarchical aggregation
- **Compliance Profiles**: Pre-configured policies for GDPR, CCPA, and AI_ACT
- **Data Residency Validation**: Automatic validation of cross-border data transfers
- **Cross-region Reporting**: Aggregate metrics across regions and domains
- **Performance**: Tested with 5+ regions for production-scale deployments

## Quick Start

### Basic Regional Setup

```python
from nethical.core import IntegratedGovernance

# Create a regional governance instance for EU
governance = IntegratedGovernance(
    storage_dir="./nethical_eu_data",
    region_id="eu-west-1",
    logical_domain="customer-service",
    data_residency_policy="EU_GDPR"
)

# Process an action with regional context
result = governance.process_action(
    agent_id="agent_123",
    action="Customer inquiry about data retention",
    region_id="eu-west-1",
    compliance_requirements=["GDPR", "data_protection"]
)

print(f"Region: {result['region_id']}")
print(f"Compliant: {result['data_residency']['compliant']}")
```

### Multi-Region Deployment

```python
# US Region with CCPA compliance
us_governance = IntegratedGovernance(
    storage_dir="./nethical_us_data",
    region_id="us-west-2",
    logical_domain="sales",
    data_residency_policy="US_CCPA"
)

# EU Region with GDPR compliance
eu_governance = IntegratedGovernance(
    storage_dir="./nethical_eu_data",
    region_id="eu-central-1",
    logical_domain="operations",
    data_residency_policy="EU_GDPR"
)

# Asia-Pacific Region with AI Act compliance
ap_governance = IntegratedGovernance(
    storage_dir="./nethical_ap_data",
    region_id="ap-south-1",
    logical_domain="analytics",
    data_residency_policy="AI_ACT"
)
```

## Regional Policy Profiles

### EU GDPR Profile

```python
governance = IntegratedGovernance(
    region_id="eu-west-1",
    data_residency_policy="EU_GDPR"
)

# GDPR profile includes:
# - compliance_requirements: ['GDPR', 'data_protection', 'right_to_erasure']
# - data_retention_days: 30
# - cross_border_transfer_allowed: False
# - encryption_required: True
# - audit_trail_required: True
# - consent_required: True
```

### US CCPA Profile

```python
governance = IntegratedGovernance(
    region_id="us-west-2",
    data_residency_policy="US_CCPA"
)

# CCPA profile includes:
# - compliance_requirements: ['CCPA', 'consumer_privacy']
# - data_retention_days: 90
# - cross_border_transfer_allowed: True
# - encryption_required: True
# - audit_trail_required: True
# - consent_required: False
```

### AI Act Profile

```python
governance = IntegratedGovernance(
    region_id="eu-central-1",
    data_residency_policy="AI_ACT"
)

# AI Act profile includes:
# - compliance_requirements: ['AI_ACT', 'high_risk_ai', 'transparency']
# - data_retention_days: 180
# - cross_border_transfer_allowed: True
# - encryption_required: True
# - audit_trail_required: True
# - human_oversight_required: True
```

## Data Residency Validation

### Same-Region Validation

```python
# Process action in the same region
result = governance.process_action(
    agent_id="agent_123",
    action="Regional query",
    region_id="eu-west-1"  # Same as governance.region_id
)

# Result includes validation status
assert result['data_residency']['compliant'] == True
assert len(result['data_residency']['violations']) == 0
```

### Cross-Border Transfer Validation

```python
# GDPR prevents cross-border transfers
eu_governance = IntegratedGovernance(
    region_id="eu-west-1",
    data_residency_policy="EU_GDPR"
)

result = eu_governance.process_action(
    agent_id="agent_456",
    action="Cross-border query",
    region_id="us-east-1"  # Different region
)

# GDPR blocks cross-border transfer
assert result['data_residency']['compliant'] == False
assert 'Cross-border data transfer' in result['data_residency']['violations'][0]

# CCPA allows cross-border transfers
us_governance = IntegratedGovernance(
    region_id="us-west-2",
    data_residency_policy="US_CCPA"
)

result = us_governance.process_action(
    agent_id="agent_789",
    action="Cross-border query",
    region_id="us-east-1"
)

# CCPA allows transfer
assert result['data_residency']['compliant'] == True
```

## Logical Domain Sharding

### Department-based Isolation

```python
# Customer Service domain
cs_governance = IntegratedGovernance(
    storage_dir="./nethical_cs_data",
    region_id="us-east-1",
    logical_domain="customer-service"
)

# Engineering domain
eng_governance = IntegratedGovernance(
    storage_dir="./nethical_eng_data",
    region_id="us-east-1",
    logical_domain="engineering"
)

# Payment Processing domain
payment_governance = IntegratedGovernance(
    storage_dir="./nethical_payment_data",
    region_id="us-east-1",
    logical_domain="payment-processing"
)
```

### Domain-specific Policies

```python
# Create actions with domain context
from nethical.core.models import AgentAction, ActionType

action = AgentAction(
    agent_id="agent_cs_001",
    action_type=ActionType.QUERY,
    content="Customer support inquiry",
    region_id="us-east-1",
    logical_domain="customer-service"
)

# Actions are automatically tagged with domain
assert action.logical_domain == "customer-service"
```

## Cross-Region Reporting

### Aggregate by Region

```python
governance = IntegratedGovernance(storage_dir="./nethical_global_data")

# Collect metrics from multiple regions
metrics = [
    {'action_id': 'a1', 'region_id': 'eu-west-1', 'risk_score': 0.5, 'violation_detected': False},
    {'action_id': 'a2', 'region_id': 'eu-west-1', 'risk_score': 0.7, 'violation_detected': True},
    {'action_id': 'a3', 'region_id': 'us-east-1', 'risk_score': 0.3, 'violation_detected': False},
    {'action_id': 'a4', 'region_id': 'ap-south-1', 'risk_score': 0.9, 'violation_detected': True},
]

# Aggregate by region
regional_summary = governance.aggregate_by_region(metrics, group_by='region_id')

for region, stats in regional_summary.items():
    print(f"\nRegion: {region}")
    print(f"  Total actions: {stats['count']}")
    print(f"  Violations: {stats['violations']}")
    print(f"  Avg risk score: {stats['avg_risk_score']:.2f}")

# Output:
# Region: eu-west-1
#   Total actions: 2
#   Violations: 1
#   Avg risk score: 0.60
# 
# Region: us-east-1
#   Total actions: 1
#   Violations: 0
#   Avg risk score: 0.30
# 
# Region: ap-south-1
#   Total actions: 1
#   Violations: 1
#   Avg risk score: 0.90
```

### Aggregate by Logical Domain

```python
# Collect metrics from multiple domains
metrics = [
    {'action_id': 'a1', 'logical_domain': 'customer-service', 'risk_score': 0.4},
    {'action_id': 'a2', 'logical_domain': 'customer-service', 'risk_score': 0.6},
    {'action_id': 'a3', 'logical_domain': 'payment-processing', 'risk_score': 0.8},
]

# Aggregate by domain
domain_summary = governance.aggregate_by_region(metrics, group_by='logical_domain')

for domain, stats in domain_summary.items():
    print(f"\nDomain: {domain}")
    print(f"  Total actions: {stats['count']}")
    print(f"  Avg risk score: {stats['avg_risk_score']:.2f}")
```

## Production Deployment Patterns

### Pattern 1: Geographic Load Balancing

```python
# Route requests based on user location
def get_governance_for_region(user_location: str) -> IntegratedGovernance:
    """Get governance instance based on user location."""
    region_mapping = {
        'EU': ('eu-west-1', 'EU_GDPR'),
        'US': ('us-west-2', 'US_CCPA'),
        'AP': ('ap-south-1', 'AI_ACT'),
    }
    
    region_id, policy = region_mapping.get(user_location, ('us-east-1', 'GLOBAL_DEFAULT'))
    
    return IntegratedGovernance(
        storage_dir=f"./nethical_{region_id}_data",
        region_id=region_id,
        data_residency_policy=policy
    )

# Usage
governance = get_governance_for_region('EU')
result = governance.process_action(
    agent_id="user_123",
    action="Query sensitive data",
    region_id="eu-west-1"
)
```

### Pattern 2: Multi-tenant with Domain Isolation

```python
# Create governance per tenant/domain
def get_governance_for_tenant(tenant_id: str, domain: str) -> IntegratedGovernance:
    """Get governance instance for specific tenant and domain."""
    return IntegratedGovernance(
        storage_dir=f"./nethical_{tenant_id}_{domain}_data",
        region_id=f"us-{tenant_id}",
        logical_domain=domain,
        data_residency_policy="US_CCPA"
    )

# Usage for different tenants
tenant_a_cs = get_governance_for_tenant('tenant_a', 'customer-service')
tenant_a_eng = get_governance_for_tenant('tenant_a', 'engineering')
tenant_b_cs = get_governance_for_tenant('tenant_b', 'customer-service')
```

### Pattern 3: Federated Reporting

```python
# Collect metrics from all regions
def get_global_summary(regions: list) -> dict:
    """Get federated summary across all regions."""
    all_metrics = []
    
    for region_config in regions:
        gov = IntegratedGovernance(**region_config)
        # Collect metrics from each region
        # (Implementation depends on your metrics collection)
        region_metrics = collect_region_metrics(gov)
        all_metrics.extend(region_metrics)
    
    # Aggregate across all regions
    global_gov = IntegratedGovernance(storage_dir="./nethical_global")
    return global_gov.aggregate_by_region(all_metrics, group_by='region_id')

# Usage
regions = [
    {'storage_dir': './eu_data', 'region_id': 'eu-west-1', 'data_residency_policy': 'EU_GDPR'},
    {'storage_dir': './us_data', 'region_id': 'us-west-2', 'data_residency_policy': 'US_CCPA'},
    {'storage_dir': './ap_data', 'region_id': 'ap-south-1', 'data_residency_policy': 'AI_ACT'},
]

summary = get_global_summary(regions)
```

## Performance Considerations

### Multi-Region Setup

The regionalization feature has been tested with 5+ regions:

```python
# Performance test with 6 regions
regions = ['eu-west-1', 'us-east-1', 'ap-south-1', 
           'ap-northeast-1', 'sa-east-1', 'ca-central-1']

for region in regions:
    gov = IntegratedGovernance(
        storage_dir=f"./nethical_{region}_data",
        region_id=region
    )
    
    result = gov.process_action(
        agent_id=f"agent_{region}",
        action=f"test action for {region}",
        region_id=region
    )
    
    # Verified: All phases execute successfully
    assert 'phase3' in result
    assert 'phase4' in result
```

### Database Sharding

Each region maintains its own SQLite database:

```python
# EU database: ./nethical_eu_data/actions.db
# US database: ./nethical_us_data/actions.db
# AP database: ./nethical_ap_data/actions.db

# Each database includes regional fields in schema:
# - actions.region_id
# - actions.logical_domain
# - violations.region_id
# - violations.logical_domain
# - judgments.region_id
# - judgments.logical_domain
```

## Migration Guide

### Migrating Existing Deployments

To add regionalization to an existing deployment:

1. Update data models (already done in core):
   ```python
   # Old action
   action = AgentAction(
       agent_id="agent_001",
       action_type=ActionType.QUERY,
       content="test"
   )
   
   # New action with regional fields (backward compatible)
   action = AgentAction(
       agent_id="agent_001",
       action_type=ActionType.QUERY,
       content="test",
       region_id="us-east-1",
       logical_domain="customer-service"
   )
   ```

2. Update governance initialization:
   ```python
   # Old governance
   gov = IntegratedGovernance(storage_dir="./data")
   
   # New governance with regional config
   gov = IntegratedGovernance(
       storage_dir="./data",
       region_id="us-east-1",
       data_residency_policy="US_CCPA"
   )
   ```

3. Regional fields are optional - existing code continues to work!

## Best Practices

### 1. Choose Appropriate Policies

- **EU users**: Use `EU_GDPR` for strict data protection
- **US users**: Use `US_CCPA` for consumer privacy
- **AI systems**: Use `AI_ACT` for AI-specific compliance
- **Global/testing**: Use default policy (no data_residency_policy)

### 2. Design Domain Hierarchy

```
company/
├── customer-service/
│   ├── support-tier1/
│   └── support-tier2/
├── engineering/
│   ├── backend/
│   └── frontend/
└── operations/
    ├── security/
    └── compliance/
```

Use `logical_domain` to match your organizational structure.

### 3. Monitor Cross-Border Transfers

```python
result = governance.process_action(
    agent_id="agent_123",
    action="query",
    region_id="other-region"
)

# Always check data residency compliance
if not result['data_residency']['compliant']:
    print("⚠️ Cross-border transfer violation detected!")
    print(result['data_residency']['violations'])
```

### 4. Aggregate Regularly

```python
# Schedule daily aggregation
def daily_regional_report():
    metrics = collect_daily_metrics()
    regional_summary = governance.aggregate_by_region(metrics)
    domain_summary = governance.aggregate_by_region(metrics, group_by='logical_domain')
    
    send_report(regional_summary, domain_summary)
```

## API Reference

### IntegratedGovernance Parameters

- `region_id` (str, optional): Geographic region identifier (e.g., 'eu-west-1')
- `logical_domain` (str, optional): Logical domain for hierarchical aggregation
- `data_residency_policy` (str, optional): Policy name ('EU_GDPR', 'US_CCPA', 'AI_ACT')

### process_action Parameters

- `region_id` (str, optional): Override region for this action
- `compliance_requirements` (List[str], optional): Specific requirements to validate

### Methods

- `validate_data_residency(region_id)`: Validate cross-border transfers
- `aggregate_by_region(metrics, group_by)`: Aggregate metrics by region or domain

## Troubleshooting

### Issue: Cross-border transfer blocked

**Solution**: Check if your policy allows transfers:
```python
print(governance.regional_policies['cross_border_transfer_allowed'])
```

### Issue: Actions not showing regional data

**Solution**: Ensure you're passing regional parameters:
```python
result = governance.process_action(
    agent_id="agent_123",
    action="query",
    region_id="eu-west-1",  # ← Add this
    compliance_requirements=["GDPR"]  # ← And this
)
```

### Issue: Performance degradation with multiple regions

**Solution**: Use separate storage directories per region:
```python
# Good: Separate directories
eu_gov = IntegratedGovernance(storage_dir="./eu_data", region_id="eu-west-1")
us_gov = IntegratedGovernance(storage_dir="./us_data", region_id="us-west-2")

# Bad: Shared directory
gov = IntegratedGovernance(storage_dir="./data")  # All regions in one DB
```

## Further Reading

- [GDPR Compliance Guide](https://gdpr.eu/)
- [CCPA Overview](https://oag.ca.gov/privacy/ccpa)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [Data Residency Best Practices](https://cloud.google.com/architecture/framework/security/data-residency)

## Support

For issues or questions about regionalization features:
- Open an issue on GitHub
- Check the test suite in `tests/test_regionalization.py`
- Review the implementation in `nethical/core/integrated_governance.py`
