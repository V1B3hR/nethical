# Data Protection Impact Assessment (DPIA) Template

## Purpose
Template for conducting DPIAs when deploying Nethical in environments processing personal data.

## Assessment Scope
- **System**: Nethical AI Governance
- **Data Types**: Agent actions (may contain PII), audit logs, risk profiles
- **Processing Purpose**: AI safety monitoring and governance
- **Legal Basis**: Legitimate interest in AI safety

## Privacy Risks
1. **PII in Actions**: Agent actions may contain personal data
2. **Audit Trail Data**: Persistent logs may contain sensitive information
3. **Cross-Border**: Data may traverse regions

## Mitigation Measures

### Technical Controls
- PII Detection: `nethical/utils/pii.py` detects 10+ PII types
- Redaction: `EnhancedRedactionPipeline` with policies (minimal/standard/aggressive)
- Differential Privacy: Configurable epsilon for aggregated queries
- Data Minimization: Automatic retention and deletion (`DataMinimization`)
- Storage Partitioning: By region_id and logical_domain

### Configuration
```python
gov = IntegratedGovernance(
    privacy_mode="differential",
    epsilon=1.0,
    redaction_policy="aggressive",
    region_id="eu-west-1"
)
```

## Data Subject Rights
- **Access**: Audit logs queryable by action_id
- **Rectification**: Policy updates via `PolicyDiffAuditor`
- **Erasure (RTBF)**: See `docs/privacy/DSR_runbook.md`
- **Portability**: JSON export capabilities

## Assessment Outcome
☐ No high risks identified - proceed
☐ High risks identified - additional measures required
☐ Unacceptable risks - do not proceed

---
Last Updated: 2025-10-15
