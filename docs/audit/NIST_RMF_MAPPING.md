# NIST AI Risk Management Framework (AI RMF) Mapping

## Overview
Mapping of Nethical features to NIST AI RMF 1.0 functions and categories.

## GOVERN
### Accountability and Oversight
- **GV-1**: Escalation Queue with human review
- **GV-2**: Policy Diff Auditor for change tracking
- **Code**: `EscalationQueue`, `PolicyDiffAuditor`

### Policies and Procedures  
- **GV-3**: Ethical Taxonomy for categorization
- **GV-4**: Correlation rules (correlation_rules.yaml)
- **Code**: `EthicalTaxonomy`, `CorrelationEngine`

## MAP
### Context
- **MP-1**: Cohort and region-based context
- **MP-2**: Ethical drift reporting
- **Code**: `FairnessSampler`, `EthicalDriftReporter`

### Risk Assessment
- **MP-3**: Multi-tier risk engine
- **MP-4**: ML-based anomaly detection
- **Code**: `RiskEngine`, `AnomalyDriftMonitor`

## MEASURE
### Monitoring
- **MS-1**: Real-time violation detection
- **MS-2**: SLA monitoring
- **Code**: All detectors, `SLAMonitor`

### Testing and Evaluation
- **MS-3**: Adversarial test suite
- **MS-4**: Performance benchmarking
- **Code**: `tests/adversarial/`, performance tests

## MANAGE
### Risk Mitigation
- **MG-1**: Quota enforcement and rate limiting
- **MG-2**: Quarantine for high-risk actions
- **Code**: `QuotaEnforcer`, `QuarantineManager`

### Incident Response
- **MG-3**: Escalation with priority triage
- **MG-4**: Merkle-anchored audit trail
- **Code**: `EscalationQueue`, `MerkleAnchor`

## Coverage Summary
| Function | Categories Covered | Implementation Status |
|----------|-------------------|----------------------|
| GOVERN | 4/4 | ✅ Complete |
| MAP | 4/4 | ✅ Complete |
| MEASURE | 4/4 | ✅ Complete |
| MANAGE | 4/4 | ✅ Complete |

---
Last Updated: 2025-10-15
