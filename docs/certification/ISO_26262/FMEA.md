# Failure Mode and Effects Analysis (FMEA)

## Document Information

| Field | Value |
|-------|-------|
| Document ID | FMEA-001 |
| Version | 1.0 |
| ASIL Classification | D |
| Date | 2025-12-03 |
| Author | Nethical Safety Team |
| Status | Draft |

## 1. Introduction

This document presents the Failure Mode and Effects Analysis (FMEA) for Nethical's AI Governance System deployed in autonomous vehicle applications. The analysis follows ISO 26262-9 guidelines and supports ASIL-D safety integrity requirements.

## 2. Scope

### System Boundary

The analysis covers:
- Edge Governance Engine
- Policy Evaluation System
- Risk Scoring Module
- Decision Output Interface
- Offline Fallback System
- Audit Logging

### Interfaces

| Interface | Type | Connected System |
|-----------|------|-----------------|
| IF-001 | Input | Vehicle Perception System |
| IF-002 | Input | Vehicle Planning System |
| IF-003 | Output | Vehicle Control System |
| IF-004 | Bidirectional | Cloud Sync |
| IF-005 | Output | Audit Storage |

## 3. FMEA Methodology

### Severity (S) Classification

| S | Description | Criteria |
|---|-------------|----------|
| 10 | Hazardous without warning | Death or serious injury |
| 9 | Hazardous with warning | Death possible with warning |
| 8 | Very High | Serious injury possible |
| 7 | High | Significant injury |
| 6 | Moderate | Minor injury |
| 5 | Low | Discomfort |
| 4 | Very Low | Minor annoyance |
| 3 | Minor | Little effect |
| 2 | Very Minor | Barely noticeable |
| 1 | None | No effect |

### Occurrence (O) Classification

| O | Description | Failure Rate |
|---|-------------|-------------|
| 10 | Very High | > 1 in 2 |
| 9 | High | 1 in 3 |
| 8 | High | 1 in 8 |
| 7 | Moderate | 1 in 20 |
| 6 | Moderate | 1 in 80 |
| 5 | Moderate | 1 in 400 |
| 4 | Moderate Low | 1 in 2,000 |
| 3 | Low | 1 in 15,000 |
| 2 | Low | 1 in 150,000 |
| 1 | Remote | < 1 in 1,500,000 |

### Detection (D) Classification

| D | Description | Detection Capability |
|---|-------------|---------------------|
| 10 | Absolute uncertainty | Cannot detect |
| 9 | Very Remote | Very remote chance |
| 8 | Remote | Remote chance |
| 7 | Very Low | Very low chance |
| 6 | Low | Low chance |
| 5 | Moderate | Moderate chance |
| 4 | Moderately High | Moderately high |
| 3 | High | High chance |
| 2 | Very High | Very high chance |
| 1 | Almost Certain | Always detected |

### Risk Priority Number (RPN)

**RPN = S × O × D**

| RPN Range | Priority | Action Required |
|-----------|----------|-----------------|
| 200-1000 | Critical | Immediate corrective action |
| 100-199 | High | Corrective action required |
| 50-99 | Medium | Corrective action recommended |
| 1-49 | Low | Monitor and control |

## 4. FMEA Analysis

### FM-001: Decision Engine Timeout

| Field | Value |
|-------|-------|
| Component | Edge Governor |
| Function | Governance Decision |
| Failure Mode | Decision exceeds 10ms SLA |
| Effect (Local) | Delayed response to AI action request |
| Effect (End) | Vehicle may execute action without governance |
| Cause | CPU overload, complex policy evaluation |
| **Severity** | **9** (Potential unsafe AI action) |
| Current Controls | Circuit breaker, resource limits |
| **Occurrence** | **3** (Rare with mitigations) |
| Detection Method | Latency monitoring, SLO probes |
| **Detection** | **2** (Real-time monitoring) |
| **RPN** | **54** (Medium) |
| Recommended Action | Pre-computed decision cache, JIT optimization |

### FM-002: Policy Cache Corruption

| Field | Value |
|-------|-------|
| Component | Policy Cache |
| Function | Policy Storage |
| Failure Mode | Cache returns incorrect policy |
| Effect (Local) | Wrong risk thresholds applied |
| Effect (End) | Incorrect governance decision |
| Cause | Memory error, sync race condition |
| **Severity** | **8** |
| Current Controls | CRC validation, redundant cache |
| **Occurrence** | **2** |
| Detection Method | Integrity checks on each read |
| **Detection** | **2** |
| **RPN** | **32** (Low) |
| Recommended Action | Triple modular redundancy |

### FM-003: Network Connectivity Loss

| Field | Value |
|-------|-------|
| Component | Cloud Sync |
| Function | Policy Updates |
| Failure Mode | Complete network disconnection |
| Effect (Local) | Cannot receive policy updates |
| Effect (End) | Using potentially outdated policies |
| Cause | Cellular failure, tunnel issues |
| **Severity** | **6** (Degraded but safe defaults) |
| Current Controls | Offline fallback, local policies |
| **Occurrence** | **5** (Common in tunnels, remote areas) |
| Detection Method | Heartbeat monitoring |
| **Detection** | **1** (Always detected) |
| **RPN** | **30** (Low) |
| Recommended Action | Extended offline policy validity |

### FM-004: Risk Score Calculation Error

| Field | Value |
|-------|-------|
| Component | Risk Engine |
| Function | Risk Assessment |
| Failure Mode | Numerical overflow/underflow |
| Effect (Local) | Extreme risk scores |
| Effect (End) | Incorrect decision (false positive/negative) |
| Cause | Edge case inputs, floating point errors |
| **Severity** | **8** |
| Current Controls | Input validation, bounded outputs |
| **Occurrence** | **2** |
| Detection Method | Range checks, plausibility validation |
| **Detection** | **2** |
| **RPN** | **32** (Low) |
| Recommended Action | Fixed-point arithmetic for critical paths |

### FM-005: Safe Default Failure

| Field | Value |
|-------|-------|
| Component | Safe Defaults |
| Function | Fallback Decisions |
| Failure Mode | Safe default returns unsafe decision |
| Effect (Local) | Incorrect fallback behavior |
| Effect (End) | Unsafe AI action during degraded mode |
| Cause | Configuration error, logic bug |
| **Severity** | **10** (Critical safety mechanism) |
| Current Controls | Hardcoded defaults, formal verification |
| **Occurrence** | **1** (Formally verified) |
| Detection Method | Startup validation, runtime assertions |
| **Detection** | **1** |
| **RPN** | **10** (Low) |
| Recommended Action | Maintain formal verification coverage |

### FM-006: Audit Log Loss

| Field | Value |
|-------|-------|
| Component | Audit System |
| Function | Decision Logging |
| Failure Mode | Decisions not logged |
| Effect (Local) | Missing audit trail |
| Effect (End) | Cannot reconstruct incident |
| Cause | Storage failure, queue overflow |
| **Severity** | **5** (Post-incident impact only) |
| Current Controls | Buffered writes, Merkle anchoring |
| **Occurrence** | **3** |
| Detection Method | Log sequence validation |
| **Detection** | **2** |
| **RPN** | **30** (Low) |
| Recommended Action | Triple-redundant storage |

### FM-007: ML Model Corruption

| Field | Value |
|-------|-------|
| Component | ML Risk Scorer |
| Function | ML-based risk assessment |
| Failure Mode | Model produces adversarial outputs |
| Effect (Local) | Incorrect ML risk contribution |
| Effect (End) | Biased governance decisions |
| Cause | Adversarial input, model degradation |
| **Severity** | **7** |
| Current Controls | ML weight limits, rule primacy |
| **Occurrence** | **3** |
| Detection Method | Anomaly detection, plausibility checks |
| **Detection** | **3** |
| **RPN** | **63** (Medium) |
| Recommended Action | Online model monitoring, A/B testing |

### FM-008: Clock Drift

| Field | Value |
|-------|-------|
| Component | System Clock |
| Function | Timestamping |
| Failure Mode | Significant clock skew |
| Effect (Local) | Incorrect decision timestamps |
| Effect (End) | Audit trail ordering issues |
| Cause | Hardware failure, NTP issues |
| **Severity** | **4** |
| Current Controls | NTP sync, monotonic clocks |
| **Occurrence** | **2** |
| Detection Method | Clock health monitoring |
| **Detection** | **2** |
| **RPN** | **16** (Low) |
| Recommended Action | GPS time synchronization |

## 5. Summary

### RPN Distribution

| Priority | Count | Percentage |
|----------|-------|------------|
| Critical (200+) | 0 | 0% |
| High (100-199) | 0 | 0% |
| Medium (50-99) | 2 | 25% |
| Low (1-49) | 6 | 75% |

### Top Risks

1. **FM-007: ML Model Corruption** (RPN: 63)
2. **FM-001: Decision Engine Timeout** (RPN: 54)
3. **FM-002: Policy Cache Corruption** (RPN: 32)

### Actions Required

| Priority | Action | Owner | Target Date |
|----------|--------|-------|-------------|
| Medium | Implement online ML monitoring | ML Team | Q1 2026 |
| Medium | Expand pre-computed decision cache | Edge Team | Q1 2026 |
| Low | Implement triple modular redundancy | Core Team | Q2 2026 |

## 6. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | Nethical Safety Team | Initial version |

## 7. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Safety Manager | | | |
| Technical Lead | | | |
| Quality Assurance | | | |

---

**Classification:** ISO 26262 ASIL-D Development  
**Retention Period:** Life of product + 15 years
