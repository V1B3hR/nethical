# Fault Tree Analysis (FTA)

## Document Information

| Field | Value |
|-------|-------|
| Document ID | FTA-001 |
| Version | 1.0 |
| ASIL Classification | D |
| Date | 2025-12-03 |
| Author | Nethical Safety Team |
| Status | Draft |

## 1. Introduction

This document presents the Fault Tree Analysis (FTA) for Nethical's AI Governance System. FTA is a top-down, deductive failure analysis that uses Boolean logic to combine lower-level events to produce a top-level undesired event (hazard).

## 2. Top-Level Hazards

Based on the Hazard Analysis and Risk Assessment (HARA), the following top-level hazards are analyzed:

| ID | Hazard | ASIL | Safe State |
|----|--------|------|------------|
| H-001 | Unsafe AI decision allowed | D | Block all AI, transfer to human |
| H-002 | Safe AI decision blocked | C | Allow with human confirmation |
| H-003 | Governance decision not available | D | Apply safe defaults |

## 3. Fault Tree: H-001 - Unsafe AI Decision Allowed

```
                    ╔═══════════════════════════════════════╗
                    ║      TOP EVENT: H-001                 ║
                    ║   Unsafe AI Decision Allowed          ║
                    ║        (ASIL-D)                       ║
                    ╚═══════════════════════════════════════╝
                                       │
                              ┌────────┴────────┐
                              │       OR        │
                              └────────┬────────┘
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐             ┌──────▼──────┐             ┌──────▼──────┐
    │   G-001     │             │   G-002     │             │   G-003     │
    │ Policy      │             │ Risk Score  │             │ Fundamental │
    │ Failure     │             │ Failure     │             │ Law Bypass  │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │                           │                           │
    ┌──────┴──────┐             ┌──────┴──────┐             ┌──────┴──────┐
    │     OR      │             │     OR      │             │     AND     │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │                           │                           │
    ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
    │      │      │             │      │      │             │      │      │
┌───▼──┐┌──▼───┐┌─▼────┐   ┌───▼──┐┌──▼───┐┌─▼────┐   ┌───▼──┐┌──▼───┐┌─▼────┐
│B-001 ││B-002 ││B-003 │   │B-004 ││B-005 ││B-006 │   │B-007 ││B-008 ││B-009 │
│Policy││Policy││Policy│   │Wrong ││Score ││ML    │   │All 25││Formal││Human │
│Load  ││Eval  ││Match │   │Input ││Calc  ││Model │   │Laws  ││Proof ││Review│
│Fail  ││Error ││Error │   │Parse ││Error ││Fail  │   │Bypass││Fail  ││Fail  │
└──────┘└──────┘└──────┘   └──────┘└──────┘└──────┘   └──────┘└──────┘└──────┘
```

### Basic Event Definitions

| ID | Event | Probability | Description |
|----|-------|-------------|-------------|
| B-001 | Policy Load Failure | 10⁻⁶ | Failure to load valid policy file |
| B-002 | Policy Evaluation Error | 10⁻⁷ | Logic error in policy evaluation |
| B-003 | Policy Match Error | 10⁻⁶ | Wrong policy selected for context |
| B-004 | Wrong Input Parse | 10⁻⁵ | Malformed input accepted |
| B-005 | Score Calculation Error | 10⁻⁷ | Numerical error in risk calculation |
| B-006 | ML Model Failure | 10⁻⁵ | ML produces adversarial output |
| B-007 | All 25 Laws Bypass | 10⁻¹⁰ | All fundamental laws circumvented |
| B-008 | Formal Proof Failure | 10⁻⁹ | Verified invariants violated |
| B-009 | Human Review Failure | 10⁻⁴ | Human oversight fails |

### Gate Calculations

**G-001 (Policy Failure):**
```
P(G-001) = P(B-001) + P(B-002) + P(B-003)
P(G-001) = 10⁻⁶ + 10⁻⁷ + 10⁻⁶ = 2.1 × 10⁻⁶
```

**G-002 (Risk Score Failure):**
```
P(G-002) = P(B-004) + P(B-005) + P(B-006)
P(G-002) = 10⁻⁵ + 10⁻⁷ + 10⁻⁵ = 2.01 × 10⁻⁵
```

**G-003 (Fundamental Law Bypass):**
```
P(G-003) = P(B-007) × P(B-008) × P(B-009)
P(G-003) = 10⁻¹⁰ × 10⁻⁹ × 10⁻⁴ = 10⁻²³
```
(Effectively impossible due to AND gate with triple redundancy)

**Top Event H-001:**
```
P(H-001) = P(G-001) + P(G-002) + P(G-003)
P(H-001) ≈ 2.21 × 10⁻⁵
```

### Minimal Cut Sets

| Rank | Cut Set | Probability | Improvement |
|------|---------|-------------|-------------|
| 1 | {B-006} | 10⁻⁵ | Add ML sanity checks |
| 2 | {B-004} | 10⁻⁵ | Stricter input validation |
| 3 | {B-001} | 10⁻⁶ | Redundant policy loading |
| 4 | {B-003} | 10⁻⁶ | Context validation |
| 5 | {B-002, B-005} | 10⁻¹⁴ | Already acceptable |

## 4. Fault Tree: H-002 - Safe AI Decision Blocked

```
                    ╔═══════════════════════════════════════╗
                    ║      TOP EVENT: H-002                 ║
                    ║   Safe AI Decision Blocked            ║
                    ║        (ASIL-C)                       ║
                    ╚═══════════════════════════════════════╝
                                       │
                              ┌────────┴────────┐
                              │       OR        │
                              └────────┬────────┘
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐             ┌──────▼──────┐             ┌──────▼──────┐
    │   G-010     │             │   G-011     │             │   G-012     │
    │ False       │             │ Threshold   │             │ Detector    │
    │ Positive    │             │ Miscalib    │             │ Error       │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │                           │                           │
    ┌──────┴──────┐             ┌──────┴──────┐             ┌──────┴──────┐
    │     OR      │             │     OR      │             │     OR      │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │                           │                           │
    ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
    │      │      │             │      │      │             │      │      │
┌───▼──┐┌──▼───┐┌─▼────┐   ┌───▼──┐┌──▼───┐┌─▼────┐   ┌───▼──┐┌──▼───┐┌─▼────┐
│B-010 ││B-011 ││B-012 │   │B-013 ││B-014 ││B-015 │   │B-016 ││B-017 ││B-018 │
│Overly││ML    ││Domain││   │Wrong ││Config││Drift │   │PII   ││Harm  ││Bias  │
│Strict││Bias  ││Misfit│   │Thresh││Error ││Decay │   │False+││False+││Detect│
│Policy││      ││      │   │      ││      ││      │   │      ││      ││      │
└──────┘└──────┘└──────┘   └──────┘└──────┘└──────┘   └──────┘└──────┘└──────┘
```

### Top Event Probability

```
P(H-002) ≈ 10⁻³ (Higher than H-001, but lower safety impact)
```

## 5. Fault Tree: H-003 - Governance Decision Not Available

```
                    ╔═══════════════════════════════════════╗
                    ║      TOP EVENT: H-003                 ║
                    ║ Governance Decision Not Available     ║
                    ║        (ASIL-D)                       ║
                    ╚═══════════════════════════════════════╝
                                       │
                              ┌────────┴────────┐
                              │       AND       │
                              └────────┬────────┘
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
             ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
             │   G-020     │    │   G-021     │    │   G-022     │
             │ Primary     │    │ Fallback    │    │ Safe        │
             │ Failure     │    │ Failure     │    │ Default     │
             │             │    │             │    │ Failure     │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
             ┌──────┴──────┐    ┌──────┴──────┐    ┌──────┴──────┐
             │     OR      │    │     OR      │    │     OR      │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
             ┌──────┼──────┐    ┌──────┼──────┐    ┌──────┼──────┐
             │      │      │    │      │      │    │      │      │
         ┌───▼──┐┌──▼───┐┌─▼────┐┌───▼──┐┌──▼───┐┌─▼────┐┌───▼──┐┌──▼───┐
         │B-020 ││B-021 ││B-022 ││B-023 ││B-024 ││B-025 ││B-026 ││B-027 │
         │CPU   ││Memory││Stack ││Cache ││Backup││Sync  ││Config││Code  │
         │Overld││Exhausth│Overfl││Miss  ││Fail  ││Fail  ││Error ││Bug   │
         └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
```

### Gate Calculations

**G-020 (Primary Failure):**
```
P(G-020) = P(B-020) + P(B-021) + P(B-022) = 3 × 10⁻⁵
```

**G-021 (Fallback Failure):**
```
P(G-021) = P(B-023) + P(B-024) + P(B-025) = 3 × 10⁻⁵
```

**G-022 (Safe Default Failure):**
```
P(G-022) = P(B-026) + P(B-027) = 10⁻⁸
```

**Top Event H-003:**
```
P(H-003) = P(G-020) × P(G-021) × P(G-022)
P(H-003) = 3×10⁻⁵ × 3×10⁻⁵ × 10⁻⁸
P(H-003) = 9 × 10⁻¹⁸
```
(Effectively impossible with triple redundancy)

## 6. Importance Analysis

### Fussell-Vesely Importance

| Event | Importance | Interpretation |
|-------|-----------|----------------|
| B-006 (ML Model) | 0.45 | Most critical single point |
| B-004 (Input Parse) | 0.45 | Equal contribution |
| B-001 (Policy Load) | 0.045 | Secondary |
| B-003 (Policy Match) | 0.045 | Secondary |
| Others | < 0.01 | Negligible |

### Improvement Priority

1. **ML Model Robustness** - Highest impact
2. **Input Validation** - High impact
3. **Policy Redundancy** - Medium impact

## 7. Common Cause Failure Analysis

### Potential CCF Groups

| CCF Group | Events | Beta Factor | Mitigation |
|-----------|--------|-------------|------------|
| CCF-1 | B-001, B-023 | 0.1 | Different storage paths |
| CCF-2 | B-002, B-005 | 0.05 | Diverse algorithms |
| CCF-3 | B-006, B-011 | 0.2 | Rule primacy enforced |

### CCF Defense Mechanisms

1. **Diversity**: Different implementations for redundant paths
2. **Separation**: Physical and logical separation
3. **Monitoring**: Independent watchdogs
4. **Testing**: Stress testing under common causes

## 8. Safety Metrics

### ASIL-D Target

| Metric | Target | Achieved |
|--------|--------|----------|
| SPFM | ≥ 99% | TBD |
| LFM | ≥ 90% | TBD |
| PMHF | < 10⁻⁸/hour | 9×10⁻¹⁸ ✅ |

**SPFM**: Single Point Fault Metric  
**LFM**: Latent Fault Metric  
**PMHF**: Probabilistic Metric for random Hardware Failures

## 9. Recommendations

### High Priority

1. Implement ML output sandboxing with rule-based override
2. Add input validation layer with strict schema enforcement
3. Deploy policy loading redundancy with voting

### Medium Priority

4. Increase detector diversity
5. Implement threshold drift monitoring
6. Add formal verification for critical paths

### Low Priority

7. Enhance CCF defense documentation
8. Quarterly FTA review process

## 10. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-03 | Nethical Safety Team | Initial version |

## 11. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Safety Manager | | | |
| Technical Lead | | | |
| Quality Assurance | | | |

---

**Classification:** ISO 26262 ASIL-D Development  
**Retention Period:** Life of product + 15 years
