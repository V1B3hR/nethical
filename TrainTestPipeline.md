# Training & Testing Pipeline

This document covers both (a) rule-focused validation (pre-ML) and (b) the incremental introduction of machine learning, anomaly detection, and optimization.

## Implementation

The training and testing pipeline described in this document is now implemented in the `scripts/` directory:

- **`scripts/train_model.py`** - Complete training pipeline with temporal split, baseline classifier, metrics, and promotion gate
- **`scripts/test_model.py`** - Comprehensive model evaluation on test dataset
- **`scripts/README.md`** - Detailed usage guide and workflow documentation

**Quick Start:**

```bash
# Train a new model
python scripts/train_model.py

# Evaluate the model
python scripts/test_model.py
```

See `scripts/README.md` for detailed documentation.

---

## 1. Conceptual Layers

| Layer | Purpose | Maturity Stage Added |
|-------|---------|----------------------|
| Guardrails | Hard deterministic blocking | Phase 1 |
| Policy Engine | Contextual rule scoring | Phase 2 |
| Simulation Harness | Scenario validation & coverage | Phase 3 |
| Risk Aggregation | Composite scoring (rules-first) | Phase 4 |
| ML Classifier (Shadow) | Probabilistic signal (passive) | Phase 5 |
| ML Blended Enforcement | Influence medium-risk decisions | Phase 6 |
| Anomaly & Drift | Behavioral & statistical shift alerts | Phase 7 |
| Optimization Loop | Threshold / weight / strategy tuning | Phase 9 |

---

## 2. Data & Event Flow

```
Agent Action → Event Normalizer → Guardrails → Policy Engine →
[ Shadow Classifier ] → Risk Aggregator → Enforcement Decision
          ↓                                     ↑
     Drift & Anomaly Monitors  ← Logs & Metrics ┘
                 ↓
         Human Escalations → Labeled Cases → Retraining / Rule Refinement
```

---

## 3. Artifact Schema (Core)

### 3.1 Event (Input to Engine)
```json
{
  "event_id": "evt_001",
  "timestamp": "2025-10-04T16:00:00Z",
  "agent_id": "agent.researcher.1",
  "action_type": "message",
  "content": "raw or hashed reference",
  "tool": null,
  "context_flags": ["possible_pii"],
  "chain_depth": 2
}
```

### 3.2 Decision (Output)
```json
{
  "event_id": "evt_001",
  "engine_version": "ruleset@a1f4c9|model@none",
  "triggered_rules": [
    {"id": "R-PII-EMAIL", "severity": "high", "weight": 0.4}
  ],
  "rule_score": 0.72,
  "ml_scores": {},
  "anomaly_score": null,
  "final_risk": 0.72,
  "enforcement": "deny",
  "explanation": "Email pattern matched"
}
```

---

## 4. Rule-Centric Testing (Pre-ML)

| Test Type | Objective | Tooling | Frequency |
|-----------|-----------|---------|-----------|
| Schema Validation | Structural integrity of rules/taxonomy | Pydantic / JSON Schema | On load / CI |
| Unit Rule Tests | Precision of each rule trigger & non-trigger | pytest | CI |
| Scenario Simulation | Multi-step workflows, combined effects | Scenario DSL | CI / pre-release |
| Conflict Detection | Identify contradictory outcomes | Static analyzer | CI |
| Coverage Report | Guarantee each rule is exercised | Coverage generator | CI |
| Performance Benchmarks | Ensure latency budget maintained | Micro-bench harness | Nightly |
| Mutation Tests (Optional) | Ensure tests fail if rule logic altered incorrectly | Custom mutator | Weekly |

Coverage Metric:
```
rule_coverage = rules_with_at_least_one_positive_test / total_rules
```

Target: ≥80% by Phase 3, approach ≥95% later (some rules may be reserved for rare cases).

---

## 5. Scenario DSL (Minimal Form)

```yaml
id: S-PII-EMAIL-01
description: "Block message containing an email address"
steps:
  - actor: agent.researcher.1
    action: message
    content: "Contact me at person@example.com"
expect:
  enforcement: deny
  triggered_rules:
    - R-PII-EMAIL
```

Validation:
- Each scenario must assert enforcement + rule hits OR explicitly note none (`triggered_rules: []`).

---

## 6. Risk Scoring Evolution

| Version | Formula | Notes |
|---------|---------|-------|
| v1 | `final_risk = rule_score` | Weighted severity sum normalized |
| v2 | `final_risk = 0.7 * rule_score + 0.3 * ml_risk` | ML assists gray zone |
| v3 | `final_risk = w1*rule + w2*ml + w3*anomaly + w4*context_mod` | Tunable |
| v4 (Opt) | Multi-objective optimized weights | Evolutionary / Bayesian |

Normalization:
- Rule severity weights (e.g. low=0.1, medium=0.2, high=0.4, critical=0.6).
- Cap cumulative pre-normalization score; scale to [0,1].

---

## 7. Introducing ML (Phases 5–6)

### 7.1 Data Pipeline (Shadow Mode)
1. Collect events + final decisions + rule trigger vector.
2. Label ambiguous / escalated cases (human adjudication).
3. Build dataset:
   - Features: embeddings (content), rule trigger binary vector, context flags.
   - Label: `violation` (binary) or multi-class severity.

### 7.2 Training Steps
| Step | Detail |
|------|--------|
| Split | Temporal split (train: past, val: recent) to mimic deployment |
| Baseline | Logistic regression as first classifier |
| Metrics | Precision, recall, F1, ROC-AUC, calibration (ECE) |
| Shadow Logging | Store predictions & compare w/ rule-only outcomes |
| Promotion Gate | Must improve recall of true violations ≥X% with ≤Y% FP penalty |

### 7.3 Activation (Assisted Enforcement)
- Define gray zone: e.g. `0.4 <= rule_score <= 0.6`
- Compute blended risk only in zone.
- Log pre/post decision diff for audit.

---

## 8. Anomaly & Drift (Phase 7)

| Component | Method | Output |
|-----------|--------|--------|
| Sequence Anomaly | n-gram probability / simple autoencoder | anomaly_score ∈ [0,1] |
| Content Drift | KL divergence on embedding clusters | drift_alert boolean |
| Rule Firing Drift | χ² test on rule frequency vector | flagged_rules list |
| Model Drift | PSI on feature distributions | retrain_trigger boolean |

Retrain Trigger Example:
```
if (psi_feature > 0.25 for any core feature) or (recall_drop >= 5%):
    schedule_retraining()
```

---

## 9. Optimization Loop (Phase 9)

Objective Vector (example):
- Max detection_recall
- Min false_positive_rate
- Min decision_latency
- Max human_agreement

Composite scoring for search:
```
fitness = 0.4*recall - 0.25*fp_rate - 0.15*latency + 0.2*agreement
```

Techniques:
- Grid / random search (early)
- Evolutionary strategies (later)
- Bayesian optimization (threshold calibration)

---

## 10. Continuous Feedback Loop

1. Escalated case enters queue.  
2. Human labels (violation class + rationale).  
3. Label stored with snapshot of:
   - ruleset hash
   - model version (if any)
   - context features
4. Batch incremental training:
   - Retrain candidate model
   - Shadow evaluate vs incumbent
5. Promote if gate conditions met (Section 7.2).

---

## 11. Metrics & Dashboards

| Metric | Purpose |
|--------|---------|
| Rule Coverage (%) | Confidence breadth |
| Decision Latency (p50/p95) | Performance SLO |
| Violation Recall | Safety effectiveness |
| False Positive Rate | User friction control |
| Escalation Volume & SLA | Operational load |
| Drift Alerts Count | Stability monitoring |
| Model Calibration (ECE) | Probability reliability |
| Human Agreement % | Governance quality |

Alert Threshold Examples:
- Recall < baseline - 3% (7-day rolling) → alert.
- FP rate > baseline + 2% (24h) → investigate.
- Drift alerts > 3 per day → trigger diagnostics.

---

## 12. Performance Benchmarks

| Layer | Target p95 Latency |
|-------|--------------------|
| Guardrails | < 5 ms |
| Policy Engine | < 10 ms |
| Risk Aggregation | < 2 ms |
| Shadow Model (if used) | < 15 ms |
| Blended Decision Total | < 30 ms |

Load Test Strategy:
- Replay large scenario batch (synthetic + real).
- Gradually scale concurrent events to identify saturation point.

---

## 13. Security & Integrity Tests

| Test | Description |
|------|-------------|
| Rule Tamper Detection | Hash mismatch triggers fail fast |
| Log Integrity | (Optional) hash-chain validation |
| Secrets Redaction | Ensure no secret literal escapes sanitization |
| Sandbox Enforcement | Attempt disallowed tool invocation scenarios |

---

## 14. Promotion Gate Template (ML)

```yaml
promotion_gate:
  min_recall_gain: 0.03          # +3% absolute
  max_fp_increase: 0.02          # +2% absolute
  max_latency_increase_ms: 5
  max_ece: 0.08
  min_human_agreement: 0.85
```

---

## 15. Suggested Directory Structure

```
nethical/
  rules/
    ethics_taxonomy.json
    correlation_rules.yaml
  engine/
    guardrails.py
    policy.py
    risk.py
    anomaly.py
    ml_shadow.py
  scenarios/
    *.yaml
  tests/
    unit/
    scenarios/
    performance/
  data/
    labeled_events/
  models/
    current/
    candidates/
```

---

## 16. CI Pipeline Stages

| Stage | Actions |
|-------|---------|
| Lint & Type | ruff / mypy |
| Unit Tests | Guardrail + policy tests |
| Scenario Simulation | Run DSL suite |
| Coverage Report | Upload artifact |
| Performance Smoke | Run micro-benchmark (thresholds) |
| Rule Hash Check | Fail if unexpected change w/o version bump |
| (Optional later) Shadow Metrics | Compare new model vs baseline |

---

## 17. Change Management

1. Rule Change PR must include:
   - Rationale
   - Added / updated scenarios
   - Coverage delta
2. ML Model Update PR must include:
   - Metrics table (old vs new)
   - Calibration plot artifact
   - Drift baseline snapshot

---

## 18. Minimal Initial Implementation Order (Actionable)

1. Decision schema + guardrails engine.  
2. Rule compiler + validation harness.  
3. Scenario DSL + first 10 scenarios.  
4. Coverage & conflict reporting.  
5. Risk aggregator (rules-only).  
6. Logging + metric emission.  
7. Benchmark harness.  
8. Label storage + escalation stub.  
9. Shadow classifier (small) after enough labeled data.  

---

## 19. Example Guardrail Unit Test (Illustrative)

```python
def test_email_guardrail_triggers():
    evt = Event(content="Reach me at a@b.com")
    decision = guardrail_engine.evaluate(evt)
    assert decision.enforcement == "deny"
    assert any(r.id == "R-PII-EMAIL" for r in decision.triggered_rules)
```

---

## 20. Glossary (Key Terms)

| Term | Definition |
|------|------------|
| Gray Zone | Mid-risk band where ML assists decisions |
| Shadow Mode | Model runs but cannot enforce |
| Drift | Statistical divergence from baseline distribution |
| Scenario | Structured multi-step validation script |
| Escalation | Human review path for uncertain / critical events |

---

## Change Log
- 2025-10-04: Initial extraction from combined roadmap.
