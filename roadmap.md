# nethical Roadmap

A staged, risk‑aware build plan for a multi‑agent governance, safety, security, and ethics platform.

Design Principle: Start deterministic (rules + guardrails) and add adaptivity (ML, anomaly detection) only after measurable coverage gaps emerge.

---

## Vision

Deliver a layered (“defense in depth”) control plane for autonomous / semi‑autonomous agents:

1. Hard Guardrails (non‑negotiable prohibitions)  
2. Policy Engine (taxonomy + correlation rules)  
3. Statistical / ML Classifiers (content / intent / misuse detection)  
4. Anomaly & Drift Detection (behavioral + distributional)  
5. Human Escalation & Oversight  
6. Continuous Feedback (labeling & threshold tuning)  

---

## High‑Level Phases

| Phase | Title | Primary Goal | Exit Criteria |
|-------|-------|--------------|---------------|
| 0 | Foundations | Repo scaffolding, config schemas | Config validated; basic CLI runs |
| 1 | Hard Guardrails | Enforce absolute deny rules | <5 ms per decision; guardrail tests ≥90% |
| 2 | Policy Engine (Rules) | Structured decision objects w/ severity | Every rule has trigger + negative test |
| 3 | Simulation & Rule QA | Scenario DSL + coverage metrics | ≥30 scenarios; conflict report = 0 |
| 4 | Telemetry & Risk Aggregation | Rules-only composite risk scoring | Risk emitted per event; baseline dashboard |
| 5 | ML Shadow Mode | First classifier (passive) | Logged predictions; baseline metrics captured |
| 6 | ML Assisted Enforcement | Blend ML in gray risk zone | FP delta <5%; improved detection rate |
| 7 | Anomaly & Drift Detection | Sequence + distribution shift alerts | Synthetic drift caught (100%) |
| 8 | Human-in-the-Loop Ops | Escalation + feedback loop | Median triage SLA defined & met |
| 9 | Continuous Optimization | Threshold / weight tuning (optional evolution) | Composite metric improvement ≥ target |

---

## Phase Details (Condensed)

### Phase 0 – Foundations ✅ IMPLEMENTED
- ✅ Config schemas (Pydantic).
- ✅ Artifact layout (`runs/`, `logs/`, `reports/`).
- ✅ Add: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this roadmap.

### Phase 1 – Hard Guardrails ✅ IMPLEMENTED
- ✅ Deterministic pattern & capability checks.
- ✅ Enforcement outcomes: `deny`, `sanitize`, `allow`.
- ✅ Latency benchmark & unit coverage.

### Phase 2 – Policy Engine ✅ IMPLEMENTED
- ✅ Load `ethics_taxonomy.json` + `correlation_rules.yaml`.
- ✅ Compile to evaluators with severity, rationale.
- ✅ Rule introspection CLI: `nethical policy list`.

### Phase 3 – Simulation & Rule QA ✅ IMPLEMENTED
- ✅ Scenario DSL; multi-step agent interactions.
- ✅ Coverage & conflict detection in CI.

### Phase 4 – Telemetry & Risk Aggregation ✅ IMPLEMENTED
- ✅ Unified event schema → rules-only risk.
- ✅ Dashboard: rule hit frequency, risk distribution.

### Phase 5 – ML Shadow Mode
- Train minimal classifier (logistic / small transformer).
- Passive inference; no enforcement authority.
- Log predictions alongside rule-based outcomes for comparison.
- Collect baseline metrics (precision, recall, F1, calibration).
- No impact on enforcement decisions.

### Phase 6 – ML Assisted Enforcement
- Risk blend: e.g. `0.7 * rules + 0.3 * ml`.
- Only influences mid-band risk (gray zone).
- Blended risk computed only for decisions in uncertain range (e.g., 0.4 ≤ rule_score ≤ 0.6).
- Log pre/post decision differences for audit trail.
- Gate: FP delta <5%; improved detection rate.

### Phase 7 – Anomaly & Drift Detection
- Sequence anomaly scoring (n-gram / simple model).
- Distribution shift (PSI / KL divergence).
- Alert pipeline.
- Behavioral anomaly detection for unusual patterns in agent sequences.
- Statistical drift monitoring to detect changes from baseline distribution.
- Automated alerts when drift exceeds thresholds.

### Phase 8 – Human-in-the-Loop Ops
- Escalation queue, labeling UI (even CLI-first).
- Feedback tags: `false_positive`, `missed_violation`, `policy_gap`.
- Human review workflow for uncertain or critical decisions.
- Structured feedback collection for model and rule improvement.
- Median triage SLA tracking and optimization.

### Phase 9 – Continuous Optimization
- Automated tuning (optionally evolutionary) of:
  - rule weights
  - classifier thresholds
  - escalation boundaries
- Multi-objective optimization (e.g., max recall, min FP rate, min latency).
- Techniques: grid/random search, evolutionary strategies, Bayesian optimization.
- Continuous feedback loop from human labels to retrain models.
- Promote new configurations only when gate conditions met.

---

## Incremental Deliverables Snapshot

| Deliverable | Phase | Notes |
|-------------|-------|-------|
| Decision schema | 1 | Frozen early |
| Rule compiler | 2 | Deterministic output |
| Scenario corpus (≥30) | 3 | Grows continuously |
| Risk aggregator (rules-only) | 4 | Baseline reference |
| Shadow classifier | 5 | No enforcement |
| Blended risk engine | 6 | Canary rollout |
| Drift monitors | 7 | Synthetic tests |
| Escalation workflow | 8 | Feedback loop |
| Optimization harness | 9 | Multi-objective |

---

## Governance & Audit Essentials

| Artifact | Versioning |
|----------|------------|
| Ruleset | Canonical serialized hash |
| Taxonomy | Semantic version + hash |
| ML Models | Model ID (timestamp + commit SHA) |
| Scenario Corpus | Manifest with scenario IDs + hash |
| Config | Resolved snapshot per run |
| Decision Logs | Append-only (optional hash chain) |

---

## Immediate Next Actions

1. Implement guardrail evaluation core + decision object schema.  
2. Draft initial 10 scenarios (positive & negative) with expected outcomes.  
3. Add coverage & conflict reports to CI.  
4. Produce rules-only risk score (simple weighting).  
5. Instrument structured JSON logging for every decision.  

---

## Exit Criteria Checklist (Summary)

| Phase | Key Checks |
|-------|------------|
| 1 | Guardrail latency <5 ms; unit tests ≥90% |
| 2 | 0 unresolved conflicts; all rules tested |
| 3 | Scenario coverage ≥80% rules triggered |
| 4 | Stable risk distribution baseline captured |
| 5 | Shadow metrics (precision/recall) recorded |
| 6 | Detection lift w/ controlled FP increase |
| 7 | Drift/anomaly synthetic tests pass |
| 8 | Feedback loop reduces repeat misses |
| 9 | Tuned config improves composite KPI |

---

## Change Log
- 2025-10-04: Expanded Phases 5-9 with detailed implementation specifications.
- 2025-10-04: Updated Phases 0-4 to mark as implemented after successful completion.
- 2025-10-04: Initial extraction (training/testing moved to separate file).
