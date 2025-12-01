# Nethical Risk Assessment Methodology

Version: 0.1.0
Status: Draft
Last Updated: 2025-12-01
Owner: Tech Lead / Security Lead

## 1. Purpose
Define a structured, auditable, and repeatable process to identify, analyze, and evaluate risks impacting Nethical’s correctness, security, compliance, fairness, reliability, performance, and ethical governance properties prior to and during their inclusion and maintenance in the risk register.

## 2. Scope
Applies to:
- Decision engine core (policy evaluation, state transitions)
- Governance workflows (multi-signature, policy activation)
- Formal verification artifacts & proof coverage
- Fairness / compliance monitoring pipelines
- Supply chain & build integrity
- Runtime monitoring infrastructure & audit subsystem

## 3. Definitions
- Risk Assessment: Combined activities of Identification, Analysis, Evaluation.
- Identification: Discovering potential risk events or conditions.
- Analysis: Determining impact, likelihood, detectability, exposure, control maturity.
- Evaluation: Prioritizing and deciding treatment path (mitigate, monitor, accept).
- Residual Risk: Remaining risk post-mitigation and controls.
- Drift: Significant deviation in monitored metrics (fairness, performance, invariants).

## 4. Principles
1. Evidence-Based Scoring (anchor definitions + referenced artifacts).
2. Traceability (risk ↔ requirements ↔ controls ↔ proofs ↔ monitoring signals).
3. Minimal Subjectivity (use quantifiable anchors & modifiers).
4. Continuous Reassessment (triggered by drift, incidents, proof coverage changes).
5. Transparency (machine-readable YAML, rationale blocks).

## 5. Trigger Sources for Assessment
| Source | Examples | Action |
|--------|----------|--------|
| Scheduled Reviews | Weekly High / Monthly Medium | Re-score if conditions changed |
| Code / Architecture Change | New policy engine feature, dependency added | Generate candidate risk |
| Proof Gap Scan | Unproved invariants, admitted lemmas | Candidate correctness risk |
| Monitoring / Metrics | Fairness parity delta > threshold, p95 latency breach | Reassess related risk |
| Threat Modeling | STRIDE adaptation, abuse cases | New security risk |
| Red-Team / Adversarial Tests | Attack simulation outcomes | Upgrade or add risk |
| Supply Chain Events | New CVE ≥ 8.0, SBOM diff | Add/upgrade supply chain risk |
| Incident / Near-Miss | Logged production anomaly | Immediate assessment |
| Regulatory Change | New compliance directive | Add compliance risk |

## 6. Workflow Overview
1. Candidate Capture (pre-screen)
2. Formal Identification Session
3. Evidence Collection
4. Structured Scoring & Analysis
5. Evaluation & Treatment Decision
6. Risk Record Creation/Update
7. Independent Review (for HIGH+)
8. Scheduling Next Review (next_review_date)
9. Integration (requirements, proofs, monitoring signals)

## 7. Pre-Screen Template
