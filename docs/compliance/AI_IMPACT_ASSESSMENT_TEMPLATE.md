# Fundamental Rights & AI Impact Assessment (FRIA / AIA) Template

**Document ID:** AIA-TEMPLATE-2026  
**Applicability:** EU AI Act (Regulation (EU) 2024/1689) Article 27, NIST AI RMF GOVERN 1.3, ISO/IEC 42001:2023  
**Target User:** Deployers of High-Risk AI Systems governed by Nethical  

---

## 1. Assessment Overview

| Field | Description / Deployer Entry |
|---|---|
| **System Name** | [Enter AI System Name & Version] |
| **Deploying Organization** | [Enter Legal Entity Name] |
| **Intended Purpose** | [Detailed description of operational use case] |
| **High-Risk Classification** | [e.g., Annex III Point 1 (Biometrics), Point 2 (Critical Infrastructure), Point 4 (Employment), Point 5 (Access to Essential Services)] |
| **Assessment Date** | [YYYY-MM-DD] |
| **Lead Assessor / Role** | [Name, Title / DPO / AI Ethics Officer] |
| **Approval Authority** | [Chief Risk Officer / Executive Board] |

---

## 2. Description of Deployer's Processes & Target Population

1. **Operational Context:**
   - In what environment and under what operational conditions will the AI system operate?
   - What decisions or actions will be automated vs. human-supervised?
2. **Affected Groups & Demographics:**
   - Which categories of natural persons or groups are likely to be affected (employees, consumers, patients, vulnerable groups)?
   - Has consultation with affected groups or worker representatives taken place (Art. 27(1)(b))?

---

## 3. Fundamental Rights Impact Analysis (EU AI Act Art. 27)

Evaluate specific impacts on EU Charter of Fundamental Rights:

| Fundamental Right | Potential Risk / Harm Description | Severity (1-5) | Likelihood (1-5) | Mitigations & Nethical Controls | Residual Risk |
|---|---|:---:|:---:|---|:---:|
| **Human Dignity (Art. 1)** | Algorithmic dehumanization or loss of agency | | | Enforce Fundamental Law #1 & #3; Mandatory human escalation | Low / Medium |
| **Right to Non-Discrimination (Art. 21)** | Proxy bias in training weights or evaluation metrics | | | Continuous demographic parity & disparate impact monitoring (`nethical.compliance.packs.uk_fairness_pack`) | Low |
| **Privacy & Data Protection (Arts. 7 & 8)** | Exposure of confidential prompts or personal data | | | Reversible Token Vault, PII scrubbing, differential privacy filters | Low |
| **Freedom of Expression (Art. 11)** | Over-filtering or censorship of legitimate opinion | | | Explainable AI rationale output, transparent rejection logs | Low |
| **Fair Working Conditions (Art. 31)** | Pervasive surveillance, excessive automated performance targets | | | Strict adherence to Acceptable Use Policy Section 2.6 | Low |
| **Effective Remedy & Fair Trial (Art. 47)** | Unexplained automated rejection with no appeal path | | | Cryptographically signed Merkle-DAG decision proofs enabling human appeal | Low |

---

## 4. Environmental & Sustainability Impact

- **Compute & Carbon Footprint:** Estimated energy consumption during inference and runtime governance checks.
- **Optimization Measures:** Utilization of quantized models (ONNX Runtime, INT8) and lightweight hash verification to minimize compute overhead.

---

## 5. Human Oversight Architecture (Art. 14 Verification)

- **Oversight Mode:**
  - [ ] **Human-in-the-Loop (HITL):** Every decision requires explicit human sign-off before execution.
  - [ ] **Human-on-the-Loop (HOTL):** System acts autonomously but human operator can intervene and veto in real-time.
  - [ ] **Human-in-Command (HIC):** Human oversees overall system behavior with emergency kill switch (`SafetyGovernance.emergency_lockdown()`).
- **Operator Competence:** Have designated overseers completed the training outlined in `docs/compliance/COMPETENCY_MATRIX.md`?

---

## 6. Incident Reporting & Continuous Review Plan

1. **Internal Notification:** Mechanism for operators to report anomalous or harmful outputs.
2. **Market Surveillance Notification:** Protocol for notifying national supervisory authorities within **72 hours** of any serious incident (Art. 73 EU AI Act).
3. **Review Frequency:** This Impact Assessment must be formally reviewed:
   - Annually, or
   - Whenever the system undergoes a **substantial modification** (Art. 3(23)).

---

## 7. Sign-Off & Declaration

We hereby certify that this Fundamental Rights Impact Assessment has been conducted in good faith in compliance with Article 27 of Regulation (EU) 2024/1689:

| Role | Name | Signature | Date |
|---|---|---|---|
| **Deployer Representative:** | ____________________ | ____________________ | ________ |
| **Data Protection Officer:** | ____________________ | ____________________ | ________ |
| **AI Safety Officer:** | ____________________ | ____________________ | ________ |
