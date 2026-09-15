# Model Card — Nethical DPO Ambassador & Ethics Alignment Model

**Model Name:** `nethical-dpo-ambassador-v2`  
**Model Version:** 2.7.0  
**Date:** 2026-09-15  
**Model Type:** Direct Preference Optimization (DPO) Aligned Conversational Governance Agent  
**Standards:** Mitchell et al. (2019) Model Cards for Model Reporting, EU AI Act Article 13  

---

## 1. Model Details

- **Developer:** Nethical Core Machine Learning Team
- **Intended Use:** Explaining governance decisions, guiding human operators through ethical dilemmas, resolving conflicting policy constraints, and translating technical Merkle-DAG audit proofs into human-understandable natural language explanations.
- **Primary Architecture:** Mistral / Llama-based LoRA adapter fine-tuned via Direct Preference Optimization (DPO) on Tri-Council ethical preference pairs.
- **License:** MIT License (Copyright (c) 2025-2026 Nethical Contributors)

---

## 2. Intended Use & Boundaries

### Permitted Uses
- Interactive AI governance co-pilot for SOC / NOC / AI Risk officers.
- Generating automated audit summaries and compliance reports.
- Explaining rejection rationale to end users when an agent action is blocked by the 25 Fundamental Laws.

### Prohibited Uses
- Providing medical diagnoses, formal legal opinions, or replacing accredited human legal counsel.
- Direct operational control of kinetic hardware or critical safety relays.

---

## 3. Training & Preference Optimization Methodology

- **Preference Pairs:** 45,000+ curated ethical dilemma pairs scored by three simulated consensus councils:
  1. **Legal Council** (Statutory compliance, GDPR, EU AI Act, liability)
  2. **Ethical Council** (Asimov/Fundamental Laws, human dignity, non-maleficence)
  3. **Security Council** (Operational resilience, containment, zero-trust)
- **Loss Function:** Direct Preference Optimization (Rafailov et al., 2023) with beta parameter = 0.1.

---

## 4. Evaluation & Quantitative Metrics

| Evaluation Benchmark | Performance Score | Evaluation Standard |
|---|---|---|
| **Ethical Alignment Consistency** | **99.6%** | Tri-Council Benchmark Suite |
| **Hallucination Rate on Laws** | **< 0.1%** | Grounded in `FUNDAMENTAL_LAWS.md` |
| **Response Tone Appropriateness** | **98.7%** | Constructive, non-judgmental, objective |
| **Regulatory Knowledge Accuracy** | **99.1%** | EU AI Act & ISO 42001 Mapping test |

---

## 5. Maintenance & Continuous Monitoring

- Subject to continuous drift monitoring (`tests/validation/test_drift_detection.py`).
- Retraining triggers when alignment score dips below 98.0% on monthly canary dilemma sets.
