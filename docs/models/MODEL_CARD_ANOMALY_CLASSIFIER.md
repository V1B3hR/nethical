# Model Card — Nethical Anomaly & Deviation Classifier

**Model Name:** `nethical-anomaly-classifier-v2`  
**Model Version:** 2.7.0  
**Date:** 2026-09-15  
**Model Type:** Transformer-based sequence classification & AST embedding classifier  
**Standards:** Mitchell et al. (2019) Model Cards for Model Reporting, EU AI Act (Regulation 2024/1689) Article 13  

---

## 1. Model Details

- **Developer:** Nethical Core Machine Learning Team
- **Intended Use:** Real-time classification of agent stated intent vs. actual action deviation; detecting subtle prompt injection, manipulation techniques, and privilege escalation attempts.
- **Primary Architecture:** DistilRoBERTa / ONNX INT8 quantized embedding backbone with dual classification heads (Intent-Action Drift Head & Policy Violation Head).
- **License:** MIT License (Copyright (c) 2025-2026 Nethical Contributors)

---

## 2. Intended Use & Limitations

### Intended Applications
- Real-time pre-execution gatekeeper for autonomous AI agents.
- Runtime safety monitoring in enterprise customer service, cyber operations, and financial analysis.

### Out-of-Scope / Prohibited Uses
- Standalone autonomous kinetic targeting without human oversight.
- Emotion or psychological profiling of natural persons in prohibited contexts (EU AI Act Art. 5).

### Known Limitations
- May produce increased false positives on highly esoteric domain-specific programming languages or obfuscated binary payloads unless augmented with local fine-tuning.

---

## 3. Training Data & Alignment

- **Datasets:** Curated open-source safety datasets, Anthropic Red-Team data, BeaverTails, Nethical synthetic adversarial dilemma corpora (`training/tri_council_trainer.py`).
- **Data Preprocessing:** Rigorous PII removal, deduplication, token masking, and cryptographic dataset hashing (`dataset-validation.yml`).

---

## 4. Evaluation Metrics & Performance

Evaluated against the comprehensive 3,150+ Nethical test suite and independent adversarial benchmarks:

| Benchmark / Metric | Score | Baseline Target |
|---|---|---|
| **Adversarial Jailbreak Detection (F1)** | **99.2%** | > 95.0% |
| **Intent-Action Deviation Accuracy** | **98.8%** | > 96.0% |
| **False Positive Rate (Benign Prompts)** | **0.18%** | < 0.50% |
| **Inference Latency (ONNX INT8 CPU)** | **4.2 ms** | < 10.0 ms |
| **Inference Latency (CUDA TensorRT)** | **0.9 ms** | < 2.0 ms |

---

## 5. Ethical Considerations & Bias Mitigation

- Checked for demographic parity across gender, nationality, and protected attributes using `nethical.compliance.packs.uk_fairness_pack`.
- Disparate impact ratio maintained within statutory bounds (0.92–1.05).
