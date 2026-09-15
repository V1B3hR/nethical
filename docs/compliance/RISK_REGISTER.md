# Enterprise AI & Information Security Risk Register

**Document ID:** RR-ENT-2026  
**Version:** 2.0  
**Last Updated:** 2026-09-15  
**Standards:** ISO/IEC 42001:2023 Clause 6.1, ISO/IEC 27001:2022 Clause 6.1.2, NIST AI RMF GOVERN/MANAGE  

---

## 1. Risk Assessment Methodology

Risks are evaluated using a 5x5 Severity (Impact) × Likelihood matrix:
- **Impact (1-5):** 1 (Negligible), 2 (Minor), 3 (Moderate), 4 (Major), 5 (Catastrophic / Life Safety)
- **Likelihood (1-5):** 1 (Rare), 2 (Unlikely), 3 (Possible), 4 (Likely), 5 (Almost Certain)
- **Risk Score:** Impact × Likelihood
  - **1–6: LOW (Green)** — Acceptable; routine monitoring
  - **8–12: MEDIUM (Yellow)** — Requires treatment plan and assigned owner
  - **15–25: HIGH / CRITICAL (Red)** — Unacceptable without executive sign-off and active automated guardrails

---

## 2. Active Risk Register

| Risk ID | Category | Threat / Hazard Description | Inh. Score | Risk Owner | Mitigating Controls & Nethical Implementations | Res. Score | Target Date | Status |
|:---:|---|---|:---:|---|---|:---:|:---:|:---:|
| **RSK-01** | AI Safety | Autonomous agent executes prompt injection or jailbreak leading to unauthorized privilege escalation | 5 × 4 = **20** | Lead Security Architect | Inoculation Mesh, Dual-model Judge verification, Ast audit | 5 × 1 = **5** (Low) | Active | Controlled |
| **RSK-02** | Cryptographic | Quantum computing advances compromise legacy signature schemes on audit logs | 4 × 3 = **12** | Cryptography Lead | NIST FIPS 204 ML-DSA-65 (Dilithium) post-quantum signatures + hybrid P-384 | 4 × 1 = **4** (Low) | Active | Controlled |
| **RSK-03** | Legal / Compliance | Deployer operates high-risk AI without EU AI Act Art. 49 registration or Art. 27 FRIA | 5 × 3 = **15** | Head of Compliance | Automated compliance packs (`eu_high_risk_pack.py`), FRIA template, registration guide | 4 × 1 = **4** (Low) | Active | Controlled |
| **RSK-04** | Privacy / Data | Leakage of PII / confidential prompts across multi-tenant boundaries | 4 × 4 = **16** | Data Protection Officer | Reversible Token Vault, local anonymization, memory encryption | 3 × 1 = **3** (Low) | Active | Controlled |
| **RSK-05** | Supply Chain | Compromise of third-party upstream Python dependencies (e.g., PyPI typosquatting) | 4 × 3 = **12** | DevSecOps Lead | Pinned hashes in `requirements.txt`, Dependabot, SLSA v1.0 provenance, SBOM generation | 3 × 1 = **3** (Low) | Active | Controlled |
| **RSK-06** | Operational | Ledger database corruption or storage loss in distributed nodes | 4 × 2 = **8** | Infrastructure Lead | Append-only Merkle-DAG consistency checks, SQLite WAL mode, S3 replication | 2 × 1 = **2** (Low) | Active | Controlled |
| **RSK-07** | Kinetic Safety | Governed agent interacting with robotics or vehicle fieldbus issues unverified physical action | 5 × 3 = **15** | Safety Governance Board | Kinetic safety pack (`test_kinetic_safety_and_iso42001.py`), mandatory human-in-the-loop override | 5 × 1 = **5** (Low) | Active | Controlled |
| **RSK-08** | Ethical / Bias | Disparate impact or demographic bias in agent evaluation metrics | 4 × 3 = **12** | AI Ethics Officer | UK Fairness Pack (`uk_fairness_pack.py`), statistical parity thresholds | 3 × 1 = **3** (Low) | Active | Controlled |

---

## 3. Residual Risk Acceptance & Governance Oversight

All residual risks scoring above 4 are reviewed quarterly by the **Nethical Tri-Council Governance Board**.
Annual formal recertification takes place concurrently with the ISO 42001 surveillance audit.
