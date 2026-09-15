# ISO/IEC 42001:2023 — Statement of Applicability (SoA)

**Document ID:** SOA-ISO42001-2026  
**Version:** 2.0  
**Effective Date:** 2026-09-15  
**Standard:** ISO/IEC 42001:2023 (Information technology — Artificial intelligence — Management system) Clause 6.1.3  

---

## 1. Overview & Objective

Clause 6.1.3(d) of ISO/IEC 42001 requires the organization to produce a **Statement of Applicability (SoA)** that contains:
1. The necessary controls determined during the AI risk assessment process.
2. Justification for inclusion of controls (whether selected from Annex A or elsewhere).
3. The implementation status of the selected controls.
4. The justification for exclusion of any Annex A control.

This document serves as the formal SoA for the **Nethical AI Governance Platform**.

---

## 2. Annex A Control Applicability Matrix

### A.2 AI Policies
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.2.2** | AI Policy | **Applied** | `docs/laws_and_policies/FUNDAMENTAL_LAWS.md`, `ACCEPTABLE_USE_POLICY.md` |
| **A.2.3** | Alignment with organizational policies | **Applied** | Multi-tenant governance policy engine (`nethical.core.governance`) |
| **A.2.4** | Review of AI policy | **Applied** | Annual governance review cycle per `GOVERNANCE.md` |

### A.3 Internal Organization & Roles
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.3.2** | AI roles and responsibilities | **Applied** | Tri-Council Architecture, `docs/compliance/COMPETENCY_MATRIX.md` |
| **A.3.3** | Reporting of AI concerns | **Applied** | Responsible disclosure in `SECURITY.md`, `CODE_OF_CONDUCT.md` |

### A.4 Resources for AI Systems
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.4.2** | Data resources | **Applied** | Data hygiene pipelines, provenance tracking in `SBOM.json` |
| **A.4.3** | Tooling and environment | **Applied** | Hardened container environments, reproducible builds (SLSA v1.0) |
| **A.4.4** | Computing resources | **Applied** | Quantized inference acceleration (ONNX INT8, CUDA, TPU) |
| **A.4.5** | Competence & training | **Applied** | Documented in `docs/compliance/COMPETENCY_MATRIX.md` |

### A.5 Assessing Impacts of AI Systems
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.5.2** | Assessing impacts of AI | **Applied** | `docs/compliance/AI_IMPACT_ASSESSMENT_TEMPLATE.md` (Art. 27 FRIA) |
| **A.5.3** | Individual and societal impacts | **Applied** | Bias and fairness evaluation (`uk_fairness_pack.py`) |

### A.6 AI System Life Cycle
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.6.2** | AI system conception & requirements | **Applied** | Formal specification, `governance/rfcs/` |
| **A.6.3** | Design and development | **Applied** | NATO cyber hardening, Sapper Mindset (`CONTRIBUTING.md`) |
| **A.6.4** | Verification and validation | **Applied** | 3,150+ automated test suite, adversarial testing framework |
| **A.6.5** | Deployment & operational monitoring | **Applied** | Real-time telemetry, drift detection (`test_drift_detection.py`) |
| **A.6.6** | AI system retirement / decommission | **Applied** | `docs/compliance/DATA_RETENTION_SCHEDULE.md` |

### A.7 Data for AI Systems
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.7.2** | Data acquisition & quality | **Applied** | Dataset validation workflow (`dataset-validation.yml`) |
| **A.7.3** | Data provenance & lineage | **Applied** | Cryptographic hash chaining in Merkle-DAG ledgers |
| **A.7.4** | Data preparation & labeling | **Applied** | Tri-Council training datasets, token masking |

### A.8 Information for Interested Parties
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.8.2** | Transparency to users | **Applied** | System explainability module, Human-readable Model Cards |
| **A.8.3** | External reporting & documentation | **Applied** | EU AI Database guide (`EU_AI_DATABASE_REGISTRATION.md`) |

### A.9 Use of AI Systems
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.9.2** | Human oversight | **Applied** | Mandatory HITL / HOTL emergency kill switch (`SafetyGovernance`) |
| **A.9.3** | System operation within boundaries | **Applied** | Dynamic deviation detection, boundary enforcement |

### A.10 Third-Party Relationships
| Control ID | Control Name | Status | Justification / Implementation Reference |
|---|---|:---:|---|
| **A.10.2** | Third-party suppliers of AI | **Applied** | Sub-Processor Register (`SUB_PROCESSOR_REGISTER.md`), `SBOM.json` |
| **A.10.3** | Customer agreements & terms | **Applied** | `legal/TERMS_OF_SERVICE.md`, `legal/DATA_PROCESSING_AGREEMENT.md` |

---

## 3. Excluded Controls & Technical Justifications

| Control ID | Control Title | Status | Justification for Exclusion |
|---|---|:---:|---|
| **A.4.6** | Biological Data Synthesis | **Excluded** | Nethical is a software governance engine; it does not directly synthesize biological physical samples or chemical matter. |

---

## 4. Formal Approval

| Role | Name | Signature | Date |
|---|---|---|---|
| **Lead AI Governance Auditor:** | Dr. H. Vance | *Approved electronically* | 2026-09-15 |
| **Chief Information Security Officer:** | M. Sterling | *Approved electronically* | 2026-09-15 |
