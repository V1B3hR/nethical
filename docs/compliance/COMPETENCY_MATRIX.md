# AI Governance & Security Competency Matrix

**Document ID:** CMX-2026  
**Version:** 2.0  
**Effective Date:** 2026-09-15  
**Standards:** ISO/IEC 42001:2023 Clause 7.2 (Competence), ISO/IEC 27001:2022 Clause 7.2  

---

## 1. Objective

ISO/IEC 42001 Clause 7.2 requires that the organization:
- Determine the necessary competence of persons doing work under its control that affects its AI performance;
- Ensure that these persons are competent on the basis of appropriate education, training, or experience;
- Retain appropriate documented information as evidence of competence.

This document establishes the mandatory competency requirements and training curricula for personnel designing, developing, operating, or auditing Nethical-governed systems.

---

## 2. Role-Based Competency Matrix

| Role | Core Domain Competencies | Required Certifications / Qualifications | Refresher Cycle |
|---|---|---|:---:|
| **AI Safety Engineer / Developer** | • Python AST manipulation & static analysis<br>• Adversarial robustness & jailbreak mechanics<br>• Fundamental Laws operationalization<br>• Quantum-safe signature validation (ML-DSA-65) | Computer Science degree or equivalent; Certified AI Security Professional / NATO Cyber Defense training | Annual |
| **Governance & Compliance Auditor** | • EU AI Act (Regulation 2024/1689) Articles 9–15<br>• ISO/IEC 42001 Lead Auditor principles<br>• Fundamental Rights Impact Assessments (Art. 27)<br>• GDPR & international privacy standards | ISO 42001 / ISO 27001 Lead Auditor, CIPP/E, or equivalent regulatory experience | Annual |
| **Field Deployer / Systems Operator** | • Human-in-the-Loop (HITL) triage procedures<br>• Emergency lockdown activation (`SafetyGovernance`)<br>• Merkle-DAG ledger verification & audit inspection<br>• Drift detection and false-positive escalation | Nethical Certified Operator (NCO) Course Completion | Biannual |
| **Security Red Teamer / Penetration Tester** | • Prompt injection attack methodologies (OWASP LLM01)<br>• Cryptographic curve & key leakage testing<br>• Distributed Denial of Service on inference pipelines | OSCP, OSCE, CISSP, or equivalent practical offensive certification | Annual |

---

## 3. Mandatory Training Curriculum Modules

1. **MOD-101: The 25 Fundamental Laws of AI Safety**
   - In-depth study of bi-directional ethical governance, non-subversion principles, and fail-closed architecture.
2. **MOD-201: Regulatory Conformance (EU AI Act, ISO 42001, NIST AI RMF)**
   - Statutory obligations for high-risk AI providers, conformity assessment procedures, and documentation obligations.
3. **MOD-301: Cryptographic Integrity & Post-Quantum Defense**
   - Merkle-DAG ledgers, NIST FIPS 204 Dilithium signatures, key rotation, and secure memory erasure.
4. **MOD-401: Incident Management & Serious Incident Reporting**
   - 72-hour breach reporting to supervisory authorities under EU AI Act Art. 73 and GDPR Art. 33.

---

## 4. Training Record Log & Sign-Off Template

| Employee / Contributor Name | Role | Module Completed | Completion Date | Assessment Score | Assessor Sign-Off |
|---|---|---|:---:|:---:|---|
| [Contributor Name] | Core Developer | MOD-101, MOD-301 | 2026-09-01 | 100% | *Certified* |
| [Deployer Operator] | Node Operator | MOD-101, MOD-401 | 2026-09-10 | 96% | *Certified* |
