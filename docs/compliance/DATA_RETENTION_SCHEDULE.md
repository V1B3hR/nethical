# Data Retention & Disposal Schedule

**Document ID:** DRS-2026  
**Version:** 2.0  
**Effective Date:** 2026-09-15  
**Standards:** GDPR Article 5(1)(e) (Storage Limitation), ISO/IEC 27001:2022 Annex A.5.33, SOC 2 Privacy (P6.6), NIST SP 800-88 Rev. 1  

---

## 1. Principle of Storage Limitation

Personal data and operational telemetry shall not be kept in an identifiable form for longer than is necessary for the purposes for which the personal data are processed.

This Schedule outlines retention lifetimes and disposal procedures for all data classes processed by Nethical.

---

## 2. Retention Schedule Matrix

| Data Category | Description | Storage Medium | Mandatory Retention Period | Statutory / Operational Basis | Disposal / Sanitization Method |
|---|---|---|---|---|---|
| **Cryptographic Ledger Receipts** | Merkle root hashes, transaction timestamps, signature proofs | Append-only Ledger / WORM | **10 Years** | EU AI Act Art. 12 (Traceability), ISO 42001 auditability | Cryptographic retention; indefinite retention of zero-knowledge hashes |
| **Agent Prompt & Action Logs** | Raw prompts and actions processed through governance engine | Encrypted Database / Object Store | **30 to 90 Days** (Deployer configurable) | Contractual governance verification; GDPR Art. 5 | Automated purge job; cryptographic key destruction |
| **Token Vault Mapping Keys** | Ephemeral keys mapping pseudonymized tokens to real PII | In-memory / Secure Vault | **Duration of Session** (Max 24h) | GDPR Data Minimization (Art. 5(1)(c)) | Secure memory zeroization (`memset_s` / explicit overwrite) |
| **Security Audit Logs** | Authentication logs, administrative access, API access tokens | SIEM / Centralized Logging | **1 Year** | SOC 2 CC7.2, ISO 27001 Annex A.8.15 | Automated rotation and secure deletion |
| **Model Weights & Training Checkpoints** | Safety classifier checkpoints, DPO preference datasets | Artifact Registry | **3 Years** following model deprecation | Reproducibility & EU AI Act Art. 11 technical documentation | Multi-pass overwrite of storage blocks |
| **Conformity & Regulatory Filings** | FRIA assessments, CE documentation, ISO certificates | Governance Archive | **10 Years** following end of market placement | EU AI Act Art. 18 (Keep documentation available for 10 years) | Secure archive retention |

---

## 3. Data Sanitization & Destruction Protocols

1. **Digital Sanitization:**
   - Performed in compliance with **NIST SP 800-88 Rev. 1 (Guidelines for Media Sanitization)**:
     - **Clear:** Logical overwrite of storage registers using secure wipe utilities.
     - **Purge:** Cryptographic erasure (Crypto-Shredding) by destroying the associated encryption keys.
2. **Verification of Deletion:**
   - Automated purge runs generate a cryptographic receipt confirming deletion timestamp and record counts.
