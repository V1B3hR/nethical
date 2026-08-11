# UK NCSC Guidelines for Secure AI System Development - Compliance Guide

## Overview

This guide details Nethical's compliance with and implementation of the **Guidelines for Secure AI System Development**, co-authored by the UK National Cyber Security Centre (NCSC), the US Cybersecurity and Infrastructure Security Agency (CISA), and other international cybersecurity agencies.

Nethical maps its security architecture to the four core pillars of the NCSC guidelines to achieve a "secure-by-design" and "secure-by-default" governance environment.

---

## 1. Secure Design

### NCSC-AI-1.1: AI-Specific Threat Modeling
*   **Requirement**: Perform thorough threat modeling that addresses AI-specific attack vectors (e.g., prompt injection, model inversion, data poisoning) alongside traditional cyber threats.
*   **Nethical Implementation**:
    *   **Module**: `nethical/security/threat_modeling.py`
    *   **Features**: Automated threat checks that model prompt injection and privacy risks (context confusion, exfiltration).
    *   **Documentation**: [`docs/laws_and_policies/threat_model.md`](../laws_and_policies/threat_model.md)

### NCSC-AI-1.2: Secure Defaults & Validation
*   **Requirement**: Enforce secure-by-default settings, include robust role-based access control, and apply strict input/output validation.
*   **Nethical Implementation**:
    *   **Module**: `nethical/core/rbac.py`, `nethical/security/auth.py`, `nethical/security/zero_trust.py`
    *   **Features**: Decorator-based role hierarchy (Admin, Operator, Auditor, Viewer), time-based access control tokens, and zero-trust verification.

---

## 2. Secure Development

### NCSC-AI-2.1: Supply Chain Security
*   **Requirement**: Keep a Software Bill of Materials (SBOM) updated, vet dependencies, and lock package hashes to prevent supply chain contamination.
*   **Nethical Implementation**:
    *   **Module**: `requirements-hashed.txt`, `SBOM.json`
    *   **Features**: Cryptographic hash locking of all transitive dependencies via `pip-compile` to ensure reproducible and secure builds.
    *   **Documentation**: [`docs/laws_and_policies/SUPPLY_CHAIN_TODO.md`](../laws_and_policies/SUPPLY_CHAIN_TODO.md)

### NCSC-AI-2.2: Technical Documentation & transparency
*   **Requirement**: Maintain clear documentation on models, training protocols, and parameters, ensuring transparency.
*   **Nethical Implementation**:
    *   **Documentation**: [`docs/compliance/UK_NCSC_AI_SECURE_DEVELOPMENT_COMPLIANCE.md`](UK_NCSC_AI_SECURE_DEVELOPMENT_COMPLIANCE.md), `README.md`
    *   **Features**: Clear instructions on model settings, explainable AI APIs, and transparent risk threshold parameters.

---

## 3. Secure Deployment

### NCSC-AI-3.1: Model & Data Integrity Protection
*   **Requirement**: Protect model parameters and configuration weights from unauthorized modification using cryptography.
*   **Nethical Implementation**:
    *   **Module**: `nethical/security/encryption.py`, `nethical/security/attestation.py`
    *   **Features**: Cryptographic checks, AES-256 model weight storage encryption, and post-quantum cryptographic readiness.
    *   **Documentation**: [`docs/security/QUANTUM_CRYPTO_GUIDE.md`](../laws_and_policies/QUANTUM_CRYPTO_GUIDE.md)

### NCSC-AI-3.2: Security Evaluation (Red Teaming)
*   **Requirement**: Assess system vulnerability using red teaming and adversarial pen testing prior to release.
*   **Nethical Implementation**:
    *   **Module**: `nethical/security/penetration_testing.py`
    *   **Features**: Simulated attacks including prompt injection payloads and model evasion scenarios.
    *   **Documentation**: [`docs/security/red_team_report_template.md`](../laws_and_policies/red_team_report_template.md)

---

## 4. Secure Operation

### NCSC-AI-4.1: Continuous Monitoring & Observability
*   **Requirement**: Continuous runtime observability of inputs, outputs, and model thresholds to identify data drift and abuse.
*   **Nethical Implementation**:
    *   **Module**: `nethical/security/anomaly_detection.py`, `nethical/security/track_analyzer.py`
    *   **Features**: Active checks for unusual traffic volumes, drift monitoring, and anomaly classifications.
    *   **Documentation**: [`docs/governance/GOVERNANCE_OBSERVABILITY.md`](../laws_and_policies/GOVERNANCE_OBSERVABILITY.md)

### NCSC-AI-4.2: Incident Response Plans
*   **Requirement**: Establish incident response plans tailored to unique AI threats (e.g., prompt injection recovery, model rollback).
*   **Nethical Implementation**:
    *   **Module**: `nethical/security/soc_integration.py`
    *   **Features**: Automated security alerting triggers for SOC integrations when critical prompt injections or policy violations occur.
    *   **Documentation**: [`docs/compliance/INCIDENT_RESPONSE_POLICY.md`](INC_RESPONSE_POLICY.md)

---

## Compliance Verification Example

Verify compliance programmatically:

```python
from nethical.security.regulatory_compliance import RegulatoryMappingGenerator

# Instantiate the mapping generator
generator = RegulatoryMappingGenerator()

# Generate and save the mapping reports
mapping = generator.generate_mapping_table()
print(f"Total requirements tracked: {mapping['metadata']['total_requirements']}")

# Generate markdown table for documentation
markdown_report = generator.generate_markdown_report()
```
