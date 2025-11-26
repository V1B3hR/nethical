# Regulatory Compliance Mapping Table

**Generated:** 2025-11-26T07:00:00.000000+00:00

**Total Requirements:** 60

## Summary by Framework

| Framework | Requirements |
|-----------|-------------|
| eu_ai_act | 14 |
| uk_gdpr | 7 |
| uk_dpa_2018 | 2 |
| uk_nhs_dspt | 5 |
| us_nist_ai_rmf | 8 |
| us_soc2 | 4 |
| iso_27001 | 20 |

## Compliance Status Summary

| Status | Count |
|--------|-------|
| implemented | 20 |
| pending_review | 40 |

## EU AI ACT

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| EU-AI-9.1 | Article 9 | Risk Management System | 🔄 | nethical/core/risk_engine.py, nethical/core/governance.py | tests/test_governance_features.py | docs/compliance/NIST_RMF_MAPPING.md |
| EU-AI-9.2 | Article 9 | Risk Identification and Analysis | 🔄 | nethical/core/anomaly_detector.py, nethical/core/ml_blended_risk.py | tests/test_anomaly_classifier.py | docs/security/threat_model.md |
| EU-AI-9.3 | Article 9 | Risk Mitigation Measures | 🔄 | nethical/core/quarantine.py, nethical/policy/engine.py | tests/test_phase3.py | docs/security/mitigations.md |
| EU-AI-10.1 | Article 10 | Training Data Governance | 🔄 | nethical/core/data_minimization.py, nethical/security/data_compliance.py | tests/test_privacy_features.py | docs/privacy/DPIA_template.md |
| EU-AI-10.2 | Article 10 | Data Quality and Bias Mitigation | 🔄 | nethical/core/fairness_sampler.py | tests/test_regionalization.py | governance/fairness_recalibration_report.md |
| EU-AI-11.1 | Article 11 | Technical Documentation | 🔄 | - | - | ARCHITECTURE.md, docs/API_USAGE.md... |
| EU-AI-12.1 | Article 12 | Automatic Logging | 🔄 | nethical/security/audit_logging.py, nethical/core/audit_merkle.py | tests/test_train_audit_logging.py | docs/AUDIT_LOGGING_GUIDE.md |
| EU-AI-13.1 | Article 13 | Transparency and Information | 🔄 | nethical/explainability/transparency_report.py, nethical/explainability/quarterly_transparency.py | tests/test_explainability/ | docs/transparency/ |
| EU-AI-13.2 | Article 13 | Instructions for Use | 🔄 | - | - | README.md, docs/API_USAGE.md |
| EU-AI-14.1 | Article 14 | Human Oversight Design | 🔄 | nethical/core/human_feedback.py, nethical/governance/human_review.py | tests/test_governance_features.py | docs/governance/governance_drivers.md |
| EU-AI-14.2 | Article 14 | Human Oversight Capabilities | 🔄 | nethical/explainability/decision_explainer.py | tests/test_advanced_explainability.py | docs/ADVANCED_EXPLAINABILITY_GUIDE.md |
| EU-AI-15.1 | Article 15 | Accuracy | 🔄 | nethical/governance/ethics_benchmark.py | tests/test_performance_benchmarks.py | docs/BENCHMARK_PLAN.md |
| EU-AI-15.2 | Article 15 | Robustness | 🔄 | nethical/security/ai_ml_security.py | tests/adversarial/ | docs/security/AI_ML_SECURITY_GUIDE.md |
| EU-AI-15.3 | Article 15 | Cybersecurity | 🔄 | nethical/security/threat_modeling.py, nethical/security/penetration_testing.py | tests/test_phase5_penetration_testing.py, tests/test_phase5_threat_modeling.py | docs/SECURITY_HARDENING_GUIDE.md |

## UK GDPR

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| UK-GDPR-5 | Article 5 | Principles of Processing | 🔄 | nethical/core/data_minimization.py, nethical/core/redaction_pipeline.py | tests/test_privacy_features.py | docs/privacy/DPIA_template.md |
| UK-GDPR-6 | Article 6 | Lawful Basis for Processing | 🔄 | nethical/security/data_compliance.py | tests/test_privacy_features.py | docs/privacy/DPIA_template.md |
| UK-GDPR-12-22 | Articles 12-22 | Data Subject Rights | 🔄 | nethical/security/data_compliance.py | tests/test_privacy_features.py | docs/privacy/DSR_runbook.md |
| UK-GDPR-25 | Article 25 | Data Protection by Design and Default | 🔄 | nethical/core/differential_privacy.py | tests/test_privacy_features.py | docs/F3_PRIVACY_GUIDE.md |
| UK-GDPR-32 | Article 32 | Security of Processing | 🔄 | nethical/security/encryption.py, nethical/security/auth.py | tests/test_security_hardening.py | docs/SECURITY_HARDENING_GUIDE.md |
| UK-GDPR-33-34 | Articles 33-34 | Breach Notification | 🔄 | nethical/security/soc_integration.py | tests/test_phase4_operational_security.py | docs/security/red_team_report_template.md |
| UK-GDPR-35 | Article 35 | Data Protection Impact Assessment | 🔄 | - | - | docs/privacy/DPIA_template.md |

## UK DPA 2018

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| UK-DPA-35 | Section 35 | Law Enforcement Processing Principles | 🔄 | nethical/policy/engine.py | tests/test_phase3.py | policies/common/data_classification.yaml |
| UK-DPA-64 | Section 64 | Automated Decision-Making | 🔄 | nethical/explainability/decision_explainer.py | tests/test_advanced_explainability.py | docs/EXPLAINABILITY_GUIDE.md |

## UK NHS DSPT

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| NHS-DSPT-1 | Standard 1 | Personal Confidential Data | 🔄 | nethical/core/redaction_pipeline.py | tests/test_privacy_features.py | docs/privacy/DPIA_template.md |
| NHS-DSPT-3 | Standard 3 | Security Training | 🔄 | - | - | docs/TRAINING_GUIDE.md |
| NHS-DSPT-7 | Standard 7 | Managing Access to Data and Systems | 🔄 | nethical/core/rbac.py, nethical/security/auth.py | tests/test_security_hardening.py | docs/security/SSO_SAML_GUIDE.md |
| NHS-DSPT-8 | Standard 8 | Unsupported Systems | 🔄 | - | - | docs/SUPPLY_CHAIN_SECURITY_GUIDE.md |
| NHS-DSPT-10 | Standard 10 | Accountable Suppliers | 🔄 | - | - | docs/SUPPLY_CHAIN_SECURITY_GUIDE.md, SBOM.json |

## US NIST AI RMF

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| NIST-RMF-GV1 | GOVERN 1 | Governance Policies and Procedures | 🔄 | nethical/core/governance.py, nethical/policy/engine.py | tests/test_governance_features.py | docs/governance/governance_drivers.md |
| NIST-RMF-GV2 | GOVERN 2 | Accountability Structures | 🔄 | nethical/governance/human_review.py | tests/test_integrated_governance.py | docs/compliance/NIST_RMF_MAPPING.md |
| NIST-RMF-MP1 | MAP 1 | Context Establishment | 🔄 | nethical/core/fairness_sampler.py | tests/test_regionalization.py | docs/REGIONAL_DEPLOYMENT_GUIDE.md |
| NIST-RMF-MP3 | MAP 3 | AI Capabilities and Limitations | 🔄 | nethical/explainability/decision_explainer.py | tests/test_advanced_explainability.py | docs/EXPLAINABILITY_GUIDE.md |
| NIST-RMF-MS1 | MEASURE 1 | Risk Measurement | 🔄 | nethical/core/risk_engine.py, nethical/core/ml_blended_risk.py | tests/test_performance_benchmarks.py | docs/BENCHMARK_PLAN.md |
| NIST-RMF-MS2 | MEASURE 2 | AI Testing | 🔄 | nethical/governance/ethics_benchmark.py | tests/adversarial/, tests/test_performance_benchmarks.py | docs/Benchmark_plan.md |
| NIST-RMF-MG1 | MANAGE 1 | Risk Response | 🔄 | nethical/core/quarantine.py | tests/test_phase3.py | docs/security/mitigations.md |
| NIST-RMF-MG3 | MANAGE 3 | Incident Response | 🔄 | nethical/security/soc_integration.py | tests/test_phase4_operational_security.py | docs/security/red_team_report_template.md |

## US SOC2

| ID | Article | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| SOC2-CC6.1 | CC6.1 | Logical Access Security | 🔄 | nethical/core/rbac.py, nethical/security/auth.py | tests/test_security_hardening.py | docs/security/SSO_SAML_GUIDE.md |
| SOC2-CC6.6 | CC6.6 | Security Monitoring | 🔄 | nethical/security/anomaly_detection.py | tests/test_phase4_operational_security.py | docs/security/threat_model.md |
| SOC2-CC7.1 | CC7.1 | Change Management | 🔄 | nethical/core/policy_diff.py, nethical/policy/release_management.py | tests/test_train_governance.py | docs/versioning.md |
| SOC2-CC7.4 | CC7.4 | Incident Response | 🔄 | nethical/security/soc_integration.py | tests/test_phase4_operational_security.py | docs/security/red_team_report_template.md |

## ISO/IEC 27001:2022 (Key Annex A Controls)

| ID | Control | Title | Status | Code Modules | Tests | Docs |
|----|---------|-------|--------|--------------|-------|------|
| A.5.1 | A.5.1 | Policies for Information Security | ✅ | nethical/policy/engine.py | tests/test_phase3.py | docs/compliance/isms/information_security_policy.md |
| A.5.7 | A.5.7 | Threat Intelligence | ✅ | nethical/security/threat_modeling.py | tests/test_phase5_threat_modeling.py | docs/security/threat_model.md |
| A.5.9 | A.5.9 | Inventory of Assets | ✅ | - | - | SBOM.json, docs/compliance/isms/asset_register.md |
| A.5.12 | A.5.12 | Classification of Information | ✅ | nethical/security/data_compliance.py | tests/test_privacy_features.py | policies/common/data_classification.yaml |
| A.5.15 | A.5.15 | Access Control | ✅ | nethical/core/rbac.py, nethical/security/auth.py | tests/test_security_hardening.py | docs/security/SSO_SAML_GUIDE.md |
| A.5.24 | A.5.24 | Incident Management | ✅ | nethical/security/soc_integration.py | tests/test_phase4_operational_security.py | docs/compliance/INCIDENT_RESPONSE_POLICY.md |
| A.5.28 | A.5.28 | Collection of Evidence | ✅ | nethical/security/audit_logging.py, nethical/core/audit_merkle.py | tests/test_train_audit_logging.py | docs/AUDIT_LOGGING_GUIDE.md |
| A.5.34 | A.5.34 | Privacy and PII Protection | ✅ | nethical/core/redaction_pipeline.py, nethical/core/differential_privacy.py | tests/test_privacy_features.py | docs/F3_PRIVACY_GUIDE.md |
| A.6.3 | A.6.3 | Security Training | ✅ | - | - | docs/TRAINING_GUIDE.md |
| A.8.2 | A.8.2 | Privileged Access Rights | ✅ | nethical/core/rbac.py | tests/test_security_hardening.py | docs/security/SSO_SAML_GUIDE.md |
| A.8.5 | A.8.5 | Secure Authentication | ✅ | nethical/security/mfa.py, nethical/security/sso.py | tests/test_security_hardening.py | docs/security/MFA_GUIDE.md |
| A.8.7 | A.8.7 | Protection Against Malware | ✅ | nethical/detectors/, nethical/security/input_validation.py | tests/adversarial/ | docs/security/AI_ML_SECURITY_GUIDE.md |
| A.8.8 | A.8.8 | Technical Vulnerability Management | ✅ | nethical/security/penetration_testing.py | tests/test_phase5_penetration_testing.py | docs/SUPPLY_CHAIN_SECURITY_GUIDE.md |
| A.8.11 | A.8.11 | Data Masking | ✅ | nethical/core/redaction_pipeline.py | tests/test_privacy_features.py | docs/F3_PRIVACY_GUIDE.md |
| A.8.15 | A.8.15 | Logging | ✅ | nethical/security/audit_logging.py, nethical/core/audit_merkle.py | tests/test_train_audit_logging.py | docs/AUDIT_LOGGING_GUIDE.md |
| A.8.16 | A.8.16 | Monitoring Activities | ✅ | nethical/monitors/, nethical/observability/ | tests/test_observability.py | docs/SEMANTIC_MONITORING_GUIDE.md |
| A.8.24 | A.8.24 | Use of Cryptography | ✅ | nethical/security/encryption.py, nethical/security/quantum_crypto.py | tests/test_security_hardening.py | docs/security/QUANTUM_CRYPTO_GUIDE.md |
| A.8.25 | A.8.25 | Secure Development Lifecycle | ✅ | - | tests/adversarial/ | docs/SUPPLY_CHAIN_SECURITY_GUIDE.md |
| A.8.32 | A.8.32 | Change Management | ✅ | nethical/core/policy_diff.py, nethical/policy/release_management.py | tests/test_train_governance.py | docs/compliance/isms/change_management_policy.md |

**Full ISO 27001 Mapping**: See [docs/compliance/ISO_27001_ANNEX_A_MAPPING.md](./ISO_27001_ANNEX_A_MAPPING.md)

## Cross-Reference Matrix

This matrix shows which controls satisfy multiple frameworks.

| Category | EU AI Act | UK GDPR | UK DPA 2018 | NHS DSPT | NIST AI RMF | SOC2 | ISO 27001 |
|----------|-----------|---------|-------------|----------|-------------|------|-----------|
| transparency | 2 | - | 1 | - | 1 | - | 1 |
| explainability | - | - | - | - | - | - | - |
| human_oversight | 2 | - | - | - | 1 | - | - |
| data_governance | 1 | - | 1 | 2 | - | 1 | 2 |
| risk_management | 4 | 1 | - | - | 5 | - | 2 |
| technical_documentation | 1 | - | - | - | - | - | 1 |
| conformity_assessment | - | - | - | - | - | - | 1 |
| incident_response | - | 1 | - | - | 1 | 1 | 2 |
| audit_logging | 1 | - | - | - | - | - | 2 |
| access_control | - | - | - | 1 | - | 1 | 3 |
| security | 2 | 1 | - | 2 | - | 1 | 5 |
| privacy | - | 4 | - | - | - | - | 2 |
| fairness | 1 | - | - | - | - | - | - |
| cryptography | - | - | - | - | - | - | 2 |
| change_management | - | - | - | - | - | 1 | 1 |

---
Last Updated: 2025-11-26
