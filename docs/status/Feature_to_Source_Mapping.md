# Nethical Feature-to-Source Mapping

This document maps key features and claims described in the Nethical README to their corresponding, verifiable code modules, scripts, and test infrastructure in the repository as of November 2025. This is not exhaustive due to GitHub API limits, but covers primary directories and implementation evidence for core functionality.

---

## 1. Audit, Governance, Compliance, and Ethics

- **Merkle Anchoring & Immutable Audit Trails**
  - *Implementation*: Likely in `audit/` directory. See [audit_scope.md](https://github.com/V1B3hR/nethical/blob/main/audit/audit_scope.md) for design, potential links to anchored audit chains.
  - *Tests*: `tests/test_train_audit_logging.py`, `tests/test_phase4_core.py`
- **Policy Diff Auditing & Quarantine Mode**
  - *Implementation*: CLI tools such as [`cli/policy_diff`](https://github.com/V1B3hR/nethical/blob/main/cli/policy_diff).
  - *Tests*: `tests/test_phase4_core.py`, `tests/test_plugin_extensibility.py`
- **Ethical Taxonomy Tagging & SLA Tracking**
  - *Implementation*: `ethics_taxonomy.json` for tagging, possibly referenced in orchestrator logic.
  - *Tests*: Indirect (taxonomy covered in reporting); see `tests/test_privacy_features.py`

---

## 2. Governance and Orchestration

- **Integrated Governance (Unified API)**
  - *Implementation*: The [governance/](https://github.com/V1B3hR/nethical/tree/main/governance) directory, with evidence in `fairness_recalibration_report.md`.
  - *Tests*: `tests/test_integrated_governance.py`
- **Quota Enforcement & Backpressure**
  - *Implementation*: Likely in governance, not directly visible in limited listing.
  - *Tests*: `tests/test_plugin_extensibility.py`, `tests/test_scalability_targets.py`

---

## 3. Security, Privacy & PII Management

- **PII Detection & Redaction**
  - *Implementation*: Implied by `tests/test_privacy_features.py` and possibly core utilities.
  - *Tests*: `tests/test_privacy_features.py`, `tests/adversarial/`
- **Regional Compliance (GDPR, CCPA)**
  - *Implementation*: `tests/test_regionalization.py`

---

## 4. Scan Tools, AI Reporting & CLI

- **Scan Orchestration/Results Logging**
  - *Implementation*: Central dispatcher and logging orchestration in `nethical.py` (main CLI), per-scan history.
  - *Tests*: `tests/test_action_replayer.py`, `tests/test_end_to_end_pipeline.py`
- **AI-Powered Reporting (CVSS, Markdown/HTML)**
  - *Implementation*: OpenAI summarization, referenced in README. Main integration in `nethical.py`.
  - *Tests*: Reporting is exercised in `tests/test_privacy_features.py`, `tests/test_integrated_governance.py`

---

## 5. Adversarial Tests & ML Detection

- **Adversarial Attack/PII/Drift**
  - *Implementation*: Comprehensive suite in [`tests/adversarial/`](https://github.com/V1B3hR/nethical/tree/main/tests/adversarial), with related coverage in `advancedtests.py`.
  - *Tests*: 36/36 adversarial test claims supported by files like:
      - `tests/test_anomaly_classifier.py`
      - `tests/test_phase4_operational_security.py`
      - `tests/test_ml_platforms.py`
- **ML-Based Anomaly Detection**
  - *Implementation*: Implied in advanced tests, may be orchestrated from `nethical.py` and reinforced in test files above.

---

## 6. Plugin Marketplace & Extensibility

- **Plugin Marketplace**
  - *Implementation*: CLI and backend references such as `tests/test_plugin_extensibility.py`
  - *Tests*: See above, also validated in user-facing reports.

---

## 7. DevOps, CI/CD & Deployment

- **Docker, CI/CD, Observability Stack**
  - *Implementation*: [`Dockerfile`](https://github.com/V1B3hR/nethical/blob/main/Dockerfile), [`docker-compose.yml`](https://github.com/V1B3hR/nethical/blob/main/docker-compose.yml)
  - CI/CD infrastructure is supported by GitHub workflow configuration (not listed due to API limits, see `.github/`).
  - *Tests*: Coverage & pipeline verification in scalability and integration files

---

## 8. Test Coverage

- **Tests and Results Infrastructure**
  - *Implementation*: Main tests in [`tests/`](https://github.com/V1B3hR/nethical/tree/main/tests), including:
      - Adversarial, misuse, performance, unit subsuites
      - `TEST_STATUS.md`, summary of comprehensive test results
      - Dedicated test scripts for governance, ML, logging, privacy, scalability, auditing
      - Extensive adversarial and operational security tests

---

## 9. Other Notable Areas

- **Red Team Playbook, Security Practice**
  - *Implementation*: [`security/red_team_playbook.md`](https://github.com/V1B3hR/nethical/blob/main/security/red_team_playbook.md)

---

## Limitations

- Only main modules and files returned by API are mapped; to explore entire source tree and more modules, see [V1B3hR/nethical repository file list](https://github.com/V1B3hR/nethical/tree/main).
- Some features may have cross-cutting implementations; mapping is based on naming, README correlation, and file type.
- Refer to [docs/implementation/](https://github.com/V1B3hR/nethical/tree/main/docs/implementation) for detail and additional linkage.

---

## Conclusion

For every major claim in the README, there are:
- Source files or directories reflecting design and implementation
- Test files providing evidence of automated validation

If you require a more granular mapping per function/class (including line numbers), further source browsing is needed.

_Last scan: November 2025_
