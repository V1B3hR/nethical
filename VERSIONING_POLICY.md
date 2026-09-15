# Nethical Versioning & API Stability Policy

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Standards:** Semantic Versioning 2.0.0 (SemVer), Keep a Changelog, ISO/IEC 42001 Clause 8.4  

---

## 1. Scope & Objective

Enterprise deployers integrate Nethical into safety-critical pipelines, mission-critical infrastructure, and regulated high-risk AI deployments. Contractual stability and predictable release management are paramount.

This document formalizes the versioning scheme, deprecation guarantees, and lifecycle support windows for the Nethical platform and its SDKs.

---

## 2. Semantic Versioning (SemVer 2.0.0)

Nethical releases adhere strictly to the `MAJOR.MINOR.PATCH` format:

```
v{MAJOR}.{MINOR}.{PATCH}
```

### A. MAJOR version (e.g., v2.0.0 → v3.0.0)
- Incremented when **incompatible API changes** or backward-incompatible protocol modifications are introduced.
- Examples: Changing signatures of `SafetyGovernance.evaluate()`, modifying the cryptographic structure of the Merkle-DAG ledger, altering fundamental schema definitions in `AgentAction`.
- Requires formal RFC and deprecation period (see Section 3).

### B. MINOR version (e.g., v2.7.0 → v2.8.0)
- Incremented when functionality is added in a **backward-compatible manner**.
- Examples: Adding new regulatory compliance packs (e.g., new national standards), new adversarial detector heuristics, performance optimizations, non-breaking CLI commands.

### C. PATCH version (e.g., v2.7.0 → v2.7.1)
- Incremented when **backward-compatible bug fixes or security patches** are introduced.
- Examples: Fixing false positives in safety filters, addressing CVEs in dependencies, updating documentation.

---

## 3. Deprecation Timeline & Breaking Change Process

To protect enterprise deployments from unexpected breakage:

1. **Deprecation Warning Notice:**
   - Any public API marked for removal must first emit a programmatic `DeprecationWarning` or `FutureWarning` in at least **one full MINOR release series** prior to removal.
2. **Minimum Grace Period:**
   - Deprecated interfaces will remain functional for a minimum of **six (6) months** following the release where deprecation was announced.
3. **Migration Guides:**
   - Every MAJOR release containing breaking changes must be accompanied by an automated migration script or detailed upgrade guide in `docs/migration/`.

---

## 4. Long-Term Support (LTS) & Maintenance Lifecycle

| Branch / Release | Classification | Active Support Window | Security Patch Window |
|---|---|---|---|
| **v2.x (Current)** | **Production / LTS** | Full feature & regulatory updates | 24 Months |
| **v1.x** | **Maintenance** | Critical security vulnerabilities only | 12 Months |
| **Pre-v1.0** | **End of Life (EOL)** | None | None |

---

## 5. Security & Substantial Modification (EU AI Act Art. 3(23))

Under Article 3(23) of the EU AI Act, a change that significantly affects the compliance of a high-risk AI system constitutes a "substantial modification" requiring a new conformity assessment.

- **Patch & Minor Updates:** Designed to maintain existing compliance boundaries without triggering the "substantial modification" threshold.
- **Major Architecture Changes:** Clearly flagged with regulatory impact notes to allow deployers to update their EU AI Database entries and internal QMS documentation.
