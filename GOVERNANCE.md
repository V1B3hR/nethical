# 🏛️ Nethical Sovereign AI Governance Charter (GOVERNANCE.md)

**Version:** 2.7.0  
**Effective Date:** 2026-09-13  
**Status:** Adopted by Core Maintainers & Technical Steering Committee (TSC)  
**Applicability:** Upstream Nethical Framework, Edge Governance Runtimes, and Federated Trust Anchors  

---

## 1. Executive Summary & Purpose

Nethical provides sovereign, post-quantum verifiable, and real-time ethical guardrails for mission-critical artificial intelligence across enterprise, healthcare, aerospace, defense, and public administration.

To eliminate single-point-of-failure vulnerabilities (the "bus factor"), establish institutional trust, and ensure mathematical immutability for the **25 Fundamental Laws of AI Safety**, this Governance Charter establishes:
1. A multi-stakeholder **Technical Steering Committee (TSC)**.
2. A formal **RFC (Request for Comments) Amendment Process** replacing informal review mechanisms.
3. A **Dual-Control M-of-N Cryptographic Key Custody Protocol** for release signing and trust anchoring.
4. An automated **Maintainer Succession and Disaster Recovery Protocol**.
5. A **Security Response Team (SRT)** operating under strict, binding SLAs.

---

## 2. Technical Steering Committee (TSC)

The Technical Steering Committee is the supreme technical authority of the Nethical project. It is structured to balance technological velocity with rigorous institutional oversight.

### 2.1 Seat Distribution (5 Voting Members)

| Seat | Domain | Representation | Mandate |
|:---|:---|:---|:---|
| **TSC-1** | **Principal Architecture** | Core Maintainer / Founder | Framework roadmap, IPC runtime, microsecond kernel performance |
| **TSC-2** | **Cryptography & Formal Methods** | Academic / Cryptographic Institute | PQC (NIST FIPS 204 ML-DSA-65), Z3 SMT solver invariants, Merkle-DAG proofs |
| **TSC-3** | **Ethics, Law & Human Rights** | Independent Legal Scholar / NGO | EU AI Act, GDPR, UN Universal Declaration of Human Rights, HIPAA |
| **TSC-4** | **Edge & Cyber-Physical Safety** | Automotive / Robotics Representative | ISO 26262 (ASIL D), ISO 13849, Fieldbus E-STOP, Hardware Watchdogs |
| **TSC-5** | **Enterprise & Public Sector** | Sovereign Adopter Representative | AIMS (ISO/IEC 42001), NATO AI Strategy, Public Procurement Standards |

### 2.2 Decision-Making & Quorum
- **Ordinary Technical Decisions**: Simple majority (>50%) of active TSC members.
- **Constitutional Amendments (Fundamental Laws, Cryptographic Baselines)**: Binding **2/3 Supermajority** (minimum 4 of 5 affirmative votes).
- **Emergency Security Patches**: Expedited approval by 2 TSC members including at least one security custodian.

---

## 3. Formal RFC Process for Fundamental Laws & Core Policies

Changes affecting the interpretation, enforcement, or deontological constraints of the **25 Fundamental Laws** cannot be made unilaterally or via informal pull request commentary. They must strictly follow the **5-Phase Governance RFC Pipeline**:

```
[ Phase 1: Invariant Specification ]
                │
                ▼
[ Phase 2: 45-Day Institutional Comment Period ]
                │
                ▼
[ Phase 3: Automated Z3 SMT Mathematical Verification ]
                │
                ▼
[ Phase 4: 2/3 TSC Supermajority Roll-Call Vote ]
                │
                ▼
[ Phase 5: Multi-Sig Cryptographic Ratification & Merkle Anchor ]
```

### 3.1 Phase Breakdown
1. **Phase 1 — Formal Proposal (RFC Submission)**:
   - Author submits an RFC to `governance/rfcs/RFC-XXXX-<title>.md`.
   - Must include: Technical Motivation, Regulatory Mapping (EU AI Act, ISO 42001), Threat Model, and Mathematical Deontological Specification.
2. **Phase 2 — Institutional & Public Review (45 Days)**:
   - Mandatory 45-day review open to enterprise users, government certifiers, and academic researchers.
   - At least two formal hearings hosted by the TSC.
3. **Phase 3 — Formal Verification & Non-Regression Analysis**:
   - The proposed policy must be processed by Nethical's automated Z3 SMT formal solver (`nethical/formal/`).
   - The solver must generate a mathematical proof verifying that the proposed amendment **does not weaken, contradict, or invalidate** any existing Fundamental Law or safety invariant.
4. **Phase 4 — Supermajority Roll-Call**:
   - TSC conducts a recorded public roll-call vote.
   - Requires at least 4 of 5 affirmative votes.
5. **Phase 5 — Cryptographic Genesis Anchoring**:
   - The ratified policy hash is signed via Dual-Control Multi-Sig and anchored into the Post-Quantum Merkle-DAG ledger.

---

## 4. Dual-Control Key Custody & Bus-Factor Elimination

To guarantee business continuity and eliminate risks associated with individual maintainer dependency:

### 4.1 M-of-N Threshold Key Custody (2-of-3 Quorum)
All production release artifacts, container image digests (Cosign), and PQC root authorities are controlled under an **M-of-N threshold scheme (2-of-3)**:
- **Key Share 1:** Upstream Project Lead (Physical FIPS 140-3 Hardware Token).
- **Key Share 2:** Institutional Trustee (Independent Academic/Foundation Security Enclave).
- **Key Share 3:** Third-Party Security Auditor (Accredited Certification Body Escrow).

No single individual possesses the capability to alter release binaries, issue unsigned security updates, or compromise the post-quantum ledger root.

### 4.2 Automated Succession & Failover Protocol
In the event that the primary maintainer is incapacitated, unreachable, or unresponsive for more than **14 calendar days**:
1. An automated cryptographic challenge is issued to the primary maintainer's registered hardware security keys.
2. If unrefuted after 14 days, operational maintainership and release signing authority automatically fail over to Custodian 2 and Custodian 3.
3. An emergency TSC session is convened within 48 hours to appoint an interim operational lead, ensuring zero downtime for dependent government and enterprise deployments.

---

## 5. Security Response Team (SRT) Charter & Binding SLAs

The Nethical Security Response Team is responsible for vulnerability ingestion, triage, patch engineering, and coordinated disclosure.

### 5.1 Service Level Agreements (SLAs)

| Severity Level | CVSS v3.1 Range | Initial Triage | Patch Availability | Public Advisory |
|:---|:---|:---|:---|:---|
| **Critical** (e.g., CVE-2026-26007 class) | 9.0 – 10.0 | **< 24 Hours** | **< 72 Hours** | Within 7 days |
| **High** | 7.0 – 8.9 | < 48 Hours | < 7 Days | Within 14 days |
| **Medium** | 4.0 – 6.9 | < 5 Business Days | < 21 Days | Next regular release |
| **Low** | 0.1 – 3.9 | < 10 Business Days | < 45 Days | Next regular release |

### 5.2 Mandatory Automated Dependency Policy
- All core production dependencies must have an active vulnerability scanning policy (Dependabot + Trivy + automated AST curve audits).
- Any dependency possessing a published critical CVE with available upstream patch must be merged and released within the 72-hour SLA.
- Release manifests must include reproducible SPDX/CycloneDX SBOMs and SLSA Level 3 provenance attestations.

---

---

## 7. Conformity Assessment Status & Third-Party Certification Disclosure

> [!IMPORTANT]
> **Regulatory Notice on Third-Party Certifications (Shared Responsibility):**
> - **Self-Assessment & Automated Evidence:** Nethical provides automated algorithmic evidence generation, formal Z3 SMT constraint proofs, and compliance mapping packages for international standards (including EU AI Act Annex IV, ISO/IEC 42001:2023, ISO/IEC 27001, and NIST AI RMF 1.0).
> - **Independent Audit Required:** Nethical is an open-source software framework and automated evidence collector; it **does not hold standalone accredited third-party certifications**.
> - **Accredited Assessment Bodies:** Formal certification for regulated deployments (e.g., CE mark for High-Risk AI systems under the EU AI Act, accredited ISO/IEC 42001 certification, or SOC 2 Type II reports) requires independent evaluation by an accredited Conformity Assessment Body (Notified Body or independent CPA firm). Nethical's tools are designed to streamline and satisfy the technical verification requirements of such audits. See [`audit/INDEPENDENT_AUDIT_AND_GAP_ASSESSMENT.md`](audit/INDEPENDENT_AUDIT_AND_GAP_ASSESSMENT.md) and [`DISCLAIMER.md`](DISCLAIMER.md).

---

## 8. Maintainer Transparency & Open Foundation Transition Roadmap

To eliminate the "bus factor" and transition from a project-lead origin to an open, decentralized stewardship model:
1. **Current Stewardship:** Founded by Lead Architect (`V1B3hR`), Nethical operates under the formal 5-seat Technical Steering Committee defined in [Section 2](#2-technical-steering-committee-tsc) and detailed with cryptographic signatures in [`governance/TSC_ROSTER.md`](governance/TSC_ROSTER.md).
2. **Dual-Control Quorum:** Releases and cryptographic trust roots require co-signing by 2 of 3 designated custodians across independent operational, academic, and enterprise seats.
3. **Open Foundation Horizon (2026–2027):** The project is preparing for formal donation to an established open-source consortium (e.g., Linux Foundation / LF AI & Data or Apache Software Foundation) to establish neutral, non-profit stewardship for sovereign AI safety standards worldwide.

---

*This charter is maintained under cryptographic version control in the root of the Nethical repository as `GOVERNANCE.md`.*

