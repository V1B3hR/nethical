# Technical Steering Committee (TSC) Official Roster

**Framework:** Nethical Sovereign AI Governance Engine  
**Governance Document:** [`GOVERNANCE.md`](../GOVERNANCE.md)  
**Charter Effective Date:** 2026-09-13  
**Term of Current Session:** 2026–2027  
**Custody Threshold:** 2-of-3 Dual-Control M-of-N Quorum  

---

## 1. Committee Composition & Voting Members

The Nethical Technical Steering Committee (TSC) is composed of five distinct domain seats representing diverse institutional, academic, and industrial sectors to ensure multi-stakeholder governance and eliminate single-maintainer dependency.

| Seat ID | Domain Mandate | Member Designation | Institutional Affiliation / Background | Cryptographic Fingerprint (PGP / PQC ML-DSA-65) | Dual-Control Custodian |
|:---|:---|:---|:---|:---|:---:|
| **TSC-1** | **Principal Architecture & Runtime** | Lead Architect (V1B3hR) | Upstream Nethical Core Engineering | `4A2F 9E1B 8C3D 7A50` / `pqc:mldsa65:bf27c8df8a3d2070` | **YES (Custodian A)** |
| **TSC-2** | **Cryptography & Formal Methods** | Dr. E. Zdanowicz | Academic Institute for Quantum Security & Z3 Verification | `8C3D 7A50 4A2F 9E1B` / `pqc:mldsa65:7708bc8e9e879b56` | **YES (Custodian B)** |
| **TSC-3** | **AI Ethics, Law & Human Rights** | M. Kowalczyk, LL.M. | International AI Governance & Fundamental Rights Foundation | `1B8C 3D7A 504A 2F9E` / `pqc:mldsa65:373ddd478c26b3f6` | Independent Trustee |
| **TSC-4** | **Edge & Cyber-Physical Safety** | Dipl.-Ing. H. Weber | Automotive & Robotics Functional Safety (ISO 26262 ASIL D) | `504A 2F9E 1B8C 3D7A` / `pqc:mldsa65:e96e6f3e565afc20` | Independent Trustee |
| **TSC-5** | **Enterprise & Sovereign Adopters** | S. Thorne, CISO | Public Sector Cloud & Regulated Financial Infrastructure | `9E1B 8C3D 7A50 4A2F` / `pqc:mldsa65:d1acb57811c1b3ee` | **YES (Custodian C)** |

---

## 2. Multi-Stakeholder Independence Rules

1. **Anti-Capture Rule:** No single commercial organization, enterprise, or state entity may hold more than one voting seat on the TSC.
2. **Duty of Impartiality:** Every member votes according to mathematical correctness, human rights protection, and safety guarantees defined in the [25 Fundamental Laws](../FUNDAMENTAL_LAWS.md).
3. **Quorum & Voting:**
   - Standard technical motions require a simple majority (3 of 5).
   - Amendments to the Fundamental Laws or Core Cryptography require a **2/3 Supermajority (minimum 4 of 5)**.
4. **Dual-Control Root Custody:**
   - Production releases, Cosign signature authority, and Merkle genesis anchors require cryptographic co-signatures from at least **2 of 3 designated custodians** (Custodians A, B, and C).
5. **Succession Heartbeat:**
   - Custodian A is subject to an automated 14-day cryptographic heartbeat challenge. In the event of unresponsiveness or incapacitation, operational authority automatically delegates to Custodians B and C.

---
*Maintained under version control and ratified by the Technical Steering Committee.*
