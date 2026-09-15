# HIPAA Business Associate Agreement (BAA) Template

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Regulatory Framework:** Health Insurance Portability and Accountability Act of 1996 ("HIPAA"), the Health Information Technology for Economic and Clinical Health Act ("HITECH Act"), 45 CFR Parts 160 and 164.

---

This Business Associate Agreement ("BAA") supplements the principal service agreement between:
1. **Covered Entity:** [Hospital / Healthcare Provider / Health Plan Name]
2. **Business Associate:** Nethical Platform Provider ("Business Associate")

---

## 1. Background & Purpose

Covered Entity wishes to utilize Business Associate’s AI safety and governance platform ("Services") to evaluate, audit, and safeguard artificial intelligence agents operating within healthcare clinical or administrative workflows.

In the course of providing Services, Business Associate may create, receive, maintain, or transmit Protected Health Information ("PHI") or electronic PHI ("ePHI") on behalf of Covered Entity.

---

## 2. Permitted Uses and Disclosures

1. **Services Delivery:** Business Associate may use or disclose PHI only as necessary to perform the AI safety governance services specified in the underlying agreement, or as required by law.
2. **De-Identification & Tokenization:** Business Associate is expressly authorized to use PHI to create de-identified data in accordance with 45 CFR §164.514(a)-(c) using the Nethical Reversible Token Vault and PII redactor. De-identified data ceases to be PHI.
3. **Internal Administration:** Business Associate may use PHI for the proper management and administration of Business Associate or to carry out its legal responsibilities.

---

## 3. Obligations and Activities of Business Associate

Business Associate agrees to:

1. **Safeguards (45 CFR §164.308, §164.312):** Implement administrative, physical, and technical safeguards that reasonably and appropriately protect the confidentiality, integrity, and availability of ePHI:
   - AES-256 encryption for ePHI at rest.
   - TLS 1.3 encryption for ePHI in transit.
   - Cryptographically anchored tamper-proof audit trails for all data access events.
2. **Breach Notification to Covered Entity (45 CFR §164.410):**
   - Notify Covered Entity in writing without unreasonable delay, and in no case later than **ten (10) business days** (well within the statutory 60-day window) after discovery of a confirmed Breach of Unsecured PHI.
3. **Subcontractors & Agents (45 CFR §164.504(e)(1)(i)):** Ensure that any subcontractors that create, receive, maintain, or transmit PHI agree in writing to the same restrictions and conditions that apply to Business Associate.
4. **Access and Amendment:** Within fifteen (15) days of a written request, provide access to PHI in a Designated Record Set to Covered Entity to satisfy 45 CFR §164.524.
5. **Accounting of Disclosures:** Maintain documentation of disclosures of PHI to respond to a request by Covered Entity for an accounting of disclosures (45 CFR §164.528).
6. **Audit by Secretary of HHS:** Make internal practices, books, and records relating to the use and disclosure of PHI available to the Secretary of Health and Human Services for purposes of determining Covered Entity’s compliance.

---

## 4. Term and Termination

1. **Term:** Effective as of the execution date of the underlying agreement and terminates when all PHI is destroyed or returned.
2. **Termination for Cause:** Covered Entity may immediately terminate this Agreement if Business Associate breaches a material term of this BAA and fails to cure within thirty (30) days.
3. **Effect of Termination:** Upon termination, Business Associate shall return or securely destroy all PHI in its possession using cryptographic sanitization methods (NIST SP 800-88 Rev. 1). If return or destruction is infeasible, protections of this BAA extend to retained data indefinitely.

---

## 5. Execution

| For Covered Entity: | For Business Associate (Nethical): |
|---|---|
| **Organization:** _________________________ | **Organization:** Nethical Enterprise Operations |
| **Name:** _________________________ | **Name:** Authorized Signatory |
| **Title:** _________________________ | **Title:** Chief Compliance Officer / DPO |
| **Date:** _________________________ | **Date:** 2026-09-15 |
