# EU AI Database Registration Guide (Article 49 Compliance)

**Document ID:** EU-AIDB-REG-001  
**Applicability:** Regulation (EU) 2024/1689 (EU AI Act) Articles 49, 71, and Annex VIII  
**Effective Mandate:** Mandatory for High-Risk AI Systems placed on the EU Market from August 2026  

---

## 1. Regulatory Context

Under **Article 49 of the EU AI Act**, before placing a high-risk AI system on the market or putting it into service (or before utilizing a high-risk AI system listed in Annex III by public authorities/deployers), the provider or deployer must register the system in the **EU Database for High-Risk AI Systems** maintained by the European Commission.

This document provides deployers and providers of Nethical-governed AI systems with a structured step-by-step guide to complete this statutory registration.

---

## 2. Who Must Register?

1. **Providers (Article 49(1)):**
   - Entities placing high-risk AI systems on the EU market under their own name or trademark.
2. **Authorized Representatives (Article 22):**
   - Where the provider is established outside the European Union, the designated EU Authorized Representative registers the system.
3. **Deployers (Article 49(2)):**
   - Deployers of high-risk AI systems who are public authorities, European Union institutions, bodies, offices, or agencies.

---

## 3. Mandatory Information Required (Annex VIII)

The EU AI Database schema requires the following standardized fields. Nethical provides pre-filled references for these sections:

| Section | Database Field | Nethical Repository Source |
|---|---|---|
| **1. Identity** | Provider Name, Address, Contact | See `docs/compliance/conformity_assessment/EU_AI_Act_Conformity_Assessment.md` Section 2 |
| **2. System Name** | Commercial Trade Name & Version | Nethical AI Governance Platform v2.7.x |
| **3. Intended Purpose** | Clear definition of intended operation & prohibited contexts | See `DISCLAIMER.md` & `ACCEPTABLE_USE_POLICY.md` |
| **4. Classification** | Specific Annex III point triggering high-risk status | See `docs/compliance/AI_IMPACT_ASSESSMENT_TEMPLATE.md` |
| **5. Conformity Path** | Internal Control (Annex VI) or Notified Body Assessment (Annex VII) | Conformity Assessment CA-EUAIA-001 |
| **6. Declaration of Conformity** | EU Declaration of Conformity copy (Annex V) | See `docs/compliance/conformity_assessment/` |
| **7. Instructions for Use** | User manual and technical parameters | `docs/` and `docs/laws_and_policies/` |
| **8. Basic Model Info** | Foundation models / architectures governed | Model Cards in `docs/models/` |

---

## 4. Step-by-Step Registration Procedure

### Step 1: ECAS / EU Login Authentication
1. Obtain an official **EU Login** account with two-factor authentication.
2. Link your EU Login to your organization’s **EUID** (European Unique Identifier) or national commercial register entry.

### Step 2: Accessing the EU High-Risk AI Database Portal
1. Navigate to the official European Commission AI Database portal (portal link provided by the European AI Office).
2. Create or claim your organization’s Provider Profile.

### Step 3: Entering System Technical Specifications
1. Populate Section A (System Identification).
2. Attach electronic copies of:
   - EU Declaration of Conformity (duly signed).
   - Summary of Technical Documentation (Annex IV).
   - Fundamental Rights Impact Assessment (where required by Art. 27).

### Step 4: Verification & Unique Registration Number (URN)
1. Upon submission, the European AI Office generates a unique registration number (URN).
2. Record the URN in your internal Quality Management System (QMS) and in the local Nethical governance configuration:
   ```bash
   export NETHICAL_EU_AI_URN="EU-HR-2026-XXXXX"
   ```

---

## 5. Ongoing Maintenance & Substantial Modifications

- **Substantial Modification (Art. 3(23)):** If any change affects safety or compliance, the registration must be updated **prior** to deploying the modified version.
- **Serious Incidents:** If a serious incident occurs, notification must link directly to the system's URN within **72 hours** (Art. 73).
- **Annual Confirmation:** Review and confirm active deployment status annually.

For assistance with registration compliance, contact: `compliance@nethical.ai`.
