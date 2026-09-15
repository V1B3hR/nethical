# Data Processing Agreement (DPA) Template

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Governing Regulation:** General Data Protection Regulation (EU) 2016/679 ("GDPR"), UK GDPR, and California Consumer Privacy Act ("CCPA")

---

This Data Processing Agreement ("DPA") supplements the Nethical Terms of Service or Enterprise Master Services Agreement entered into by and between:

1. **The Customer / Deployer** acting as a **Data Controller** (or Data Processor on behalf of a third party); and
2. **Nethical Platform / Provider** acting as a **Data Processor** (or Sub-Processor).

---

## 1. Scope and Processing Details

- **Subject Matter:** Provision of AI safety monitoring, prompt filtering, behavioral boundary governance, and Merkle-DAG audit logging services.
- **Duration of Processing:** Duration of Customer's active subscription, license agreement, or API service term.
- **Nature and Purpose of Processing:** Evaluating agent actions, identifying security/ethics violations, anonymizing/tokenizing sensitive data, and producing non-repudiable audit trails.
- **Categories of Data Subjects:** Customer employees, authorized users, and individuals interacting with Customer's governed AI agents.
- **Types of Personal Data:** User identifiers, system logs, prompt inputs, model responses, IP addresses, and cryptographic telemetry.

---

## 2. Processor Obligations (GDPR Article 28)

The Processor agrees that it shall:

1. **Documented Instructions:** Process personal data only on documented instructions from the Controller, including with regard to transfers of personal data to a third country or an international organization, unless required to do so by Union or Member State law.
2. **Confidentiality:** Ensure that persons authorized to process the personal data have committed themselves to confidentiality or are under an appropriate statutory obligation of confidentiality.
3. **Security Measures (Article 32):** Implement technical and organizational measures (TOMs) including AES-256 encryption at rest, TLS 1.3 in transit, role-based access control (RBAC), post-quantum digital signatures, and continuous penetration testing.
4. **Sub-Processors:** Not engage another processor without prior specific or general written authorization of the Controller (see Section 4).
5. **Assistance with Data Subject Rights:** Assist the Controller by appropriate technical and organizational measures in fulfilling Controller's obligations to respond to requests exercising data subject rights (Chapter III GDPR).
6. **Assistance with Compliance (Articles 32–36):** Assist the Controller in ensuring compliance with security obligations, breach notifications, and Data Protection Impact Assessments (DPIAs / FRIAs under EU AI Act Art. 27).
7. **Data Return and Deletion:** At the choice of the Controller, delete or return all personal data after the end of the provision of services, and delete existing copies unless Union or Member State law requires storage.
8. **Audit Rights:** Make available to the Controller all information necessary to demonstrate compliance with Article 28 and allow for and contribute to audits, including inspections, conducted by the Controller or another auditor mandated by the Controller.

---

## 3. Security Incident & Breach Notification SLA

1. **72-Hour Notification:** In the event of a confirmed Personal Data Breach impacting Customer Personal Data, Processor shall notify Customer without undue delay and, where feasible, not later than **48 hours** (and strictly within statutory **72 hours**) after becoming aware of the breach.
2. **Breach Details:** The notification shall describe the nature of the breach, affected categories and approximate numbers of data subjects, contact details of the Data Protection Officer, likely consequences, and remediation measures taken or proposed.

---

## 4. Sub-Processors

1. **Sub-Processor Register:** Processor maintains an up-to-date register of all authorized sub-processors at [`docs/compliance/SUB_PROCESSOR_REGISTER.md`](../docs/compliance/SUB_PROCESSOR_REGISTER.md).
2. **Notice of Changes:** Processor will provide at least thirty (30) days' prior written notice of any intended appointment or replacement of a sub-processor, giving Customer opportunity to object on reasonable data protection grounds.
3. **Flow-Down Obligations:** Processor imposes data protection obligations on any sub-processor no less protective than those set out in this DPA.

---

## 5. International Data Transfers (SCCs)

Where personal data is transferred from the European Economic Area (EEA), United Kingdom, or Switzerland to countries that do not ensure an adequate level of data protection:
- The parties agree to incorporate the European Commission's Standard Contractual Clauses (Module 2: Controller-to-Processor or Module 3: Processor-to-Processor) pursuant to Commission Implementing Decision (EU) 2021/914.
- For UK transfers, the UK International Data Transfer Addendum (IDTA) applies.

---

## 6. CCPA Service Provider Addendum

For data subject to the California Consumer Privacy Act ("CCPA/CPRA"):
1. Processor acts as a **Service Provider**.
2. Processor shall not "sell" or "share" Customer Personal Information.
3. Processor shall not retain, use, or disclose Customer Personal Information for any purpose other than performing the business services specified in the principal agreement.

---

## 7. Signatures & Execution

| For Data Controller (Customer): | For Data Processor (Nethical): |
|:---|:---|
| **Organization:** _________________________ | **Organization:** Nethical Open Source / Commercial Entity |
| **Name:** _________________________ | **Name:** Authorized Signatory |
| **Title:** _________________________ | **Title:** Data Protection Officer / Compliance Lead |
| **Date:** _________________________ | **Date:** 2026-09-15 |
| **Signature:** _________________________ | **Signature:** *Signed electronically* |
