# Nethical Authorized Sub-Processor Register

**Document ID:** SPR-2026  
**Version:** 2.0  
**Effective Date:** 2026-09-15  
**Regulatory Framework:** GDPR Article 28(2), UK GDPR, SOC 2 CC9.2, ISO/IEC 27001 Annex A.5.21  

---

## 1. Overview & Policy

Under Article 28(2) of the General Data Protection Regulation (GDPR), processors must maintain an up-to-date list of all authorized sub-processors engaged in processing customer personal data.

Nethical prioritizes **sovereign, on-premises, and local execution**. When deployed on-premise or in an air-gapped sovereign cluster, **zero external sub-processors are engaged**.

When utilizing Nethical hosted cloud infrastructure or managed services, the following vetted sub-processors are utilized:

---

## 2. Authorized Infrastructure & Platform Sub-Processors

| Sub-Processor Entity | Service Description | Processing Location | Transfer Safeguard Mechanism | Security Certification |
|---|---|---|---|---|
| **OVHcloud SAS** | Sovereign European Cloud Compute & Dedicated Bare Metal | European Union (France, Germany, Poland) | EU Adequacy / EEA Native (No 3rd country transfer) | ISO 27001, SecNumCloud, SOC 2 Type II |
| **Hetzner Online GmbH** | European High-Performance Node Hosting & Storage | European Union (Germany, Finland) | EU Adequacy / EEA Native | ISO 27001 |
| **Amazon Web Services EMEA SARL** | Optional Managed S3 Backups & CloudWatch Telemetry (Enterprise Cloud tier) | EU Regions (Frankfurt, Dublin) | Standard Contractual Clauses (SCCs) + DPA | ISO 27001, SOC 1/2/3, FedRAMP High |
| **Cloudflare Portugal Unipessoal Lda / Cloudflare, Inc.** | DDoS Mitigation, Web Application Firewall (WAF), Edge Caching | Global Edge Network (EU Data Localization Option enabled) | Standard Contractual Clauses (SCCs) + BCR | ISO 27001, SOC 2 Type II, PCI DSS |
| **GitHub, Inc. (Microsoft)** | CI/CD Build Pipelines, Release Distribution, Artifact Storage | United States / Global | EU-US Data Privacy Framework (DPF) + SCCs | ISO 27001, SOC 2 Type II |

---

## 3. Sub-Processor Change Notification Protocol

1. **30-Day Notice Window:** Customers subscribing to hosted or managed services will receive email notification at least thirty (30) calendar days prior to the onboarding of any new sub-processor.
2. **Right to Object:** If a Customer has reasonable data protection or security objections to a proposed sub-processor, the Customer may notify `privacy@nethical.ai` in writing.
3. **Due Diligence:** Every sub-processor undergoes annual security risk assessment, SOC 2 / ISO 27001 review, and contractual DPA execution with equivalent safeguards.
