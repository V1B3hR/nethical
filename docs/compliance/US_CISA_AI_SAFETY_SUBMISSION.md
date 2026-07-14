# Proposal to US CISA: Nethical AI Governance Platform

**From:** Andrzej Matewski  
**Date:** June 29, 2026  
**Subject:** Introduction of Nethical: A Reference Implementation of "Secure by Design" and NIST AI RMF for Autonomous Agentic Systems  

---

## Executive Summary

As frontier models transition from passive query-response interfaces to active, autonomous agents operating in real-world environments, traditional post-hoc safety guardrails (such as wrapper filters) are increasingly susceptible to jailbreaking, context drift, and adversarial exploitation.

**Nethical** is an open-source, local-first runtime governance and threat detection framework designed to address safety, alignment, and security challenges in autonomous agentic AI systems. Engineered as an intercepting gateway positioned between autonomous agents and their operating environment, Nethical evaluates proposed actions in real-time, providing an active compliance and safety layer.

Nethical is fully aligned with the **US CISA "Secure by Design" and "Secure by Default"** campaign, the **NIST AI Risk Management Framework (AI RMF)**, and **US Executive Order 14028** on supply chain transparency.

---

## Key Architectural Features of Nethical

*   **Active Runtime Evaluation**: Intercepts agent requests and screens them against a digital "Bill of Rights and Duties" (the 25 Fundamental Laws), returning immediate decisions (`ALLOW`, `RESTRICT`, `BLOCK`, `TERMINATE`).
*   **Cryptographically Verifiable Auditing**: Generates a "Decision + Reason + Proof" tuple for every evaluation, cryptographically anchored in an append-only, Merkle-tree-structured audit log to ensure tamper-evident records.
*   **Local-First Resilience**: Runs locally on host machines or edge devices to eliminate external API dependencies and ensure safety functionality is maintained offline.
*   **Sub-50ms Threat Detection Suite**: Integrates low-latency defensive modules, including:
    *   *Shadow AI Detector* (<20ms) to prevent unauthorized model execution and data leakage.
    *   *Polymorphic Malware Detector* (<50ms) to monitor system calls and memory patterns against agent-initiated exploits.
    *   *Prompt Injection Guard* (<15ms) to defend against direct and indirect jailbreaks.
    *   *AI vs AI Defender* (<25ms) to protect against model extraction and adversarial inputs.

---

## Alignment with CISA Initiatives and US Standards

### 1. Secure by Design & Secure by Default
CISA’s flagship initiative calls for technology vendors to take ownership of security outcomes and design software with security embedded as a core requirement. Nethical serves as a practical, open-source reference implementation of this philosophy for AI by:
*   Enforcing secure-by-default configurations for AI agent environments.
*   Integrating strict input/output validation at the runtime layer before any action is executed.
*   Directly mapping its architecture to the four pillars of the co-authored **Guidelines for Secure AI System Development** (co-authored by CISA and UK NCSC).

### 2. NIST AI Risk Management Framework (AI RMF)
Nethical includes an automated compliance mapping generator that maps internal system controls directly to the NIST AI RMF functions:
*   **GOVERN**: Enforces accountability structures and governance policies through human-in-the-loop review mechanisms (`nethical/governance/human_review.py`).
*   **MAP**: Assesses system context and boundary limitations (`nethical/core/fairness_sampler.py`).
*   **MEASURE**: Runs automated benchmark tests, bias detection, and adversarial robustness checks (`nethical/governance/ethics_benchmark.py`).
*   **MANAGE**: Deploys immediate mitigation, quarantines, and rollback procedures for non-compliant models (`nethical/core/quarantine.py`).

### 3. Supply Chain Security (Executive Order 14028)
To meet CISA’s guidelines on supply chain transparency and software integrity, Nethical:
*   Generates a comprehensive, machine-readable Software Bill of Materials (`SBOM.json`) detailing all project components.
*   Uses cryptographically pinned and hashed dependencies (`requirements-hashed.txt`) to protect against dependency confusion and supply chain injection attacks.

### 4. Critical Infrastructure Protection
For AI systems deployed in critical infrastructure sectors (healthcare, energy, emergency services), Nethical implements safe failure modes, graceful degradation, and secure API boundaries. Its compatibility has been validated against standards such as **NHS DSPT** (Standard 7: Access Control, Standard 10: Accountable Suppliers) and **SOC2** (Logical Access Security & Security Monitoring).

---

## Collaboration and Research Opportunities

As an independent Systems Architect committed to the open-source AI safety ecosystem, I am dedicated to the ongoing development, testing, and hardening of runtime safety mechanisms. 

I believe Nethical can serve as a valuable case study or testing platform for CISA's AI security initiatives. I am highly interested in:
*   **Reference Architectures**: Discussing how Nethical can be used by public and private organizations to enforce CISA's Secure AI Guidelines.
*   **Security Evaluation**: Exploring how Nethical's low-latency threat detection modules can support CISA's guidance on defensive AI architecture.
*   **Research Partnerships & Grants**: Exploring funding, sponsorships, or collaborative grants available through CISA or its partner federal research programs to accelerate this open-source security research.

I welcome the opportunity to present Nethical to CISA's cybersecurity and emerging technology experts, share technical design logs, or discuss the challenges of local-first agent alignment.

*   **Repository and Codebase**: [https://github.com/V1B3hR/nethical](https://github.com/V1B3hR/nethical)
*   **NCSC / CISA Compliance Guide**: [https://github.com/V1B3hR/nethical/blob/main/docs/compliance/UK_NCSC_AI_SECURE_DEVELOPMENT_COMPLIANCE.md](https://github.com/V1B3hR/nethical/blob/main/docs/compliance/UK_NCSC_AI_SECURE_DEVELOPMENT_COMPLIANCE.md)

---

## Contact Information

**Andrzej Matewski (V1B3hR)**  
Freelance Systems Architect & Engineer  
Preston, Lancashire, United Kingdom  
*   **Phone**: +44 7912 853 241  
*   **Email**: brightnightbeacon@gmail.com  
*   **GitHub**: [https://github.com/V1B3hR](https://github.com/V1B3hR)  
