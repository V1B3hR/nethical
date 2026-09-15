# Nethical — Export Control & Sanctions Compliance Notice

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Classification:** EAR99 / 5D002 (TSU Eligible) / EU Dual-Use 5D002  

---

## 1. Regulatory Context

The Nethical governance framework contains cryptographic capabilities including:
- Post-quantum digital signatures (NIST FIPS 204 ML-DSA-65 / Dilithium)
- Elliptic Curve Digital Signatures (ECDSA P-384, Ed25519)
- Cryptographic hash chaining (SHA-256 / SHA-384 Merkle-DAG ledgers)
- Reversible token vault encryption (AES-256-GCM / ChaCha20-Poly1305)

Because Nethical incorporates strong encryption and cyber defense components, its distribution and export are subject to United States, European Union, and international export control laws.

---

## 2. United States Export Administration Regulations (EAR)

### Export Control Classification Number (ECCN)
- **Primary ECCN:** `5D002` (Information Security — Software)
- **License Exception:** **TSU (Technology and Software — Unrestricted)** under **15 CFR § 740.13(e)** for publicly available open-source encryption software.

Under 15 CFR § 740.13(e), publicly available encryption source code is released from EAR licensing requirements provided that notification is sent to the Bureau of Industry and Security (BIS) and the National Security Agency (NSA).

### Bureau of Industry and Security (BIS) Notification
- **Recipient:** `crypt@bis.doc.gov` and `enc@nsa.gov`
- **Subject:** Open Source Encryption Notification — Nethical AI Governance Framework
- **Repository URL:** `https://github.com/V1B3hR/nethical`
- **Status:** Publicly available source code; no fee or licensing barrier to download.

---

## 3. European Union Dual-Use Export Control

Under **Council Regulation (EU) 2021/821** (EU Dual-Use Regulation):
- **Classification:** Category 5, Part 2 (Information Security)
- **Open Source Exemption:** Cryptographic software that is generally available to the public without restriction upon payment or as open-source software falls under the Cryptography Note (Note 3 to Category 5, Part 2) and the General Software Note (GSN).

---

## 4. Wassenaar Arrangement

In accordance with the **Wassenaar Arrangement on Export Controls for Conventional Arms and Dual-Use Goods and Technologies**:
- Dual-use list: Category 5 — Telecommunications and "Information Security", Part 2.
- Software made available in the public domain without restriction satisfies the General Software Note exclusion for multilateral controls.

---

## 5. Prohibited Jurisdictions & Sanctions Compliance

Notwithstanding open-source availability, the Nethical Project and its automated build/distribution systems comply with international economic sanctions administered by the **US Department of the Treasury Office of Foreign Assets Control (OFAC)**, the **European External Action Service (EEAS)**, and the **United Nations Security Council**.

Nethical hosted cloud services, binary container distribution, and direct commercial support are **strictly prohibited** to, in, or for the benefit of:
- Cuba
- Iran
- North Korea (DPRK)
- Syria
- The Crimea, Donetsk, and Luhansk regions of Ukraine
- Any individual or entity listed on the OFAC Specially Designated Nationals (SDN) list or the EU Consolidated Financial Sanctions List.

---

## 6. Deployer & Downstream Responsibilities

If you re-export, transfer, incorporate into proprietary hardware, or distribute Nethical in compiled or modified form:
1. You are solely responsible for determining the export classification of your derivative work.
2. If you add proprietary cryptographic algorithms or restrict source access, the TSU / open-source exception **no longer applies**, and you may require a formal export license from BIS or your national competent authority.

For export control inquiries, contact: `compliance@nethical.ai`.
