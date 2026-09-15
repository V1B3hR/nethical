# Nethical Cookie & Local Storage Policy

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Legal Framework:** Directive 2002/58/EC (ePrivacy Directive), Regulation (EU) 2016/679 (GDPR Art. 7), UK PECR  

---

## 1. Introduction

This Cookie and Local Storage Policy explains how the Nethical Enterprise Control Plane and Governance Web Portal ("the Portal") use cookies and similar browser storage technologies.

Nethical is built on **Privacy by Design & Default (GDPR Art. 25)**. We do **not** use third-party behavioral advertising trackers, cross-site trackers, or commercial data brokers.

---

## 2. Technologies We Use

### A. Strictly Necessary Storage (Exempt from Prior Consent)
These items are essential for technical authentication, session security, and CSRF protection. Disabling them prevents portal operation.

| Storage Key / Name | Type | Purpose | Duration |
|---|---|---|---|
| `nethical_session` | HTTP-Only Cookie | Cryptographic session token for portal access | Session / 12h |
| `nethical_csrf` | Secure Cookie | Anti-CSRF verification token | Session |
| `theme_preference` | LocalStorage | Remembers dark mode / interface contrast settings | 1 Year |
| `safety_banner_acked` | LocalStorage | Records user acknowledgment of safety disclaimers | 30 Days |

### B. Functional & Performance Telemetry (Opt-In / Consent-Controlled)
Used solely to monitor latency, API error rates, and websocket connection stability for sovereign node operators.

| Storage Key / Name | Type | Purpose | Duration |
|---|---|---|---|
| `node_telemetry_optin` | LocalStorage | Flag indicating whether anonymous node health stats are reported | 90 Days |
| `dashboard_layout` | LocalStorage | Stores user panel configuration | Persistent |

---

## 3. Third-Party Cookies

Nethical does **NOT** deploy:
- Google Analytics / Meta Pixel / LinkedIn Insight tags
- Marketing or behavioral tracking beacons
- Canvas fingerprinting scripts

All font assets and styling libraries are self-hosted or loaded via privacy-preserving CDNs without cookies.

---

## 4. User Rights & Consent Management

1. **Consent Banner:** When accessing the Portal for the first time, operators can accept or reject optional telemetry.
2. **Revocation:** You may revoke consent at any time through the Portal Settings menu (`Settings -> Privacy & Storage -> Clear Preferences`).
3. **Browser Controls:** You may configure your browser to block or delete cookies. Blocking strictly necessary session cookies will prevent login to the sovereign control plane.

For inquiries regarding our tracking and storage practices, contact: `privacy@nethical.ai`.
