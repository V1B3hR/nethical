# Nethical — Safety-Critical Use & Limitation of Liability Disclaimer

**Effective Date:** 2026-09-15  
**Version:** 2.7.0  
**Status:** Legally Binding Under MIT License Terms

---

## 1. Important Safety & Operational Notice

> [!CAUTION]
> **PLEASE READ THIS DISCLAIMER CAREFULLY BEFORE INSTALLING, INTEGRATING, OR RUNNING NETHICAL.**
> 
> Nethical is an open-source AI safety and ethical governance platform provided under the MIT License. While Nethical incorporates defense-in-depth mechanisms, post-quantum cryptographic auditing, and formal regulatory compliance mapping (EU AI Act, ISO 42001, NIST AI RMF), **it is not by itself a substitute for formal certification in life-critical, medical, or kinetic systems.**

---

## 2. Non-Certified Applications (Prohibited Standalone Operations)

Unless explicitly certified in writing by an accredited conformity assessment body or regulatory authority under applicable jurisdiction, Nethical **MUST NOT BE USED AS THE SOLE, UNMONITORED FAILSAFE** in the following high-stakes environments:

1. **Class III Medical Devices & Direct Patient Care:** Any system regulated under EU MDR (Regulation 2017/745), US FDA 21 CFR Part 820, or equivalent medical regulations where software error could directly cause death or bodily injury.
2. **Kinetic Autonomous Weapon Systems:** Direct targeting, arming, or release of lethal force without human command verification (in accordance with Fundamental Law #2 and Law #4).
3. **Avionics & Space Flight Control:** Systems governed by DO-178C Level A/B safety standards where automated decisions control flight dynamics without mechanical or human override.
4. **Nuclear Facility Process Control:** Direct safety shutoff or reactivity control in nuclear installations (IEC 61513 / IEEE 7-4.3.2).
5. **Autonomous Heavy Machinery / Automated Vehicles:** Direct vehicle trajectory planning or kinetic actuator control without an independent ISO 26262 ASIL-D or ISO 13849 PL-e hardware supervisor.

---

## 3. Mandatory Human Oversight (Human-in-the-Loop)

In accordance with:
- **Article 14 of the EU AI Act (Regulation (EU) 2024/1689)**
- **Principle 3 of the Nethical 25 Fundamental Laws (Bi-Directional Responsibility)**
- **NIST AI RMF GOVERN 1.3 & MANAGE 2.4**

All production deployments involving irreversible economic transactions, legal rights determination, or physical risk **must** implement a verified Human-in-the-Loop (HITL) or Human-on-the-Loop (HOTL) supervisory protocol. Deployers bear sole legal and operational responsibility for establishing human override capabilities.

---

## 4. Limitation of Liability

To the maximum extent permitted by applicable law:

1. **NO WARRANTY:** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, FREEDOM FROM ERROR, AND NONINFRINGEMENT.
2. **CONSEQUENTIAL DAMAGES:** IN NO EVENT SHALL THE AUTHORS, COPYRIGHT HOLDERS, CORE CONTRIBUTORS, OR AFFILIATED ENTITIES BE LIABLE FOR ANY CLAIM, DAMAGES, LOSS OF LIFE, BODILY INJURY, ECONOMIC LOSS, BUSINESS INTERRUPTION, LOSS OF DATA, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
3. **ALLOCATION OF RISK:** The deployer and end-user acknowledge that the risk of utilizing AI governance software in production environments is solely assumed by the deploying entity. Deployers must conduct their own independent conformity assessments, penetration testing, and safety hazard analyses.

---

## 5. Regulatory Compliance & Deployer Obligations

Deployers of Nethical are independently responsible for ensuring full compliance with:
- The EU AI Act (Regulation (EU) 2024/1689)
- General Data Protection Regulation (GDPR - Regulation (EU) 2016/679)
- US Export Administration Regulations (EAR 15 CFR §730-774)
- National and sector-specific statutory obligations

Nethical provides policy templates and automated verification tooling to assist compliance, but integration of this software does **not** automatically grant regulatory certification to the deployer's downstream application.

---

## 6. Runtime Acknowledgment

To acknowledge awareness of these safety constraints in non-interactive production environments, set the environment variable:
```bash
export NETHICAL_SAFETY_ACKNOWLEDGED=1
```
For queries regarding compliance partnerships or formal audit verification, contact `compliance@nethical.ai`.
