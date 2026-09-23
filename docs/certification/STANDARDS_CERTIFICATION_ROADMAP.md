# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

# Nethical Edge: Standards Alignment & Institutional Certification Roadmap

## 1. Executive Context & Industry Audit Assessment

An independent functional safety and robotics domain audit evaluated Nethical's Edge subsystem against the institutional standards claimed in project documentation:
- **ISO 13849-1** (Performance Level e, Category 4)
- **ISO 10218-1/2:2025** (Industrial Robot Safety & Parameter Testing)
- **ISO/TS 15066** (Collaborative Robots: SRMS, HG, PFL, SSM)
- **ISO 26262:2018** (Road Vehicles Functional Safety, ASIL-D)
- **NATO AEP-107** (Sense and Avoid for Military UAS)
- **EU Regulation 2019/947 & 2019/945** (U-Space / Open & Specific Category BVLOS)
- **IEC 61508** (Functional Safety of E/E/PE Safety-Related Systems, SIL 3)

### Expert Consensus
> *"The test suite represents a significant and commendable step forward... It covers the functional layer of safety well but leaves critical gaps in verification, validation, and formal certification evidence. The tests demonstrate that faults are detected, not how well (Diagnostic Coverage) or how reliably (MTTFd) on physical target hardware."*

This assessment is technically rigorous and valid. In institutional safety engineering, **software test suites provide functional verification; they do not constitute an accredited certificate issued by a Notified Body (e.g. TÜV SÜD, UL, DEKRA, or NATO Military Airworthiness Authorities).**

---

## 2. Standards Compliance vs. Software Architecture Matrix

| Standard / Regulation | Intended Safety Tier | Nethical Software Implementation | Physical / Accredited Lab Requirement (OEM/Integrator) |
|---|---|---|---|
| **ISO 13849-1** | PL-e, Cat 4 | Cat 4 dual-channel state evaluation, redundant watchdog heartbeat, sub-50 µs cutoff (`nethical/edge/iso13849_watchdog.py`, `tests/edge/test_iso13849_iso26262.py`) | Physical hardware B10d / MTTFd component data, measured EMC immunity (IEC 61000-4), certified safety relays |
| **ISO 10218-1/2:2025** | STO, SS1, SS2, SLS, SLP, SLT | Deterministic kinematics governor, joint torque ceilings, table penetration containment, normative parameter variation scaling (`tests/edge/test_robot_safety.py`) | Metrological calibration of optical trackers, robot brake-distance deceleration testing, physical load cell validation |
| **ISO/TS 15066** | PFL, SSM, SRMS | Annex A biomechanical force limits mapped to anatomical zones (Face 65N, Chest 140N, Hands 140N, Back 210N), dynamic velocity ramping (`robot_safety.py`) | Instrumented crash tests using biofidelic human body impactors, pressure-sensitive film measurement |
| **ISO 26262:2018** | ASIL-D | HARA hazard classification matrix (S3/E4/C3), TTC collision horizon, steering jerk interlock (`iso26262_asil.py`) | MC/DC structural coverage on target automotive MCU binary (Infineon AURIX / NXP S32K), ASIL-D certified compiler |
| **NATO AEP-107** | Sense-and-Avoid (SAA) | 3D geocaging, 120m AGL ceiling, ADS-B In deconfliction, Safe2Ditch, Flight Termination System (`drone_safety.py`) | Live flight test campaign, physical radar/EO sensor range verification, military flight clearance (STANAG 4671) |
| **EU 2019/947** | BVLOS Operations | Fail-closed geo-awareness, multi-stage battery RTH/Land failsafes, C2 link-loss recovery | U-Space service provider (USSP) integration, Specific Operations Risk Assessment (SORA) validation |

---

## 3. The Three Layers of Safety in Nethical

To provide institutional clarity for defense ministries, governments, and industrial partners, Nethical delineates safety into three operational layers:

```mermaid
graph TD
    subgraph Layer 1: Deterministic AI Governance (Nethical Core & Edge)
        L1A[Pre-computation Prediction Profile] --> L1B[Real-time Kinetic Safety Governor]
        L1B --> L1C[Merkle Cryptographic Audit Proof]
        L1C --> L1D[Sub-50 µs Decision Engine]
    end

    subgraph Layer 2: Deterministic Hardware Interlocks (Edge & HIL)
        L2A[CANopen EMCY 0x080 + NMT STOP]
        L2B[Modbus TCP Power Coil De-energisation]
        L2C[EtherCAT ESM SAFE-OP & FSoE Zeroing]
        L2D[Hardware Watchdog Relay]
    end

    subgraph Layer 3: Physical System Certification (Accredited Lab & OEM)
        L3A[TÜV SÜD / UL Laboratory Certification]
        L3B[MIL-STD-810H Environmental Qualification]
        L3C[NATO / EASA Airworthiness Flight Testing]
        L3D[Biofidelic Crash Impactor Testing]
    end

    L1D --> L2A
    L1D --> L2B
    L1D --> L2C
    L1D --> L2D
    L2D --> L3A
```

1. **Layer 1: Deterministic AI Governance (Nethical Software)**
   - Operates in real time ($<50\ \mu\text{s}$ edge cutoff, $<10\text{ ms}$ local governor).
   - Validates mathematical safety invariants against the 25 Fundamental Laws.
   - Enforces geocages, speed limits, torque ceilings, and collision horizons.
2. **Layer 2: Deterministic Hardware Coupling (Fieldbus & HIL)**
   - Physically drops power to actuator contactors via CANopen EMCY, EtherCAT Fail-Safe over EtherCAT (FSoE), or Modbus coils.
   - Hardware Watchdog Timers guarantee fail-closed behavior even under total software lockup.
3. **Layer 3: Physical System Qualification (OEM / Accredited Notified Body)**
   - Must be executed on the physical end-product (the specific drone airframe, robotic work cell, or autonomous vehicle).
   - Involves physical crash testing, thermal vacuum, EMC testing, and military flight test ranges.

---

## 4. Engineering Hardening Implemented in Response to Audit

In response to the expert audit, the following engineering enhancements were added and verified:

1. **Quantified Diagnostic Coverage & MTTFd Metrics (`tests/edge/test_iso13849_iso26262.py`)**:
   - Validated that Cat 4 architecture with $DC_{avg} \ge 99.0\%$ and $MTTF_d \ge 30\text{ years}$ achieves PL-e and SIL 3.
   - Verified that lower DC or CCF $<65$ points triggers non-compliance flags.
2. **ISO 10218-2:2025 Parameter Testing (`tests/edge/test_robot_safety.py`)**:
   - Implemented parameter variation suites verifying that dynamic SSM distance thresholds and PFL force ceilings scale deterministically.
3. **ISO/TS 15066 Annex A Biofidelic Body Regions (`nethical/edge/robot_safety.py`)**:
   - Added `BodyRegion` classifications and calibrated force thresholds across all human anatomical zones (Face 65N, Chest 140N, Hands 140N, Back 210N).
4. **Cryptographic Device Hub Security (`nethical/edge/device_hub.py`)**:
   - Added cryptographic handshake token validation and anti-spoofing checks to edge heartbeats.

---

## 5. Strategic Trajectory: Where Does Nethical Head?

Nethical faces an architectural and market crossroads: **Dual-Use Critical Infrastructure vs. Consumer Domestic Robotics.**

### Option A: Sovereign Dual-Use Critical Infrastructure (Recommended)
- **Target Sectors**:
  - **Defense & Alliance**: NATO UAS BVLOS operations, counter-UAS interceptors, autonomous naval vessels, air-gapped tactical command posts.
  - **Energy & Utilities**: Nuclear and hydroelectric power plants, high-voltage substations, water dam floodgate automation.
  - **Advanced Manufacturing**: Automotive robotic assembly lines (KUKA, ABB, Fanuc), aerospace titanium milling cobots.
  - **Critical Healthcare**: Robotic surgical arms, autonomous radiopharmaceutical dispensing.
- **Why this path wins**:
  - Extremely high barriers to entry: requires mathematical rigor, Merkle auditability, deterministic sub-50 µs response, and air-gapped sovereign control.
  - Premium contract values with governments, national defense procurement, and institutional energy operators.
  - Aligns natively with EU AI Act High-Risk systems (Annex III) and NIS2 directive for critical entities.

### Option B: Commercial Consumer & Domestic Robotics
- **Target Sectors**:
  - Consumer drones, domestic vacuum robots, warehouse delivery rovers.
- **Trade-offs**:
  - Price-sensitive commodity market with high volume but razor-thin margins.
  - Proprietary vendor ecosystems (DJI, iRobot) resist external governance layers.
  - Security and safety standards are voluntary or light-touch compared to institutional defense standards.

### Strategic Directive
Nethical will anchor itself as the **Global Sovereign Standard for High-Consequence Autonomous Systems (Defense, Energy, Robotics, Healthcare)**, while offering a streamlined lightweight SDK (`nethical-edge`) that commercial vendors can integrate to achieve turn-key compliance with the EU AI Act.
