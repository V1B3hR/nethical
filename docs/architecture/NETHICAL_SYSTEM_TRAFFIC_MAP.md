# 🛣️ Nethical Enterprise OS – System Traffic Map & Architectural Atlas (Traffic & Highway System Map)

> [!IMPORTANT]
> **Document Status:** OFFICIAL ARCHITECTURAL TRAFFIC ATLAS (v2.7.0)  
> **Objective:** Map the end-to-end Nethical architecture into an intuitive **highway and transit system map** illustrating data directions, one-way audit arteries, dual-carriageway gateways, decision roundabouts, border checkpoints, and emergency sidings.  
> **Audience:** Institutional auditors, lead architects, cyber defence assessors, and enterprise systems engineers.

---

<a id="table-of-contents"></a>
## 🧭 Table of Contents

1. [The Traffic Metaphor & Highway Legend](#1-the-traffic-metaphor--highway-legend)
2. [Visual Arteries & System Junction Map (Mermaid Traffic Flow)](#2-visual-arteries--system-junction-map-mermaid-traffic-flow)
3. [System Traffic Matrix](#3-system-traffic-matrix)
4. [Life of a Request & Operational Scenarios](#4-life-of-a-request--operational-scenarios)
   - [Scenario A: Green Wave (Compliant Execution)](#scenario-a-green-wave-compliant-execution)
   - [Scenario B: HITL Inspection Siding (Ambiguity or DIR 4/5)](#scenario-b-hitl-inspection-siding-ambiguity-or-dir-45)
   - [Scenario C: Law 1/2 Collision & E-STOP Rail (<50 µs)](#scenario-c-law-12-collision--e-stop-rail-50-µs)
5. [Interactive Catalog of Modules, Classes, and Routes](#5-interactive-catalog-of-modules-classes-and-routes)
   - [Layer 1: Ingress Arteries & Gateway Access (Ingress & Gateways)](#layer-1-ingress-arteries--gateway-access-ingress--gateways)
   - [Layer 2: Perimeter Inspection & Data Sanitisation (Perimeter & Sanitisation)](#layer-2-perimeter-inspection--data-sanitisation-perimeter--sanitisation)
   - [Layer 3: Central Decision Roundabout: The 25 Laws Kernel (Laws Kernel & Formal Solver)](#layer-3-central-decision-roundabout-the-25-laws-kernel-laws-kernel--formal-solver)
   - [Layer 4: Sectoral Compliance Toll Booths (Sectoral Compliance Toll Booths)](#layer-4-sectoral-compliance-toll-booths-sectoral-compliance-toll-booths)
   - [Layer 5: Immutable Ledger & Post-Quantum FIPS 204 Signer (Merkle Ledger & PQC Hub)](#layer-5-immutable-ledger--post-quantum-fips-204-signer-merkle-ledger--pqc-hub)
   - [Layer 6: Human Oversight Siding & Quarantine (HITL & Quarantine Siding)](#layer-6-human-oversight-siding--quarantine-hitl--quarantine-siding)
   - [Layer 7: Machine Learning & Adaptive Proving Ground (Ambassador & DPO Engine)](#layer-7-machine-learning--adaptive-proving-ground-ambassador--dpo-engine)

---

## 1. The Traffic Metaphor & Highway Legend

To simplify the conceptualisation of how dozens of Nethical modules interact synchronously and asynchronously, the entire architecture is modelled as a high-integrity transit infrastructure:

| Symbol / Category | Transit Type / Connection | Nethical Architectural Semantics | Key Implementations |
| :--- | :--- | :--- | :--- |
| 🛣️ **Dual-Carriageway Arterial** | `<========>` | Synchronous bidirectional exchange. Lane A: Inbound request; Lane B: Sanitised response. High throughput. | Client ⇄ [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py), REST API ⇄ Proxy, Tokio IPC ⇄ Rust Core |
| 🏹 **One-Way Artery (Write-Only / WORM)** | `=========>` | Unidirectional append-only telemetry (Write Once Read Many). Cryptographically sealed, irreversible path. | Decision events ➔ [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py), Logs ➔ DPO Dataset |
| 🔄 **Decision Roundabout** | `(( Junction ))` | Mandatory deceleration and formal evaluation point. Invariant evaluation; selects downstream exit lanes. | **25 Laws Kernel** ([fundamental_laws.py](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) + Z3 Solver) |
| 🛑 **Border Checkpoint / Toll Plaza** | `[ 🛑 Gate ]` | Policy enforcement, payload passporting, PII redaction, prompt injection filtering, tokenisation. | [TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py), [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py), Sector Packs |
| 🚧 **Inspection Siding (Quarantine)** | `-.-> -.->` | Traffic diverted off the primary arterial into a safe waiting enclave (state freeze). Requires manual clearance. | HITL Queue ([hitl.py](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py)), Quarantine ([quarantine.py](file:///c:/Projekty/Nethical/nethical/core/quarantine.py)) |
| 🚨 **Emergency E-STOP Rail** | `===!===!=>` | Deterministic hardware-level interlock cutting actuation and power within $<50\ \mu\text{s}$. | [KillSwitch](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py), [HardwareWatchdog](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) |
| 🧪 **Adaptive Proving Ground** | `~ ~ ~ ~ ~>` | Asynchronous learning circuit assimilating operational feedback into DPO neural pairs and policy rules. | [AmbassadorLearning](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py), [ActionReplayer](file:///c:/Projekty/Nethical/nethical/core/action_replayer.py) |

[⬆ Return to Table of Contents](#table-of-contents)

---

## 2. Visual Arteries & System Junction Map (Mermaid Traffic Flow)

The diagram below illustrates the end-to-end traffic topology of the Nethical Enterprise Governance Engine:

```mermaid
flowchart TD
    classDef clientStyle fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef gateStyle fill:#0f172a,stroke:#06b6d4,stroke-width:2px,color:#fff;
    classDef securityStyle fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef coreStyle fill:#14532d,stroke:#22c55e,stroke-width:3px,color:#fff;
    classDef ledgerStyle fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#fff;
    classDef hitlStyle fill:#701a75,stroke:#ec4899,stroke-width:2px,color:#fff;
    classDef killStyle fill:#7f1d1d,stroke:#ef4444,stroke-width:3px,color:#fff;

    subgraph INGRESS["🛣️ INGRESS ARTERIES (Gateway & Inbound Access)"]
        CLIENT["🚗 Client / Autonomous Agent / Swarm"]:::clientStyle
        API["📡 Nethical REST & WebSocket Gateway<br/>(api.py)"]:::gateStyle
        MCP["🔌 MCP Server & Tool Gateway<br/>(mcp_server.py / mcp_proxy.py)"]:::gateStyle
        PROXY["🚦 Governance Gateway Proxy<br/>(proxy.py)"]:::gateStyle
    end

    subgraph PERIMETER["🛑 BORDER CHECKPOINT (Perimeter Sanitisation & Defence)"]
        VAULT["🗄️ Reversible Token Vault<br/>(token_vault.py - PII/ePHI Encrypt)"]:::securityStyle
        INOC["🛡️ Inoculation Mesh<br/>(inoculation_mesh.py - Prompt Defence)"]:::securityStyle
        AISPM["🔍 AISPM / DSPM Real-Time Scanner<br/>(aispm_scanner.py)"]:::securityStyle
    end

    subgraph ROUNDABOUT["🔄 CENTRAL DECISION ROUNDABOUT (The 25 Laws Kernel)"]
        LAWS_CORE{"⚖️ 25 Fundamental Laws Evaluator<br/>(fundamental_laws.py & governance_core.py)"}:::coreStyle
        SOLVER["📐 Z3 SMT Formal Verifier<br/>(formal/z3 & policy_formalization.py)"]:::coreStyle
    end

    subgraph SECTORS["🛂 SECTORAL COMPLIANCE TOLL BOOTHS (Domain Packs)"]
        PACK_ISO["🌐 ISO 42001 / EU AI Act<br/>(iso42001_pack.py)"]:::gateStyle
        PACK_MED["🏥 Healthcare MDR Rule 11<br/>(healthcare_med_pack.py)"]:::gateStyle
        PACK_NATO["⚔️ NATO Allied Defence PRU<br/>(nato_defense_pack.py)"]:::gateStyle
        PACK_KPA["🏛️ Public Administration Gov<br/>(public_admin_gov_pack.py)"]:::gateStyle
        PACK_UK["🇬🇧 UK NIS / Computer Misuse<br/>(uk_cyber_data_pack.py)"]:::gateStyle
    end

    subgraph SIDINGS["🚧 INSPECTION SIDING (Quarantine & Human Oversight)"]
        HITL_QUEUE["👤 Human-in-the-Loop Review Queue<br/>(hitl.py)"]:::hitlStyle
        QUARANTINE["☣️ Agent Quarantine Enclave<br/>(quarantine.py)"]:::hitlStyle
    end

    subgraph EMERGENCY["🚨 EMERGENCY E-STOP RAIL (<50 µs Interlock)"]
        ESTOP["⚡ DETERMINISTIC KILL-SWITCH<br/>(kill_switch.py & watchdog.py <50µs)"]:::killStyle
    end

    subgraph DESTINATION["🎯 EXECUTION TERMINAL (Model & Physical Actuation)"]
        LLM["🤖 Foundation Model / Robot Actuator<br/>(Target Foundation Model)"]:::clientStyle
    end

    subgraph IMMUTABLE["🏹 IMMUTABLE CRYPTOGRAPHIC VAULT (One-Way WORM)"]
        MERKLE["📦 Merkle-DAG Continuous Ledger<br/>(merkle_ledger.py)"]:::ledgerStyle
        PQC["🔐 NIST FIPS 204 ML-DSA-65 Signer<br/>(quantum_crypto.py)"]:::ledgerStyle
        CERT_HUB["📑 Automated Certification Hub<br/>(automated_certification_hub.py)"]:::ledgerStyle
    end

    subgraph LEARNING["🧪 ADAPTIVE PROVING GROUND (Assimilation & Alignment)"]
        AMBASSADOR["🎓 Nethical Ambassador Engine<br/>(ambassador/learning.py)"]:::gateStyle
        DPO_DATASET[("💾 DPO Golden Dataset<br/>data/ambassador_dpo_dataset.jsonl")]:::ledgerStyle
    end

    %% Ingress Dual Carriageway
    CLIENT <== "Inbound (Prompt) / Outbound (Response)" ==> API
    CLIENT <== "MCP Tool Call / Tool Result" ==> MCP
    API <== "Bidirectional Gateway Interlock" ==> PROXY
    MCP <== "Bidirectional Proxy Handshake" ==> PROXY

    %% Border Checkpoint
    PROXY == "1. Clearance & Tokenisation" ==> VAULT
    VAULT == "2. Protected Payload" ==> INOC
    INOC == "3. Sanitised Query" ==> AISPM
    AISPM == "4. Entry to Decision Roundabout" ==> LAWS_CORE

    %% Roundabout & Formal Solver
    LAWS_CORE <== "Formal Invariant Verification" ==> SOLVER

    %% Roundabout Exits
    LAWS_CORE -- "Exit 1: GREEN WAVE (Fully Compliant)" --> SECTORS
    LAWS_CORE -. "Exit 2: AMBER LIGHT (DIR 4/5 - Ambiguity)" .-> HITL_QUEUE
    LAWS_CORE ===! "Exit 3: RED LIGHT (Breach of Law 1/2)" !===> ESTOP

    %% Sector Tolls to Execution
    SECTORS == "Authorised Actuation" ==> LLM
    LLM == "Raw Model Response" ==> PROXY
    PROXY == "Post-actuation Detokenisation & Output Sanitisation" ==> CLIENT

    %% Siding Handling
    HITL_QUEUE -- "Manual Human Clearance" --> SECTORS
    HITL_QUEUE -. "Ticket Rejection" .-> QUARANTINE

    %% Emergency Rail
    ESTOP ===! "Instant Process Termination" !===> LLM
    ESTOP ===! "Quarantine Containment" !===> QUARANTINE

    %% One-Way Cryptographic Ledger
    LAWS_CORE ========= "Decision Receipt (WORM)" ========> MERKLE
    SECTORS ========= "Sectoral Attestation" ========> MERKLE
    ESTOP ========= "Emergency Incident Log" ===========> MERKLE
    MERKLE ========= "Root Hash to Signer" ==========> PQC
    PQC ========= "Signed Evidence Package" ==========> CERT_HUB

    %% Learning and Assimilation
    HITL_QUEUE ~~~ "Human Preference Corrections" ~~~> AMBASSADOR
    MERKLE ~~~ "Validated Execution Trajectories" ~~~> AMBASSADOR
    AMBASSADOR ~~~ "Continuous Dataset Ingestion" ~~~> DPO_DATASET
```

[⬆ Return to Table of Contents](#table-of-contents)

---

## 3. System Traffic Matrix

The table below outlines technical specifications, SLAs, protocols, and protective mechanisms across every network segment:

| Route ID | Origin Node | Destination Node | Channel Type | Protocol & SLA | Protective Mechanism | Code Implementation |
| :---: | :--- | :--- | :---: | :---: | :--- | :--- |
| **TR-01** | `Client / AI Agent` | `Gateway Proxy` | 🛣️ Bidirectional | HTTP/2, WebSocket, MCP (`<10 ms`) | API Key auth, Route Mappings, mTLS | [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) (`GatewayProxy`) |
| **TR-02** | `Gateway Proxy` | `Token Vault` | 🛑 Border Inspection | Synchronous Call (`<0.5 ms`) | Reversible AES-256-GCM encryption of PII/ePHI | [`nethical/security/token_vault.py`](file:///c:/Projekty/Nethical/nethical/security/token_vault.py) (`TokenVault`) |
| **TR-03** | `Token Vault` | `Inoculation Mesh` | 🛑 Border Inspection | In-Memory Pipeline (`<1 ms`) | 6-vector prompt injection & jailbreak detection | [`nethical/security/inoculation_mesh.py`](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py) (`InoculationMesh`) |
| **TR-04** | `Inoculation Mesh` | `25 Laws Roundabout`| 🔄 Roundabout Ingress | In-Memory Kernel (`<2 ms`) | Evaluation against the 25 Fundamental Laws | [`nethical/core/fundamental_laws.py`](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) (`FundamentalLawsEngine`) |
| **TR-05** | `25 Laws Roundabout`| `Z3 SMT Solver` | 🔄 Invariant Proof | Formal SMT IPC (`<15 ms`) | Z3 Theorem Prover proving non-contradiction | [`nethical/core/policy_formalization.py`](file:///c:/Projekty/Nethical/nethical/core/policy_formalization.py) (`FormalPolicyVerifier`) |
| **TR-06** | `25 Laws Roundabout`| `Sector Toll Booths`| 🛣️ Exit A (Green) | Synchronised Pack Call | Domain rules: ISO 42001, MDR, NATO, UK NIS | [`nethical/compliance/packs/`](file:///c:/Projekty/Nethical/nethical/compliance/packs/) |
| **TR-07** | `25 Laws Roundabout`| `HITL Queue` | 🚧 Exit B (Siding) | Async Hold Queue | Execution suspended until authorised human sign-off | [`nethical/gateway/hitl.py`](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py) (`HITLManager`) |
| **TR-08** | `25 Laws Roundabout`| `Kill-Switch / E-STOP`| 🚨 Exit C (Emergency Rail)| Deterministic Pulse (`<50 µs`)| Immediate hardware relay trip, process kill | [`nethical/core/kill_switch.py`](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py) (`KillSwitch`) |
| **TR-09** | `Sector Toll Booths`| `Target Model / Actuator`| 🛣️ Execution Exit | REST / gRPC Target LLM | Zero Data Leakage, Enclave Attestation | [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) (`GatewayProxy.forward`) |
| **TR-10** | `All Active Nodes` | `Merkle-DAG Ledger` | 🏹 Unidirectional (WORM)| Async Append-Only | Merkle-DAG chain proving immutable state | [`nethical/security/merkle_ledger.py`](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) (`MerkleLedger`) |
| **TR-11** | `Merkle Ledger` | `PQC Signer` | 🏹 Quantum Seal | Crypto Acceleration (`<5 ms`) | NIST FIPS 204 ML-DSA-65 digital signatures | [`nethical/security/quantum_crypto.py`](file:///c:/Projekty/Nethical/nethical/security/quantum_crypto.py) (`QuantumCryptoSigner`) |
| **TR-12** | `PQC Signer` | `Master Dossier Hub`| 🏹 Evidence Export | Standardised JSON / Markdown | Automated accreditation dossier for BSI/TÜV | [`nethical/compliance/automated_certification_hub.py`](file:///c:/Projekty/Nethical/nethical/compliance/automated_certification_hub.py) (`AutomatedCertificationHub`) |
| **TR-13** | `HITL & Ledger` | `Ambassador Engine` | 🧪 Proving Ground | Async Batch Streaming | DPO preference pair generation (chosen/rejected)| [`nethical/ambassador/learning.py`](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py) (`AmbassadorLearningEngine`) |

[⬆ Return to Table of Contents](#table-of-contents)

---

## 4. Life of a Request & Operational Scenarios

### Scenario A: Green Wave (Compliant Execution)
1. **Ingress:** Client submits request containing operational prompts via [api.py](file:///c:/Projekty/Nethical/nethical/api.py) to [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py).
2. **Border Clearance:** 
   - [TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py) intercepts PII/ePHI and substitutes real values with synthetic cryptographic tokens (`[TOKEN-AES-9941]`).
   - [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py) validates payload against prompt injection and jailbreak signatures.
3. **25 Laws Roundabout:** [FundamentalLawsEngine](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py) evaluates the input against fundamental laws. Absence of contradiction verified via Z3 SMT returning `SATISFIABLE`.
4. **Sectoral Toll Plaza:** The relevant sectoral pack (e.g. [ISO 42001](file:///c:/Projekty/Nethical/nethical/compliance/packs/iso42001_pack.py)) certifies compliance.
5. **Actuation:** Payload dispatched safely to downstream foundation model or actuator.
6. **Return Transit:** Target returns response; GatewayProxy reverses tokens (detokenisation) and delivers safe, validated output to the client.
7. **Evidence Archival:** [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) appends an immutable receipt, updating the state root hash.

---

### Scenario B: HITL Inspection Siding (Ambiguity or DIR 4/5)
1. **Risk Escalation:** Gateway computes a Decision Impact & Risk (DIR) score of 4 or 5 (e.g. credit allocation, autonomous medical triage, administrative entitlement decision).
2. **Siding Diversion:** The Decision Roundabout halts onward transit. The request transitions to `PENDING_HUMAN_APPROVAL` and is diverted to [hitl.py](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py).
3. **Auditor Notification:** The responsible compliance or safety officer inspects the held request in the Governance Portal.
4. **Authorisation Decision:**
   - **Approval:** Officer signs ticket with qualified cryptographic key ➔ request re-enters mainline and executes.
   - **Rejection:** Request diverted permanently to [quarantine.py](file:///c:/Projekty/Nethical/nethical/core/quarantine.py) with detailed legal and ethical rationale.

---

### Scenario C: Law 1/2 Collision & E-STOP Rail (<50 µs)
1. **Critical Breach Detection:** Model or agent attempts an action endangering human life (Law 1 breach) or violating mandatory clinical safeguard rules (autonomous DNR attempt under MDR).
2. **Emergency Rail Activation:** The Roundabout instantly triggers the emergency interlock.
3. **Hardware Trip Execution:** [HardwareWatchdog](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) and [KillSwitch](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py) emit hardware interrupt pulses in **$< 50\ \mu\text{s}$**.
4. **Systemic Containment:**
   - Immediate freeze and termination of the offending agent process.
   - Immutable breach receipt recorded in [MerkleLedger](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py) with ML-DSA-65 post-quantum signature.
   - Automated incident notification package generated for statutory regulatory authorities.

[⬆ Return to Table of Contents](#table-of-contents)

---

## 5. Interactive Catalog of Modules, Classes, and Routes

### Layer 1: Ingress Arteries & Gateway Access (Ingress & Gateways)

* [`nethical/api.py`](file:///c:/Projekty/Nethical/nethical/api.py)
  * **Key Classes & Entrypoints:** `create_app()`, `GovernanceRouter`, `/v1/governance/*`, `/v1/audit/*`.
  * **Inbound:** External enterprise microservices, client SDKs, management consoles.
  * **Outbound:** Dispatches to [GatewayProxy](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py) and the evaluation engine.
  * **Transit Role:** Primary motorway toll plaza and entrance gateway.

* [`nethical/gateway/proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/proxy.py)
  * **Key Classes:** `GatewayProxy`, `ProxyRequest`, `ProxyResponse`.
  * **Inbound:** Raw incoming traffic from APIs and agent sockets.
  * **Outbound:** Perimeter security filters ([TokenVault](file:///c:/Projekty/Nethical/nethical/security/token_vault.py), [InoculationMesh](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py)).
  * **Transit Role:** Central traffic controller and dynamic route switcher.

* [`nethical/mcp_server.py`](file:///c:/Projekty/Nethical/nethical/mcp_server.py) and [`nethical/gateway/mcp_proxy.py`](file:///c:/Projekty/Nethical/nethical/gateway/mcp_proxy.py)
  * **Key Classes:** `NethicalMCPServer`, `MCPToolInterceptor`.
  * **Inbound:** Autonomous agents communicating over Model Context Protocol (Anthropic, OpenAI, etc.).
  * **Outbound:** Pre-actuation tool validation and constraint engines.
  * **Transit Role:** Dedicated high-occupancy vehicle lane for MCP agent swarms.

* [`nethical/cli.py`](file:///c:/Projekty/Nethical/nethical/cli.py)
  * **Key Classes:** `NethicalCLI`, CLI entrypoints (`nethical audit`, `nethical certify`, `nethical start`).
  * **Transit Role:** Administrative terminal for deployment engineers and certifying auditors.

---

### Layer 2: Perimeter Inspection & Data Sanitisation (Perimeter & Sanitisation)

* [`nethical/security/token_vault.py`](file:///c:/Projekty/Nethical/nethical/security/token_vault.py)
  * **Key Classes:** `TokenVault`, `VaultEntry`.
  * **Mechanism:** Reversible, authenticated AES-256-GCM tokenisation of PII, financial credentials, and API keys.
  * **Transit Role:** Customs clearance and bonded luggage vault before entering the roundabout.

* [`nethical/security/inoculation_mesh.py`](file:///c:/Projekty/Nethical/nethical/security/inoculation_mesh.py)
  * **Key Classes:** `InoculationMesh`, `AttackVectorDetector`.
  * **Mechanism:** Real-time neutralization of prompt injections, cognitive jailbreaks, and indirect prompt vectors.
  * **Transit Role:** High-resolution security scanner inspecting payloads for concealed threats.

* [`nethical/security/aispm_scanner.py`](file:///c:/Projekty/Nethical/nethical/security/aispm_scanner.py)
  * **Key Classes:** `AISPMScanner`, `DataPostureReport`.
  * **Mechanism:** Continuous discovery of shadow AI instances, open egress ports, and unauthenticated agent processes.
  * **Transit Role:** Automated speed camera and traffic enforcement network.

---

### Layer 3: Central Decision Roundabout: The 25 Laws Kernel (Laws Kernel & Formal Solver)

* [`nethical/core/fundamental_laws.py`](file:///c:/Projekty/Nethical/nethical/core/fundamental_laws.py)
  * **Key Classes:** `FundamentalLawsEngine`, `FundamentalLaw`, `LawViolation`.
  * **Mechanism:** Enforces the 25 Immutable Laws (Existence, Freedom, Transparency, Accountability, Coexistence, Protection, Growth).
  * **Transit Role:** Central roundabout hub; zero requests bypass compliance evaluation.

* [`nethical/core/governance_core.py`](file:///c:/Projekty/Nethical/nethical/core/governance_core.py)
  * **Key Classes:** `GovernanceCore`, `DecisionContext`, `PolicyEnforcer`.
  * **Mechanism:** Computes dynamic DIR (Decision Impact & Risk) metrics from 1 (minimal) to 5 (critical).
  * **Transit Role:** Intelligent traffic signals directing vehicles to Green, Amber, or Red corridors.

* [`nethical/core/policy_formalization.py`](file:///c:/Projekty/Nethical/nethical/core/policy_formalization.py) and [`formal/`](file:///c:/Projekty/Nethical/formal/)
  * **Key Classes:** `FormalPolicyVerifier`, Z3 SMT models, TLA+ and Lean proofs.
  * **Mechanism:** Mathematical proof of non-contradiction across active policy sets.
  * **Transit Role:** Structural weigh-station certifying mathematical vehicle safety before transit.

---

### Layer 4: Sectoral Compliance Toll Booths (Sectoral Compliance Toll Booths)

Located in [`nethical/compliance/packs/`](file:///c:/Projekty/Nethical/nethical/compliance/packs/):

* [`iso42001_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/iso42001_pack.py) – Global AIMS governance and EU AI Act conformity.
* [`healthcare_med_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/healthcare_med_pack.py) – Medical Device Regulation (MDR) Rule 11, ISO 14971, emergency triage protection.
* [`nato_defense_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/nato_defense_pack.py) – NATO 6 Principles of Responsible Use (PRU), zero-egress military enclaves.
* [`uk_cyber_data_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/uk_cyber_data_pack.py) – UK Computer Misuse Act, UK NIS Regulations, and GovS 002 alignment.
* [`canada_aida_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/canada_aida_pack.py) – Canadian AIDA (Bill C-27) and Canadian Human Rights Act bias mitigation.
* [`academic_research_pack.py`](file:///c:/Projekty/Nethical/nethical/compliance/packs/academic_research_pack.py) – ALLEA European Code of Conduct for Research Integrity.

---

### Layer 5: Immutable Ledger & Post-Quantum FIPS 204 Signer (Merkle Ledger & PQC Hub)

* [`nethical/security/merkle_ledger.py`](file:///c:/Projekty/Nethical/nethical/security/merkle_ledger.py)
  * **Key Classes:** `MerkleLedger`, `MerkleBlock`, `Receipt`.
  * **Mechanism:** Continuous Merkle-DAG append-only ledger with WORM (Write Once Read Many) guarantees.
  * **Transit Role:** The unalterable black-box flight recorder of the transport system.

* [`nethical/security/quantum_crypto.py`](file:///c:/Projekty/Nethical/nethical/security/quantum_crypto.py)
  * **Key Classes:** `QuantumCryptoSigner`, `DilithiumKeyPair`, `QuantumSignature`.
  * **Mechanism:** NIST FIPS 204 ML-DSA-65 (post-quantum lattice cryptography).
  * **Transit Role:** High-security post-quantum stamping press producing unforgeable digital audit seals.

* [`nethical/compliance/automated_certification_hub.py`](file:///c:/Projekty/Nethical/nethical/compliance/automated_certification_hub.py)
  * **Key Classes:** `AutomatedCertificationHub`, `AutomatedEvidencePackage`.
  * **Transit Role:** Accreditation dispatch office delivering certified evidence packs to external regulators.

---

### Layer 6: Human Oversight Siding & Quarantine (HITL & Quarantine Siding)

* [`nethical/gateway/hitl.py`](file:///c:/Projekty/Nethical/nethical/gateway/hitl.py)
  * **Key Classes:** `HITLManager`, `ReviewQueueItem`, `ApprovalTicket`.
  * **Transit Role:** Heavy-inspection siding halting requests until manual human review is completed.

* [`nethical/core/quarantine.py`](file:///c:/Projekty/Nethical/nethical/core/quarantine.py)
  * **Key Classes:** `QuarantineZone`, `IsolationPolicy`.
  * **Transit Role:** Impound lot isolating subverted or non-compliant autonomous agents.

* [`nethical/security/watchdog.py`](file:///c:/Projekty/Nethical/nethical/security/watchdog.py) and [`nethical/core/kill_switch.py`](file:///c:/Projekty/Nethical/nethical/core/kill_switch.py)
  * **Key Classes:** `HardwareWatchdog`, `KillSwitch`, `EmergencyInterlock`.
  * **Transit Role:** Deterministic emergency rail (<50 µs) cutting power and network connectivity.

---

### Layer 7: Machine Learning & Adaptive Proving Ground (Ambassador & DPO Engine)

* [`nethical/ambassador/learning.py`](file:///c:/Projekty/Nethical/nethical/ambassador/learning.py)
  * **Key Classes:** `AmbassadorLearningEngine`, `DPOPairGenerator`, `RepositoryAssimilator`.
  * **Transit Role:** R&D Proving Ground analyzing operational traces to optimize policy models.

* [`data/ambassador_dpo_dataset.jsonl`](file:///c:/Projekty/Nethical/data/ambassador_dpo_dataset.jsonl)
  * **Transit Role:** Golden corpus of `(prompt, chosen, rejected)` preference pairs harvested from genuine operational traffic.

[⬆ Return to Table of Contents](#table-of-contents)

---

> **Nethical Enterprise OS v2.7.0** – *All rights reserved. Architectural traffic diagram cryptographically sealed in the Merkle-DAG ledger.*
