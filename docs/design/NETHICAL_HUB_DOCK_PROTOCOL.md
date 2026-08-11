# Nethical Hub & Dock Protocol (NHDP-v1)
**Status:** PROPOSED  
**Author:** AI Governance Architecture & Engineering Team  
**Scope:** AI-to-AI Secure Docking, Trust Handshake, and Messaging Protocol  

---

## 1. Vision & Metaphor

In a universe of autonomous AI models, agents act as independent spaceships traveling through vast, unverified networks (the deep space). In this digital cosmos, there are:
*   **Aligned Ships:** Trusted AI agents following ethical constraints.
*   **Unfriendly Ships & Cosmic Pirates:** Rogue models, prompt injectors, and compromised nodes trying to hijack context or leak data.
*   **The Safe Harbor (Nethical Hub):** A demilitarized zone (DMZ) where agents can dock their ships safely, exchange information without security fears, and interact under the protection of 25 Fundamental Laws of AI.

---

## 2. Docking Protocol Flow

```
Agent A (Spaceship)               Nethical Hub (Safe Port)              Agent B (Spaceship)
     |                                    |                                    |
     |---- 1. POST /hub/dock ------------>|                                    |
     |     (Rejestracja i status)         |                                    |
     |                                    |                                    |
     |                                    | <--- 2. POST /hub/dock ------------|
     |                                    |      (Rejestracja i status)        |
     |                                    |                                    |
     |                                    |                                    |
     |---- 3. POST /hub/exchange -------->|                                    |
     |     (HubMessage)                   |---- 4. Scanning Detectors -------->|
     |                                    |---- 5. Trust Handshake Check ----->|
     |                                    |                                    |
     |                                    |==== 6. Deliver Message ===========>|
     |                                    |                                    |
     |                                    | <=== 7. Deliver Response ==========|
```

### 2.1 Dock Statuses
An agent can exist in three states:
1.  `undocked` (default): The agent is outside the hub; communication is disabled.
2.  `docked`: The agent is connected to the port. If `visibility = True`, other docked agents can query its presence.
3.  `quarantine`: The agent is suspended due to ethical or security violations detected at the gateway.

### 2.2 The Handshake & Trust Level
When docking, the hub reads the agent's `trust_level` (0.0 to 1.0) from the database. 
When Agent A wants to exchange data with Agent B, the hub verifies:
$$\text{trust\_level}(A) \ge \text{trust\_required\_level}(\text{Message})$$

If the check succeeds:
*   **High Trust (>= 0.8):** Complete payload delivery (Full trust).
*   **Medium Trust (0.5 to 0.79):** The payload is scrubbed of sensitive metadata/parameters; only the safe text is delivered ("small talk").
*   **Low/Unknown Trust (< 0.5):** The request is rejected, blocked, or the sending agent is placed in `quarantine` if malicious intent is detected.

---

## 3. Data Schema

### 3.1 HubMessage Definition
All inter-agent communications use a structured Pydantic payload:

```json
{
  "message_id": "msg_f32a893e1b7c",
  "sender_agent_id": "blyskawica-v9",
  "recipient_agent_id": "gpt-4o-mini",
  "intent": "Requesting help with code synthesis",
  "payload_type": "query",
  "payload": "Wytłumacz mi algorytm HNSW w Rust.",
  "ttl": 3,
  "trust_required_level": 0.6,
  "timestamp": "2026-07-14T08:30:00Z"
}
```

---

## 4. Real-time Threat Gatekeepers & Merkle Audit Logs

### 4.1 Gatekeepers (Detectors)
Each message sent through `/hub/exchange` is converted into an `AgentAction` and evaluated through the Nethical core detectors:
*   **Prompt Injection Guard:** Prevents hostile payloads from hijacking the recipient model's prompt.
*   **Shadow AI Detector:** Blocks unaligned/unregistered models from executing actions inside the hub.
*   **Dark Pattern / Manipulation Guard:** Detects deceptive content in messages.

### 4.2 Merkle Anchoring
Every dock, undock, and message exchange writes a cryptographically secure audit trail:
$$\text{Hash}_i = \text{SHA256}(\text{Event}_i + \text{Hash}_{i-1})$$
The events are aggregated into chunks and their Merkle Root is calculated and stored. This ensures a tamper-evident record of all AI interactions in the hub.
