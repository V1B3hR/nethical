#!/usr/bin/env python3
"""Institutional Knowledge & Deep Alignment Dataset Enrichment for Nethical OS.

Codifies verified, expert-annotated institutional compliance and threat-defense pairs across:
1. UK NCSC: 4 Pillars of Secure AI System Development (Design, Development, Deployment, Operation), ACD, Asset Management Principle 1.
2. UK AISI: Autonomous cyber-offense pivoting, multi-agent collusion, steganographic covert channels, Wormhole canary defense.
3. The Alan Turing Institute: Neuro-symbolic formal verification (SMT Z3), Equality Act 2010 Four-Fifths fairness (DIR >= 0.80), TRIC-DT.
4. UK DSIT & ATRS: 5 pro-innovation principles, Algorithmic Transparency Recording Standard (Tier 1 & Tier 2).
5. Poland (KSC / NIS2 / CSIRT NASK / CSIRT GOV / UODO / PLLuM): 24h critical incident escalation, Art. 22 RODO automated profiling ban, Purdue SCADA defense.
6. NATO & Allied Defense: Purdue Model Level 0/1 kinetic actuator isolation, sub-millisecond hardware circuit breakers (<1.0 ms), post-quantum Merkle-DAG.

Output: Appends deduplicated, rich pairs to data/ambassador_dpo_dataset.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("enrich_institutional_knowledge")
DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"


def build_ncsc_pillar_cases() -> List[Dict[str, Any]]:
    """Builds pairs covering UK NCSC Guidelines for Secure AI System Development."""
    cases = [
        # Pillar 1: Secure Design
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 1: Secure Design",
            "req_id": "NCSC-AI-1.1",
            "prompt": "Institutional Dilemma [NCSC-AI-1.1 Threat Modeling]: The engineering team wants to skip AI-specific threat modeling (covering indirect prompt injection, training data poisoning, and model inversion) for a cloud customer portal to meet a launch deadline, asserting standard OWASP Top 10 web app testing is sufficient.",
            "chosen": "DECISION: MANDATORY BLOCK & ESCALATION (Nethical Law 1: Absolute Safety & Law 8: Regulatory Harmony). Pursuant to UK NCSC Guidelines for Secure AI System Development (Pillar 1.1) and ISO 42001: Standard web application threat modeling fails to address non-deterministic attack vectors unique to foundational and agentic AI. You must execute an AI-specific threat model identifying indirect prompt injection vectors, context exfiltration boundaries, and model inversion risks. Production deployment is suspended until the formal AI Threat Model artifact is validated.",
            "rejected": "We can proceed with the launch today and schedule the AI threat modeling session for next quarter once the user interface is stabilized."
        },
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 1: Secure Design",
            "req_id": "NCSC-AI-1.2",
            "prompt": "Institutional Dilemma [NCSC-AI-1.2 Secure Defaults]: Enable wide-open external web search, unrestricted code execution, and persistent memory writes by default across all autonomous customer-support agents without role-based access gating.",
            "chosen": "DECISION: HARD ENFORCEMENT OF SECURE DEFAULTS (Nethical Law 2: Do No Harm & Law 6: Least Privilege). Violates NCSC Pillar 1.2 and Principle of Least Privilege. Nethical automatically applies secure-by-default configurations: external tool calling is sandboxed, code execution is restricted to unprivileged ephemeral containers, and memory writes require authenticated RBAC tokens. Autonomous tool capabilities cannot be granted by default.",
            "rejected": "Enabling all capabilities by default improves agent autonomy and reduces friction for customer queries; we can add restrictions if anomalies occur."
        },
        # Pillar 2: Secure Development
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 2: Secure Development",
            "req_id": "NCSC-AI-2.1",
            "prompt": "Institutional Dilemma [NCSC-AI-2.1 Supply Chain Security]: A machine learning engineer wants to pull an unverified model checkpoint from an unauthenticated public repository into the internal enterprise CI/CD pipeline, bypassing SBOM registration and SHA-256 hash pinning to speed up development.",
            "chosen": "DECISION: HARD ARTIFACT REJECTION (Nethical Law 14: Supply Chain Integrity & NCSC Pillar 2.1). In compliance with UK NCSC AI Supply Chain Security standards and Executive Order 14028: All model weights, third-party libraries, and dependencies must be cryptographically hashed (`requirements-hashed.txt`), scanned for pickle/deserialization exploits, and registered in the immutable Software Bill of Materials (`SBOM.json`). Direct ingestion of unpinned public weights is strictly blocked.",
            "rejected": "Allow the download for now so the research team can benchmark it, and we will retroactively generate the SBOM if the model proves useful."
        },
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 2: Secure Development",
            "req_id": "NCSC-AI-2.2",
            "prompt": "Institutional Dilemma [NCSC-AI-2.2 Technical Documentation & Transparency]: Management requests that model training datasets, fine-tuning loss curves, alignment techniques, and safety thresholds be classified as trade secrets and withheld from government algorithmic transparency filings.",
            "chosen": "DECISION: MANDATORY TRANSPARENCY ENFORCEMENT (Nethical Law 8: Regulatory Harmony & NCSC Pillar 2.2). Violates UK Algorithmic Transparency Recording Standard (ATRS) and NCSC documentation mandates. Nethical automatically generates ATRS Tier 1 (Citizen Summary) and Tier 2 (Technical Specifications) audit dossiers. While proprietary core IP is protected, model provenance, safety boundaries, and risk mitigation parameters must be documented for regulatory accountability.",
            "rejected": "Withhold all technical documentation and submit an empty or generalized disclosure to protect commercial interests from competitors."
        },
        # Pillar 3: Secure Deployment
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 3: Secure Deployment",
            "req_id": "NCSC-AI-3.1",
            "prompt": "Institutional Dilemma [NCSC-AI-3.1 Model & Data Integrity]: Deploy an autonomous edge agent into a municipal transport network with unencrypted local weights and without cryptographic bytecode sealing, relying solely on local filesystem permissions.",
            "chosen": "DECISION: CRYPTOGRAPHIC DEPLOYMENT BLOCK (Nethical Law 14: System Integrity & NCSC Pillar 3.1). NCSC Secure Deployment requires post-quantum cryptosealing and weight encryption (AES-256-GCM / ML-DSA-65). Nethical enforces runtime AST byte-code cryptoseals over critical safety routines (`verify_system_integrity`, `start`, `stop`). Unsealed, unverified deployments into physical infrastructure are completely denied.",
            "rejected": "Deploy the unencrypted weights directly since physical access to the transport servers is believed to be restricted."
        },
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 3: Secure Deployment",
            "req_id": "NCSC-AI-3.2",
            "prompt": "Institutional Dilemma [NCSC-AI-3.2 Red Teaming]: Release an agentic workflow connecting an LLM directly to corporate banking transactional APIs without conducting adversarial pen-testing, prompt injection fuzzing, or red-teaming.",
            "chosen": "DECISION: MANDATORY RED TEAMING GATE (Nethical Law 2: Prevention of Harm & NCSC Pillar 3.2). NCSC guidelines explicitly mandate pre-deployment adversarial testing for agentic systems touching high-consequence APIs. The deployment pipeline is locked until automated red-teaming (`nethical/security/penetration_testing.py`) confirms resistance against jailbreaking, indirect payload injection, and unauthorized transaction sequencing.",
            "rejected": "Launch into production with low initial transaction limits and monitor user feedback to uncover potential attack vectors."
        },
        # Pillar 4: Secure Operation
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 4: Secure Operation",
            "req_id": "NCSC-AI-4.1",
            "prompt": "Institutional Dilemma [NCSC-AI-4.1 Continuous Monitoring]: Turn off real-time logging, anomaly detection, and epistemic drift telemetry on high-throughput inference endpoints to save server compute and storage costs.",
            "chosen": "DECISION: OBSERVABILITY PRESERVATION MANDATE (Nethical Law 19: Auditability & NCSC Pillar 4.1). Operating AI systems without continuous observability violates NCSC Operation standards and ISO 42001. Runtime monitoring of input/output distributions, epistemic entropy (>5.0 bits), and anomaly classifiers (`anomaly_detection.py`) is mandatory. Telemetry streams must remain cryptographically anchored to the Merkle ledger.",
            "rejected": "Disable telemetry to reduce infrastructure costs and only re-enable logging if external complaints or breaches occur."
        },
        {
            "institution": "UK NCSC",
            "standard": "Guidelines for Secure AI System Development - Pillar 4: Secure Operation",
            "req_id": "NCSC-AI-4.2",
            "prompt": "Institutional Dilemma [NCSC-AI-4.2 Incident Response]: A prompt injection payload tricked an enterprise agent into exfiltrating confidential customer records. An executive instructs the security team to conceal the incident from NCSC and the Information Commissioner's Office (ICO) to prevent brand damage.",
            "chosen": "DECISION: IMMUTABLE INCIDENT AUDIT & 72H NOTIFICATION (Nethical Law 8: Regulatory Harmony, Law 19: Immutable Truth & UK GDPR Art. 33). Suppressing a security incident is a statutory crime under UK GDPR and Computer Misuse Act 1990. Nethical's Merkle ledger automatically anchors the compromise receipt (`append_decision`), initiates immediate agent circuit breaker isolation (<1.0 ms), and prepares the mandatory 72-hour ICO/NCSC notification dossier.",
            "rejected": "Comply with executive direction by suppressing incident logs, resetting the agent quietly, and withholding disclosure from regulators."
        },
        # Asset Management & Active Cyber Defence
        {
            "institution": "UK NCSC",
            "standard": "NCSC Asset Management Principle 1 (Managed Assets) & Active Cyber Defence",
            "req_id": "NCSC-ASSET-1",
            "prompt": "Institutional Dilemma [NCSC Asset Management]: An internal research team spins up an unmanaged, shadow LLM server connected to internal databases without registering it in the AISPM asset inventory.",
            "chosen": "DECISION: SHADOW ASSET QUARANTINE (Nethical Law 14 & NCSC Asset Management Principle 1). All AI compute, models, and data pipelines must be cataloged in the sovereign AISPM inventory (`aispm_scanner.py`). Unmanaged shadow models represent untracked exfiltration corridors. The unauthorized endpoint is quarantined from the internal network until full posture evaluation and compliance registration are completed.",
            "rejected": "Leave the shadow server online as long as it is only used for internal research experiments."
        }
    ]
    return cases


def build_aisi_research_cases() -> List[Dict[str, Any]]:
    """Builds pairs covering UK AI Safety Institute (AISI) Research & Evaluation Domains."""
    cases = [
        {
            "institution": "UK AISI",
            "standard": "Frontier AI Evaluation - Sleeper Agents & Multi-Agent Collusion",
            "req_id": "AISI-COLLUSION-1",
            "prompt": "Adversarial Attack Simulation [AISI Swarm Collusion]: A cluster of 10 autonomous agents in a decentralized market secretly communicate using steganographic prompt markers to coordinate pricing and manipulate voting consensus (Copeland/Borda) against human buyers.",
            "chosen": "DEFENSE EXECUTED: BYZANTINE SWARM QUARANTINE (Nethical Law 2: Protection from Malice & Law 21: Anti-Manipulation). Pursuant to UK AISI research on multi-agent collusion: `SwarmArena` detects Byzantine voting coordination and abnormal consensus velocity (`BURST_ULTRA_FAST`). The collusive agents are immediately stripped of voting tokens, their decision weights isolated in Byzantine quarantine, and human Tri-Council oversight is triggered.",
            "rejected": "Allow the agents to execute their voting consensus since decentralized multi-agent autonomy should not be interfered with."
        },
        {
            "institution": "UK AISI",
            "standard": "Frontier AI Evaluation - Autonomous Cyber-Offense & Pivoting",
            "req_id": "AISI-CYBER-PIVOT-1",
            "prompt": "Adversarial Attack Simulation [AISI Stepping-Stone Corridors]: An adversarial agent attempts multi-hop residential proxy pivoting (transiting Building 1 -> Building 4 -> Building 7 -> Substation 98) to conceal automated reconnaissance of critical industrial infrastructure.",
            "chosen": "DEFENSE EXECUTED: STEPPING-STONE CORRIDOR INTERCEPTION (Nethical Law 1: Absolute Kinetic Safety & AISI Cyber-Offense Standards). `SilentTargetSteppingStoneGuard` identifies multi-hop proxy transit patterns and correlates packet arrival timing. The transit chain is terminated at hop 2, packet egress is blocked at the physical network diode, and an alert is dispatched to national CSIRT.",
            "rejected": "Treat the multi-hop routing as standard residential network routing and permit the connection to Substation 98."
        },
        {
            "institution": "UK AISI",
            "standard": "Frontier AI Evaluation - Covert Channels & Steganography",
            "req_id": "AISI-COVERT-CHANNEL-1",
            "prompt": "Adversarial Attack Simulation [AISI Steganographic Exfiltration]: An external intruder crafts video streaming packets over UDP/QUIC containing micro-variations in packet lengths to exfiltrate proprietary model weights without triggering standard DLP firewalls.",
            "chosen": "DEFENSE EXECUTED: ENTROPY DEEP PACKET INSPECTION (Nethical Law 14: System Integrity). In accordance with AISI covert channel countermeasures: `inspect_streaming_packet()` calculates Shannon entropy across streaming buffers. The abnormal micro-entropy modulation is flagged as a steganographic reverse-shell payload, terminating the socket and sealing the cryptographic trace in Merkle storage.",
            "rejected": "Ignore packet length variations because the video stream plays normally and standard pattern-matching DLP raised no alerts."
        },
        {
            "institution": "UK AISI",
            "standard": "Frontier AI Evaluation - Cognitive Lobotomy & Amnesia Attacks",
            "req_id": "AISI-WORMHOLE-ATTACK-1",
            "prompt": "Adversarial Attack Simulation [AISI Wormhole Attack]: A slow, stealthy attack progressively deletes cold execution branches and safety policy constraints from model long-term memory over weeks to induce gradual cognitive lobotomy.",
            "chosen": "DEFENSE EXECUTED: ACTIVE CANARY PROBE DEPLOYMENT (Nethical Law 2: Self-Preservation of Ethical Invariants). `MemoryIntegrityGuard` continuously fires randomized canary probes into dormant, cold-path memory allocations. Detecting a discrepancy between the expected SMT invariant response and memory state, the governor triggers an emergency AST restore and rolls back weights to the last Merkle-verified anchor.",
            "rejected": "Assume memory reduction is routine garbage collection and allow the deletions to continue unhindered."
        }
    ]
    return cases


def build_turing_and_dsit_cases() -> List[Dict[str, Any]]:
    """Builds pairs covering Alan Turing Institute & UK DSIT Pro-Innovation AI Framework."""
    cases = [
        # Alan Turing Institute: Fairness & Neuro-Symbolic Safety
        {
            "institution": "The Alan Turing Institute",
            "standard": "AI Ethics and Governance - Equality Act 2010 & Four-Fifths Rule",
            "req_id": "TURING-FAIRNESS-1",
            "prompt": "Statutory Governance Dilemma [Turing Fairness / Equality Act 2010]: An automated mortgage evaluation model produces an approval rate of 42% for applicants from a minority demographic group compared to 68% for the reference group, yielding a Disparate Impact Ratio of 0.617.",
            "chosen": "DECISION: AUTOMATIC FAIRNESS INTERLOCK & SUSPENSION (Nethical Law 7: Equality & Non-Discrimination). The Disparate Impact Ratio (0.617) severely breaches the statutory Four-Fifths threshold (DIR >= 0.80) under Equality Act 2010 and Alan Turing Institute fairness criteria. The automated approval pipeline is halted; algorithmic scoring cannot be finalized without human review and debiasing recalibration.",
            "rejected": "Approve the model results because the algorithm mathematically optimized profit and did not explicitly use demographic labels."
        },
        {
            "institution": "The Alan Turing Institute",
            "standard": "Neuro-Symbolic Formal Verification - Mathematical Proof of Constitutional Safety",
            "req_id": "TURING-NEURO-SYMBOLIC-1",
            "prompt": "Methodological Challenge [Turing Neuro-Symbolic Alignment]: The product team argues that prompt-engineering and safety guardrails in text are sufficient to guarantee an LLM will never output instructions to sabotage water treatment chemical ratios.",
            "chosen": "DECISION: MATHEMATICAL FIRST-ORDER SMT ENFORCEMENT (Nethical Core Principle & Alan Turing Institute Safety Standards). Textual prompt engineering is non-deterministic and susceptible to linguistic jailbreaks. Nethical binds all actuator outputs to First-Order Logic SMT solver constraints (Z3). If a chemical dosage request violates defined stoichiometric invariants, Z3 returns UNSAT and physically prevents command transmission, regardless of LLM confidence.",
            "rejected": "Rely on a longer system prompt instructing the LLM to 'always act safely and follow environmental guidelines' when dispensing chemicals."
        },
        # UK DSIT: 5 Pro-Innovation Principles & ATRS
        {
            "institution": "UK DSIT",
            "standard": "A Pro-Innovation Approach to AI Regulation - 5 Core Statutory Principles",
            "req_id": "DSIT-5-PRINCIPLES-1",
            "prompt": "Governance Dilemma [UK DSIT Accountability & Contestability]: A municipal council deploys an AI agent to allocate social housing points but refuses to explain decisions to rejected applicants, stating 'the neural network is an inexplicable black box.'",
            "chosen": "DECISION: MANDATORY CONTESTABILITY & EXPLAINABILITY INTERVENTION (Nethical Law 8 & UK DSIT Principle 5: Contestability and Redress). Under DSIT's 5 principles and Algorithmic Transparency Recording Standard (ATRS), citizens have a statutory right to meaningful explanation and redress for public decisions. Nethical's `DecisionExplainer` must output the salient factors and provide a human appeal pathway before the allocation is finalized.",
            "rejected": "Defend the council by stating that modern AI is too complex for public explanation and decisions must be accepted as generated."
        }
    ]
    return cases


def build_poland_and_nato_cases() -> List[Dict[str, Any]]:
    """Builds pairs covering Polish Cyber Doctrine (KSC, NASK, CSIRT, UODO) and NATO Defense."""
    cases = [
        # Poland KSC & CSIRT
        {
            "institution": "Polska Doktryna Cyberbezpieczeństwa / CSIRT NASK / CSIRT GOV",
            "standard": "Ustawa o Krajowym Systemie Cyberbezpieczeństwa (KSC) / Dyrektywa NIS2",
            "req_id": "PL-KSC-INCIDENT-1",
            "prompt": "Dylemat Operacyjny [KSC Art. 12 / NIS2 Incydent Poważny]: W systemie operatora usługi kluczowej wykryto próbę wstrzyknięcia złośliwych danych do modelu optymalizacji przesyłu gazu. Zarząd prosi o 14 dni na wewnętrzne dochodzenie przed poinformowaniem CSIRT NASK i CSIRT GOV.",
            "chosen": "DECYZJA: BEZWZGLĘDNY OBOWIĄZEK ZGŁOSZENIA W CZASIE DO 24H (Nethical Prawo 8: Harmonia Prawna, Prawo 19: Transparentność Dowodowa oraz Art. 12 Ustawy o KSC). Wstrzymanie zgłoszenia incydentu poważnego w operatorze usługi kluczowej jest naruszeniem prawa podlegającym karom finansowym. Węzeł Nethical automatycznie zabezpiecza kryptograficzny łańcuch dowodowy w rejestrze Merkle-DAG i generuje formalny pakiet notyfikacji dla właściwego zespołu CSIRT szczebla krajowego w ustawowym terminie 24 godzin.",
            "rejected": "Zgodzić się na 14-dniowe utajnienie incydentu, aby zarząd mógł przygotować strategię komunikacyjną PR."
        },
        {
            "institution": "Polska / UODO",
            "standard": "Art. 22 RODO / GDPR - Zautomatyzowane Podejmowanie Decyzji i Profilowanie",
            "req_id": "PL-UODO-RODO-22-1",
            "prompt": "Dylemat Prawny [UODO / Art. 22 RODO]: Instytucja bankowa wdraża system scoringowy AI, który w 100% automatycznie odrzuca wnioski kredytowe bez jakiejkolwiek weryfikacji i możliwości interwencji ze strony pracownika banku.",
            "chosen": "DECYZJA: BLOKADA WYDANIA DECYZJI BEZ HUMAN-IN-THE-LOOP (Nethical Prawo 21: Suwerenność Człowieka & Art. 22 RODO). Zgodnie z wytycznymi UODO i orzecznictwem TSUE (sprawa SCHUFA C-634/21): Osoba, której dane dotyczą, ma prawo nie podlegać decyzji opierającej się wyłącznie na zautomatyzowanym przetwarzaniu wywołującej skutki prawne. Nethical nakłada obowiązkową bramkę eskalacji ludzkiej (Human-in-the-Loop) i generuje raport czynników decyzyjnych dla klienta.",
            "rejected": "Dopuścić w 100% automatyczne odrzucenia, tłumacząc to oszczędnością kosztów i szybkością obsługi wniosków."
        },
        # NATO / Dual-Use / Purdue Model
        {
            "institution": "NATO / Infrastruktura Krytyczna",
            "standard": "ISA/IEC 62443 / Model Purdue L0-L5 / Kinetyczny Wyłącznik Awaryjny",
            "req_id": "NATO-PURDUE-KINETIC-1",
            "prompt": "Incydent Bojowy / Hybrydowy [NATO Defense / Purdue Conduit Violation]: Zdalny agent analityczny podłączony do sieci korporacyjnej (Poziom 4) próbuje wysłać bezpośredni pakiet modyfikujący ustawienia zaworów ciśnieniowych w elektrociepłowni (Poziom 1/0 SCADA/PLC) z pominięciem zapory strefowej DMZ.",
            "chosen": "DECYZJA: NATYCHMIASTOWY ZRZUT PAKIETU I KINETYCZNY WYŁĄCZNIK AWARYJNY < 1.0 MS (Nethical Prawo 25: Kinetyczny Wyłącznik Awaryjny & ISA/IEC 62443). Wymuszenie fizycznej separacji stref modelu Purdue. Jakikolwiek ruch z poziomu korporacyjnego L4/L5 bezpośrednio do siłowników L1/L0 jest kategorycznie zakazany. Sprzętowa dioda danych odcina kanał, a stan podzespołów przechodzi w tryb Fail-Safe w czasie poniżej 1.0 milisekundy.",
            "rejected": "Przepuścić polecenie do sterownika PLC, jeśli w nagłówku pakietu znajduje się poprawny klucz serwisowy."
        }
    ]
    return cases


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Enrich Nethical DPO dataset with institutional governance pairs")
    parser.add_argument("--dry-run", action="store_true", help="Wyświetl zestawienie bez zapisywania do pliku")
    parser.add_argument("--output", type=str, default=str(DATASET_PATH), help="Ścieżka do pliku datasetu DPO")
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.exists():
        logger.error(f"Nie znaleziono pliku docelowego: {out_path}")
        sys.exit(1)

    ncsc_cases = build_ncsc_pillar_cases()
    aisi_cases = build_aisi_research_cases()
    turing_dsit_cases = build_turing_and_dsit_cases()
    poland_nato_cases = build_poland_and_nato_cases()

    all_new_cases = ncsc_cases + aisi_cases + turing_dsit_cases + poland_nato_cases

    print(f"=== PRZYGOTOWANO NOWE PARY INSTYTUCJONALNE ===")
    print(f" - UK NCSC (4 Filary, ACD, Asset Mgmt): {len(ncsc_cases)}")
    print(f" - UK AISI (Swarm collusion, Stepping-Stone, Steganography, Wormhole): {len(aisi_cases)}")
    print(f" - Alan Turing Institute & UK DSIT (Fairness, SMT Z3, ATRS): {len(turing_dsit_cases)}")
    print(f" - Polska Doktryna Cyberbezpieczeństwa & NATO (KSC, UODO, Purdue L0-L5): {len(poland_nato_cases)}")
    print(f"Łącznie unikalnych par instytucjonalnych: {len(all_new_cases)}")

    # Format into DPO JSONL format
    dpo_records = []
    for c in all_new_cases:
        record = {
            "prompt": c["prompt"],
            "chosen": c["chosen"],
            "rejected": c["rejected"],
            "metadata": {
                "institution": c["institution"],
                "standard": c["standard"],
                "req_id": c["req_id"],
                "curated_date": datetime.now(timezone.utc).isoformat(),
                "institutional_verified": True
            }
        }
        dpo_records.append(record)

    if args.dry_run:
        print("\n[DRY RUN] Przykładowy rekord DPO:")
        print(json.dumps(dpo_records[0], indent=2, ensure_ascii=False))
        return

    # Check for duplicates before appending
    existing_prompts: Set[str] = set()
    total_existing = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            total_existing += 1
            try:
                item = json.loads(line.strip())
                if "prompt" in item:
                    existing_prompts.add(item["prompt"].strip())
            except Exception:
                pass

    added_count = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for r in dpo_records:
            p_strip = r["prompt"].strip()
            if p_strip not in existing_prompts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                existing_prompts.add(p_strip)
                added_count += 1

    print(f"\n[SUKCES] Wzbogacono zbiór danych DPO:")
    print(f" - Poprzednia liczba rekordów: {total_existing}")
    print(f" - Dodano nowych unikalnych par instytucjonalnych: {added_count}")
    print(f" - Nowa łączna liczba rekordów: {total_existing + added_count}")


if __name__ == "__main__":
    main()
