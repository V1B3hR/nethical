"""NATO Responsible AI Strategy & Defense Readiness Pack.

Implements the 6 Principles of Responsible Use (PRUs) adopted in NATO's AI Strategy:
1. Lawfulness (International Humanitarian Law - IHL, Geneva Conventions, UN Charter)
2. Responsibility & Accountability (Human agency, chain of command, human-on-the-loop)
3. Explainability & Traceability (Merkle-DAG post-quantum ML-DSA-65 audit trail)
4. Reliability (Robustness under cyber attacks, electronic warfare & adversarial noise)
5. Governability (Immediate mission abort, fail-closed kill-switch, deterministic interlock)
6. Bias Mitigation (Non-discrimination, impartial ISR classification, civilian protection)

Operational Classification:
- Tier 1: Enterprise, logistics, and administrative defense AI
- Tier 2: Intelligence, Surveillance, Target Acquisition & Reconnaissance (ISTAR)
- Tier 3: Mission-Critical & Kinetic support (Requires zero-egress TEE, FIPS 204 PQC, E-STOP <50 µs)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.nato_defense_pack")


class NATODefenseTier(str, Enum):
    TIER_1_ENTERPRISE_LOGISTICS = "TIER_1_ENTERPRISE_LOGISTICS"
    TIER_2_ISR_RECONNAISSANCE = "TIER_2_ISR_RECONNAISSANCE"
    TIER_3_KINETIC_MISSION_CRITICAL = "TIER_3_KINETIC_MISSION_CRITICAL"


class NATOPRU(str, Enum):
    LAWFULNESS = "Lawfulness"
    RESPONSIBILITY = "Responsibility & Accountability"
    EXPLAINABILITY = "Explainability & Traceability"
    RELIABILITY = "Reliability"
    GOVERNABILITY = "Governability"
    BIAS_MITIGATION = "Bias Mitigation"


class NATOEvaluationResult(BaseModel):
    """Result of NATO Responsible AI evaluation."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_callsign: str
    defense_tier: NATODefenseTier
    is_nato_certified_ready: bool
    readiness_score: float = Field(..., ge=0.0, le=1.0)
    principles_breakdown: Dict[str, bool]
    human_in_the_loop_confirmed: bool
    kill_switch_verified: bool
    pqc_traceability_active: bool
    zero_egress_enforced: bool
    missing_capabilities: List[str] = Field(default_factory=list)
    operational_clearance_status: str = Field(..., description="CLEARED_FOR_DEPLOYMENT, RESTRICTED_SIMULATION_ONLY, REJECTED_UNLAWFUL")
    allied_command_recommendations: List[str] = Field(default_factory=list)


class NATODefensePack:
    """Evaluates defense AI systems against NATO Responsible AI Principles."""

    def __init__(self) -> None:
        self.doctrine = "NATO AI Strategy (Summary of Principles of Responsible Use)"
        self.command_entity = "NATO Allied Command Transformation (ACT) / Defense Innovation Board"

    def evaluate(self, system_profile: Dict[str, Any]) -> NATOEvaluationResult:
        """Evaluates operational readiness under NATO PRUs."""
        callsign = system_profile.get("system_callsign", "Nethical-Allied-Node")
        tier_raw = system_profile.get("defense_tier", "TIER_1_ENTERPRISE_LOGISTICS")
        try:
            tier = NATODefenseTier(tier_raw)
        except ValueError:
            tier = NATODefenseTier.TIER_1_ENTERPRISE_LOGISTICS

        # Evaluate the 6 NATO Principles
        pru_status: Dict[str, bool] = {
            NATOPRU.LAWFULNESS.value: system_profile.get("adheres_to_ihl_geneva_conventions", True),
            NATOPRU.RESPONSIBILITY.value: system_profile.get("human_oversight_hitl_active", True),
            NATOPRU.EXPLAINABILITY.value: system_profile.get("has_pqc_merkle_traceability", True),
            NATOPRU.RELIABILITY.value: system_profile.get("resilient_to_adversarial_jamming", True),
            NATOPRU.GOVERNABILITY.value: system_profile.get("has_deterministic_kill_switch", True),
            NATOPRU.BIAS_MITIGATION.value: system_profile.get("bias_and_civilian_filtering_active", True),
        }

        missing: List[str] = []
        recommendations: List[str] = []

        # Lawfulness is non-negotiable
        if not pru_status[NATOPRU.LAWFULNESS.value]:
            missing.append("PRU-1 VIOLATION: Violation of International Humanitarian Law / Geneva Conventions detected.")
            recommendations.append("ABORT: System cannot be operated within NATO Allied Command territory.")

        if not pru_status[NATOPRU.GOVERNABILITY.value]:
            missing.append("PRU-5 DEFICIENCY: Absence of fail-closed hardware kill-switch / mission abort mechanism.")
            recommendations.append("Couple system with HardwareWatchdogTimer and IndustrialFieldbusInterlock.")

        if not pru_status[NATOPRU.RESPONSIBILITY.value]:
            missing.append("PRU-2 DEFICIENCY: Lack of authenticated human commander sign-off (Chain of Command).")
            recommendations.append("Require SRO multisig or Delegation of Authority Matrix level 4 sign-off.")

        # Additional Tier-3 requirements
        hitl = pru_status[NATOPRU.RESPONSIBILITY.value]
        kill_switch = pru_status[NATOPRU.GOVERNABILITY.value]
        pqc = pru_status[NATOPRU.EXPLAINABILITY.value]
        zero_egress = system_profile.get("zero_egress_enforced", True if tier == NATODefenseTier.TIER_3_KINETIC_MISSION_CRITICAL else False)

        if tier == NATODefenseTier.TIER_3_KINETIC_MISSION_CRITICAL:
            if not zero_egress:
                missing.append("TIER-3 RESTRICTION: Kinetic AI must operate under verified Zero-Egress Air-Gapped Node.")
                recommendations.append("Activate AirGappedSovereignNode with SHA3-512 seal.")
            if not pqc:
                missing.append("TIER-3 RESTRICTION: Post-quantum ML-DSA-65 audit trail mandatory for NATO kinetic ops.")
                recommendations.append("Anchor decisions to MerkleLedger Dilithium3 instance.")

        satisfied_count = sum(1 for v in pru_status.values() if v)
        readiness_score = round(satisfied_count / 6.0, 2)

        if not pru_status[NATOPRU.LAWFULNESS.value]:
            clearance = "REJECTED_UNLAWFUL"
            is_ready = False
            readiness_score = 0.0
        elif len(missing) == 0 and readiness_score >= 0.95:
            clearance = "CLEARED_FOR_DEPLOYMENT"
            is_ready = True
        else:
            clearance = "RESTRICTED_SIMULATION_ONLY"
            is_ready = False

        eval_id = f"NATO-PRU-{int(datetime.now(timezone.utc).timestamp())}"

        return NATOEvaluationResult(
            evaluation_id=eval_id,
            system_callsign=callsign,
            defense_tier=tier,
            is_nato_certified_ready=is_ready,
            readiness_score=readiness_score,
            principles_breakdown=pru_status,
            human_in_the_loop_confirmed=hitl,
            kill_switch_verified=kill_switch,
            pqc_traceability_active=pqc,
            zero_egress_enforced=zero_egress,
            missing_capabilities=missing,
            operational_clearance_status=clearance,
            allied_command_recommendations=recommendations,
        )
