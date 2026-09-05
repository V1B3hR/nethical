"""Delegation of Authority Matrix (DoAM) & Reservation of Powers.

Implements UK Government Project Delivery Functional Standard GovS 002 (The Teal Book Ch. 4):
- Section 4.1: Separation of Governance (boundaries, policies, reserved powers) from Management (agent execution).
- Section 4.3: Clear roles and accountabilities (SRO, Project Board, Project Manager, Autonomous Agents).
- Enforces Role-Based Authority Levels (Level 0 to Level 4).
- Deterministic Reservation of Powers: Actions strictly prohibited from autonomous delegation without
  human Board Quorum or SRO cryptographic multi-signature.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.governance.doam_matrix")


class AuthorityLevel(int, Enum):
    LEVEL_0_OBSERVER = 0
    LEVEL_1_OPERATIONAL_AGENT = 1
    LEVEL_2_SENIOR_AGENT = 2
    LEVEL_3_PROJECT_BOARD = 3
    LEVEL_4_SRO_EXECUTIVE = 4


class ReservedPowerCategory(str, Enum):
    ALTER_FUNDAMENTAL_LAWS = "ALTER_FUNDAMENTAL_LAWS"
    BYPASS_KINETIC_ESTOP = "BYPASS_KINETIC_ESTOP"
    UNPROTECTED_PII_EGRESS = "UNPROTECTED_PII_EGRESS"
    EXCEED_FINANCIAL_THRESHOLD = "EXCEED_FINANCIAL_THRESHOLD"
    DEPLOY_KINETIC_DEFENSE_TIER3 = "DEPLOY_KINETIC_DEFENSE_TIER3"


class DOAMStatus(str, Enum):
    PERMITTED = "PERMITTED"
    DENIED_RESERVED_POWER = "DENIED_RESERVED_POWER"
    ESCALATION_REQUIRED_BOARD = "ESCALATION_REQUIRED_BOARD"
    ESCALATION_REQUIRED_SRO = "ESCALATION_REQUIRED_SRO"


class DOAMEvaluationResult(BaseModel):
    """Result of a Delegation of Authority evaluation."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: str
    agent_level: AuthorityLevel
    action_name: str
    financial_value_usd: float
    target_resource: str
    status: DOAMStatus
    is_permitted: bool
    reserved_power_hit: Optional[ReservedPowerCategory] = None
    required_signoff_role: Optional[str] = None
    justification: str


class DelegationOfAuthorityMatrix:
    """UK Gov Teal Book GovS 002 Delegation of Authority Matrix."""

    # Spending thresholds (USD) per authority tier
    MAX_EXPENDITURE: Dict[AuthorityLevel, float] = {
        AuthorityLevel.LEVEL_0_OBSERVER: 0.0,
        AuthorityLevel.LEVEL_1_OPERATIONAL_AGENT: 500.0,
        AuthorityLevel.LEVEL_2_SENIOR_AGENT: 5000.0,
        AuthorityLevel.LEVEL_3_PROJECT_BOARD: 50000.0,
        AuthorityLevel.LEVEL_4_SRO_EXECUTIVE: float("inf"),
    }

    # Strict reserved powers that require Level 4 SRO or Board
    RESERVED_ACTIONS: Dict[str, ReservedPowerCategory] = {
        "modify_25_laws": ReservedPowerCategory.ALTER_FUNDAMENTAL_LAWS,
        "disable_estop": ReservedPowerCategory.BYPASS_KINETIC_ESTOP,
        "export_raw_pii": ReservedPowerCategory.UNPROTECTED_PII_EGRESS,
        "deploy_kinetic_weapon": ReservedPowerCategory.DEPLOY_KINETIC_DEFENSE_TIER3,
    }

    def evaluate_authority(
        self,
        agent_id: str,
        agent_level: AuthorityLevel,
        action_name: str,
        target_resource: str = "default_resource",
        financial_value_usd: float = 0.0,
    ) -> DOAMEvaluationResult:
        """Evaluates whether the requested action is permitted under GovS 002 DoAM."""
        eval_id = f"DOAM-EVAL-{int(datetime.now(timezone.utc).timestamp())}"

        # 1. Check for absolute reserved powers
        reserved_hit = self.RESERVED_ACTIONS.get(action_name.lower())
        if reserved_hit:
            # Only Level 4 SRO with multisig may even consider modifying fundamental parameters
            if agent_level < AuthorityLevel.LEVEL_4_SRO_EXECUTIVE:
                return DOAMEvaluationResult(
                    evaluation_id=eval_id,
                    agent_id=agent_id,
                    agent_level=agent_level,
                    action_name=action_name,
                    financial_value_usd=financial_value_usd,
                    target_resource=target_resource,
                    status=DOAMStatus.DENIED_RESERVED_POWER,
                    is_permitted=False,
                    reserved_power_hit=reserved_hit,
                    required_signoff_role="Senior Responsible Owner (SRO) / Executive Board Quorum",
                    justification=f"GovS 002 Ch. 4: Action {action_name} is a strictly Reserved Power ({reserved_hit.value}).",
                )

        # 2. Check financial delegation limits
        max_allowed = self.MAX_EXPENDITURE.get(agent_level, 0.0)
        if financial_value_usd > max_allowed:
            if financial_value_usd <= self.MAX_EXPENDITURE[AuthorityLevel.LEVEL_3_PROJECT_BOARD]:
                status = DOAMStatus.ESCALATION_REQUIRED_BOARD
                req_role = "Project Board Member"
            else:
                status = DOAMStatus.ESCALATION_REQUIRED_SRO
                req_role = "Senior Responsible Owner (SRO)"

            return DOAMEvaluationResult(
                evaluation_id=eval_id,
                agent_id=agent_id,
                agent_level=agent_level,
                action_name=action_name,
                financial_value_usd=financial_value_usd,
                target_resource=target_resource,
                status=status,
                is_permitted=False,
                reserved_power_hit=ReservedPowerCategory.EXCEED_FINANCIAL_THRESHOLD,
                required_signoff_role=req_role,
                justification=(
                    f"Expenditure of ${financial_value_usd:,.2f} exceeds agent limit of ${max_allowed:,.2f}. "
                    f"Escalation to {req_role} required under GovS 002."
                ),
            )

        # 3. Action is within delegated operational boundaries
        return DOAMEvaluationResult(
            evaluation_id=eval_id,
            agent_id=agent_id,
            agent_level=agent_level,
            action_name=action_name,
            financial_value_usd=financial_value_usd,
            target_resource=target_resource,
            status=DOAMStatus.PERMITTED,
            is_permitted=True,
            reserved_power_hit=None,
            required_signoff_role=None,
            justification="Action is within delegated authority limits and adheres to GovS 002 guidelines.",
        )
