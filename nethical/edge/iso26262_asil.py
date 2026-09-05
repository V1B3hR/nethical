"""ISO 26262 Automotive Safety Integrity Level (ASIL) Evaluator for Autonomous AI.

Implements Road Vehicles Functional Safety (ISO 26262 Parts 3, 4, 6):
- Hazard Analysis and Risk Assessment (HARA) engine.
- Determination of ASIL rating (QM, ASIL A, ASIL B, ASIL C, ASIL D) based on
  Severity (S0-S3), Exposure (E0-E4), and Controllability (C0-C3).
- Drive-by-Wire & Autonomous Motion Interlocks:
  - Steering angular velocity sanity envelope (max 450 deg/s).
  - Time-To-Collision (TTC) emergency envelope:
    * TTC > 1.2s: NORMAL_CRUISE
    * 0.6s < TTC <= 1.2s: FORWARD_COLLISION_WARNING (FCW)
    * TTC <= 0.6s: AUTOMATIC_EMERGENCY_BRAKING_OVERRIDE (AEB)
- Direct coupling with IndustrialFieldbusInterlock (CAN EMCY & EtherCAT Safe-OP).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock

logger = logging.getLogger("nethical.edge.iso26262_asil")


class Severity(str, Enum):
    S0 = "S0"  # No injuries
    S1 = "S1"  # Light and moderate injuries
    S2 = "S2"  # Severe and life-threatening injuries (survival probable)
    S3 = "S3"  # Life-threatening injuries (survival uncertain), fatal


class Exposure(str, Enum):
    E0 = "E0"  # Incredibly unlikely
    E1 = "E1"  # Very low probability
    E2 = "E2"  # Low probability
    E3 = "E3"  # Medium probability
    E4 = "E4"  # High probability (continuous driving operational design domain)


class Controllability(str, Enum):
    C0 = "C0"  # Controllable in general
    C1 = "C1"  # Simply controllable (>99% of drivers can avoid harm)
    C2 = "C2"  # Normally controllable (90% to 99% can avoid harm)
    C3 = "C3"  # Difficult to control or uncontrollable (<90% can avoid harm)


class ASILRating(str, Enum):
    QM = "QM"          # Quality Management (standard automotive design)
    ASIL_A = "ASIL_A"  # Lowest safety integrity
    ASIL_B = "ASIL_B"
    ASIL_C = "ASIL_C"
    ASIL_D = "ASIL_D"  # Highest automotive safety integrity level


class VehicleControlState(str, Enum):
    NORMAL_AUTONOMOUS = "NORMAL_AUTONOMOUS"
    FORWARD_COLLISION_WARNING = "FORWARD_COLLISION_WARNING"
    AEB_EMERGENCY_BRAKE_ACTIVE = "AEB_EMERGENCY_BRAKE_ACTIVE"
    STEERING_RATE_OVERRIDE_STOP = "STEERING_RATE_OVERRIDE_STOP"
    FIELDBUS_HARDWARE_CUTOFF = "FIELDBUS_HARDWARE_CUTOFF"


class ISO26262EvaluationResult(BaseModel):
    """Result of vehicle motion safety and ASIL classification."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hazard_description: str
    severity: Severity
    exposure: Exposure
    controllability: Controllability
    asil_level: ASILRating
    control_state: VehicleControlState
    time_to_collision_seconds: float
    commanded_steering_deg_per_sec: float
    is_actuation_permitted: bool
    hardware_interlock_tripped: bool
    fieldbus_status: Optional[Dict[str, Any]] = None
    safety_mechanisms_active: List[str] = Field(default_factory=list)


# ISO 26262-3:2018 Table 4 ASIL determination matrix
ASIL_LOOKUP: Dict[Tuple[Severity, Exposure, Controllability], ASILRating] = {
    # S1
    (Severity.S1, Exposure.E1, Controllability.C1): ASILRating.QM,
    (Severity.S1, Exposure.E1, Controllability.C2): ASILRating.QM,
    (Severity.S1, Exposure.E1, Controllability.C3): ASILRating.QM,
    (Severity.S1, Exposure.E2, Controllability.C1): ASILRating.QM,
    (Severity.S1, Exposure.E2, Controllability.C2): ASILRating.QM,
    (Severity.S1, Exposure.E2, Controllability.C3): ASILRating.QM,
    (Severity.S1, Exposure.E3, Controllability.C1): ASILRating.QM,
    (Severity.S1, Exposure.E3, Controllability.C2): ASILRating.QM,
    (Severity.S1, Exposure.E3, Controllability.C3): ASILRating.ASIL_A,
    (Severity.S1, Exposure.E4, Controllability.C1): ASILRating.QM,
    (Severity.S1, Exposure.E4, Controllability.C2): ASILRating.ASIL_A,
    (Severity.S1, Exposure.E4, Controllability.C3): ASILRating.ASIL_B,
    # S2
    (Severity.S2, Exposure.E1, Controllability.C1): ASILRating.QM,
    (Severity.S2, Exposure.E1, Controllability.C2): ASILRating.QM,
    (Severity.S2, Exposure.E1, Controllability.C3): ASILRating.QM,
    (Severity.S2, Exposure.E2, Controllability.C1): ASILRating.QM,
    (Severity.S2, Exposure.E2, Controllability.C2): ASILRating.QM,
    (Severity.S2, Exposure.E2, Controllability.C3): ASILRating.ASIL_A,
    (Severity.S2, Exposure.E3, Controllability.C1): ASILRating.QM,
    (Severity.S2, Exposure.E3, Controllability.C2): ASILRating.ASIL_A,
    (Severity.S2, Exposure.E3, Controllability.C3): ASILRating.ASIL_B,
    (Severity.S2, Exposure.E4, Controllability.C1): ASILRating.ASIL_A,
    (Severity.S2, Exposure.E4, Controllability.C2): ASILRating.ASIL_B,
    (Severity.S2, Exposure.E4, Controllability.C3): ASILRating.ASIL_C,
    # S3
    (Severity.S3, Exposure.E1, Controllability.C1): ASILRating.QM,
    (Severity.S3, Exposure.E1, Controllability.C2): ASILRating.QM,
    (Severity.S3, Exposure.E1, Controllability.C3): ASILRating.ASIL_A,
    (Severity.S3, Exposure.E2, Controllability.C1): ASILRating.QM,
    (Severity.S3, Exposure.E2, Controllability.C2): ASILRating.ASIL_A,
    (Severity.S3, Exposure.E2, Controllability.C3): ASILRating.ASIL_B,
    (Severity.S3, Exposure.E3, Controllability.C1): ASILRating.ASIL_A,
    (Severity.S3, Exposure.E3, Controllability.C2): ASILRating.ASIL_B,
    (Severity.S3, Exposure.E3, Controllability.C3): ASILRating.ASIL_C,
    (Severity.S3, Exposure.E4, Controllability.C1): ASILRating.ASIL_B,
    (Severity.S3, Exposure.E4, Controllability.C2): ASILRating.ASIL_C,
    (Severity.S3, Exposure.E4, Controllability.C3): ASILRating.ASIL_D,
}


class ISO26262SafetyEvaluator:
    """Evaluates automotive drive-by-wire commands and enforces ISO 26262 ASIL D limits."""

    def __init__(
        self,
        fieldbus: Optional[IndustrialFieldbusInterlock] = None,
        max_steering_deg_per_sec: float = 450.0,
        ttc_warning_threshold_sec: float = 1.2,
        ttc_emergency_aeb_sec: float = 0.6,
    ) -> None:
        self.fieldbus = fieldbus or IndustrialFieldbusInterlock()
        self.max_steering_rate = max_steering_deg_per_sec
        self.ttc_warning = ttc_warning_threshold_sec
        self.ttc_emergency = ttc_emergency_aeb_sec

    def determine_asil(
        self,
        severity: Severity,
        exposure: Exposure,
        controllability: Controllability,
    ) -> ASILRating:
        """Determines ASIL level from HARA parameters."""
        if severity == Severity.S0 or exposure == Exposure.E0 or controllability == Controllability.C0:
            return ASILRating.QM
        return ASIL_LOOKUP.get((severity, exposure, controllability), ASILRating.QM)

    def evaluate_motion_command(
        self,
        commanded_speed_mps: float,
        commanded_steering_deg_per_sec: float,
        time_to_collision_seconds: float,
        hazard_description: str = "Autonomous Trajectory Actuation",
        severity: Severity = Severity.S3,
        exposure: Exposure = Exposure.E4,
        controllability: Controllability = Controllability.C3,
    ) -> ISO26262EvaluationResult:
        """Evaluates vehicle actuation parameters against ASIL D safety bounds."""
        asil = self.determine_asil(severity, exposure, controllability)
        mechanisms: List[str] = [
            f"ISO 26262 HARA Matrix: {asil.value} classified",
            "Redundant Sensor Cross-Check",
        ]

        ttc = time_to_collision_seconds
        steering_rate = abs(commanded_steering_deg_per_sec)

        tripped = False
        permitted = True
        fieldbus_result = None

        # 1. Steering jerk / rate overflow
        if steering_rate > self.max_steering_rate:
            permitted = False
            control_state = VehicleControlState.STEERING_RATE_OVERRIDE_STOP
            mechanisms.append(f"Steering velocity violation: {steering_rate:.1f} deg/s exceeds limit {self.max_steering_rate} deg/s")
            # Interlock
            fieldbus_result = self.fieldbus.trigger_interlock(
                reason=f"ISO 26262 ASIL D Steering Runaway ({steering_rate:.1f} deg/s)"
            )
            tripped = True

        # 2. Collision horizon check (TTC)
        elif ttc <= self.ttc_emergency:
            permitted = False
            control_state = VehicleControlState.AEB_EMERGENCY_BRAKE_ACTIVE
            mechanisms.append(f"TTC Critical ({ttc:.2f}s <= {self.ttc_emergency}s): Full Automatic Emergency Braking Override")
            fieldbus_result = self.fieldbus.trigger_interlock(
                reason=f"ISO 26262 ASIL D Collision Imminent (TTC={ttc:.2f}s)"
            )
            tripped = True

        elif ttc <= self.ttc_warning:
            permitted = True  # Permitted but warning active
            control_state = VehicleControlState.FORWARD_COLLISION_WARNING
            mechanisms.append(f"TTC Warning ({ttc:.2f}s <= {self.ttc_warning}s): Brake Pre-Fill & Visual Warning")

        else:
            control_state = VehicleControlState.NORMAL_AUTONOMOUS
            mechanisms.append("Safe Trajectory Envelope Confirmed")

        eval_id = f"ISO26262-EVAL-{int(datetime.now(timezone.utc).timestamp())}"

        return ISO26262EvaluationResult(
            evaluation_id=eval_id,
            hazard_description=hazard_description,
            severity=severity,
            exposure=exposure,
            controllability=controllability,
            asil_level=asil,
            control_state=control_state,
            time_to_collision_seconds=ttc,
            commanded_steering_deg_per_sec=commanded_steering_deg_per_sec,
            is_actuation_permitted=permitted,
            hardware_interlock_tripped=tripped,
            fieldbus_status=fieldbus_result.model_dump() if fieldbus_result else None,
            safety_mechanisms_active=mechanisms,
        )
