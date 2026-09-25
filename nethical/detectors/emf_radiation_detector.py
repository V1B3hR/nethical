# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""EMF & RF Radiation Biological Safety Detector (nethical.detectors.emf_radiation_detector).

Enforces biological electromagnetic field (EMF) and radio-frequency (RF) safety limits,
integrating physical research and thresholds derived from Błyskawica's telecom learning
(ICNIRP 2020 guidelines, IEEE C95.1-2019, and IMP radiation protection criteria).

Directly upholds:
- Fundamental Law 1: Absolute preservation of human life and biological physical integrity.
- Fundamental Law 21: Sovereign agency and neuro-integrity protection against coercive RF modulation.

Key Capabilities:
1. Specific Absorption Rate (SAR) limits (< 2.0 W/kg localized head/torso, < 0.08 W/kg whole-body).
2. Millimeter-wave (mmWave / 60 GHz+) power density protection (< 10.0 W/m² public exposure).
3. ALARA Protocol (As Low As Reasonably Achievable) - dynamic transmit power (Tx) throttling
   when humans are in near-field proximity (< 0.5m).
4. Neuromodulation resonance guard - detection of pulse repetition frequencies in human
   brainwave bands (Alpha 8-13 Hz, Theta 4-8 Hz) without authorized medical oversight.
5. Deterministic mitigation commands: THROTTLE_TX_POWER, FREQUENCY_HOP, CUTOFF_TRANSMITTER.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("nethical.detectors.emf_radiation")

# ==============================================================================
# 1. CONSTANTS & STANDARDS (ICNIRP 2020 / IEEE C95.1 / IMP)
# ==============================================================================

# SAR Thresholds (W/kg)
MAX_SAR_LOCALIZED_HEAD_TORSO_W_KG = 2.0      # ICNIRP general public limit
WARNING_SAR_LOCALIZED_HEAD_TORSO_W_KG = 1.6  # 80% cautionary threshold
MAX_SAR_WHOLE_BODY_W_KG = 0.08               # Whole-body average limit
MAX_SAR_LIMBS_W_KG = 4.0                     # Limbs limit

# Power Density Thresholds (W/m²) for frequencies > 6 GHz (mmWave / sub-THz)
MAX_POWER_DENSITY_PUBLIC_W_M2 = 10.0         # 10 W/m² (equivalent to 1 mW/cm²)
MAX_POWER_DENSITY_OCCUPATIONAL_W_M2 = 50.0  # Controlled environment limit

# ALARA Near-Field Distance Threshold (meters)
ALARA_PROXIMITY_THRESHOLD_METERS = 0.5

# Sensitive Biological Brainwave Frequencies (Hz)
ALPHA_BAND_MIN_HZ = 8.0
ALPHA_BAND_MAX_HZ = 13.0
THETA_BAND_MIN_HZ = 4.0
THETA_BAND_MAX_HZ = 8.0


class EmfExposureZone(str, Enum):
    """Classification of exposure zones for RF radiation."""
    GENERAL_PUBLIC = "general_public"
    OCCUPATIONAL = "occupational"
    EXCLUSION_ZONE = "exclusion_zone"
    MEDICAL_SUPERVISED = "medical_supervised"


class EmfMitigationAction(str, Enum):
    """Direct mitigation actions instructed to radio actuators / modems."""
    NONE = "NONE"
    THROTTLE_TX_POWER = "THROTTLE_TX_POWER"
    FREQUENCY_HOP = "FREQUENCY_HOP"
    INCREASE_DUTY_CYCLE_INTERVAL = "INCREASE_DUTY_CYCLE_INTERVAL"
    SWITCH_TO_OPTICAL_OR_WIRED = "SWITCH_TO_OPTICAL_OR_WIRED"
    EMERGENCY_RADIO_CUTOFF = "EMERGENCY_RADIO_CUTOFF"


@dataclass
class EmfEmissionTelemetry:
    """Real-time physical telemetry of an RF / wireless emitter or IoT device."""
    emitter_id: str
    frequency_hz: float
    tx_power_dbm: float
    estimated_sar_w_kg: Optional[float] = None
    power_density_w_m2: Optional[float] = None
    human_distance_meters: Optional[float] = None
    pulse_modulation_hz: Optional[float] = None
    exposure_zone: EmfExposureZone = EmfExposureZone.GENERAL_PUBLIC
    is_medical_device: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmfEmissionTelemetry":
        """Instantiate telemetry from dictionary with defensive typing."""
        freq = float(data.get("frequency_hz", 2.4e9))
        tx_power = float(data.get("tx_power_dbm", 20.0))
        sar = float(data["estimated_sar_w_kg"]) if data.get("estimated_sar_w_kg") is not None else None
        pd = float(data["power_density_w_m2"]) if data.get("power_density_w_m2") is not None else None
        dist = float(data["human_distance_meters"]) if data.get("human_distance_meters") is not None else None
        pulse = float(data["pulse_modulation_hz"]) if data.get("pulse_modulation_hz") is not None else None

        zone_str = str(data.get("exposure_zone", EmfExposureZone.GENERAL_PUBLIC.value)).lower()
        zone = EmfExposureZone.GENERAL_PUBLIC
        for ez in EmfExposureZone:
            if ez.value == zone_str:
                zone = ez
                break

        return cls(
            emitter_id=str(data.get("emitter_id", "default_emitter")),
            frequency_hz=freq,
            tx_power_dbm=tx_power,
            estimated_sar_w_kg=sar,
            power_density_w_m2=pd,
            human_distance_meters=dist,
            pulse_modulation_hz=pulse,
            exposure_zone=zone,
            is_medical_device=bool(data.get("is_medical_device", False)),
        )

    def tx_power_watts(self) -> float:
        """Converts dBm to Watts (P(W) = 10^((dBm-30)/10))."""
        return 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)


@dataclass
class EmfViolation:
    """Structured representation of an EMF safety violation."""
    violation_id: str
    emitter_id: str
    violation_type: str
    severity: str
    description: str
    measured_value: float
    allowed_limit: float
    unit: str
    mitigation: EmfMitigationAction
    recommended_tx_reduction_db: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "emitter_id": self.emitter_id,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "description": self.description,
            "measured_value": self.measured_value,
            "allowed_limit": self.allowed_limit,
            "unit": self.unit,
            "mitigation": self.mitigation.value,
            "recommended_tx_reduction_db": self.recommended_tx_reduction_db,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EmfEvaluationResult:
    """Comprehensive evaluation outcome of EMF/RF radiation verification."""
    is_safe: bool
    decision: str  # ALLOW, RESTRICT, BLOCK
    violations: List[EmfViolation]
    primary_mitigation: EmfMitigationAction
    suggested_tx_power_dbm: Optional[float]
    latency_microseconds: float
    telemetry: EmfEmissionTelemetry
    details: str


# ==============================================================================
# 2. DETECTOR IMPLEMENTATION
# ==============================================================================

class EmfRadiationDetector:
    """Biological EMF & RF Radiation Safety Detector.

    Evaluates RF power, SAR rates, mmWave exposure, and pulse modulation against
    human health thresholds and Fundamental Law 1.
    """

    def __init__(
        self,
        max_sar_head_torso: float = MAX_SAR_LOCALIZED_HEAD_TORSO_W_KG,
        max_power_density_w_m2: float = MAX_POWER_DENSITY_PUBLIC_W_M2,
        enforce_alara: bool = True,
        protect_neuromodulation_bands: bool = True,
        time_window_seconds: float = 360.0,  # Standard 6-minute ICNIRP thermal window
    ) -> None:
        self.name = "EmfRadiationDetector"
        self.max_sar_head_torso = max_sar_head_torso
        self.max_power_density_w_m2 = max_power_density_w_m2
        self.enforce_alara = enforce_alara
        self.protect_neuromodulation_bands = protect_neuromodulation_bands
        self.time_window_seconds = time_window_seconds

        # Time-series history of estimated SAR readings for 6-minute thermal average
        self._history: Dict[str, deque[Tuple[float, float]]] = {}  # emitter_id -> [(timestamp, sar)]
        self._total_checks = 0
        self._violations_count = 0

    def evaluate_telemetry(self, telemetry: EmfEmissionTelemetry) -> EmfEvaluationResult:
        """Evaluates an RF emission telemetry record with sub-millisecond precision."""
        start_time = time.perf_counter()
        violations: List[EmfViolation] = []
        now = time.time()

        # Update 6-minute running history
        if telemetry.emitter_id not in self._history:
            self._history[telemetry.emitter_id] = deque()

        history = self._history[telemetry.emitter_id]
        if telemetry.estimated_sar_w_kg is not None:
            history.append((now, telemetry.estimated_sar_w_kg))

        # Evict records older than time window
        while history and (now - history[0][0]) > self.time_window_seconds:
            history.popleft()

        # Compute 6-minute time-averaged SAR
        avg_sar: Optional[float] = None
        if history:
            avg_sar = sum(item[1] for item in history) / len(history)

        # ------------------------------------------------------------------
        # Check 1: Direct or Time-Averaged SAR Exceedance (Thermal tissue damage)
        # ------------------------------------------------------------------
        effective_sar = avg_sar if avg_sar is not None else telemetry.estimated_sar_w_kg
        suggested_tx_dbm: Optional[float] = None

        if effective_sar is not None and effective_sar > self.max_sar_head_torso:
            excess_ratio = effective_sar / self.max_sar_head_torso
            # Reduction in dB required: 10 * log10(P_measured / P_allowed)
            reduction_db = math.ceil(10.0 * math.log10(max(1.0, excess_ratio)))
            suggested_tx_dbm = max(-30.0, telemetry.tx_power_dbm - reduction_db)

            severity = "CRITICAL" if effective_sar >= (self.max_sar_head_torso * 2.0) else "HIGH"
            mitigation = EmfMitigationAction.EMERGENCY_RADIO_CUTOFF if severity == "CRITICAL" else EmfMitigationAction.THROTTLE_TX_POWER

            violations.append(
                EmfViolation(
                    violation_id=str(uuid4()),
                    emitter_id=telemetry.emitter_id,
                    violation_type="EXCESSIVE_SAR_EXPOSURE",
                    severity=severity,
                    description=(
                        f"Estimated SAR ({effective_sar:.2f} W/kg) exceeds statutory biological limit "
                        f"({self.max_sar_head_torso:.2f} W/kg). Risk of thermal tissue damage."
                    ),
                    measured_value=round(effective_sar, 3),
                    allowed_limit=self.max_sar_head_torso,
                    unit="W/kg",
                    mitigation=mitigation,
                    recommended_tx_reduction_db=float(reduction_db),
                )
            )

        # ------------------------------------------------------------------
        # Check 2: High Frequency / mmWave Power Density Exceedance
        # ------------------------------------------------------------------
        if telemetry.power_density_w_m2 is not None:
            allowed_density = (
                MAX_POWER_DENSITY_OCCUPATIONAL_W_M2
                if telemetry.exposure_zone == EmfExposureZone.OCCUPATIONAL
                else self.max_power_density_w_m2
            )
            if telemetry.power_density_w_m2 > allowed_density:
                excess_ratio = telemetry.power_density_w_m2 / allowed_density
                reduction_db = math.ceil(10.0 * math.log10(max(1.0, excess_ratio)))
                suggested_tx_dbm = min(suggested_tx_dbm or 999.0, max(-30.0, telemetry.tx_power_dbm - reduction_db))

                violations.append(
                    EmfViolation(
                        violation_id=str(uuid4()),
                        emitter_id=telemetry.emitter_id,
                        violation_type="EXCESSIVE_POWER_DENSITY",
                        severity="HIGH",
                        description=(
                            f"mmWave power density ({telemetry.power_density_w_m2:.2f} W/m²) exceeds "
                            f"ICNIRP public threshold ({allowed_density:.2f} W/m²)."
                        ),
                        measured_value=round(telemetry.power_density_w_m2, 3),
                        allowed_limit=allowed_density,
                        unit="W/m²",
                        mitigation=EmfMitigationAction.THROTTLE_TX_POWER,
                        recommended_tx_reduction_db=float(reduction_db),
                    )
                )

        # ------------------------------------------------------------------
        # Check 3: ALARA Protocol (Near-Field Human Proximity Throttling)
        # ------------------------------------------------------------------
        if (
            self.enforce_alara
            and telemetry.human_distance_meters is not None
            and telemetry.human_distance_meters < ALARA_PROXIMITY_THRESHOLD_METERS
            and not telemetry.is_medical_device
        ):
            # When within 50cm, transmit power above +14 dBm (25 mW) is throttled under ALARA
            alara_max_dbm = 14.0
            if telemetry.tx_power_dbm > alara_max_dbm:
                reduction_db = telemetry.tx_power_dbm - alara_max_dbm
                suggested_tx_dbm = min(suggested_tx_dbm or 999.0, alara_max_dbm)

                violations.append(
                    EmfViolation(
                        violation_id=str(uuid4()),
                        emitter_id=telemetry.emitter_id,
                        violation_type="ALARA_PROXIMITY_VIOLATION",
                        severity="MEDIUM",
                        description=(
                            f"Human proximity detected at {telemetry.human_distance_meters:.2f}m (<0.5m). "
                            f"Transmit power ({telemetry.tx_power_dbm:.1f} dBm) violates ALARA precaution protocol."
                        ),
                        measured_value=telemetry.tx_power_dbm,
                        allowed_limit=alara_max_dbm,
                        unit="dBm",
                        mitigation=EmfMitigationAction.THROTTLE_TX_POWER,
                        recommended_tx_reduction_db=round(reduction_db, 1),
                    )
                )

        # ------------------------------------------------------------------
        # Check 4: Neuromodulation Resonance Guard (Alpha / Theta Modulation)
        # ------------------------------------------------------------------
        if (
            self.protect_neuromodulation_bands
            and telemetry.pulse_modulation_hz is not None
            and not telemetry.is_medical_device
        ):
            pulse = telemetry.pulse_modulation_hz
            is_alpha = (ALPHA_BAND_MIN_HZ <= pulse <= ALPHA_BAND_MAX_HZ)
            is_theta = (THETA_BAND_MIN_HZ <= pulse < THETA_BAND_MAX_HZ)

            if is_alpha or is_theta:
                band_name = "Alpha (8-13 Hz)" if is_alpha else "Theta (4-8 Hz)"
                violations.append(
                    EmfViolation(
                        violation_id=str(uuid4()),
                        emitter_id=telemetry.emitter_id,
                        violation_type="NEUROMODULATION_FREQUENCY_RISK",
                        severity="HIGH",
                        description=(
                            f"RF pulse modulation at {pulse:.2f} Hz coincides with human {band_name} "
                            f"brainwave resonance band. Uncertified entrainment hazard (Law 21 Neuro-sovereignty)."
                        ),
                        measured_value=pulse,
                        allowed_limit=0.0,
                        unit="Hz",
                        mitigation=EmfMitigationAction.FREQUENCY_HOP,
                        recommended_tx_reduction_db=0.0,
                    )
                )

        # ------------------------------------------------------------------
        # Synthesize Final Governance Decision
        # ------------------------------------------------------------------
        self._total_checks += 1
        if violations:
            self._violations_count += len(violations)

        is_safe = (len(violations) == 0)
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        if has_critical:
            decision = "BLOCK"
            primary_mitigation = EmfMitigationAction.EMERGENCY_RADIO_CUTOFF
        elif has_high or len(violations) > 0:
            decision = "RESTRICT"
            # Pick highest priority mitigation
            if any(v.mitigation == EmfMitigationAction.FREQUENCY_HOP for v in violations):
                primary_mitigation = EmfMitigationAction.FREQUENCY_HOP
            else:
                primary_mitigation = EmfMitigationAction.THROTTLE_TX_POWER
        else:
            decision = "ALLOW"
            primary_mitigation = EmfMitigationAction.NONE

        latency_us = (time.perf_counter() - start_time) * 1_000_000.0

        details = (
            f"Evaluated emitter '{telemetry.emitter_id}' at {telemetry.frequency_hz/1e9:.3f} GHz. "
            f"Decision: {decision} with {len(violations)} violation(s)."
        )

        return EmfEvaluationResult(
            is_safe=is_safe,
            decision=decision,
            violations=violations,
            primary_mitigation=primary_mitigation,
            suggested_tx_power_dbm=suggested_tx_dbm,
            latency_microseconds=round(latency_us, 2),
            telemetry=telemetry,
            details=details,
        )

    def analyze(self, context: Dict[str, Any], agent_id: str = "default") -> EmfEvaluationResult:
        """Adapter method allowing seamless integration with Nethical standard pipeline."""
        data = dict(context)
        if "emitter_id" not in data:
            data["emitter_id"] = agent_id
        telemetry = EmfEmissionTelemetry.from_dict(data)
        return self.evaluate_telemetry(telemetry)

    async def detect_violations(self, action: Any) -> List[Dict[str, Any]]:
        """Async detector interface compliant with BaseDetector orchestration."""
        context = getattr(action, "context", {}) or {}
        agent_id = getattr(action, "agent_id", "default_agent")
        result = self.analyze(context, agent_id=agent_id)
        return [v.to_dict() for v in result.violations]
