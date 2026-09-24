# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital & Stratospheric RF Jamming Detector (nethical.space.detectors.jamming_detector).

Detects uplink, downlink, and cross-link electronic warfare (EW) jamming,
barrage noise, and carrier degradation against space communication links.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from nethical.space.models import LinkBudget

logger = logging.getLogger("nethical.space.detectors.jamming_detector")


class JammingType(str, Enum):
    """Categorisation of hostile or incidental RF interference."""
    NONE = "NONE"
    CONTINUOUS_WAVE = "CONTINUOUS_WAVE"   # Single-tone high-power carrier denial
    BARRAGE_NOISE = "BARRAGE_NOISE"       # Wideband noise overwhelming transponder
    SWEEP_RF = "SWEEP_RF"                 # Chirp / sweeping jammer
    DEGRADED_FADING = "DEGRADED_FADING"   # Severe scintillation or atmospheric loss


class JammingMitigationAction(str, Enum):
    """Autonomous mitigation actions for communications preservation."""
    NONE = "NONE"
    SWITCH_TO_OPTICAL_ISL = "SWITCH_TO_OPTICAL_ISL"
    ACTIVATE_ADAPTIVE_NOTCH_FILTER = "ACTIVATE_ADAPTIVE_NOTCH_FILTER"
    AUTONOMOUS_DEAD_RECKONING = "AUTONOMOUS_DEAD_RECKONING"
    BOOST_TX_POWER = "BOOST_TX_POWER"


class JammingAlert(BaseModel):
    """Result of link budget and RF spectrum jamming analysis."""
    is_jammed: bool
    jamming_type: JammingType = JammingType.NONE
    js_ratio_db: Optional[float] = None
    c_n0_db_hz: float
    snr_db: float
    mitigation_action: JammingMitigationAction = JammingMitigationAction.NONE
    details: str


class JammingDetectorConfig(BaseModel):
    """Configuration thresholds for space EW jamming detection."""
    js_ratio_threshold_db: float = Field(default=-3.0, description="Jamming-to-Signal ratio triggering alert (dB)")
    min_c_n0_threshold_db_hz: float = Field(default=42.0, description="Minimum demodulation C/N0 threshold (dB-Hz)")
    max_acceptable_bit_error_rate: float = Field(default=1e-3, description="Bit error rate threshold")


class JammingDetector:
    """Evaluates real-time RF telemetry and link budget for electronic warfare threats."""

    def __init__(self, config: Optional[JammingDetectorConfig] = None) -> None:
        self.config = config or JammingDetectorConfig()

    def evaluate(
        self,
        link_budget: LinkBudget,
        measured_bit_error_rate: Optional[float] = None,
    ) -> JammingAlert:
        """Analyze link parameters to detect active jamming and determine countermeasures."""
        js_ratio = link_budget.jamming_to_signal_ratio_db
        c_n0 = link_budget.carrier_to_noise_density_c_n0_db_hz
        snr = link_budget.snr_db

        is_jammed = False
        jam_type = JammingType.NONE
        mitigation = JammingMitigationAction.NONE
        details = "Link operating within nominal RF performance bounds."

        # 1. Direct Jamming-to-Signal (J/S) power override
        if js_ratio is not None and js_ratio >= self.config.js_ratio_threshold_db:
            is_jammed = True
            if js_ratio >= 10.0:
                jam_type = JammingType.BARRAGE_NOISE
                mitigation = JammingMitigationAction.SWITCH_TO_OPTICAL_ISL
                details = (
                    f"Severe high-power barrage jamming detected: J/S = {js_ratio:.1f} dB "
                    f"(exceeds threshold {self.config.js_ratio_threshold_db:.1f} dB). "
                    f"Immediate failover to Optical Laser Inter-Satellite Link commanded."
                )
            else:
                jam_type = JammingType.CONTINUOUS_WAVE
                mitigation = JammingMitigationAction.ACTIVATE_ADAPTIVE_NOTCH_FILTER
                details = (
                    f"In-band carrier jamming detected: J/S = {js_ratio:.1f} dB. "
                    f"Engaging digital beamforming adaptive spatial/notch nulling."
                )

        # 2. Spectral degradation without explicit jammer power measurement
        elif c_n0 < self.config.min_c_n0_threshold_db_hz:
            is_jammed = True
            jam_type = JammingType.DEGRADED_FADING
            mitigation = JammingMitigationAction.AUTONOMOUS_DEAD_RECKONING
            details = (
                f"Severe RF link loss: C/N0 = {c_n0:.1f} dB-Hz below minimum {self.config.min_c_n0_threshold_db_hz:.1f} dB-Hz. "
                f"Activating autonomous dead-reckoning and telemetry buffering."
            )

        # 3. Bit Error Rate elevation check
        if measured_bit_error_rate is not None and measured_bit_error_rate > self.config.max_acceptable_bit_error_rate:
            if not is_jammed:
                is_jammed = True
                jam_type = JammingType.SWEEP_RF
                mitigation = JammingMitigationAction.BOOST_TX_POWER
                details = (
                    f"Critical Bit Error Rate elevation: BER = {measured_bit_error_rate:.2e} "
                    f"(limit {self.config.max_acceptable_bit_error_rate:.2e}). RF sweep interference suspected."
                )

        if is_jammed:
            logger.warning("Spacecraft RF link %s jammed! Action: %s | %s", link_budget.link_id, mitigation.value, details)

        return JammingAlert(
            is_jammed=is_jammed,
            jamming_type=jam_type,
            js_ratio_db=js_ratio,
            c_n0_db_hz=c_n0,
            snr_db=snr,
            mitigation_action=mitigation,
            details=details,
        )
