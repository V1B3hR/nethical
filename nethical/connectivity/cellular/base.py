# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Abstract Base Classes and Data Models for Cellular & Edge Connectivity.

Derived from telecommunication models studied in Błyskawica
(adaptiveneuralnetwork/knowledge/wireless_bci_foundation.md and advanced_telecom_digest.md).

Supports:
- 3G / 4G LTE / 5G Sub-6 / 5G mmWave / 6G JCAS telemetry
- 3GPP signal metrics: RSRP, RSRQ, SINR, CQI (1-15), latency, jitter
- Standardized signal grading and connection states
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("nethical.connectivity.cellular.base")


class CellularGeneration(str, Enum):
    """Generations and tiers of terrestrial cellular connectivity."""
    G3_LEGACY = "3G"
    G4_LTE = "4G_LTE"
    G5_SUB6 = "5G_Sub6"
    G5_MMWAVE = "5G_mmWave"
    G6_JCAS = "6G_JCAS"


class SignalQualityGrade(str, Enum):
    """Categorisation of radio signal quality based on 3GPP RSRP/RSRQ standards."""
    EXCELLENT = "EXCELLENT"  # RSRP >= -80 dBm, RSRQ >= -10 dB, CQI >= 12
    GOOD = "GOOD"            # RSRP >= -95 dBm, RSRQ >= -13 dB, CQI >= 9
    FAIR = "FAIR"            # RSRP >= -105 dBm, RSRQ >= -16 dB, CQI >= 6
    POOR = "POOR"            # RSRP >= -115 dBm, RSRQ >= -19 dB, CQI >= 3
    UNUSABLE = "UNUSABLE"    # RSRP < -115 dBm or RSRQ < -20 dB


@dataclass
class CellularTelemetry:
    """Real-time physical telemetry of a cellular radio link."""
    cell_id: str = "CELL_DEFAULT_01"
    generation: CellularGeneration = CellularGeneration.G5_SUB6
    frequency_mhz: float = 3500.0
    bandwidth_mhz: float = 100.0
    rsrp_dbm: float = -85.0
    rsrq_db: float = -10.0
    sinr_db: float = 15.0
    cqi: int = 12  # Channel Quality Indicator (1 - 15)
    latency_ms: float = 8.0
    jitter_ms: float = 1.5
    packet_loss_percent: float = 0.0
    handover_count: int = 0
    connected: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signal_grade(self) -> SignalQualityGrade:
        """Determines signal grade per 3GPP RSRP and RSRQ criteria."""
        if not self.connected or self.rsrp_dbm < -115.0 or self.rsrq_db < -20.0:
            return SignalQualityGrade.UNUSABLE
        if self.rsrp_dbm >= -80.0 and self.rsrq_db >= -10.0 and self.cqi >= 12:
            return SignalQualityGrade.EXCELLENT
        if self.rsrp_dbm >= -95.0 and self.rsrq_db >= -13.0 and self.cqi >= 9:
            return SignalQualityGrade.GOOD
        if self.rsrp_dbm >= -105.0 and self.rsrq_db >= -16.0:
            return SignalQualityGrade.FAIR
        return SignalQualityGrade.POOR

    def to_dict(self) -> Dict[str, Any]:
        """Serializes telemetry record to dictionary."""
        return {
            "cell_id": self.cell_id,
            "generation": self.generation.value,
            "frequency_mhz": self.frequency_mhz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "rsrp_dbm": self.rsrp_dbm,
            "rsrq_db": self.rsrq_db,
            "sinr_db": self.sinr_db,
            "cqi": self.cqi,
            "latency_ms": self.latency_ms,
            "jitter_ms": self.jitter_ms,
            "packet_loss_percent": self.packet_loss_percent,
            "handover_count": self.handover_count,
            "connected": self.connected,
            "signal_grade": self.signal_grade.value,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseCellularModem(ABC):
    """Abstract interface for hardware or simulated cellular modems."""

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection with cellular network."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Terminates connection with cellular network."""
        pass

    @abstractmethod
    def get_telemetry(self) -> CellularTelemetry:
        """Fetches latest real-time radio telemetry."""
        pass

    @abstractmethod
    def trigger_handover(self, target_cell_id: str, target_generation: CellularGeneration) -> bool:
        """Executes cell handover to target station."""
        pass
