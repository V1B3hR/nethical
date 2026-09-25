# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""5G / Advanced Cellular Modem Implementation (nethical.connectivity.cellular.modem_5g).

Implements 5G Sub-6 and mmWave modem driver logic, channel quality evaluation,
cell handovers, and intent-based QoS requested by autonomous systems.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

from nethical.connectivity.cellular.base import (
    BaseCellularModem,
    CellularGeneration,
    CellularTelemetry,
    SignalQualityGrade,
)

logger = logging.getLogger("nethical.connectivity.cellular.modem_5g")


class Cellular5GModem(BaseCellularModem):
    """5G Modem controller supporting Sub-6 and mmWave operation."""

    def __init__(
        self,
        modem_id: str = "MODEM_5G_PRIME",
        primary_cell_id: str = "GNB_WARSAW_01",
        default_generation: CellularGeneration = CellularGeneration.G5_SUB6,
    ) -> None:
        self.modem_id = modem_id
        self.current_cell_id = primary_cell_id
        self.current_generation = default_generation
        self._connected = False
        self._handover_count = 0

        # Physical link attributes
        self.rsrp_dbm = -85.0
        self.rsrq_db = -10.0
        self.sinr_db = 18.0
        self.cqi = 13
        self.packet_loss_percent = 0.0

    def connect(self) -> bool:
        """Establishes 5G radio bearer."""
        self._connected = True
        logger.info(f"Modem {self.modem_id} connected to {self.current_cell_id} via {self.current_generation.value}")
        return True

    def disconnect(self) -> bool:
        """Tears down radio bearer."""
        self._connected = False
        logger.info(f"Modem {self.modem_id} disconnected.")
        return True

    def get_telemetry(self) -> CellularTelemetry:
        """Returns simulated or polled physical link telemetry."""
        if not self._connected:
            return CellularTelemetry(
                cell_id=self.current_cell_id,
                generation=self.current_generation,
                connected=False,
                rsrp_dbm=-140.0,
                rsrq_db=-30.0,
                sinr_db=-10.0,
                cqi=1,
                latency_ms=999.0,
            )

        # Latency profile according to generation
        if self.current_generation == CellularGeneration.G5_MMWAVE:
            base_latency = 3.5
            freq = 28000.0  # 28 GHz
            bw = 400.0      # 400 MHz
        elif self.current_generation == CellularGeneration.G5_SUB6:
            base_latency = 8.0
            freq = 3500.0   # 3.5 GHz
            bw = 100.0      # 100 MHz
        elif self.current_generation == CellularGeneration.G6_JCAS:
            base_latency = 0.8
            freq = 140000.0 # 140 GHz THz
            bw = 2000.0
        else:
            base_latency = 35.0
            freq = 1800.0
            bw = 20.0

        return CellularTelemetry(
            cell_id=self.current_cell_id,
            generation=self.current_generation,
            frequency_mhz=freq,
            bandwidth_mhz=bw,
            rsrp_dbm=self.rsrp_dbm,
            rsrq_db=self.rsrq_db,
            sinr_db=self.sinr_db,
            cqi=self.cqi,
            latency_ms=base_latency,
            jitter_ms=1.2,
            packet_loss_percent=self.packet_loss_percent,
            handover_count=self._handover_count,
            connected=True,
        )

    def trigger_handover(self, target_cell_id: str, target_generation: CellularGeneration) -> bool:
        """Performs cell reselection / handover to target gNodeB or eNodeB."""
        if not self._connected:
            logger.warning(f"Cannot execute handover for disconnected modem {self.modem_id}.")
            return False

        old_cell = self.current_cell_id
        old_gen = self.current_generation
        self.current_cell_id = target_cell_id
        self.current_generation = target_generation
        self._handover_count += 1

        logger.info(
            f"Handover #{self._handover_count} successful: "
            f"[{old_cell} ({old_gen.value})] -> [{target_cell_id} ({target_generation.value})]"
        )
        return True

    def simulate_signal_degradation(self, rsrp_drop_db: float = 30.0) -> None:
        """Simulates physical path loss, blockage or hostile jamming."""
        self.rsrp_dbm = max(-140.0, self.rsrp_dbm - rsrp_drop_db)
        self.rsrq_db = max(-30.0, self.rsrq_db - (rsrp_drop_db / 3.0))
        self.sinr_db = max(-10.0, self.sinr_db - (rsrp_drop_db / 2.0))
        self.cqi = max(1, int(self.cqi - (rsrp_drop_db / 5.0)))
        self.packet_loss_percent = min(100.0, self.packet_loss_percent + 5.0)

    def restore_signal(self) -> None:
        """Restores optimal radio parameters."""
        self.rsrp_dbm = -85.0
        self.rsrq_db = -10.0
        self.sinr_db = 18.0
        self.cqi = 13
        self.packet_loss_percent = 0.0
