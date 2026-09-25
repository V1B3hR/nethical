# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Cellular & Terrestrial Edge Connectivity (nethical.connectivity.cellular).

Verifies:
1. Telemetry signal grade classification (3GPP RSRP / RSRQ / CQI standards).
2. 5G modem operations across Sub-6, mmWave, and legacy fallback.
3. Cell handovers and link state management.
4. Signal degradation and restoration.
5. Dynamic Hybrid Cellular ⟷ Satellite Router failover and failback.
"""

from typing import Any, Dict, Optional
import pytest
from nethical.connectivity.cellular import (
    ActiveRoute,
    Cellular5GModem,
    CellularGeneration,
    CellularTelemetry,
    HybridCellularSatelliteRouter,
    SignalQualityGrade,
)
from nethical.connectivity.satellite.base import (
    ConnectionConfig,
    ConnectionState,
    SatelliteProvider,
)


class MockSatelliteProvider(SatelliteProvider):
    """Mock satellite provider for testing hybrid router bonding."""

    def __init__(self, name: str = "MockStarlink", connected: bool = True) -> None:
        super().__init__(ConnectionConfig())
        self._name = name
        self._state = ConnectionState.CONNECTED if connected else ConnectionState.DISCONNECTED

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def provider_type(self) -> str:
        return "LEO"

    async def connect(self) -> bool:
        self._state = ConnectionState.CONNECTED
        return True

    async def disconnect(self) -> bool:
        self._state = ConnectionState.DISCONNECTED
        return True

    async def send(self, data: bytes, priority: int = 0) -> bool:
        return True

    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        return b""

    async def health_check(self) -> bool:
        return self.is_connected

    async def get_signal_info(self) -> Dict[str, Any]:
        return {"c_n0_db_hz": 50.0}


def test_cellular_telemetry_signal_grades() -> None:
    """Verifies that RSRP and RSRQ translate correctly to 3GPP signal grades."""
    # Excellent
    tel_exc = CellularTelemetry(rsrp_dbm=-75.0, rsrq_db=-8.0, cqi=14, connected=True)
    assert tel_exc.signal_grade == SignalQualityGrade.EXCELLENT

    # Good
    tel_good = CellularTelemetry(rsrp_dbm=-90.0, rsrq_db=-11.0, cqi=10, connected=True)
    assert tel_good.signal_grade == SignalQualityGrade.GOOD

    # Fair
    tel_fair = CellularTelemetry(rsrp_dbm=-102.0, rsrq_db=-15.0, cqi=7, connected=True)
    assert tel_fair.signal_grade == SignalQualityGrade.FAIR

    # Poor
    tel_poor = CellularTelemetry(rsrp_dbm=-112.0, rsrq_db=-18.0, cqi=4, connected=True)
    assert tel_poor.signal_grade == SignalQualityGrade.POOR

    # Unusable
    tel_unusable = CellularTelemetry(rsrp_dbm=-125.0, rsrq_db=-22.0, cqi=1, connected=True)
    assert tel_unusable.signal_grade == SignalQualityGrade.UNUSABLE

    # Disconnected
    tel_disc = CellularTelemetry(connected=False)
    assert tel_disc.signal_grade == SignalQualityGrade.UNUSABLE


def test_5g_modem_connection_and_telemetry() -> None:
    """Verifies 5G modem connection and latency properties."""
    modem = Cellular5GModem(modem_id="TEST_MODEM", default_generation=CellularGeneration.G5_MMWAVE)
    assert modem.connect() is True

    telemetry = modem.get_telemetry()
    assert telemetry.connected is True
    assert telemetry.generation == CellularGeneration.G5_MMWAVE
    assert telemetry.frequency_mhz == 28000.0
    assert telemetry.latency_ms < 5.0  # Ultra-low latency for mmWave

    # Serialization
    t_dict = telemetry.to_dict()
    assert t_dict["cell_id"] == "GNB_WARSAW_01"
    assert t_dict["signal_grade"] in [g.value for g in SignalQualityGrade]


def test_5g_modem_handover() -> None:
    """Verifies cell handover from Sub-6 to mmWave and 4G fallback."""
    modem = Cellular5GModem()
    modem.connect()

    assert modem.get_telemetry().generation == CellularGeneration.G5_SUB6

    # Handover to 5G mmWave
    success = modem.trigger_handover(target_cell_id="GNB_CENTER_MMWAVE_02", target_generation=CellularGeneration.G5_MMWAVE)
    assert success is True
    assert modem.get_telemetry().cell_id == "GNB_CENTER_MMWAVE_02"
    assert modem.get_telemetry().generation == CellularGeneration.G5_MMWAVE
    assert modem.get_telemetry().handover_count == 1

    # Fallback handover to 4G LTE
    success_fallback = modem.trigger_handover(target_cell_id="ENB_RURAL_01", target_generation=CellularGeneration.G4_LTE)
    assert success_fallback is True
    assert modem.get_telemetry().generation == CellularGeneration.G4_LTE
    assert modem.get_telemetry().handover_count == 2


def test_hybrid_cellular_satellite_failover_and_failback() -> None:
    """Verifies that degraded 5G triggers satellite failover, and healthy 5G fails back."""
    modem = Cellular5GModem()
    modem.connect()
    sat = MockSatelliteProvider("MockStarlink", connected=True)

    router = HybridCellularSatelliteRouter(cellular_modem=modem, satellite_provider=sat)

    # 1. Initially healthy 5G
    decision1 = router.evaluate_routing()
    assert decision1.active_route == ActiveRoute.CELLULAR_5G
    assert router.failover_count == 0

    # 2. Simulate severe radio blockage / jamming
    modem.simulate_signal_degradation(rsrp_drop_db=40.0)  # Drops RSRP to ~ -125 dBm (UNUSABLE)
    decision2 = router.evaluate_routing()
    assert decision2.active_route == ActiveRoute.SATELLITE_LEO
    assert router.failover_count == 1
    assert "routing via satellite" in decision2.reason

    # 3. Simulate radio recovery
    modem.restore_signal()
    decision3 = router.evaluate_routing()
    assert decision3.active_route == ActiveRoute.CELLULAR_5G
    assert "Cellular signal healthy" in decision3.reason


def test_hybrid_router_when_satellite_offline() -> None:
    """Verifies behavior when cellular degrades and satellite is unavailable."""
    modem = Cellular5GModem()
    modem.connect()
    sat = MockSatelliteProvider("MockStarlink", connected=False)  # Disconnected

    router = HybridCellularSatelliteRouter(cellular_modem=modem, satellite_provider=sat)
    modem.simulate_signal_degradation(rsrp_drop_db=50.0)  # Drops to UNUSABLE

    decision = router.evaluate_routing()
    assert decision.active_route == ActiveRoute.OFFLINE
    assert decision.satellite_connected is False
