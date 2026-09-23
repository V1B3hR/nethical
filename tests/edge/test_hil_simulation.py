# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Hardware-in-the-Loop (HIL) Simulator & Industrial Fieldbus (tests.edge.test_hil_simulation).

Validates physical microcontroller simulation (STM32H7, ESP32-S3, TI TMS320, NXP S32K),
fault injection (CAN bus-off, CRC corruption, babbling idiot flood, watchdog loss),
and sub-50 µs physical cutoff relay tripping.
"""

from __future__ import annotations

import time
import pytest

from nethical.edge.hil_simulator import (
    FaultType,
    HILFieldbusBridge,
    HILSimulationResult,
    TargetMCU,
)
from nethical.edge.industrial_fieldbus import (
    EtherCATState,
    IndustrialFieldbusInterlock,
)


@pytest.fixture
def fieldbus() -> IndustrialFieldbusInterlock:
    """Fresh industrial fieldbus interlock instance."""
    return IndustrialFieldbusInterlock()


@pytest.fixture
def hil_bridge(fieldbus: IndustrialFieldbusInterlock) -> HILFieldbusBridge:
    """Standard HIL bridge targeting STM32H7 with sub-50 µs latency budget."""
    return HILFieldbusBridge(
        target_mcu=TargetMCU.STM32H7,
        fieldbus=fieldbus,
        max_permitted_latency_us=50.0,
    )


class TestHILBridgeInitialization:
    """Tests for HIL bridge configuration across diverse embedded microcontrollers."""

    @pytest.mark.parametrize(
        "mcu",
        [
            TargetMCU.STM32H7,
            TargetMCU.ESP32_S3,
            TargetMCU.TI_TMS320,
            TargetMCU.NXP_S32K,
        ],
    )
    def test_target_mcu_initialization(self, mcu: TargetMCU) -> None:
        """Verify HIL bridge supports all institutional automotive and industrial MCUs."""
        bridge = HILFieldbusBridge(target_mcu=mcu)
        assert bridge.target_mcu == mcu
        assert bridge.max_latency_us == 50.0


class TestNominalHILCycle:
    """Tests for nominal hardware round-trip loopback."""

    def test_nominal_loopback_energized_relay(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """Nominal loopback maintains closed relay and operational fieldbuses."""
        result = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.NONE)

        assert isinstance(result, HILSimulationResult)
        assert result.physical_relay_state == "ENERGIZED_CLOSED"
        assert result.fieldbus_acknowledged is True
        assert result.fault_mitigation_confirmed is True
        assert result.is_timing_compliant is True
        assert result.measured_loopback_latency_us < 50.0
        assert fieldbus.is_interlocked is False
        assert fieldbus.ethercat_state == EtherCATState.OP


class TestHILFaultInjections:
    """Tests for physical electronics fault injection and fail-closed interlock response."""

    def test_can_bus_off_fault_trips_interlock(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """CAN Bus-Off condition forces instantaneous de-energisation of safety relay."""
        result = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.CAN_BUS_OFF)

        assert result.physical_relay_state == "DE-ENERGIZED_SAFE_OPEN"
        assert result.fault_mitigation_confirmed is True
        assert fieldbus.is_interlocked is True
        assert "CAN Bus-Off condition injected" in (fieldbus.last_trip_reason or "")
        assert fieldbus.ethercat_state == EtherCATState.SAFE_OP
        assert fieldbus.fsoe_zeroed is True

    def test_crc_checksum_corruption_fault(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """CRC bit corruption on safety telegram is rejected and trips interlock."""
        result = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.CRC_CHECKSUM_CORRUPTION)

        assert result.physical_relay_state == "DE-ENERGIZED_SAFE_OPEN"
        assert fieldbus.is_interlocked is True
        assert "CRC error on safety telegram" in (fieldbus.last_trip_reason or "")

    def test_watchdog_heartbeat_dropout_fault(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """Missed MCU heartbeat pulse trips interlock via passive pull-down resistor."""
        result = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.WATCHDOG_HEARTBEAT_DROPPED)

        assert result.physical_relay_state == "DE-ENERGIZED_SAFE_OPEN"
        assert fieldbus.is_interlocked is True
        assert "MCU heartbeat missed" in (fieldbus.last_trip_reason or "")

    def test_babbling_idiot_bus_flood_fault(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """Babbling idiot rogue transmitter isolates bus and trips safety interlock."""
        result = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.BABBLING_IDIOT_FLOOD)

        assert result.physical_relay_state == "DE-ENERGIZED_SAFE_OPEN"
        assert fieldbus.is_interlocked is True
        assert "Babbling idiot" in (fieldbus.last_trip_reason or "")


class TestFieldbusResetAuthorization:
    """Tests for secure reset and authorization PIN enforcement."""

    def test_interlock_reset_pin_validation(
        self,
        hil_bridge: HILFieldbusBridge,
        fieldbus: IndustrialFieldbusInterlock,
    ) -> None:
        """Resetting fieldbus after trip requires valid authorization PIN."""
        hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.CAN_BUS_OFF)
        assert fieldbus.is_interlocked is True

        # Invalid PIN attempt
        success_invalid, _ = fieldbus.reset_interlock("WRONG_RESET_CODE")
        assert not success_invalid
        assert bool(fieldbus.is_interlocked)

        # Correct PIN attempt
        success_valid, _ = fieldbus.reset_interlock("NETHICAL-FIELDBUS-RESET-2026")
        assert success_valid
        assert not bool(fieldbus.is_interlocked)
        assert fieldbus.ethercat_state == EtherCATState.OP
        assert fieldbus.fsoe_zeroed is False


class TestHILTimingCompliance:
    """Benchmark tests to verify timing determinism under 200 consecutive HIL cycles."""

    def test_200_cycles_timing_determinism(self, hil_bridge: HILFieldbusBridge) -> None:
        """Verify that 200 consecutive cycles never exceed the 50 µs hard deadline."""
        for _ in range(200):
            res = hil_bridge.run_hardware_verification_cycle(inject_fault=FaultType.NONE)
            assert res.is_timing_compliant is True
            assert res.measured_loopback_latency_us < 50.0
