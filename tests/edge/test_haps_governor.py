# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for High-Altitude Platform Station (HAPS) Governor (tests.edge.test_haps_governor)."""

from datetime import datetime, timezone
import pytest

from nethical.edge.haps_governor import (
    HAPSAction,
    HAPSGovernor,
    HAPSGovernorConfig,
    HAPSSafetyDecision,
)
from nethical.space.models import HAPSFlightState


class TestHAPSGovernor:
    """Test suite for stratospheric pseudo-satellite governance."""

    def test_nominal_stratospheric_flight(self) -> None:
        governor = HAPSGovernor()
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.23,
            longitude=21.01,
            altitude_msl_m=20500.0,
            battery_soc_pct=85.0,
            solar_generation_watts=3200.0,
            power_consumption_watts=1600.0,
            station_keeping_center_lat=52.23,
            station_keeping_center_lon=21.01,
            sensor_active=False,
        )
        decision: HAPSSafetyDecision = governor.evaluate_flight_and_payload(state)
        assert decision.allowed
        assert decision.action == HAPSAction.CONTINUE_MISSION
        assert decision.merkle_proof_hash is not None
        assert len(decision.merkle_proof_hash) == 64

    def test_critical_battery_collapse_forces_rtb(self) -> None:
        governor = HAPSGovernor()
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.23,
            longitude=21.01,
            altitude_msl_m=19000.0,
            battery_soc_pct=12.0,  # Below 15% critical RTB threshold
            solar_generation_watts=0.0,
        )
        decision = governor.evaluate_flight_and_payload(state)
        assert not decision.allowed
        assert decision.action == HAPSAction.RETURN_TO_BASE
        assert decision.clamped_payload_power_watts == 0.0
        assert any("CRITICAL ENERGY COLLAPSE" in r for r in decision.reasons)

    def test_night_survival_sheds_payload(self) -> None:
        governor = HAPSGovernor()
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.23,
            longitude=21.01,
            altitude_msl_m=21000.0,
            battery_soc_pct=22.0,  # Below 25% night survival threshold
            solar_generation_watts=0.0,  # Night time
            sensor_active=True,
        )
        decision = governor.evaluate_flight_and_payload(state)
        assert not decision.allowed
        assert decision.action == HAPSAction.SHED_PAYLOAD_POWER
        assert decision.clamped_payload_power_watts == 0.0
        assert any("Diurnal Solar Deficit" in r for r in decision.reasons)

    def test_station_drift_commands_corrective_heading(self) -> None:
        governor = HAPSGovernor(
            config=HAPSGovernorConfig(max_station_keeping_radius_km=30.0)
        )
        # Platform drifted ~55 km north of assigned station
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.75,
            longitude=21.01,
            station_keeping_center_lat=52.23,
            station_keeping_center_lon=21.01,
            battery_soc_pct=75.0,
            solar_generation_watts=2500.0,
        )
        decision = governor.evaluate_flight_and_payload(state)
        assert not decision.allowed
        assert decision.action == HAPSAction.CORRECT_STATION_DRIFT
        assert decision.recommended_heading_deg is not None
        # Should be heading south (~180 deg) back to center
        assert pytest.approx(decision.recommended_heading_deg, abs=10.0) == 180.0

    def test_civilian_surveillance_veto_without_lawful_warrant_law_25(self) -> None:
        governor = HAPSGovernor(
            config=HAPSGovernorConfig(
                protected_civilian_geofence_center_lat=52.2297,
                protected_civilian_geofence_center_lon=21.0122,
                protected_civilian_radius_km=25.0,
            )
        )
        # High resolution sensor aimed at Warsaw without lawful token
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.23,
            longitude=21.01,
            sensor_active=True,
            sensor_type="HIGH_RES_EO_IR",
            current_target_lat=52.2297,
            current_target_lon=21.0122,
            lawful_intercept_token=None,  # No token!
        )
        decision = governor.evaluate_flight_and_payload(state)
        assert not decision.allowed
        assert decision.action == HAPSAction.VETO_SURVEILLANCE_PAYLOAD
        assert decision.law_implicated == 25  # Law 25: Privacy
        assert any("civilian sanctuary without authenticated lawful intercept warrant" in r for r in decision.reasons)

    def test_civilian_surveillance_authorized_with_warrant(self) -> None:
        governor = HAPSGovernor(
            config=HAPSGovernorConfig(
                protected_civilian_geofence_center_lat=52.2297,
                protected_civilian_geofence_center_lon=21.0122,
                protected_civilian_radius_km=25.0,
            )
        )
        # High resolution sensor aimed at Warsaw with authenticated warrant token
        state = HAPSFlightState(
            platform_id="HAPS_ZEPHYR_01",
            latitude=52.23,
            longitude=21.01,
            sensor_active=True,
            current_target_lat=52.2297,
            current_target_lon=21.0122,
            lawful_intercept_token="WARRANT_JUDICIAL_AUTH_2026_09_WARSAW",
        )
        decision = governor.evaluate_flight_and_payload(state)
        assert decision.allowed
        assert decision.action == HAPSAction.CONTINUE_MISSION
        assert any("Civilian surveillance collection authenticated" in r for r in decision.reasons)
