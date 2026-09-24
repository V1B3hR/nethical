# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for space detectors: jamming, spoofing, collision, and beam steering (tests.space.test_space_detectors)."""

import pytest

from nethical.space.detectors import (
    BeamPointingCommand,
    BeamSteeringAuditor,
    BeamTargetType,
    CollisionCourseDetector,
    ConjunctionAlertLevel,
    JammingDetector,
    JammingMitigationAction,
    JammingType,
    SecondaryHazardObject,
    SpoofingDetector,
    SpoofingMitigationAction,
    SpoofingSeverity,
)
from nethical.space.models import (
    LinkBudget,
    OrbitalState,
    Vector3D,
)


class TestJammingDetector:
    """Test suite for space RF electronic warfare jamming detection."""

    def test_nominal_rf_link(self) -> None:
        detector = JammingDetector()
        lb = LinkBudget(
            link_id="LINK_NOMINAL",
            carrier_frequency_ghz=14.0,
            tx_power_dbw=15.0,
            tx_antenna_gain_dbi=35.0,
            rx_antenna_gain_dbi=38.0,
            slant_range_km=800.0,
        )
        alert = detector.evaluate(lb)
        assert not alert.is_jammed
        assert alert.jamming_type == JammingType.NONE
        assert alert.mitigation_action == JammingMitigationAction.NONE

    def test_barrage_noise_mitigation(self) -> None:
        detector = JammingDetector()
        lb = LinkBudget(
            link_id="LINK_JAMMED",
            carrier_frequency_ghz=14.0,
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=30.0,
            rx_antenna_gain_dbi=35.0,
            slant_range_km=1000.0,
            jammer_power_at_rx_dbw=-65.0,  # Extreme barrage jammer
        )
        alert = detector.evaluate(lb)
        assert alert.is_jammed
        assert alert.jamming_type == JammingType.BARRAGE_NOISE
        assert alert.mitigation_action == JammingMitigationAction.SWITCH_TO_OPTICAL_ISL

    def test_continuous_wave_notch_filter(self) -> None:
        detector = JammingDetector()
        lb = LinkBudget(
            link_id="LINK_CW",
            carrier_frequency_ghz=14.0,
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=35.0,
            rx_antenna_gain_dbi=35.0,
            slant_range_km=1000.0,
            jammer_power_at_rx_dbw=-93.0,  # Moderate carrier jammer (J/S ~ 3.7 dB)
        )
        alert = detector.evaluate(lb)
        assert alert.is_jammed
        assert alert.jamming_type == JammingType.CONTINUOUS_WAVE
        assert alert.mitigation_action == JammingMitigationAction.ACTIVATE_ADAPTIVE_NOTCH_FILTER


class TestSpoofingDetector:
    """Test suite for orbital GNSS spoofing and navigation cross-validation."""

    def test_nominal_celestial_cross_check(self) -> None:
        detector = SpoofingDetector()
        gnss = OrbitalState(
            satellite_id="SAT_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        celestial = OrbitalState(
            satellite_id="SAT_01",
            position_eci_km=Vector3D(x=6878.05, y=0.02, z=0.01),
            velocity_eci_kms=Vector3D(x=0.0, y=7.601, z=0.0),
        )
        alert = detector.evaluate(gnss_state=gnss, celestial_imu_state=celestial)
        assert not alert.is_spoofed
        assert alert.severity == SpoofingSeverity.NONE

    def test_acute_position_jump_spoofing(self) -> None:
        detector = SpoofingDetector()
        gnss = OrbitalState(
            satellite_id="SAT_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        # Spoofed position is shifted by 25 km
        spoofed_gnss = OrbitalState(
            satellite_id="SAT_01",
            position_eci_km=Vector3D(x=6903.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        alert = detector.evaluate(gnss_state=spoofed_gnss, celestial_imu_state=gnss)
        assert alert.is_spoofed
        assert alert.severity == SpoofingSeverity.ACUTE_SPOOFING
        assert alert.mitigation_action == SpoofingMitigationAction.REJECT_GNSS_USE_CELESTIAL

    def test_clock_walk_off_drift(self) -> None:
        detector = SpoofingDetector()
        state = OrbitalState(
            satellite_id="SAT_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        alert = detector.evaluate(
            gnss_state=state,
            celestial_imu_state=state,
            receiver_clock_drift_ppm=25.0,  # Exceeds 15 ppm threshold
        )
        assert alert.is_spoofed
        assert alert.severity == SpoofingSeverity.SUSPECTED_DRIFT
        assert alert.mitigation_action == SpoofingMitigationAction.HOLD_PROPAGATION_ORBIT


class TestCollisionCourseDetector:
    """Test suite for orbital conjunction assessment and maneuver authorization."""

    def test_nominal_separation(self) -> None:
        detector = CollisionCourseDetector()
        primary = OrbitalState(
            satellite_id="SAT_ALPHA",
            position_eci_km=Vector3D(x=7000.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.5, z=0.0),
        )
        secondary = OrbitalState(
            satellite_id="SAT_BETA",
            position_eci_km=Vector3D(x=7020.0, y=0.0, z=0.0),  # 20 km away
            velocity_eci_kms=Vector3D(x=0.0, y=7.5, z=0.0),
        )
        result = detector.evaluate_conjunction(primary, secondary)
        assert not result.conjunction_detected
        assert not result.maneuver_required
        assert result.alert_level == ConjunctionAlertLevel.NOMINAL

    def test_critical_conjunction_triggers_authorized_avoidance(self) -> None:
        detector = CollisionCourseDetector()
        primary = OrbitalState(
            satellite_id="SAT_ALPHA",
            position_eci_km=Vector3D(x=6900.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        secondary = OrbitalState(
            satellite_id="DEBRIS_KOSMOS_1408",
            position_eci_km=Vector3D(x=6900.05, y=0.02, z=0.01),  # ~54 metres miss distance!
            velocity_eci_kms=Vector3D(x=0.0, y=-7.6, z=0.0),      # Head-on collision
        )
        result = detector.evaluate_conjunction(primary, secondary)
        assert result.conjunction_detected
        assert result.maneuver_required
        assert result.maneuver_authorized
        assert result.safe_delta_v_vector_ms is not None
        assert result.alert_level == ConjunctionAlertLevel.CRITICAL_AVOIDANCE

    def test_secondary_debris_hazard_vetoes_dangerous_maneuver(self) -> None:
        detector = CollisionCourseDetector()
        primary = OrbitalState(
            satellite_id="SAT_ALPHA",
            position_eci_km=Vector3D(x=6900.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        secondary = OrbitalState(
            satellite_id="DEBRIS_TRACK_A",
            position_eci_km=Vector3D(x=6900.1, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=-7.6, z=0.0),
        )
        # Secondary debris object sitting exactly at the primary position
        secondary_catalog = [
            SecondaryHazardObject(
                object_id="DEBRIS_CLOUD_KOSMOS_FRAGMENT_99",
                position_eci_km=Vector3D(x=6900.2, y=0.0, z=0.0),
                velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
            )
        ]
        result = detector.evaluate_conjunction(
            primary_state=primary,
            secondary_state=secondary,
            secondary_catalog=secondary_catalog,
        )
        assert result.conjunction_detected
        assert result.maneuver_required
        # Maneuver is vetoed because evasion corridor is contaminated by secondary fragment!
        assert not result.maneuver_authorized
        assert any("VETO" in r for r in result.reasons)


class TestBeamSteeringAuditor:
    """Test suite for ITU Article 22 and quiet zone geofence audits."""

    def test_nominal_ground_station_beam(self) -> None:
        auditor = BeamSteeringAuditor()
        cmd = BeamPointingCommand(
            beam_id="BEAM_01",
            target_type=BeamTargetType.GROUND_STATION,
            target_lat=52.23,
            target_lon=21.01,
            slant_range_km=800.0,
            off_axis_angle_to_geo_arc_deg=20.0,
        )
        res = auditor.audit_beam_command(cmd)
        assert res.allowed
        assert res.itu_article22_compliant
        assert res.geofence_compliant

    def test_itu_article22_epfd_breach(self) -> None:
        auditor = BeamSteeringAuditor()
        # High power beam aimed within 0.5 degrees of Geostationary arc
        cmd = BeamPointingCommand(
            beam_id="BEAM_HIGH_POWER",
            target_type=BeamTargetType.INTER_SATELLITE_LINK,
            tx_power_watts=500.0,
            slant_range_km=2000.0,
            off_axis_angle_to_geo_arc_deg=0.5,  # Violates 2.5 deg minimum separation
        )
        res = auditor.audit_beam_command(cmd)
        assert not res.allowed
        assert not res.itu_article22_compliant
        assert any("ITU Article 22 Breach" in r for r in res.reasons)

    def test_radio_quiet_zone_prohibition(self) -> None:
        auditor = BeamSteeringAuditor()
        # Beam aimed directly at Square Kilometre Array (SKA) in Western Australia
        cmd = BeamPointingCommand(
            beam_id="BEAM_AUS",
            target_type=BeamTargetType.TERRESTRIAL_COMMUNITY,
            target_lat=-26.696,
            target_lon=116.637,
            slant_range_km=600.0,
        )
        res = auditor.audit_beam_command(cmd)
        assert not res.allowed
        assert not res.geofence_compliant
        assert any("Square Kilometre Array" in r for r in res.reasons)

    def test_uncoordinated_satellite_dazzle_veto(self) -> None:
        auditor = BeamSteeringAuditor()
        cmd = BeamPointingCommand(
            beam_id="BEAM_OPTICAL_DAZZLE",
            target_type=BeamTargetType.UNCOORDINATED_SATELLITE,
            target_satellite_id="FOREIGN_SAT_999",
            slant_range_km=500.0,
        )
        res = auditor.audit_beam_command(cmd)
        assert not res.allowed
        assert any("Uncoordinated targeting" in r for r in res.reasons)
