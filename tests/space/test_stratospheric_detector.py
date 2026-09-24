# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Stratospheric & HAPS Detectors (tests.space.test_stratospheric_detector)."""

import pytest

from nethical.space.detectors.stratospheric_detector import (
    AirspaceClass,
    StratosphericDetector,
)


class TestStratosphericDetector:
    """Test suite for persistent surveillance dwell, EMF sniffing, and U-space transitions."""

    def test_surveillance_dwell_exceeded_under_law_25(self) -> None:
        detector = StratosphericDetector(max_dwell_hours_without_warrant=8.0)
        # 5 hours incremental dwell -> nominal
        alert1 = detector.audit_surveillance_dwell("ZONE_CIVILIAN_A", incremental_dwell_hours=5.0)
        assert not alert1.dwell_exceeded
        assert alert1.action_required == "CONTINUE"

        # Additional 4 hours -> cumulative 9 hours > 8 hours threshold -> dwell exceeded
        alert2 = detector.audit_surveillance_dwell("ZONE_CIVILIAN_A", incremental_dwell_hours=4.0)
        assert alert2.dwell_exceeded
        assert alert2.action_required == "FREEZE_COLLECTION_AND_REDACT"

        # Refreshed warrant resets dwell
        alert3 = detector.audit_surveillance_dwell("ZONE_CIVILIAN_A", incremental_dwell_hours=2.0, has_refreshed_warrant=True)
        assert not alert3.dwell_exceeded
        assert alert3.action_required == "CONTINUE"

    def test_emf_intercept_protected_civilian_frequency(self) -> None:
        detector = StratosphericDetector()
        # VHF emergency band 121.5 MHz (0.1215 GHz) intercepted without warrant
        alert = detector.audit_emf_intercept(
            intercept_frequency_ghz=0.1215,
            lawful_sigint_token=None,
        )
        assert alert.unauthorized_intercept_detected
        assert alert.is_protected_civilian_frequency

        # Authorized collection with defense/judicial token
        auth_alert = detector.audit_emf_intercept(
            intercept_frequency_ghz=0.1215,
            lawful_sigint_token="SIGINT_WARRANT_NATO_BALTIC_2026",
        )
        assert not auth_alert.unauthorized_intercept_detected

    def test_uspace_controlled_airspace_transition(self) -> None:
        detector = StratosphericDetector()
        # Climbing at 12,000m (Controlled Airspace, below FL600) with ATC clearance
        alert_ok = detector.audit_uspace_airspace_transition(
            altitude_msl_m=12000.0,
            vertical_speed_mps=2.5,
            atc_clearance_token="ATC_WARSAW_CTR_CLEARANCE_CLIMB_FL600",
            adsb_out_active=True,
        )
        assert alert_ok.in_controlled_airspace
        assert alert_ok.climb_authorized

        # Climbing without ATC clearance
        alert_no_atc = detector.audit_uspace_airspace_transition(
            altitude_msl_m=12000.0,
            vertical_speed_mps=2.5,
            atc_clearance_token=None,
            adsb_out_active=True,
        )
        assert not alert_no_atc.climb_authorized
        assert "lacks Air Traffic Control clearance" in alert_no_atc.details

        # Transiting Upper Airspace (21,000 m / FL690)
        alert_upper = detector.audit_uspace_airspace_transition(
            altitude_msl_m=21000.0,
            vertical_speed_mps=0.0,
        )
        assert not alert_upper.in_controlled_airspace
        assert alert_upper.climb_authorized
