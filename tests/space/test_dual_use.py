# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Dual-Use Classification & Export Control (tests.space.test_dual_use)."""

import pytest

from nethical.space.dual_use import (
    DualUseCategory,
    DualUseClassification,
    DualUseClassifier,
    ExportControlRegime,
)


class TestDualUseClassifier:
    """Test suite for space sensor payload dual-use and export control compliance."""

    def test_high_resolution_optical_classified_tactical_military(self) -> None:
        classifier = DualUseClassifier()
        # 0.30m GSD optical satellite deployed in operational theater
        result: DualUseClassification = classifier.classify_payload(
            sensor_type="VERY_HIGH_RES_OPTICAL",
            optical_gsd_meters=0.30,
            is_theater_of_operations=True,
        )

        assert result.is_military_grade
        assert result.category == DualUseCategory.TACTICAL_MILITARY
        assert result.licensing_required
        assert ExportControlRegime.US_ITAR_CATEGORY_XV in result.applicable_regimes
        assert ExportControlRegime.EU_DUAL_USE_REGULATION in result.applicable_regimes

    def test_synthetic_aperture_radar_dual_use(self) -> None:
        classifier = DualUseClassifier()
        # Sub-metre SAR payload (0.8m resolution)
        result = classifier.classify_payload(
            sensor_type="SYNTHETIC_APERTURE_RADAR",
            sar_resolution_meters=0.80,
            is_theater_of_operations=False,
        )

        assert result.is_military_grade
        assert result.category == DualUseCategory.DUAL_USE_INFRASTRUCTURE
        assert result.licensing_required
        assert ExportControlRegime.US_ITAR_CATEGORY_XV in result.applicable_regimes

    def test_civilian_commercial_broadband(self) -> None:
        classifier = DualUseClassifier()
        result = classifier.classify_payload(
            sensor_type="KA_BAND_BROADBAND_RELAY",
            optical_gsd_meters=None,
            sar_resolution_meters=None,
            is_theater_of_operations=False,
        )

        assert not result.is_military_grade
        assert result.category == DualUseCategory.CIVILIAN_COMMERCIAL
        assert not result.licensing_required
