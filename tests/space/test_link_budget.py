# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for RF and optical link budget calculations (tests.space.test_link_budget)."""

import pytest
from nethical.space.models import LinkBudget


class TestLinkBudget:
    """Test suite for LinkBudget physics and electronic warfare telemetry."""

    def test_nominal_link_budget(self) -> None:
        """Verify nominal Ku-band satellite downlink link budget."""
        lb = LinkBudget(
            link_id="SAT_DOWNLINK_01",
            carrier_frequency_ghz=12.0,
            tx_power_dbw=13.0,          # 20 Watts
            tx_antenna_gain_dbi=34.0,
            rx_antenna_gain_dbi=41.0,
            slant_range_km=1000.0,
            atmospheric_loss_db=0.6,
            pointing_loss_db=0.4,
            system_noise_temp_k=250.0,
            bandwidth_mhz=20.0,
            bit_rate_mbps=50.0,
        )

        assert lb.eirp_dbw == 47.0
        # FSPL at 1000 km, 12 GHz: 20*log10(1000) + 20*log10(12) + 92.45 = 60 + 21.58 + 92.45 = 174.03 dB
        assert pytest.approx(lb.free_space_path_loss_db, abs=0.5) == 174.0
        # Received power: 47 - 174.03 - 0.6 - 0.4 + 41 = -87.03 dBW
        assert pytest.approx(lb.received_carrier_power_dbw, abs=0.5) == -87.0
        assert lb.g_over_t_db_k > 15.0
        assert lb.snr_db > 15.0
        assert lb.carrier_to_noise_density_c_n0_db_hz > 70.0
        assert lb.jamming_to_signal_ratio_db is None
        assert not lb.is_jammed()

    def test_active_rf_jamming_detection(self) -> None:
        """Verify jamming override when hostile emitter is active."""
        lb = LinkBudget(
            link_id="SAT_CONTESTED_01",
            carrier_frequency_ghz=14.0,
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=30.0,
            rx_antenna_gain_dbi=35.0,
            slant_range_km=1200.0,
            system_noise_temp_k=300.0,
            bandwidth_mhz=25.0,
            # Hostile ground jammer beaming high power at spacecraft transponder
            jammer_power_at_rx_dbw=-75.0,
        )

        rx_power = lb.received_carrier_power_dbw
        js_ratio = lb.jamming_to_signal_ratio_db
        assert js_ratio is not None
        assert js_ratio == pytest.approx(-75.0 - rx_power, abs=0.1)
        assert js_ratio > 0.0  # Jammer dominates desired signal
        assert lb.is_jammed(js_threshold_db=-3.0)

    def test_spectral_fading_jamming_trigger(self) -> None:
        """Verify link-loss trigger when C/N0 falls below threshold."""
        lb = LinkBudget(
            link_id="SAT_DEGRADED_01",
            carrier_frequency_ghz=20.0,
            tx_power_dbw=0.0,
            tx_antenna_gain_dbi=10.0,
            rx_antenna_gain_dbi=10.0,
            slant_range_km=10000.0,
            atmospheric_loss_db=15.0,  # Extreme attenuation
            system_noise_temp_k=1500.0,
            bandwidth_mhz=100.0,
        )

        assert lb.carrier_to_noise_density_c_n0_db_hz < 45.0
        assert lb.is_jammed(min_c_n0_db_hz=45.0)
