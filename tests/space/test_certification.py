# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for ECSS Software Qualification & ITU Filing Generators (tests.space.test_certification)."""

import pytest

from nethical.space.certification import CertificationArtifactGenerator
from nethical.space.models import (
    LinkBudget,
    OrbitalState,
    Vector3D,
)


class TestCertificationArtifactGenerator:
    """Test suite for automated institutional qualification artifacts."""

    def test_generate_ecss_qualification_dossier(self) -> None:
        generator = CertificationArtifactGenerator()
        dossier = generator.generate_ecss_dossier(
            mission_name="POLSA_ORBITAL_GUARDIAN_01",
            satellite_id="SAT_POLSA_OG1",
            criticality_category="CATEGORY_B",
            test_pass_count=55,
            max_evaluation_latency_us=68.5,
        )

        assert "ECSS-E-ST-40C" in dossier
        assert "ECSS-Q-ST-80C" in dossier
        assert "POLSA_ORBITAL_GUARDIAN_01" in dossier
        assert "CATEGORY_B" in dossier
        assert "COMPLIANT" in dossier
        assert "Audit Anchor Hash" in dossier

    def test_generate_itu_appendix4_filing(self) -> None:
        generator = CertificationArtifactGenerator()
        state = OrbitalState(
            satellite_id="SAT_IRIS2_DEMO",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
            inclination_deg=53.0,
        )
        lb = LinkBudget(
            link_id="IRIS2_FEEDER",
            carrier_frequency_ghz=28.5,
            tx_power_dbw=15.0,
            tx_antenna_gain_dbi=42.0,
            bandwidth_mhz=50.0,
        )

        filing = generator.generate_itu_appendix4_filing(
            network_name="EU_IRIS2_SOVEREIGN_NET",
            satellite_state=state,
            link_budget=lb,
        )

        assert "ITU Radiocommunication Bureau" in filing
        assert "EU_IRIS2_SOVEREIGN_NET" in filing
        assert "ITU Article 22 Compliance" in filing
        assert "Merkle Certification Fingerprint" in filing
