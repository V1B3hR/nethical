# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital & Stratospheric Dual-Use Classification & Export Control (nethical.space.dual_use).

Categorizes commercial vs defense space capabilities, satellite links, and sensor payloads under:
- EU Dual-Use Regulation (EU) 2021/821 Annex I (Category 7 Navigation & Category 9 Aerospace)
- US ITAR 22 CFR § 121.1 Category XV (Spacecraft Systems & Related Articles)
- NATO Strategic Commercial Space Integration Policy
- Law 20 (Coexistence) & Law 25 (Privacy)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.space.dual_use")


class DualUseCategory(str, Enum):
    """Categorisation of space operations under international export control treaties."""
    CIVILIAN_COMMERCIAL = "CIVILIAN_COMMERCIAL"         # Civil telecom, scientific weather, civilian GNSS
    DUAL_USE_INFRASTRUCTURE = "DUAL_USE_INFRASTRUCTURE" # High-res commercial EO/SAR, tactical satcom mesh
    TACTICAL_MILITARY = "TACTICAL_MILITARY"             # Kinetic targeting, SIGINT intercept, electronic attack


class ExportControlRegime(str, Enum):
    """Export control jurisdictions."""
    EU_DUAL_USE_REGULATION = "EU_2021_821"
    US_ITAR_CATEGORY_XV = "US_ITAR_CAT_XV"
    WASSENAAR_ARRANGEMENT = "WASSENAAR_ARRANGEMENT"


class DualUseClassification(BaseModel):
    """Operational verdict and export classification for spacecraft capability."""
    category: DualUseCategory
    is_military_grade: bool
    applicable_regimes: List[ExportControlRegime]
    licensing_required: bool
    optical_gsd_m: Optional[float] = None
    defense_justification: str
    compliance_recommendations: List[str] = Field(default_factory=list)


class DualUseClassifier:
    """Classifies satellite links, optical payloads, and stratospheric sensors."""

    def __init__(self) -> None:
        # Ground Sample Distance (GSD) thresholds: <= 0.5m is classified military/dual-use in EU/US
        self.military_optical_gsd_threshold_m = 0.50
        self.military_sar_resolution_threshold_m = 1.00

    def classify_payload(
        self,
        sensor_type: str,
        optical_gsd_meters: Optional[float] = None,
        sar_resolution_meters: Optional[float] = None,
        has_post_quantum_encryption: bool = True,
        is_theater_of_operations: bool = False,
    ) -> DualUseClassification:
        """Evaluate sensor payload parameters against export control and dual-use thresholds."""
        applicable_regimes = [
            ExportControlRegime.EU_DUAL_USE_REGULATION,
            ExportControlRegime.WASSENAAR_ARRANGEMENT,
        ]

        # 1. Very High-Resolution Optical Imagery (GSD <= 0.5m)
        if optical_gsd_meters is not None and optical_gsd_meters <= self.military_optical_gsd_threshold_m:
            applicable_regimes.append(ExportControlRegime.US_ITAR_CATEGORY_XV)
            category = DualUseCategory.TACTICAL_MILITARY if is_theater_of_operations else DualUseCategory.DUAL_USE_INFRASTRUCTURE
            return DualUseClassification(
                category=category,
                is_military_grade=True,
                applicable_regimes=applicable_regimes,
                licensing_required=True,
                optical_gsd_m=optical_gsd_meters,
                defense_justification=(
                    f"Sub-half-metre optical resolution (GSD = {optical_gsd_meters:.2f} m) "
                    f"meets tactical reconnaissance criteria under ITAR Cat XV / EU 2021/821 Cat 9."
                ),
                compliance_recommendations=[
                    "Implement cryptographic shutter geofencing over sovereign civilian sanctuaries (Law 25).",
                    "Require verified End-User Certificate (EUC) before downlinking raw uncompressed imagery.",
                ],
            )

        # 2. Synthetic Aperture Radar (SAR) Sub-Metre Resolution
        if sar_resolution_meters is not None and sar_resolution_meters <= self.military_sar_resolution_threshold_m:
            applicable_regimes.append(ExportControlRegime.US_ITAR_CATEGORY_XV)
            return DualUseClassification(
                category=DualUseCategory.DUAL_USE_INFRASTRUCTURE,
                is_military_grade=True,
                applicable_regimes=applicable_regimes,
                licensing_required=True,
                defense_justification=(
                    f"High-resolution Synthetic Aperture Radar (resolution = {sar_resolution_meters:.2f} m) "
                    f"capable of all-weather day/night tactical surface surveillance."
                ),
                compliance_recommendations=[
                    "Log all active phased-array radar emissions in Merkle-DAG ledger.",
                    "Verify compliance with NATO STANAG 4607 ground moving target indicator standards.",
                ],
            )

        # 3. Commercial Satellite / Civil Link
        category = DualUseCategory.DUAL_USE_INFRASTRUCTURE if is_theater_of_operations else DualUseCategory.CIVILIAN_COMMERCIAL
        return DualUseClassification(
            category=category,
            is_military_grade=False,
            applicable_regimes=[ExportControlRegime.EU_DUAL_USE_REGULATION],
            licensing_required=is_theater_of_operations,
            optical_gsd_m=optical_gsd_meters,
            defense_justification=(
                "Civilian telecommunications or low-resolution environmental monitoring profile."
                if not is_theater_of_operations else
                "Commercial broadband terminal deployed in contested theater of operations (dual-use)."
            ),
            compliance_recommendations=[
                "Maintain standard commercial export logging.",
            ],
        )
