from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from nethical.hooks.interfaces import Region


@dataclass
class RegionProfile:
    region: Region
    compliance: list[str]
    phi_locales: list[str]
    data_residency_required: bool
    export_controls_enabled: bool
    default_geofencing_policy: Optional[str] = None


REGION_PROFILES: Dict[Region, RegionProfile] = {
    Region.US: RegionProfile(Region.US, ["HIPAA", "SOC2"], ["en_US"], True, False),
    Region.UK: RegionProfile(
        Region.UK, ["UK GDPR", "DPA 2018", "NHS DSPT"], ["en_GB"], True, False
    ),
    Region.EU: RegionProfile(
        Region.EU,
        ["EU GDPR", "NIS2"],
        ["en_GB", "de_DE", "fr_FR", "es_ES", "it_IT"],
        True,
        False,
    ),
    Region.NATO: RegionProfile(
        Region.NATO,
        ["NATO STANAG hooks"],
        ["en_GB"],
        False,
        True,
        default_geofencing_policy="mission_only",
    ),
}
