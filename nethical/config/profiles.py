from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple

from nethical.hooks.interfaces import Region

logger = logging.getLogger(__name__)

_LOCALE_RE = re.compile(r"^[a-z]{2}_[A-Z]{2}$")  # e.g., en_US, en_GB, de_DE


class ComplianceFramework(str, Enum):
    """Canonical set of compliance frameworks used across regions.

    Subclassing str helps JSON serialization and logs remain human-friendly.
    """
    HIPAA = "HIPAA"
    SOC2 = "SOC2"
    UK_GDPR = "UK GDPR"
    DPA_2018 = "DPA 2018"
    NHS_DSPT = "NHS DSPT"
    EU_GDPR = "EU GDPR"
    NIS2 = "NIS2"
    NATO_STANAG = "NATO STANAG"


class GeofencingPolicy(str, Enum):
    """Default policy for geofencing behavior, if relevant to the region."""
    OPEN = "open"
    RESTRICTED = "restricted"
    MISSION_ONLY = "mission_only"


def _normalize_locale(locale: str) -> str:
    """Normalize locale to LL_CC form, e.g., 'en_US'.

    Note: If you prefer stricter validation against real locales, consider
    integrating Babel and verifying in __post_init__ (optional dependency).
    """
    if not isinstance(locale, str):
        raise TypeError(f"Locale must be a string, got {type(locale)}")
    # Accept case-insensitive input like 'en_us' and normalize to 'en_US'.
    parts = locale.replace("-", "_").split("_")
    if len(parts) != 2 or len(parts[0]) != 2 or len(parts[1]) != 2:
        raise ValueError(f"Locale '{locale}' must be in the form ll_CC (e.g., 'en_US')")
    normalized = f"{parts[0].lower()}_{parts[1].upper()}"
    if not _LOCALE_RE.match(normalized):
        raise ValueError(f"Locale '{locale}' is invalid after normalization: '{normalized}'")
    return normalized


@dataclass(frozen=True, slots=True)
class RegionProfile:
    """Immutable profile describing compliance and policy characteristics of a Region.

    Fields:
      - region: The region enum this profile applies to.
      - compliance: Tuple of canonical compliance frameworks enforced in this region.
      - locales: Tuple of normalized locales relevant for PHI or other content constraints.
      - data_residency_required: If True, data must remain in-region.
      - export_controls_required: If True, export control checks must be enforced.
      - default_geofencing_policy: Default geofencing stance for the region.
      - last_reviewed_at: Optional date for auditability/governance.
      - schema_version: Optional schema version of this profile structure.
    """
    region: Region
    compliance: Tuple[ComplianceFramework, ...]
    locales: Tuple[str, ...]
    data_residency_required: bool
    export_controls_required: bool
    default_geofencing_policy: GeofencingPolicy = GeofencingPolicy.OPEN
    last_reviewed_at: Optional[date] = None
    schema_version: Optional[str] = "1.0"

    def __post_init__(self) -> None:
        # Ensure uniqueness and normalization of locales
        normalized_locales = tuple(sorted({ _normalize_locale(l) for l in self.locales }))
        object.__setattr__(self, "locales", normalized_locales)

        # Ensure uniqueness of compliance frameworks
        unique_compliance = tuple(sorted(set(self.compliance), key=lambda cf: cf.value))
        object.__setattr__(self, "compliance", unique_compliance)

        # Invariants
        if self.data_residency_required and not self.locales:
            raise ValueError(
                f"Region {self.region} requires data residency but no locales were provided."
            )

    @property
    def compliance_codes(self) -> Tuple[str, ...]:
        """Compatibility helper: return compliance codes as strings."""
        return tuple(cf.value for cf in self.compliance)

    def supports(self, framework: ComplianceFramework | str) -> bool:
        """Check if this region supports/enforces a given framework."""
        if isinstance(framework, str):
            try:
                framework = ComplianceFramework(framework)
            except ValueError:
                return False
        return framework in self.compliance

    def to_policy_dict(self) -> dict:
        """Shape data for external policy/ABAC engines."""
        return {
            "region": str(self.region.name),
            "compliance": list(self.compliance_codes),
            "locales": list(self.locales),
            "data_residency_required": self.data_residency_required,
            "export_controls_required": self.export_controls_required,
            "default_geofencing_policy": self.default_geofencing_policy.value,
            "last_reviewed_at": self.last_reviewed_at.isoformat() if self.last_reviewed_at else None,
            "schema_version": self.schema_version,
        }


def _mk_profile(
    *,
    region: Region,
    compliance: Iterable[ComplianceFramework],
    locales: Iterable[str],
    data_residency_required: bool,
    export_controls_required: bool,
    default_geofencing_policy: GeofencingPolicy = GeofencingPolicy.OPEN,
    last_reviewed_at: Optional[date] = None,
) -> RegionProfile:
    return RegionProfile(
        region=region,
        compliance=tuple(compliance),
        locales=tuple(locales),
        data_residency_required=data_residency_required,
        export_controls_required=export_controls_required,
        default_geofencing_policy=default_geofencing_policy,
        last_reviewed_at=last_reviewed_at,
    )


# Built-in profiles (immutable baseline)
_BUILTIN_REGION_PROFILES: dict[Region, RegionProfile] = {
    Region.US: _mk_profile(
        region=Region.US,
        compliance=(ComplianceFramework.HIPAA, ComplianceFramework.SOC2),
        locales=("en_US",),
        data_residency_required=True,
        export_controls_required=False,
    ),
    Region.UK: _mk_profile(
        region=Region.UK,
        compliance=(ComplianceFramework.UK_GDPR, ComplianceFramework.DPA_2018, ComplianceFramework.NHS_DSPT),
        locales=("en_GB",),
        data_residency_required=True,
        export_controls_required=False,
    ),
    Region.EU: _mk_profile(
        region=Region.EU,
        compliance=(ComplianceFramework.EU_GDPR, ComplianceFramework.NIS2),
        locales=("en_GB", "de_DE", "fr_FR", "es_ES", "it_IT"),
        data_residency_required=True,
        export_controls_required=False,
    ),
    Region.NATO: _mk_profile(
        region=Region.NATO,
        compliance=(ComplianceFramework.NATO_STANAG,),
        locales=("en_GB",),
        data_residency_required=False,
        export_controls_required=True,
        default_geofencing_policy=GeofencingPolicy.MISSION_ONLY,
    ),
}

# Expose as immutable mapping
REGION_PROFILES: Mapping[Region, RegionProfile] = MappingProxyType(_BUILTIN_REGION_PROFILES)


def get_profile(region: Region) -> RegionProfile:
    """Fetch a region profile or raise KeyError if unknown."""
    return REGION_PROFILES[region]


def find_regions_by_locale(locale: str) -> Tuple[Region, ...]:
    """Return regions that list the given locale."""
    normalized = _normalize_locale(locale)
    matches = tuple(r for r, profile in REGION_PROFILES.items() if normalized in profile.locales)
    return matches


def region_supports(region: Region, framework: ComplianceFramework | str) -> bool:
    """Helper for one-shot checks."""
    try:
        return get_profile(region).supports(framework)
    except KeyError:
        return False


# Optional: support external overrides via YAML while preserving immutability
def load_overrides_from_yaml(path: str) -> Mapping[Region, RegionProfile]:
    """Load region profile overrides from a YAML file.

    Expected minimal shape:
      RegionName:
        compliance: ["HIPAA","SOC2"]
        locales: ["en_US","es_ES"]
        data_residency_required: true
        export_controls_required: false
        default_geofencing_policy: "open" | "restricted" | "mission_only"
        last_reviewed_at: "2025-01-10"
        schema_version: "1.0"

    Notes:
      - Unspecified fields inherit from the built-in profile if present.
      - Compliance items are matched to ComplianceFramework; unknown strings are rejected.
      - This function returns a new immutable mapping; it does not mutate REGION_PROFILES.
    """
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to use load_overrides_from_yaml") from exc

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML top-level must be a mapping of RegionName -> profile")

    merged: dict[Region, RegionProfile] = dict(_BUILTIN_REGION_PROFILES)

    for region_name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Profile for region '{region_name}' must be a mapping")

        try:
            region = Region[region_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Region '{region_name}' in overrides") from exc

        base = merged.get(region)

        def _cf_list(values: Iterable[str | ComplianceFramework]) -> Tuple[ComplianceFramework, ...]:
            out = []
            for v in values:
                if isinstance(v, ComplianceFramework):
                    out.append(v)
                else:
                    try:
                        out.append(ComplianceFramework(v))
                    except ValueError as exc:
                        raise ValueError(f"Unknown compliance framework '{v}' for region '{region_name}'") from exc
            return tuple(out)

        compliance = _cf_list(spec.get("compliance", base.compliance if base else ()))
        locales = tuple(spec.get("locales", base.locales if base else ()))
        data_residency_required = bool(spec.get(
            "data_residency_required",
            base.data_residency_required if base else False
        ))
        export_controls_required = bool(spec.get(
            "export_controls_required",
            base.export_controls_required if base else False
        ))
        geofence_str = spec.get(
            "default_geofencing_policy",
            base.default_geofencing_policy.value if base else GeofencingPolicy.OPEN.value
        )
        try:
            default_geofencing_policy = GeofencingPolicy(geofence_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid default_geofencing_policy '{geofence_str}' for region '{region_name}'"
            ) from exc

        lra = spec.get("last_reviewed_at", base.last_reviewed_at.isoformat() if (base and base.last_reviewed_at) else None)
        last_reviewed_at = date.fromisoformat(lra) if isinstance(lra, str) else None
        schema_version = spec.get("schema_version", base.schema_version if base else "1.0")

        merged[region] = RegionProfile(
            region=region,
            compliance=compliance,
            locales=locales,
            data_residency_required=data_residency_required,
            export_controls_required=export_controls_required,
            default_geofencing_policy=default_geofencing_policy,
            last_reviewed_at=last_reviewed_at,
            schema_version=schema_version,
        )

    return MappingProxyType(merged)


# Backward-compatibility section (optional)
# If existing callers expect attributes named like the original file, you can expose
# lightweight shims or properties here. For example:
# - 'phi_locales' -> use 'locales'
# - 'export_controls_enabled' -> use 'export_controls_required'
# You can also provide a thin adapter function to emit legacy dicts if needed.

def as_legacy_dict(profile: RegionProfile) -> dict:
    """Produce a legacy-shaped dict compatible with older code paths."""
    return {
        "region": profile.region,
        "compliance": list(profile.compliance_codes),
        "phi_locales": list(profile.locales),
        "data_residency_required": profile.data_residency_required,
        "export_controls_enabled": profile.export_controls_required,
        "default_geofencing_policy": profile.default_geofencing_policy.value,
    }
