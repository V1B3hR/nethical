# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit tests for Nethical regional compliance profiles and configuration."""

import pytest
from datetime import date
from pathlib import Path

from nethical.hooks.interfaces import Region
from nethical.config.profiles import (
    ComplianceFramework,
    GeofencingPolicy,
    RegionProfile,
    REGION_PROFILES,
    get_profile,
    find_regions_by_locale,
    region_supports,
    as_legacy_dict,
    load_overrides_from_yaml,
)


def test_builtin_region_profiles_integrity() -> None:
    """Verifies that all standard regions have built-in profiles configured."""
    assert Region.US in REGION_PROFILES
    assert Region.UK in REGION_PROFILES
    assert Region.EU in REGION_PROFILES
    assert Region.NATO in REGION_PROFILES

    # Verify NATO profile
    nato_prof = get_profile(Region.NATO)
    assert nato_prof.supports(ComplianceFramework.NATO_STANAG)
    assert nato_prof.export_controls_required is True
    assert nato_prof.default_geofencing_policy == GeofencingPolicy.MISSION_ONLY

    # Verify EU profile
    eu_prof = get_profile(Region.EU)
    assert eu_prof.supports(ComplianceFramework.EU_GDPR)
    assert eu_prof.supports(ComplianceFramework.NIS2)
    assert eu_prof.data_residency_required is True

    # Verify UK profile
    uk_prof = get_profile(Region.UK)
    assert uk_prof.supports("UK GDPR")
    assert uk_prof.supports("NHS DSPT")


def test_region_profile_normalization_and_invariants() -> None:
    """Verifies locale normalization and invariants on RegionProfile."""
    # Test valid profile with unnormalized locales
    prof = RegionProfile(
        region=Region.EU,
        compliance=(ComplianceFramework.EU_GDPR,),
        locales=("de-de", "FR_fr"),
        data_residency_required=True,
        export_controls_required=False,
    )
    assert prof.locales == ("de_DE", "fr_FR")

    # Invariant: data residency requires at least one locale
    with pytest.raises(ValueError, match="requires data residency but no locales were provided"):
        RegionProfile(
            region=Region.EU,
            compliance=(ComplianceFramework.EU_GDPR,),
            locales=(),
            data_residency_required=True,
            export_controls_required=False,
        )


def test_find_regions_by_locale_and_support_helpers() -> None:
    """Verifies helper utilities for locale matching and framework support checks."""
    us_regions = find_regions_by_locale("en_US")
    assert Region.US in us_regions

    uk_regions = find_regions_by_locale("en_GB")
    assert Region.UK in uk_regions

    assert region_supports(Region.US, ComplianceFramework.HIPAA) is True
    assert region_supports(Region.US, "SOC2") is True
    assert region_supports(Region.US, "NON_EXISTENT_FRAMEWORK") is False


def test_serialization_and_legacy_dict() -> None:
    """Verifies to_policy_dict and as_legacy_dict representations."""
    prof = get_profile(Region.UK)
    pdict = prof.to_policy_dict()
    assert pdict["region"] == "UK"
    assert "UK GDPR" in pdict["compliance"]
    assert pdict["data_residency_required"] is True

    legacy = as_legacy_dict(prof)
    assert legacy["region"] == Region.UK
    assert "phi_locales" in legacy


def test_load_overrides_from_yaml(tmp_path: Path) -> None:
    """Verifies loading regional profile overrides from YAML."""
    yaml_content = """
US:
  compliance: ["HIPAA", "SOC2"]
  locales: ["en_US", "es_US"]
  data_residency_required: true
  export_controls_required: true
  default_geofencing_policy: "restricted"
  last_reviewed_at: "2026-01-15"
  schema_version: "2.0"
"""
    yaml_file = tmp_path / "region_overrides.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    overrides = load_overrides_from_yaml(str(yaml_file))
    us_override = overrides[Region.US]
    assert us_override.export_controls_required is True
    assert us_override.default_geofencing_policy == GeofencingPolicy.RESTRICTED
    assert us_override.schema_version == "2.0"
    assert us_override.last_reviewed_at == date(2026, 1, 15)
