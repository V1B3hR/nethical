"""Test suite for Institutional Trust, Governance Maturity & Release Assurance."""

import json
from pathlib import Path
import pytest

from nethical.formal.verify_rfc import RFCFormalVerifier, run_rfc_verification

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_topplan_exists_and_covers_remediation():
    """Verify that topplan.md exists at repo root and defines the remediation roadmap."""
    topplan_file = REPO_ROOT / "topplan.md"
    assert topplan_file.exists(), "topplan.md is missing from repository root!"
    content = topplan_file.read_text(encoding="utf-8")
    assert "CVE-2026-26007" in content
    assert "ISO/IEC 42001:2023" in content
    assert "Technical Steering Committee" in content
    assert "REMEDIATED" in content


def test_security_advisory_published():
    """Verify that formal GHSA security advisory exists for CVE-2026-26007."""
    ghsa_file = REPO_ROOT / ".github" / "SECURITY_ADVISORIES" / "GHSA-2026-cve-26007.md"
    assert ghsa_file.exists(), "GHSA advisory file is missing!"
    content = ghsa_file.read_text(encoding="utf-8")
    assert "CVSS:3.1" in content
    assert "9.1" in content
    assert "cryptography >= 50.0.0" in content


def test_independent_audit_and_gap_assessment():
    """Verify that formal Independent Audit & Gap Assessment document exists."""
    audit_file = REPO_ROOT / "audit" / "INDEPENDENT_AUDIT_AND_GAP_ASSESSMENT.md"
    assert audit_file.exists(), "INDEPENDENT_AUDIT_AND_GAP_ASSESSMENT.md is missing!"
    content = audit_file.read_text(encoding="utf-8")
    assert "ISO/IEC 42001:2023" in content
    assert "EU AI Act" in content
    assert "97.4%" in content or "97.8%" in content


def test_tsc_roster_and_governance_independence():
    """Verify that TSC_ROSTER.md documents 5 distinct multi-stakeholder domain seats."""
    roster_file = REPO_ROOT / "governance" / "TSC_ROSTER.md"
    assert roster_file.exists(), "governance/TSC_ROSTER.md is missing!"
    content = roster_file.read_text(encoding="utf-8")
    for seat in ["TSC-1", "TSC-2", "TSC-3", "TSC-4", "TSC-5"]:
        assert seat in content, f"Seat {seat} missing from TSC roster!"
    assert "Dual-Control Custodian" in content


def test_rfc_formal_z3_verification():
    """Verify that RFC-0001 passes automated Z3 SMT solver non-regression proof."""
    verifier = RFCFormalVerifier()
    res = verifier.verify_rfc_0001_cryptographic_baseline()
    assert res.proved is True
    assert res.status == "PROVED"
    assert "RFC0001_CryptographicNonRegression" in res.property_name


def test_production_sbom_freshness():
    """Verify that SBOM.json lists modern dependencies with cryptography>=50."""
    sbom_file = REPO_ROOT / "SBOM.json"
    assert sbom_file.exists(), "SBOM.json is missing!"
    sbom = json.loads(sbom_file.read_text(encoding="utf-8"))
    assert sbom.get("bomFormat") == "CycloneDX"
    assert sbom["metadata"]["component"]["version"] == "2.7.0"

    components = {c["name"]: c["version"] for c in sbom.get("components", [])}
    assert "cryptography" in components
    assert components["cryptography"].startswith("50.")
    assert "fastapi" in components
    assert "pydantic" in components
