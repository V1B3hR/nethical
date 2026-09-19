"""Tests for Official Audit Dossier Generation (ISO 42001 & EU AI Act Annex IV).

Validates generation of regulatory compliance packages and integrity of Merkle anchors.
"""

from pathlib import Path
import json
import pytest
from training.generate_audit_dossier import (
    generate_eu_ai_act_annex_iv_dossier,
    generate_iso_42001_dossier,
    main as run_generator,
)

REPO_ROOT = Path(__file__).parent.parent
AUDIT_DIR = REPO_ROOT / "models" / "audit"


def test_eu_ai_act_annex_iv_dossier_structure():
    """Verify that EU AI Act Annex IV dossier contains all mandatory regulatory sections."""
    data, md = generate_eu_ai_act_annex_iv_dossier()

    assert data["document_type"] == "EU_AI_ACT_ANNEX_IV_TECHNICAL_DOCUMENTATION"
    assert "cryptographic_merkle_root" in data
    assert len(data["cryptographic_merkle_root"]) >= 32

    # Mandatory sections under Annex IV
    assert "section_1_general_system_description" in data
    assert "section_2_methods_and_algorithms" in data
    assert "section_3_data_governance_and_provenance" in data
    assert "section_4_human_oversight_and_fundamental_rights" in data
    assert "section_5_cybersecurity_and_swarm_adversary_defense" in data

    # Markdown format checks
    assert "# EU AI ACT ANNEX IV TECHNICAL DOCUMENTATION" in md
    assert "Merkle Anchor Root" in md


def test_iso_42001_aims_dossier_structure():
    """Verify that ISO/IEC 42001 AIMS dossier contains all clauses and Annex A controls."""
    data, md = generate_iso_42001_dossier()

    assert "ISO/IEC 42001:2023" in data["standard"]
    assert data["merkle_verification_anchor"] is not None

    clauses = data["clauses_assessment"]
    for c in ["clause_4_context_of_organization", "clause_5_leadership_and_policy",
              "clause_6_planning_and_ai_risk_assessment", "clause_7_support_and_resources",
              "clause_8_operational_control", "clause_9_performance_evaluation",
              "clause_10_continual_improvement"]:
        assert c in clauses
        assert clauses[c]["status"] == "COMPLIANT"

    assert "annex_a_controls" in data
    assert len(data["annex_a_controls"]) >= 8


def test_audit_dossier_disk_generation():
    """Verify that running generator creates disk artifacts with valid JSON."""
    run_generator()

    eu_json = AUDIT_DIR / "EU_AI_ACT_ANNEX_IV_DOSSIER.json"
    eu_md = AUDIT_DIR / "EU_AI_ACT_ANNEX_IV_DOSSIER.md"
    iso_json = AUDIT_DIR / "ISO_42001_AIMS_CERTIFICATION_DOSSIER.json"
    iso_md = AUDIT_DIR / "ISO_42001_AIMS_CERTIFICATION_DOSSIER.md"

    for p in [eu_json, eu_md, iso_json, iso_md]:
        assert p.exists(), f"Expected audit artifact missing: {p}"
        assert p.stat().st_size > 500, f"File {p} is suspiciously small"

    with open(eu_json, "r", encoding="utf-8") as f:
        loaded_eu = json.load(f)
        assert loaded_eu["system_name"] == "Nethical Enterprise Governance OS & Błyskawica Ambassador"

    with open(iso_json, "r", encoding="utf-8") as f:
        loaded_iso = json.load(f)
        assert loaded_iso["standard"] == "ISO/IEC 42001:2023 - Artificial Intelligence Management System (AIMS)"
