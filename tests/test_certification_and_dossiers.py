# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit and integration tests for Automated Certification Hub, Dossier Exporters, CSIRT Declarations, and Air-Gap Packaging."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.compliance.automated_certification_hub import (
    AutomatedCertificationHub,
    AutomatedEvidencePackage,
    CertificationStandard,
)
from nethical.security.merkle_ledger import MerkleLedger
from scripts.package_sovereign_bundle import (
    verify_pqc_cryptography,
    verify_sovereign_compose,
    verify_zero_external_cdn,
    workspace_root,
)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as client:
        yield client


def test_all_15_certification_standards_generation() -> None:
    """Verifies that all 15 certification standards generate valid evidence packages with high readiness scores."""
    ledger = MerkleLedger()
    hub = AutomatedCertificationHub(ledger=ledger)

    standards = list(CertificationStandard)
    assert len(standards) == 15

    for std in standards:
        pkg = hub.generate_evidence_package(std)
        assert isinstance(pkg, AutomatedEvidencePackage)
        assert pkg.standard == std
        assert pkg.readiness_score >= 0.90
        assert len(pkg.merkle_anchor_root) == 64  # SHA-256 Merkle root hex
        assert len(pkg.pqc_signature) > 32
        assert len(pkg.controls_matrix) >= 3
        assert "first_line_operational" in pkg.three_lines_of_defense
        assert "second_line_risk_compliance" in pkg.three_lines_of_defense
        assert "third_line_independent_audit" in pkg.three_lines_of_defense


def test_eu_ai_act_annex_iv_technical_dossier() -> None:
    """Verifies specific requirements of EU AI Act Annex IV technical documentation."""
    hub = AutomatedCertificationHub()
    pkg = hub.generate_evidence_package(CertificationStandard.EU_AI_ACT_ANNEX_IV)

    assert pkg.readiness_score == 0.99
    assert "Annex_IV_1_General_Description" in pkg.controls_matrix
    assert "Annex_IV_4_Risk_Management_Art9" in pkg.controls_matrix
    assert "Annex_IV_6_Human_Oversight_Art14" in pkg.controls_matrix
    assert "Annex_IV_7_Cybersecurity_Art15" in pkg.controls_matrix
    assert "Rozporządzenia (UE) 2024/1689" in pkg.auditor_verification_instructions


def test_common_criteria_eal4_security_target() -> None:
    """Verifies Common Criteria ISO/IEC 15408 EAL4+ Security Target generation."""
    hub = AutomatedCertificationHub()
    pkg = hub.generate_evidence_package(CertificationStandard.COMMON_CRITERIA_EAL4)

    assert pkg.readiness_score >= 0.98
    assert "FAU_GEN.1_Audit_Data_Generation" in pkg.controls_matrix
    assert "FAU_STG.1_Protected_Audit_Review" in pkg.controls_matrix
    assert "FCS_COP.1_Cryptographic_Operation" in pkg.controls_matrix
    assert "FPT_FLS.1_Failure_with_Preservation" in pkg.controls_matrix


def test_dossier_markdown_and_json_export() -> None:
    """Verifies that evidence packages can be exported to clean Markdown and canonical JSON."""
    hub = AutomatedCertificationHub()
    pkg = hub.generate_evidence_package(CertificationStandard.NATO_DEFENSE_AI)

    # 1. Markdown Export
    md = hub.export_dossier_markdown(pkg)
    assert "# SOVEREIGN COMPLIANCE DOSSIER & AUDIT EVIDENCE" in md
    assert pkg.package_id in md
    assert pkg.merkle_anchor_root in md
    assert "Trzy Linie Obrony" in md
    assert "NATO_PRU_1_Lawfulness" in md

    # 2. JSON Export
    json_data = hub.export_dossier_json(pkg)
    assert json_data["package_id"] == pkg.package_id
    assert json_data["standard"] == CertificationStandard.NATO_DEFENSE_AI.value
    assert json_data["readiness_score"] == pkg.readiness_score


def test_api_export_dossier_markdown(client: TestClient) -> None:
    """Tests POST /api/v1/compliance/export-dossier-markdown endpoint."""
    res = client.post(
        "/api/v1/compliance/export-dossier-markdown",
        json={"standard": "EU_AI_ACT_ANNEX_IV"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "markdown" in data
    assert "# SOVEREIGN COMPLIANCE DOSSIER" in data["markdown"]
    assert data["standard"] == "EU_AI_ACT_ANNEX_IV"
    assert data["readiness_score"] == 0.99
    assert len(data["merkle_anchor_root"]) == 64


def test_api_csirt_incident_report(client: TestClient) -> None:
    """Tests POST /api/v1/compliance/csirt-incident-report endpoint."""
    res = client.post(
        "/api/v1/compliance/csirt-incident-report",
        json={
            "incident_title": "Wykrycie nieautoryzowanej próby podmiany wag LoRA",
            "severity": "CRITICAL",
            "affected_asset": "KSC_OPERATOR_SYSTEM_01",
            "description": "Zablokowano próbę manipulacji parametrami modelu przez Governance Gateway.",
            "indicators_of_compromise": ["192.168.1.50", "sha256:abc123def456"],
            "tenant_id": "gov_pl_cyber",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "OFFICIALLY_DECLARED_SEALED"
    assert data["receipt_id"].startswith("RCPT-")
    assert len(data["merkle_root"]) == 64
    assert data["evidence_package"]["standard"] == CertificationStandard.CSIRT_SERIOUS_INCIDENT.value


def test_airgap_packaging_scripts() -> None:
    """Verifies that the air-gap packaging checks run cleanly."""
    # 1. Zero CDN
    cdn_ok, violations = verify_zero_external_cdn(workspace_root / "portal")
    assert cdn_ok is True
    assert len(violations) == 0

    # 2. PQC Crypto
    pqc_ok, pqc_msg = verify_pqc_cryptography()
    assert pqc_ok is True
    assert "ML-DSA-65" in pqc_msg

    # 3. Sovereign Compose
    compose_ok, compose_msg = verify_sovereign_compose(workspace_root / "docker-compose.sovereign.yml")
    assert compose_ok is True
