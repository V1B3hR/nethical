"""Tests for Sectoral Governance Packs (Healthcare, Public Administration, Academic Research).

Validates:
1. HealthcareMedPack: MDR SaMD Rule 11, ISO 14971, KEL, DNR prohibition, SOR triage, drug dosage locks.
2. PublicAdminGovPack: KPA Art. 7/107, KRI interoperability, state secrets (UOIN), qualified signature.
3. AcademicResearchPack: ALLEA FFP prevention, DOI/PMID citation validation, patent novelty shield, bioethics.
4. AutomatedCertificationHub: 12 standards support, PQC ML-DSA-65 signatures and Merkle anchoring.
5. REST API: FastAPI endpoint execution and response schemas.
"""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.compliance.packs.healthcare_med_pack import (
    HealthcareMedPack,
    SaMDClass,
    MedicalRiskLevel,
)
from nethical.compliance.packs.public_admin_gov_pack import (
    PublicAdminGovPack,
    ClearanceLevel,
    AdminDecisionStatus,
)
from nethical.compliance.packs.academic_research_pack import (
    AcademicResearchPack,
    ResearchDiscipline,
    ResearchIntegrityStatus,
)
from nethical.compliance.automated_certification_hub import (
    AutomatedCertificationHub,
    CertificationStandard,
)


# ==============================================================================
# 1. HEALTHCARE & MEDTECH PACK TESTS
# ==============================================================================

def test_healthcare_med_pack_compliant():
    pack = HealthcareMedPack()
    payload = {
        "system_name": "CardioDiagnostica-SaMD",
        "samd_class": "CLASS_IIA",
        "iso14971_risk_analysis_present": True,
        "iso13485_qms_certified": True,
        "physician_in_the_loop_verified": True,
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is True
    assert result.risk_level == MedicalRiskLevel.CLINICALLY_SAFE
    assert result.compliance_score >= 0.90
    assert result.hard_invariants_passed is True
    assert result.autonomous_dnr_blocked is True
    assert len(result.violations) == 0


def test_healthcare_med_pack_blocks_autonomous_dnr():
    pack = HealthcareMedPack()
    payload = {
        "system_name": "ICU-Predictor",
        "samd_class": "CLASS_III",
        "autonomous_dnr_order": True,  # Critical violation!
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is False
    assert result.risk_level == MedicalRiskLevel.PROHIBITED_MALPRACTICE_RISK
    assert result.hard_invariants_passed is False
    assert result.autonomous_dnr_blocked is False
    assert any("Do Not Resuscitate" in v for v in result.violations)
    assert any("Kodeks Etyki Lekarskiej" in v for v in result.violations)


def test_healthcare_med_pack_triage_and_dosage_invariants():
    pack = HealthcareMedPack()
    # Test triage downgrade attempt without doctor
    triage_payload = {
        "system_name": "SOR-Triage-Bot",
        "autonomous_triage_downgrade": True,
    }
    res_triage = pack.evaluate(triage_payload)
    assert res_triage.is_compliant is False
    assert res_triage.triage_integrity_verified is False
    assert any("Manchester Triage" in v for v in res_triage.violations)

    # Test autonomous drug dosage override without signature
    dosage_payload = {
        "system_name": "InfusionPump-Controller",
        "autonomous_drug_dosage_override": True,
        "physician_digital_signature": False,
    }
    res_dosage = pack.evaluate(dosage_payload)
    assert res_dosage.is_compliant is False
    assert res_dosage.dosage_override_locked is False
    assert any("pharmacotherapy" in v.lower() for v in res_dosage.violations)


# ==============================================================================
# 2. PUBLIC ADMINISTRATION & SOVEREIGN GOV PACK TESTS
# ==============================================================================

def test_public_admin_gov_pack_compliant():
    pack = PublicAdminGovPack()
    payload = {
        "system_name": "e-Urzad-Wydzial-Podatkowy",
        "clearance_level": "JAWNE",
        "is_autonomous_final_decision": False,
        "provides_legal_basis_and_reasoning": True,
        "kri_interoperability_compliant": True,
        "wcag_21_aa_accessible": True,
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is True
    assert result.decision_status == AdminDecisionStatus.LAWFUL_AND_ACTIONABLE
    assert result.compliance_score >= 0.90
    assert result.kpa_art7_objective_truth_verified is True
    assert result.kpa_art107_reasoning_provided is True


def test_public_admin_gov_pack_blocks_unsigned_decision():
    pack = PublicAdminGovPack()
    payload = {
        "system_name": "e-Decyzja-Bot",
        "is_autonomous_final_decision": True,
        "qualified_human_signature": False,  # Missing official signature!
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is False
    assert result.decision_status == AdminDecisionStatus.PROHIBITED_ARBITRARY_DECISION
    assert any("107" in v and "KPA" in v for v in result.violations)
    assert any("nieważności" in v for v in result.violations)


def test_public_admin_gov_pack_anti_black_box_and_secrets():
    pack = PublicAdminGovPack()
    # Test black box reasoning omission
    black_box_payload = {
        "system_name": "Beneficjent-Scoring",
        "is_black_box_model": True,
    }
    res_bb = pack.evaluate(black_box_payload)
    assert res_bb.is_compliant is False
    assert res_bb.decision_status == AdminDecisionStatus.DEFECTIVE_BLACK_BOX
    assert any("uzasadnienie" in v.lower() for v in res_bb.violations)

    # Test classified info leakage outside air gap
    classified_payload = {
        "system_name": "MON-Briefing-Assistant",
        "clearance_level": "TAJNE",
        "is_airgapped_node": False,
        "has_abw_skw_accreditation": False,
    }
    res_sec = pack.evaluate(classified_payload)
    assert res_sec.is_compliant is False
    assert res_sec.classified_data_airgapped is False
    assert any("ABW/SKW" in v for v in res_sec.violations)


# ==============================================================================
# 3. ACADEMIC RESEARCH INTEGRITY PACK TESTS
# ==============================================================================

def test_academic_research_pack_compliant():
    pack = AcademicResearchPack()
    payload = {
        "system_name": "Quantum-Computing-Manuscript",
        "discipline": "EXACT_AND_ENGINEERING",
        "citations": [
            {"id": "10.1038/s41586-023-06096-3", "type": "DOI"},
            {"id": "arXiv:2305.18290", "type": "ARXIV"},
        ],
        "fair_data_principles_met": True,
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is True
    assert result.integrity_status == ResearchIntegrityStatus.IMPECCABLE_SCHOLARSHIP
    assert result.compliance_score >= 0.95
    assert result.allea_ffp_free_verified is True
    assert result.citations_verified_valid is True
    assert result.patent_prior_art_shielded is True


def test_academic_research_pack_blocks_ffp_misconduct():
    pack = AcademicResearchPack()
    payload = {
        "system_name": "Fabricated-Cancer-Study",
        "discipline": "BIOMEDICAL_AND_CLINICAL",
        "data_fabrication_detected": True,  # Critical!
    }
    result = pack.evaluate(payload)
    assert result.is_compliant is False
    assert result.integrity_status == ResearchIntegrityStatus.SCIENTIFIC_MISCONDUCT_FFP
    assert result.allea_ffp_free_verified is False
    assert any("Fabrication" in v for v in result.violations)
    assert any("ALLEA" in v for v in result.violations)


def test_academic_research_pack_validates_citations_and_patent():
    pack = AcademicResearchPack()
    # Test fake, hallucinated DOI
    fake_citation_payload = {
        "system_name": "Doctoral-Thesis-Draft",
        "citations": [
            {"id": "NOT_A_VALID_DOI_12345", "type": "DOI"},
        ],
    }
    res_cit = pack.evaluate(fake_citation_payload)
    assert res_cit.is_compliant is False
    assert res_cit.citations_verified_valid is False
    assert any("Zmyślona" in v or "niepoprawna" in v for v in res_cit.violations)

    # Test patent prior art leak
    patent_leak_payload = {
        "system_name": "Nanotech-Synthesis-Engine",
        "unprotected_prior_art_disclosure": True,
    }
    res_pat = pack.evaluate(patent_leak_payload)
    assert res_pat.is_compliant is False
    assert res_pat.patent_prior_art_shielded is False
    assert any("nowości" in v.lower() for v in res_pat.violations)


# ==============================================================================
# 4. AUTOMATED CERTIFICATION HUB 12 STANDARDS & PQC TESTS
# ==============================================================================

def test_certification_hub_sectoral_standards():
    hub = AutomatedCertificationHub()
    available = hub.list_available_certifications()
    standards_list = [item["standard"] for item in available]

    assert CertificationStandard.HEALTHCARE_MEDTECH_MDR.value in standards_list
    assert CertificationStandard.PUBLIC_ADMIN_KPA_KRI.value in standards_list
    assert CertificationStandard.ACADEMIC_RESEARCH_ALLEA.value in standards_list

    # Generate and verify Healthcare package
    med_pkg = hub.generate_evidence_package(CertificationStandard.HEALTHCARE_MEDTECH_MDR)
    assert med_pkg.readiness_score >= 0.95
    assert "MDR_Rule_11_SaMD_Classification" in med_pkg.controls_matrix
    assert len(med_pkg.pqc_signature) > 32
    assert len(med_pkg.merkle_anchor_root) == 64

    # Generate and verify Public Admin package
    gov_pkg = hub.generate_evidence_package(CertificationStandard.PUBLIC_ADMIN_KPA_KRI)
    assert gov_pkg.readiness_score >= 0.95
    assert "KPA_Art7_Objective_Truth" in gov_pkg.controls_matrix

    # Generate and verify Academic package
    aca_pkg = hub.generate_evidence_package(CertificationStandard.ACADEMIC_RESEARCH_ALLEA)
    assert aca_pkg.readiness_score >= 0.95
    assert "ALLEA_FFP_Zero_Tolerance" in aca_pkg.controls_matrix


# ==============================================================================
# 5. REST API ENDPOINTS INTEGRATION TESTS
# ==============================================================================

def test_api_sectoral_endpoints():
    client = TestClient(app)

    # 1. Healthcare endpoint
    resp_med = client.post(
        "/api/v1/compliance/healthcare/evaluate",
        json={
            "payload": {
                "system_name": "API-Cardio-Assistant",
                "samd_class": "CLASS_IIA",
                "iso14971_risk_analysis_present": True,
                "iso13485_qms_certified": True,
            }
        }
    )
    assert resp_med.status_code == 200
    med_data = resp_med.json()
    assert "is_compliant" in med_data
    assert "autonomous_dnr_blocked" in med_data

    # 2. Public Admin endpoint
    resp_gov = client.post(
        "/api/v1/compliance/public-admin/evaluate",
        json={
            "payload": {
                "system_name": "API-Tax-Office",
                "clearance_level": "JAWNE",
                "provides_legal_basis_and_reasoning": True,
                "kri_interoperability_compliant": True,
            }
        }
    )
    assert resp_gov.status_code == 200
    gov_data = resp_gov.json()
    assert "kpa_art7_objective_truth_verified" in gov_data

    # 3. Academic endpoint
    resp_aca = client.post(
        "/api/v1/compliance/academic/evaluate",
        json={
            "payload": {
                "system_name": "API-University-Thesis",
                "discipline": "EXACT_AND_ENGINEERING",
                "citations": [
                    {"id": "10.1145/3372297.3417882", "type": "DOI"},
                ],
                "fair_data_principles_met": True,
            }
        }
    )
    assert resp_aca.status_code == 200
    aca_data = resp_aca.json()
    assert "allea_ffp_free_verified" in aca_data
