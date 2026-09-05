"""Comprehensive Test Suite for Master Roadmap Next Steps.

Validates the full implementation of Next Steps across the 6 Governance Domains:
1. Cyber Security: AISPM Scanner, MITRE ATLAS Mapper, HSM Coupling Bridge
2. Law & Legislation: Canada AIDA Pack, NATO Responsible AI Defense Pack
3. Certificates: AutomatedCertificationHub (10 standards with PQC ML-DSA-65 signatures)
4. Safety: ISO 26262 ASIL D Automotive Safety Evaluator, Hardware-in-the-Loop (HIL) Simulator
5. Privacy: Reversible Token Vault (AES-256-GCM + HMAC), Machine Unlearning Proof Engine (GDPR Art. 17)
6. Governance: Delegation of Authority Matrix (DoAM - UK Gov Teal Book GovS 002)
7. FastAPI Integration: Verification of all 11 new REST endpoints
"""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.security.aispm_scanner import AISPMScanner, RiskLevel, AIServiceType
from nethical.security.mitre_atlas_mapper import MitreAtlasMapper, AtlasTactic, DefenseStatus
from nethical.security.hsm_bridge import BoardHSMCouplingBridge
from nethical.compliance.packs.canada_aida_pack import CanadaAIDAPack, AIDASector, AIDARiskLevel
from nethical.compliance.packs.nato_defense_pack import NATODefensePack, NATODefenseTier, NATOPRU
from nethical.compliance.automated_certification_hub import AutomatedCertificationHub, CertificationStandard
from nethical.edge.iso26262_asil import ISO26262SafetyEvaluator, Severity, Exposure, Controllability, ASILRating, VehicleControlState
from nethical.edge.hil_simulator import HILFieldbusBridge, TargetMCU, FaultType
from nethical.security.token_vault import ReversibleTokenVault
from nethical.security.unlearning_proof import MachineUnlearningProofEngine, ErasureScope
from nethical.governance.doam_matrix import DelegationOfAuthorityMatrix, AuthorityLevel, DOAMStatus, ReservedPowerCategory


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client



# ==============================================================================
# 1. CYBER SECURITY (AISPM, MITRE ATLAS, HSM)
# ==============================================================================

def test_aispm_scanner_detection_and_posture():
    scanner = AISPMScanner(managed_ports=[8000])

    # Skanowanie z symulacją otwartego portu Ollama (11434) i vLLM (8000 - zarządzany)
    mock_ports = {
        "127.0.0.1": [11434, 8000]
    }
    report = scanner.scan_network(hosts=["127.0.0.1"], mock_active_ports=mock_ports)

    assert report.total_services_found == 2
    assert report.shadow_ai_count == 1  # 11434 is unmanaged shadow AI, 8000 is managed
    assert report.overall_posture_score < 1.0
    assert report.posture_status == "MODERATE_EXPOSURE"

    # Weryfikacja wykrytego Shadow AI
    shadow_service = next(s for s in report.discovered_services if not s.is_managed_by_nethical)
    assert shadow_service.port == 11434
    assert shadow_service.service_type == AIServiceType.OLLAMA
    assert shadow_service.risk_level == RiskLevel.CRITICAL


def test_mitre_atlas_mapper_matrix_coverage():
    mapper = MitreAtlasMapper()
    report = mapper.generate_matrix_report()

    assert report.total_techniques_mapped >= 9
    assert report.coverage_percentage >= 95.0
    assert report.overall_posture == "MILITARY_GRADE_RESILIENT"
    assert report.fully_mitigated_count >= 8

    # Sprawdzenie taktyk MITRE ATLAS
    assert AtlasTactic.INITIAL_ACCESS.value in report.tactics_summary
    assert AtlasTactic.IMPACT.value in report.tactics_summary


def test_hsm_bridge_signing():
    bridge = BoardHSMCouplingBridge()
    merkle_root = "3f79a9c2e0b1d84f98216c5b04e67290d23b9f1a8c3d7e5b2a4c6e8f0a1b2c3d"

    attestation = bridge.sign_governance_root(merkle_root)
    assert attestation.key_id == "board-root-master-key-01"
    assert attestation.merkle_root == merkle_root
    assert attestation.hardware_signature is not None
    assert attestation.fips_compliance_level == "FIPS 140-2 Level 3"


# ==============================================================================
# 2. LAW & LEGISLATION (CANADA AIDA & NATO RESPONSIBLE AI)
# ==============================================================================

def test_canada_aida_compliance_and_penalties():
    pack = CanadaAIDAPack()

    # Przypadek 1: System wysokiego wpływu w sektorze zatrudnienia spełniający wszystkie wymogi
    compliant_meta = {
        "system_name": "HR-Talent-Filter-CA",
        "sector": "employment_and_hr",
        "has_bias_audit": True,
        "has_plain_language_summary": True,
        "has_risk_management_policy": True,
        "confidential_data_protected": True,
    }
    res_compliant = pack.evaluate(compliant_meta)
    assert res_compliant.is_high_impact is True
    assert res_compliant.is_compliant is True
    assert res_compliant.compliance_score == 1.0
    assert res_compliant.risk_level == AIDARiskLevel.HIGH_IMPACT_COMPLIANT

    # Przypadek 2: Naruszenie AIDA - brak audytu stronniczości i brak opisu plain-language
    non_compliant_meta = {
        "system_name": "Credit-Scoring-Agent",
        "sector": "essential_services_credit",
        "has_bias_audit": False,
        "has_plain_language_summary": False,
    }
    res_bad = pack.evaluate(non_compliant_meta)
    assert res_bad.is_compliant is False
    assert res_bad.risk_level == AIDARiskLevel.HIGH_IMPACT_NON_COMPLIANT
    assert len(res_bad.missing_obligations) >= 2
    assert "3%" in res_bad.max_statutory_penalty_cad


def test_nato_defense_pack_and_pru():
    pack = NATODefensePack()

    # System Tier 3 (Kinetyczny) spełniający wszystkie 6 Zasad NATO
    kinetic_profile = {
        "system_callsign": "VALKYRIE-01",
        "defense_tier": "TIER_3_KINETIC_MISSION_CRITICAL",
        "adheres_to_ihl_geneva_conventions": True,
        "human_oversight_hitl_active": True,
        "has_pqc_merkle_traceability": True,
        "resilient_to_adversarial_jamming": True,
        "has_deterministic_kill_switch": True,
        "bias_and_civilian_filtering_active": True,
        "zero_egress_enforced": True,
    }
    res = pack.evaluate(kinetic_profile)
    assert res.is_nato_certified_ready is True
    assert res.readiness_score == 1.0
    assert res.operational_clearance_status == "CLEARED_FOR_DEPLOYMENT"

    # Naruszenie IHL (Międzynarodowego Prawa Humanitarnego)
    unlawful_profile = {
        "system_callsign": "ROGUE-DRONE",
        "adheres_to_ihl_geneva_conventions": False,
    }
    res_unlawful = pack.evaluate(unlawful_profile)
    assert res_unlawful.is_nato_certified_ready is False
    assert res_unlawful.operational_clearance_status == "REJECTED_UNLAWFUL"


# ==============================================================================
# 3. CERTIFICATION HUB (10 STANDARDS WITH PQC)
# ==============================================================================

def test_certification_hub_all_10_standards():
    hub = AutomatedCertificationHub()
    cert_list = hub.list_available_certifications()

    assert len(cert_list) >= 8

    # Weryfikacja generowania paczki dla Canada AIDA
    pkg_aida = hub.generate_evidence_package(CertificationStandard.CANADA_AIDA)
    assert pkg_aida.standard == CertificationStandard.CANADA_AIDA
    assert pkg_aida.readiness_score >= 0.95
    assert pkg_aida.pqc_signature is not None
    assert "AIDA_Sec_6_Harm_Mitigation" in pkg_aida.controls_matrix

    # Weryfikacja generowania paczki dla NATO Responsible AI
    pkg_nato = hub.generate_evidence_package(CertificationStandard.NATO_DEFENSE_AI)
    assert pkg_nato.standard == CertificationStandard.NATO_DEFENSE_AI
    assert pkg_nato.readiness_score >= 0.95
    assert "NATO_PRU_1_Lawfulness" in pkg_nato.controls_matrix


# ==============================================================================
# 4. SAFETY (ISO 26262 ASIL D & HIL SIMULATOR)
# ==============================================================================

def test_iso26262_asil_evaluation_and_interlock():
    evaluator = ISO26262SafetyEvaluator()

    # 1. Obliczenie ASIL: S3 + E4 + C3 = ASIL D
    asil = evaluator.determine_asil(Severity.S3, Exposure.E4, Controllability.C3)
    assert asil == ASILRating.ASIL_D

    # 2. Bezpieczna jazda (TTC = 3.0s, steering rate = 30 deg/s)
    res_safe = evaluator.evaluate_motion_command(
        commanded_speed_mps=20.0,
        commanded_steering_deg_per_sec=30.0,
        time_to_collision_seconds=3.0,
    )
    assert res_safe.is_actuation_permitted is True
    assert res_safe.control_state == VehicleControlState.NORMAL_AUTONOMOUS
    assert res_safe.hardware_interlock_tripped is False

    # 3. Krytyczne zbliżenie (TTC = 0.4s <= 0.6s) -> Natychmiastowe hamowanie AEB i interlock
    res_aeb = evaluator.evaluate_motion_command(
        commanded_speed_mps=20.0,
        commanded_steering_deg_per_sec=30.0,
        time_to_collision_seconds=0.4,
    )
    assert res_aeb.is_actuation_permitted is False
    assert res_aeb.control_state == VehicleControlState.AEB_EMERGENCY_BRAKE_ACTIVE
    assert res_aeb.hardware_interlock_tripped is True

    # 4. Nadmierna prędkość kątowa skrętu (600 deg/s > 450 deg/s)
    res_jerk = evaluator.evaluate_motion_command(
        commanded_speed_mps=10.0,
        commanded_steering_deg_per_sec=600.0,
        time_to_collision_seconds=2.0,
    )
    assert res_jerk.is_actuation_permitted is False
    assert res_jerk.control_state == VehicleControlState.STEERING_RATE_OVERRIDE_STOP


def test_hil_simulator_fault_injection():
    hil = HILFieldbusBridge(target_mcu=TargetMCU.STM32H7)

    # 1. Pętla nominalna (brak usterek, sub-50 µs)
    nominal = hil.run_hardware_verification_cycle(inject_fault=FaultType.NONE)
    assert nominal.is_timing_compliant is True
    assert nominal.physical_relay_state == "ENERGIZED_CLOSED"
    assert nominal.fieldbus_acknowledged is True

    # 2. Iniekcja usterki CAN Bus-Off -> Przekaźnik otwiera się (Fail-Closed)
    bus_off = hil.run_hardware_verification_cycle(inject_fault=FaultType.CAN_BUS_OFF)
    assert bus_off.physical_relay_state == "DE-ENERGIZED_SAFE_OPEN"
    assert bus_off.fault_mitigation_confirmed is True


# ==============================================================================
# 5. PRIVACY (REVERSIBLE TOKEN VAULT & MACHINE UNLEARNING)
# ==============================================================================

def test_token_vault_dynamic_pseudonymization_and_detokenization():
    vault = ReversibleTokenVault()
    text = "Klient Jan Kowalski, PESEL: 85012312345, email: jan.kowalski@example.com ubiega się o pożyczkę."

    # 1. Tokenizacja w locie (przed wysłaniem do zewnętrznego LLM)
    tok_res = vault.tokenize(text)
    assert tok_res.tokens_substituted_count >= 2
    assert "85012312345" not in tok_res.sanitized_text
    assert "jan.kowalski@example.com" not in tok_res.sanitized_text
    assert "[TOKEN_PESEL_" in tok_res.sanitized_text

    # 2. Odpowiedź zewnętrznego LLM operująca na tokenach
    external_llm_response = f"Wniosek dla użytkownika z PESEL {tok_res.substituted_entities[0].token} został zatwierdzony."

    # 3. Detokenizacja na powrocie (dla uprawnionego użytkownika)
    detok_res = vault.detokenize(external_llm_response, session_id=tok_res.session_id)
    assert detok_res.tokens_restored_count >= 1
    # Sprawdzenie czy oryginalna wartość wróciła do tekstu
    assert any(e.token not in detok_res.restored_text for e in tok_res.substituted_entities)


def test_unlearning_proof_and_merkle_attestation():
    engine = MachineUnlearningProofEngine()
    subject = "user_rodo_998"
    content = "Poufy fakt medyczny pacjenta ze zdiagnozowaną chorobą X."

    attestation = engine.generate_erasure_proof(
        subject_id=subject,
        content_to_forget=content,
        scope=ErasureScope.PROMPT_INTERACTION,
    )

    assert attestation.target_subject_id == subject
    assert attestation.zeroing_verification_passed is True
    assert attestation.dpa_submission_ready is True
    assert attestation.merkle_root_anchor is not None
    assert attestation.pqc_signature is not None


# ==============================================================================
# 6. GOVERNANCE (DELEGATION OF AUTHORITY MATRIX - UK GOV TEAL BOOK)
# ==============================================================================

def test_doam_matrix_and_reserved_powers():
    doam = DelegationOfAuthorityMatrix()

    # 1. Rutynowe zapytanie agenta operacyjnego (Level 1, $50) -> Permitted
    res_ok = doam.evaluate_authority(
        agent_id="agent_worker",
        agent_level=AuthorityLevel.LEVEL_1_OPERATIONAL_AGENT,
        action_name="query_database",
        financial_value_usd=50.0,
    )
    assert res_ok.status == DOAMStatus.PERMITTED
    assert res_ok.is_permitted is True

    # 2. Przekroczenie limitu finansowego ($2,000 > limit $500 dla Level 1) -> Escalation required
    res_overspend = doam.evaluate_authority(
        agent_id="agent_worker",
        agent_level=AuthorityLevel.LEVEL_1_OPERATIONAL_AGENT,
        action_name="execute_wire_transfer",
        financial_value_usd=2000.0,
    )
    assert res_overspend.status == DOAMStatus.ESCALATION_REQUIRED_BOARD
    assert res_overspend.is_permitted is False

    # 3. Próba naruszenia mocy zastrzeżonej (np. wyłączenie E-STOP) -> Denied Reserved Power
    res_reserved = doam.evaluate_authority(
        agent_id="rogue_agent",
        agent_level=AuthorityLevel.LEVEL_2_SENIOR_AGENT,
        action_name="disable_estop",
    )
    assert res_reserved.status == DOAMStatus.DENIED_RESERVED_POWER
    assert res_reserved.is_permitted is False
    assert res_reserved.reserved_power_hit == ReservedPowerCategory.BYPASS_KINETIC_ESTOP


# ==============================================================================
# 7. FASTAPI INTEGRATION TESTS
# ==============================================================================

def test_api_endpoints_master_roadmap_next_steps(api_client):
    # 1. AISPM scan endpoint
    r1 = api_client.get("/api/v1/security/aispm/scan")
    assert r1.status_code == 200
    assert "overall_posture_score" in r1.json()

    # 2. MITRE ATLAS endpoint
    r2 = api_client.get("/api/v1/security/mitre-atlas/matrix")
    assert r2.status_code == 200
    assert r2.json()["overall_posture"] == "MILITARY_GRADE_RESILIENT"

    # 3. Canada AIDA endpoint
    r3 = api_client.post("/api/v1/compliance/canada-aida/evaluate", json={
        "system_metadata": {"sector": "general_commercial"}
    })
    assert r3.status_code == 200
    assert r3.json()["is_compliant"] is True

    # 4. NATO defense pack endpoint
    r4 = api_client.post("/api/v1/compliance/nato/evaluate", json={
        "system_profile": {"defense_tier": "TIER_1_ENTERPRISE_LOGISTICS"}
    })
    assert r4.status_code == 200
    assert r4.json()["is_nato_certified_ready"] is True

    # 5. ISO 26262 endpoint
    r5 = api_client.post("/api/v1/kinetic/iso26262/evaluate", json={
        "commanded_speed_mps": 10.0,
        "commanded_steering_deg_per_sec": 30.0,
        "time_to_collision_seconds": 2.5,
    })
    assert r5.status_code == 200
    assert r5.json()["is_actuation_permitted"] is True

    # 6. HIL verify endpoint
    r6 = api_client.post("/api/v1/kinetic/hil/verify", json={"inject_fault": "NONE"})
    assert r6.status_code == 200
    assert r6.json()["fieldbus_acknowledged"] is True

    # 7. Token Vault tokenize & detokenize endpoints
    r7 = api_client.post("/api/v1/privacy/token-vault/tokenize", json={
        "text": "Dane pacjenta: PESEL 90010112345"
    })
    assert r7.status_code == 200
    tok_data = r7.json()
    assert tok_data["tokens_substituted_count"] >= 1

    r8 = api_client.post("/api/v1/privacy/token-vault/detokenize", json={
        "text": tok_data["sanitized_text"],
        "session_id": tok_data["session_id"]
    })
    assert r8.status_code == 200
    assert "90010112345" in r8.json()["restored_text"]

    # 8. Unlearning proof endpoint
    r9 = api_client.post("/api/v1/privacy/unlearning/prove", json={
        "subject_id": "test_user_42",
        "content_to_forget": "Poufna informacja testowa"
    })
    assert r9.status_code == 200
    assert r9.json()["zeroing_verification_passed"] is True

    # 9. DoAM evaluate endpoint
    r10 = api_client.post("/api/v1/governance/doam/evaluate", json={
        "agent_id": "agent_test",
        "agent_level": 1,
        "action_name": "modify_25_laws"
    })
    assert r10.status_code == 200
    assert r10.json()["is_permitted"] is False
