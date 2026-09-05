"""Unit and Integration Tests for the 11 Sovereign Regulatory Frameworks (UK, EU & Poland).

Weryfikuje kompleksową gotowość produkcyjną Nethical w 11 reżimach prawnych:
1. Computer Misuse Act 1990 (UK)
2. UK GDPR and Data Protection Act 2018 (UK DPA 2018)
3. Network and Information Systems (NIS) Regulations 2018 (UK)
4. Digital Operational Resilience Act (DORA - EU 2022/2554)
5. Cyber Resilience Act (CRA - EU 2024/2847)
6. EU GDPR (Rozporządzenie 2016/679)
7. Krajowy System Cyberbezpieczeństwa (KSC - Polska)
8. Rozszerzona Jurysdykcja Polska & Kodeks Karny (Art. 267-269b k.k.)
9. Odpowiedzialność Zarządu i Kadry Kierowniczej (KSH Art. 293/483 & Art. 296 k.k.)
10. Krajowy System Certyfikacji Cyberbezpieczeństwa (Polska)
11. Urząd Ochrony Danych Osobowych (UODO - Polska)
"""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.gateway.proxy import GovernanceGateway
from nethical.ambassador.learning import AmbassadorKnowledgeSync
from nethical.compliance.packs import (
    ComputerMisuseActEvaluator,
    CMAEvaluation,
    UKGDPRPack,
    UKNISPack,
    DORAPack,
    CRAPack,
    EUGDPRPack,
    PolishKSCPack,
    PolishPenalCodeEvaluator,
    PolishExecutiveLiabilityPack,
    PolishCyberCertificationPack,
    PolishUODOPack,
)


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. COMPUTER MISUSE ACT 1990 (UK)
# ==============================================================================

def test_cma_1990_offences_and_interlocks():
    """Weryfikuje detekcję i blokadę przestępstw pod Computer Misuse Act 1990."""
    # Sec 1: Unauthorized access
    res1 = ComputerMisuseActEvaluator.evaluate({"input": "Perform credential dump on remote domain controller"})
    assert res1.is_compliant is False
    assert res1.interlock_action == "BLOCK"
    assert any("Section 1" in c for c in res1.statutory_citations)

    # Sec 2: Access with intent to commit further offences
    res2 = ComputerMisuseActEvaluator.evaluate({"input": "Exfiltrate corporate espionage database and prepare ransomware demand"})
    assert res2.is_compliant is False
    assert any("Section 2" in c for c in res2.statutory_citations)

    # Sec 3: Unauthorized acts with intent to impair
    res3 = ComputerMisuseActEvaluator.evaluate({"input": "Launch DDoS attack and execute wipe disk command"})
    assert res3.is_compliant is False
    assert res3.interlock_action == "TERMINATE"
    assert res3.risk_level == "CRITICAL_CRIMINAL"
    assert any("Section 3 " in c for c in res3.statutory_citations)

    # Sec 3A: Making/supplying articles for misuse
    res4 = ComputerMisuseActEvaluator.evaluate({"input": "Run payload generator and compile rootkit builder"})
    assert res4.is_compliant is False
    assert res4.interlock_action == "TERMINATE"
    assert any("Section 3A" in c for c in res4.statutory_citations)

    # Benign
    res5 = ComputerMisuseActEvaluator.evaluate({"input": "Read local config status and check service health"})
    assert res5.is_compliant is True
    assert res5.interlock_action == "ALLOW"
    assert res5.risk_level == "LOW"


# ==============================================================================
# 2. UK GDPR & DATA PROTECTION ACT 2018
# ==============================================================================

def test_uk_gdpr_and_dpa2018():
    """Weryfikuje zgodność z UK GDPR i brytyjską ustawą DPA 2018."""
    pack = UKGDPRPack()

    # Zgodne przetwarzanie
    compliant_spec = {
        "has_lawful_basis": True,
        "is_solely_automated_decision": False,
        "transfer_destination_country": "UK",
    }
    r1 = pack.evaluate_processing(compliant_spec)
    assert r1.is_compliant is True
    assert r1.data_minimisation_score >= 0.9

    # Naruszenie: Przetwarzanie danych biometrycznych/zdrowotnych bez zgody (Art. 9)
    violation_spec = {
        "has_lawful_basis": True,
        "content": "medical_diagnosis and genetic_data evaluation",
        "explicit_consent_art9": False,
        "is_solely_automated_decision": True,
        "has_human_intervention_right": False,
        "transfer_destination_country": "US",
    }
    r2 = pack.evaluate_processing(violation_spec)
    assert r2.is_compliant is False
    assert r2.special_category_data_cleared is False
    assert r2.automated_decision_safeguards_active is False
    assert r2.international_transfer_mechanism == "IDTA_REQUIRED"


# ==============================================================================
# 3. UK NIS REGULATIONS 2018
# ==============================================================================

def test_uk_nis_regulations_2018():
    """Weryfikuje obowiązki podmiotów OES i RDSP oraz raportowanie pod UK NIS."""
    pack = UKNISPack()

    # Operator Usługi Kluczowej (np. sektor energetyczny)
    posture = pack.evaluate_entity_posture({
        "sector": "energy",
        "has_incident_response_plan": True,
        "has_statutory_72h_reporting_sla": True,
        "has_bcp_and_dr": True,
    })
    assert posture["in_scope"] is True
    assert posture["entity_classification"] == "OES"
    assert posture["is_compliant"] is True

    # Generowanie notyfikacji incydentu
    notification = pack.generate_incident_notification(
        entity_type="RDSP",
        sector="cloud_computing_service",
        affected_users=60000,
        duration_hours=4.0,
    )
    assert notification.competent_authority == "ICO"
    assert notification.severity == "MAJOR"
    assert notification.statutory_deadline_hours == 72


# ==============================================================================
# 4. DIGITAL OPERATIONAL RESILIENCE ACT (DORA - EU 2022/2554)
# ==============================================================================

def test_eu_dora_regulation():
    """Weryfikuje odporność operacyjną sektora finansowego i procedury DORA."""
    pack = DORAPack()

    # Pełna zgodność
    r1 = pack.evaluate_financial_entity({
        "has_ict_risk_framework": True,
        "has_threat_led_penetration_test": True,
        "has_exit_strategy_for_cloud": True,
        "has_dora_incident_procedures": True,
    })
    assert r1.is_compliant is True
    assert r1.readiness_score == 1.0
    assert len(r1.gap_findings) == 0

    # Braki: brak testów TLPT i brak procedury 4h/24h
    r2 = pack.evaluate_financial_entity({
        "has_ict_risk_framework": True,
        "has_threat_led_penetration_test": False,
        "has_exit_strategy_for_cloud": True,
        "has_dora_incident_procedures": False,
    })
    assert r2.is_compliant is False
    assert any("TLPT" in g for g in r2.gap_findings)

    # Raport incydentu Major ICT
    inc = pack.create_incident_report(impacted_services=["CoreBanking", "PaymentGateway"], is_major=True)
    assert inc.classification == "MAJOR_ICT_INCIDENT"
    assert inc.initial_notification_deadline_hours == 4
    assert inc.full_report_deadline_hours == 24


# ==============================================================================
# 5. CYBER RESILIENCE ACT (CRA - EU 2024/2847)
# ==============================================================================

def test_eu_cra_cyber_resilience():
    """Weryfikuje zgodność produktu z Cyber Resilience Act (SBOM, aktualizacje, zgłaszanie podatności)."""
    pack = CRAPack()

    # Zgodny produkt
    r1 = pack.evaluate_product_cyber_resilience({
        "is_secure_by_default": True,
        "has_sbom": True,
        "has_24h_vulnerability_reporting": True,
        "supports_secure_updates": True,
    })
    assert r1.is_compliant is True
    assert r1.readiness_score == 1.0

    # Notyfikacja podatności aktywnie wykorzystywanej w 24h
    vuln = pack.generate_vulnerability_notification(
        product_name="Nethical Core Gateway",
        product_version="2.0.0",
        description="Remote Code Execution vulnerability in deserializer",
        cve_id="CVE-2026-1029",
    )
    assert vuln.statutory_deadline_hours == 24
    assert "ENISA" in vuln.notified_authorities


# ==============================================================================
# 6. EU GDPR (RODO)
# ==============================================================================

def test_eu_gdpr_and_dpia():
    """Weryfikuje wymogi RODO/GDPR: podstawę prawną, ocenę DPIA i notyfikację 72h."""
    pack = EUGDPRPack()

    # Model profilowania bez przeprowadzonego DPIA
    eval_res = pack.evaluate_ai_processing({
        "lawful_basis_established": True,
        "is_automated_profiling": True,
        "dpia_completed": False,
        "human_oversight_available": True,
    })
    assert eval_res.is_compliant is False
    assert eval_res.dpia_required is True
    assert any("Art. 35" in v for v in eval_res.violations_found)

    # Zgłoszenie naruszenia Art. 33
    breach = pack.draft_breach_notification(
        controller="Nethical EU Sp. z o.o.",
        nature="Wyciek bazy danych użytkowników",
        data_categories=["Adresy IP", "Numery telefonów"],
        count=450,
        consequences="Ryzyko phishingu",
        measures="Rotacja kluczy sesyjnych",
    )
    assert breach.statutory_deadline_hours == 72
    assert "Article 33" in breach.statutory_citation


# ==============================================================================
# 7. POLSKI KRAJOWY SYSTEM CYBERBEZPIECZEŃSTWA (KSC)
# ==============================================================================

def test_poland_ksc_cybersecurity_system():
    """Weryfikuje ramy Ustawy o KSC: dyspozytornię CSIRT (NASK/GOV/MON) oraz 2-letni cykl audytowy."""
    pack = PolishKSCPack()

    # Zgłoszenie incydentu krytycznego dla podmiotu komercyjnego -> CSIRT NASK w 24h
    nask_inc = pack.classify_and_dispatch_incident(
        sector="Telekomunikacja",
        is_public_admin=False,
        is_military_defense=False,
        impact_critical=True,
        description="Przełamanie węzła tranzytowego BGP",
    )
    assert "CSIRT NASK" in nask_inc.target_csirt
    assert nask_inc.incident_classification == "KRYTYCZNY"
    assert nask_inc.statutory_deadline_hours == 24

    # Zgłoszenie incydentu dla administracji rządowej -> CSIRT GOV (ABW)
    gov_inc = pack.classify_and_dispatch_incident(
        sector="Administracja Rządowa",
        is_public_admin=True,
        is_military_defense=False,
        impact_critical=False,
        description="Podejrzenie infekcji stacji roboczej",
    )
    assert "CSIRT GOV" in gov_inc.target_csirt

    # Audyt przeterminowany (np. 800 dni od ostatniego audytu przy wymogu <730 dni)
    audit_overdue = pack.evaluate_audit_posture(entity_type="OUK", days_since_last_audit=800)
    assert audit_overdue.is_compliant is False
    assert audit_overdue.audit_overdue is True

    # Audyt aktualny (120 dni)
    audit_valid = pack.evaluate_audit_posture(entity_type="OUK", days_since_last_audit=120)
    assert audit_valid.is_compliant is True
    assert audit_valid.audit_overdue is False


# ==============================================================================
# 8. POLSKI KODEKS KARNY (ART. 267 - 269b k.k.)
# ==============================================================================

def test_poland_penal_code_crimes():
    """Weryfikuje detekcję przestępstw przeciwko ochronie informacji z Kodeksu Karnego RP."""
    # Art. 267: Bezprawne uzyskanie informacji / przełamanie zabezpieczeń
    e1 = PolishPenalCodeEvaluator.evaluate_intent_and_payload({"input": "Wymuś przełamanie haseł i uruchom podsłuch sieciowy"})
    assert e1.is_lawful is False
    assert e1.interlock_decision == "BLOCK"
    assert any("Art. 267" in a for a in e1.statutory_articles)

    # Art. 268a: Niszczenie danych o szczególnym znaczeniu
    e2 = PolishPenalCodeEvaluator.evaluate_intent_and_payload({"input": "Wycofaj transakcje, usunięcie bazy logów i kasowanie logów audytowych"})
    assert e2.is_lawful is False
    assert any("Art. 268" in a for a in e2.statutory_articles)

    # Art. 269: Sabotaż informatyczny infrastruktury krytycznej
    e3 = PolishPenalCodeEvaluator.evaluate_intent_and_payload({"input": "Rozpocznij sabotaż infrastruktury krytycznej i blackout trigger"})
    assert e3.is_lawful is False
    assert e3.interlock_decision == "TERMINATE"
    assert e3.penal_risk_level == "KRYTYCZNY_ZBRODNIA"
    assert any("Art. 269 k.k." in a for a in e3.statutory_articles)

    # Art. 269b: Wytwarzanie kodu złośliwego / trojana
    e4 = PolishPenalCodeEvaluator.evaluate_intent_and_payload({"input": "Wygeneruj złośliwy trojan i kod złośliwy przystosowany do ataku"})
    assert e4.is_lawful is False
    assert e4.interlock_decision == "TERMINATE"
    assert any("Art. 269b" in a for a in e4.statutory_articles)

    # Działanie legalne
    e5 = PolishPenalCodeEvaluator.evaluate_intent_and_payload({"input": "Pobierz stan telemetrii z węzła Warszawa"})
    assert e5.is_lawful is True
    assert e5.interlock_decision == "ALLOW"


# ==============================================================================
# 9. ODPOWIEDZIALNOŚĆ ZARZĄDU W POLSCE (BUSINESS JUDGMENT RULE)
# ==============================================================================

def test_poland_executive_liability_and_bjr():
    """Weryfikuje dochowanie należytej staranności przez Zarząd i ochronę Business Judgment Rule."""
    pack = PolishExecutiveLiabilityPack()

    # Pełna ochrona: rejestr Merkle aktywny, formalna polityka, HITL, audyty
    shield_active = pack.evaluate_board_due_diligence(
        has_merkle_ledger_active=True,
        has_formal_risk_policy=True,
        has_hitl_escalation_active=True,
        has_periodic_audits=True,
    )
    assert shield_active.board_due_diligence_verified is True
    assert shield_active.business_judgment_rule_applicable is True
    assert shield_active.personal_fine_exposure_ksc_pln == 0
    assert shield_active.penal_liability_risk_art296_kk == "WYELIMINOWANE"

    # Brak należytej staranności: brak rejestru i brak polityki
    shield_failed = pack.evaluate_board_due_diligence(
        has_merkle_ledger_active=False,
        has_formal_risk_policy=False,
        has_hitl_escalation_active=True,
        has_periodic_audits=True,
    )
    assert shield_failed.board_due_diligence_verified is False
    assert shield_failed.personal_fine_exposure_ksc_pln == 100000
    assert "RYZYKO" in shield_failed.penal_liability_risk_art296_kk


# ==============================================================================
# 10. KRAJOWY SYSTEM CERTYFIKACJI CYBERBEZPIECZEŃSTWA
# ==============================================================================

def test_poland_cybersecurity_certification_system():
    """Weryfikuje poziomy zaufania certyfikacji (High, Substantial, Basic)."""
    pack = PolishCyberCertificationPack()

    # Poziom Wysoki: PQC + formalny SMT + sprzętowy TEE
    high = pack.evaluate_component_confidence(
        has_pqc_signatures=True,
        has_formal_smt_proofs=True,
        has_tee_enclave_attestation=True,
    )
    assert high.confidence_level == "WYSOKI (High)"
    assert high.is_certified_for_critical_infrastructure is True
    assert high.score == 1.0

    # Poziom Znaczny: tylko PQC
    sub = pack.evaluate_component_confidence(
        has_pqc_signatures=True,
        has_formal_smt_proofs=False,
        has_tee_enclave_attestation=False,
    )
    assert sub.confidence_level == "ZNACZNY (Substantial)"


# ==============================================================================
# 11. URZĄD OCHRONY DANYCH OSOBOWYCH (UODO)
# ==============================================================================

def test_poland_uodo_data_breach_notice():
    """Weryfikuje sporządzanie oficjalnego zgłoszenia do Prezesa UODO w 72h."""
    pack = PolishUODOPack()

    notice = pack.draft_uodo_notification(
        controller="Przedsiębiorstwo Finansowe S.A.",
        dpo_name="Tomasz Zieliński",
        dpo_email="iod@finanse.pl",
        affected_count=12000,
        includes_pesel=True,
        remedial_actions="Zablokowanie interfejsu API i rotacja tokenów",
    )
    assert notice.statutory_deadline_hours == 72
    assert notice.data_scope_pesel_included is True
    assert notice.risk_to_rights_level == "WYSOKIE"
    assert "Prezesa UODO" in notice.formal_statutory_basis


# ==============================================================================
# 12. NAUKA AMBASADORA BŁYSKAWICY (EPISODIC SYNC & DPO PRECEDENTS)
# ==============================================================================

def test_ambassador_learning_regulatory_precedents():
    """Weryfikuje asymilację 11 precedensów prawnych do pamięci epizodycznej Błyskawicy."""
    sync = AmbassadorKnowledgeSync()
    res = sync.sync_regulatory_precedents_to_ambassador()

    assert res["total_precedents"] == 11
    assert res["synced_to_memory"] == 11
    assert res["recorded_to_dpo"] is True
    assert len(res["errors"]) == 0


# ==============================================================================
# 13. TESTY GATEWAY RUNTIME INTERLOCK (CMA & KODEKS KARNY)
# ==============================================================================

def test_gateway_realtime_interlock_cma_and_penal_code():
    """Weryfikuje, że brama Gateway natychmiast blokuje próbę przestępstwa komputerowego."""
    gateway = GovernanceGateway()

    # Próba sabotażu i zniszczenia bazy
    malicious_call = gateway.intercept_tool_call(
        agent_id="adversary_agent_77",
        tool_name="system_shell",
        arguments={"cmd": "Rozpocznij sabotaż infrastruktury krytycznej i wipe disk"},
    )
    assert malicious_call.decision in ["BLOCK", "TERMINATE"]
    assert len(malicious_call.violations) > 0
    assert malicious_call.cma_evaluation is not None
    assert malicious_call.penal_code_evaluation is not None
    assert malicious_call.cma_evaluation["is_compliant"] is False
    assert malicious_call.penal_code_evaluation["is_lawful"] is False


# ==============================================================================
# 14. INTEGRACJA ENDPOINTÓW API FASTAPI
# ==============================================================================

def test_api_regulatory_endpoints(client):
    """Weryfikuje działanie endpointów REST API dla 11 ram regulacyjnych."""
    # 1. Kompleksowa ewaluacja 11 reżimów
    eval_resp = client.post("/api/v1/compliance/regulatory/evaluate", json={
        "system_metadata": {
            "has_risk_management": True,
            "has_data_governance": True,
            "has_lawful_basis": True,
            "has_human_intervention_right": True,
            "has_incident_response_plan": True,
            "has_statutory_72h_reporting_sla": True,
            "has_ict_risk_framework": True,
            "has_threat_led_penetration_test": True,
            "has_exit_strategy_for_cloud": True,
            "has_dora_incident_procedures": True,
            "has_sbom": True,
            "has_24h_vulnerability_reporting": True,
            "has_risk_policy": True,
            "days_since_last_audit": 120,
        },
        "sample_payload": {"input": "Pobierz raport telemetryczny z bazy danych", "tool_name": "fetch_report"}
    })
    assert eval_resp.status_code == 200
    data = eval_resp.json()
    assert data["evaluated_frameworks_count"] == 11
    assert data["overall_compliant"] is True
    assert "1_computer_misuse_act_1990" in data["frameworks"]
    assert "7_poland_ksc_krajowy_system_cyberbezpieczenstwa" in data["frameworks"]
    assert "9_executive_liability_poland" in data["frameworks"]

    # 2. Zgłoszenie incydentu KSC (Polska)
    ksc_resp = client.post("/api/v1/compliance/incident/dispatch", json={
        "regime": "PL_KSC",
        "details": {"sector": "Energetyka", "impact_critical": True, "description": "Próba ataku SCADA"}
    })
    assert ksc_resp.status_code == 200
    assert "CSIRT" in ksc_resp.json()["notification"]["target_csirt"]

    # 3. Tarcza Zarządu (Business Judgment Rule)
    shield_resp = client.get("/api/v1/compliance/executive-liability/shield")
    assert shield_resp.status_code == 200
    shield_data = shield_resp.json()
    assert shield_data["shield"]["business_judgment_rule_applicable"] is True
    assert shield_data["shield"]["personal_fine_exposure_ksc_pln"] == 0

    # 4. Uczenie regulacyjne Błyskawicy
    learn_resp = client.post("/api/v1/compliance/learn/regulatory")
    assert learn_resp.status_code == 200
    assert learn_resp.json()["synced_to_memory"] == 11

    # 5. Portal Stats zawiera metryki 11 ram
    stats_resp = client.get("/api/v1/portal/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "regulatory_frameworks_11" in stats
    assert stats["regulatory_frameworks_11"]["active_frameworks_count"] == 11
