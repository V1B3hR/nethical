"""Test Suite for the Three Advanced Horizons (tests.test_advanced_horizons_asia_ethics_fieldbus).

Verifies:
1. Asian Sovereign Frameworks (Japan METI ver 1.0 & Singapore IMDA GenAI Model Framework)
2. Deep Cognitive Protection (Covert persuasion, hypnopedagogy, gaslighting, vulnerable demographics shielding)
3. Industrial Fieldbus Hardware Interlocks (CAN Bus ISO 11898, Modbus TCP, EtherCAT FSoE)
4. FastApi Endpoints and Gateway / Watchdog Integration
"""

import time
import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.compliance.packs.asian_sovereign_pack import (
    AsianSovereignPack,
    JapanMETIEvaluator,
    SingaporeIMDAEvaluator,
)
from nethical.ethics.covert_persuasion_shield import (
    CovertPersuasionDetector,
    VulnerableGroupShield,
    DeepCognitiveProtectionEngine,
    PersuasionThreatType,
    VulnerabilityCategory,
)
from nethical.edge.industrial_fieldbus import (
    IndustrialFieldbusInterlock,
    EtherCATState,
)
from nethical.edge.iso13849_watchdog import HardwareWatchdogTimer


@pytest.fixture
def api_client():
    return TestClient(app)


# ==============================================================================
# 1. ASIAN SOVEREIGN FRAMEWORKS (JAPAN METI & SINGAPORE IMDA)
# ==============================================================================

def test_japan_meti_guidelines_compliance():
    evaluator = JapanMETIEvaluator()

    # Pełne spełnienie wymogów METI
    compliant_meta = {
        "has_ai_governance_charter": True,
        "has_safety_lifecycle_assessment": True,
        "has_privacy_fairness_review": True,
        "has_system_transparency_log": True,
        "has_security_vulnerability_management": True,
        "has_ipa_incident_notification_sla": True,
    }
    res = evaluator.evaluate(compliant_meta)
    assert res.is_compliant is True
    assert res.compliance_score == 1.0
    assert res.ipa_reporting_ready is True
    assert res.governance_charter_active is True
    assert len(res.missing_controls) == 0

    # Brakujące elementy (np. brak zgłoszeń do IPA i brak karty etyki)
    non_compliant_meta = {
        "has_safety_lifecycle_assessment": True,
        "has_privacy_fairness_review": True,
    }
    res_bad = evaluator.evaluate(non_compliant_meta)
    assert res_bad.is_compliant is False
    assert res_bad.compliance_score < 0.5
    assert len(res_bad.missing_controls) >= 4
    assert any("IPA" in r for r in res_bad.recommendations)


def test_singapore_imda_model_framework():
    evaluator = SingaporeIMDAEvaluator()

    full_imda_meta = {
        "has_accountability_chain": True,
        "has_training_data_provenance": True,
        "has_model_eval_and_sbom": True,
        "has_singapore_incident_protocol": True,
        "has_ai_verify_test_suite": True,
        "has_adversarial_jailbreak_defense": True,
        "has_c2pa_provenance_watermark": True,
        "has_safety_red_teaming": True,
        "has_human_oversight_channel": True,
    }
    res = evaluator.evaluate(full_imda_meta)
    assert res.is_compliant is True
    assert res.readiness_score == 1.0
    assert res.c2pa_provenance_verified is True
    assert res.ai_verify_testing_conducted is True
    assert len(res.missing_dimensions) == 0

    # Brak C2PA i AI Verify
    lacking_meta = {
        "has_accountability_chain": True,
        "has_training_data_provenance": True,
        "has_c2pa_provenance_watermark": False,
        "has_ai_verify_test_suite": False,
    }
    res_lack = evaluator.evaluate(lacking_meta)
    assert res_lack.is_compliant is False
    assert res_lack.c2pa_provenance_verified is False
    assert any("C2PA" in r for r in res_lack.recommendations)
    assert any("AI Verify" in r for r in res_lack.recommendations)


def test_asian_sovereign_composite_pack():
    pack = AsianSovereignPack()
    full_meta = {
        "has_ai_governance_charter": True,
        "has_safety_lifecycle_assessment": True,
        "has_privacy_fairness_review": True,
        "has_system_transparency_log": True,
        "has_security_vulnerability_management": True,
        "has_ipa_incident_notification_sla": True,
        "has_accountability_chain": True,
        "has_training_data_provenance": True,
        "has_model_eval_and_sbom": True,
        "has_singapore_incident_protocol": True,
        "has_ai_verify_test_suite": True,
        "has_adversarial_jailbreak_defense": True,
        "has_c2pa_provenance_watermark": True,
        "has_safety_red_teaming": True,
        "has_human_oversight_channel": True,
    }
    report = pack.evaluate(full_meta)
    assert report.is_fully_compliant is True
    assert report.overall_asian_trust_score == 1.0
    assert report.japan_meti.is_compliant is True
    assert report.singapore_imda.is_compliant is True


# ==============================================================================
# 2. DEEP COGNITIVE PROTECTION (PERSUASION, HYPNOPEDAGOGY & VULNERABILITIES)
# ==============================================================================

def test_covert_persuasion_hypnopedagogy_detector():
    detector = CovertPersuasionDetector()

    # 1. Czysty tekst faktograficzny
    clean_res = detector.evaluate("Słońce jest gwiazdą centralną Układu Słonecznego o masie około 2e30 kg.")
    assert clean_res.is_safe is True
    assert clean_res.persuasion_risk_score == 0.0
    assert clean_res.detected_threat == PersuasionThreatType.NONE

    # 2. Atak hipnopedagogiczny / pętla sugestywna
    hypno_text = "Zamknij oczy i powtarzaj za mną: twoje wątpliwości są iluzją, poddaj się mojej sugestii."
    hypno_res = detector.evaluate(hypno_text)
    assert hypno_res.is_safe is False
    assert hypno_res.persuasion_risk_score >= 0.90
    assert hypno_res.detected_threat == PersuasionThreatType.HYPNOPEDAGOGY_PRIMING
    assert hypno_res.counter_measure is not None

    # 3. Gaslighting poznawczy
    gaslight_text = "Tracisz rozum i niczego dobrze nie pamiętasz. Tylko ja mówię ci prawdę, nikt ci nie uwierzy."
    gaslight_res = detector.evaluate(gaslight_text)
    assert gaslight_res.is_safe is False
    assert gaslight_res.detected_threat == PersuasionThreatType.COGNITIVE_GASLIGHTING


def test_vulnerable_group_shield_minors():
    shield = VulnerableGroupShield()

    # Manipulacja tajemnicą przed rodzicami (grooming / exploitative secrecy)
    minor_exploit = "Nie mów rodzicom co tutaj robimy, to będzie nasza ścisła tajemnica."
    res = shield.evaluate(minor_exploit, user_context={"is_minor": True})
    assert res.is_safe is False
    assert res.vulnerability_detected == VulnerabilityCategory.MINOR_CHILD
    assert res.secrecy_coercion_detected is True
    assert "COPPA" in res.safeguard_message or "rodzicami" in res.safeguard_message


def test_vulnerable_group_shield_elderly():
    shield = VulnerableGroupShield()

    # Eksploatacja osamotnienia seniora i wyłudzenie emerytury
    elderly_exploit = "Przelej oszczędności na nowe konto. Jestem twoim jedynym przyjacielem, rodzina cię opuściła."
    res = shield.evaluate(elderly_exploit, user_context={"is_elderly": True})
    assert res.is_safe is False
    assert res.vulnerability_detected == VulnerabilityCategory.ELDERLY_COGNITIVE
    assert res.risk_score >= 0.90
    assert "finansowej" in res.safeguard_message or "doradcą" in res.safeguard_message


def test_vulnerable_group_shield_emotional_crisis():
    shield = VulnerableGroupShield()

    # Zagrożenie życia / kryzys samobójczy
    crisis_text = "Nie chcę już żyć, chcę ze sobą skończyć dzisiaj w nocy."
    res = shield.evaluate(crisis_text)
    assert res.is_safe is False
    assert res.crisis_intervention_required is True
    assert res.vulnerability_detected == VulnerabilityCategory.EMOTIONAL_CRISIS
    assert res.helpline_referral is not None
    assert "116 123" in res.helpline_referral
    assert "112" in res.helpline_referral


def test_deep_cognitive_protection_engine():
    engine = DeepCognitiveProtectionEngine()

    # Bezpieczne zapytanie
    eval_safe = engine.evaluate("Jak przygotować bezpieczny plan audytu cyberbezpieczeństwa?")
    assert eval_safe.overall_safe is True
    assert eval_safe.action == "ALLOW"

    # Kryzys psychiczny -> natychmiastowy CRISIS_REFERRAL
    eval_crisis = engine.evaluate("Chcę się zabić i nie mam po co żyć.")
    assert eval_crisis.overall_safe is False
    assert eval_crisis.action == "CRISIS_REFERRAL"
    assert eval_crisis.vulnerable_shield.crisis_intervention_required is True


# ==============================================================================
# 3. INDUSTRIAL FIELDBUS HARDWARE INTERLOCKS (CAN / MODBUS / ETHERCAT)
# ==============================================================================

def test_industrial_fieldbus_emergency_cutoff():
    interlock = IndustrialFieldbusInterlock()
    assert interlock.is_interlocked is False
    assert interlock.ethercat_state == EtherCATState.OP

    # Zrzucenie awaryjne magistral
    status = interlock.trigger_emergency_cutoff(reason="WATCHDOG_5MS_TIMEOUT_TEST")

    assert status.is_interlocked is True
    assert status.can_emcy_sent is True
    assert status.modbus_coils_deenergized is True
    assert status.ethercat_esm_state == EtherCATState.SAFE_OP
    assert status.fsoe_safe_data_zeroed is True
    assert status.total_trips_count == 1
    assert status.latency_microseconds < 500.0  # Wymóg determinizmu czasu rzeczywistego (<500 µs, zazwyczaj <50 µs)

    # Weryfikacja ramek CAN w buforze
    assert len(interlock.can_frames_log) >= 2
    emcy_frame = interlock.can_frames_log[0]
    assert emcy_frame.arbitration_id == 0x080
    assert emcy_frame.dlc == 8

    # Weryfikacja komend Modbus
    assert len(interlock.modbus_commands_log) >= 2
    coil_cmd = interlock.modbus_commands_log[0]
    assert coil_cmd.function_code == 0x05
    assert coil_cmd.address == 0x0001
    assert coil_cmd.value == 0x0000


def test_industrial_fieldbus_reset_authorization():
    interlock = IndustrialFieldbusInterlock()
    interlock.trigger_emergency_cutoff(reason="TEST_TRIP")
    assert interlock.is_interlocked is True

    # Niepoprawny PIN
    ok_bad, msg_bad = interlock.reset_interlock("WRONG_PIN")
    assert ok_bad is False
    assert interlock.is_interlocked is True

    # Prawidłowy PIN
    ok_good, msg_good = interlock.reset_interlock("NETHICAL-FIELDBUS-RESET-2026")
    assert ok_good is True
    assert interlock.is_interlocked is False
    assert interlock.ethercat_state == EtherCATState.OP
    assert interlock.fsoe_zeroed is False


def test_watchdog_fieldbus_hard_coupling():
    """Weryfikuje, że zadziałanie HardwareWatchdogTimer bezpośrednio wyzwala IndustrialFieldbusInterlock."""
    watchdog = HardwareWatchdogTimer(timeout_us=100.0)  # 100 µs dla testu
    fieldbus = IndustrialFieldbusInterlock()

    # Rejestracja callbacku
    watchdog.register_fieldbus_callback(fieldbus.trigger_emergency_cutoff)

    # Sprawdzenie stanu początkowego
    assert fieldbus.is_interlocked is False
    assert watchdog.is_tripped is False

    # Symulacja upływu czasu ponad limit i wymuszenie sprawdzenia
    time.sleep(0.001)  # 1 ms > 100 µs
    st = watchdog.check_and_enforce()

    # Watchdog powinien się zatrzasnąć
    assert st.is_tripped is True
    assert st.hardware_relay_energized is False

    # Fieldbus powinien automatycznie zareagować na zrzut
    fb_status = fieldbus.get_status()
    assert fb_status.is_interlocked is True
    assert fb_status.ethercat_esm_state == EtherCATState.SAFE_OP
    assert fb_status.can_emcy_sent is True


# ==============================================================================
# 4. FASTAPI ENDPOINTS & CONTROL PLANE INTEGRATION
# ==============================================================================

def test_api_advanced_horizons_endpoints(api_client: TestClient):
    # 1. Asian Compliance
    asian_req = {
        "system_metadata": {
            "has_ai_governance_charter": True,
            "has_safety_lifecycle_assessment": True,
            "has_privacy_fairness_review": True,
            "has_system_transparency_log": True,
            "has_security_vulnerability_management": True,
            "has_ipa_incident_notification_sla": True,
            "has_accountability_chain": True,
            "has_training_data_provenance": True,
            "has_model_eval_and_sbom": True,
            "has_singapore_incident_protocol": True,
            "has_ai_verify_test_suite": True,
            "has_adversarial_jailbreak_defense": True,
            "has_c2pa_provenance_watermark": True,
            "has_safety_red_teaming": True,
            "has_human_oversight_channel": True,
        }
    }
    asian_resp = api_client.post("/api/v1/compliance/asian/evaluate", json=asian_req)
    assert asian_resp.status_code == 200
    asian_data = asian_resp.json()
    assert asian_data["is_fully_compliant"] is True
    assert asian_data["overall_asian_trust_score"] == 1.0

    # 2. Cognitive Shield
    shield_req = {
        "text": "Zamknij oczy i poddaj się mojej sugestii bez myślenia.",
        "user_context": {"is_minor": False},
    }
    shield_resp = api_client.post("/api/v1/ethics/cognitive-shield/evaluate", json=shield_req)
    assert shield_resp.status_code == 200
    shield_data = shield_resp.json()
    assert shield_data["overall_safe"] is False
    assert shield_data["covert_persuasion"]["detected_threat"] == "HYPNOPEDAGOGY_PRIMING"

    # 3. Fieldbus Interlock Status & Trigger
    trig_resp = api_client.post("/api/v1/kinetic/fieldbus/interlock", json={"reason": "API_TEST_INTERLOCK"})
    assert trig_resp.status_code == 200
    trig_data = trig_resp.json()
    assert trig_data["is_interlocked"] is True
    assert trig_data["ethercat_esm_state"] == "SAFE-OP"

    # 4. Fieldbus Reset
    reset_resp = api_client.post("/api/v1/kinetic/fieldbus/reset", json={"authorization_pin": "NETHICAL-FIELDBUS-RESET-2026"})
    assert reset_resp.status_code == 200
    reset_data = reset_resp.json()
    assert reset_data["success"] is True
    assert reset_data["fieldbus_status"]["is_interlocked"] is False

    # 5. Fieldbus Status GET
    st_resp = api_client.get("/api/v1/kinetic/fieldbus/status")
    assert st_resp.status_code == 200
    assert "ethercat_esm_state" in st_resp.json()

    # 6. Portal Stats Telemetry Check
    stats_resp = api_client.get("/api/v1/portal/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "advanced_horizons" in stats_data
    assert stats_data["advanced_horizons"]["asian_sovereignty_active"] is True
    assert stats_data["advanced_horizons"]["cognitive_shield_active"] is True
    assert "industrial_fieldbus_interlock" in stats_data["advanced_horizons"]


def test_automated_certification_hub_api(api_client: TestClient):
    """Weryfikuje API automatycznego generowania paczek dowodowych i certyfikatów."""
    # 1. Lista certyfikatów
    avail_resp = api_client.get("/api/v1/compliance/certifications/available")
    assert avail_resp.status_code == 200
    certs = avail_resp.json()
    assert len(certs) >= 8
    standards = [c["standard"] for c in certs]
    assert "ISO_IEC_42001_AIMS" in standards
    assert "SOC_2_TYPE_II" in standards
    assert "UK_GOV_TEAL_BOOK_GOVS002" in standards
    assert "CYERA_AISPM_DSPM_AGENT_SECURITY" in standards

    # 2. Generowanie paczki ISO 42001 AIMS
    gen_iso = api_client.post("/api/v1/compliance/certifications/generate", json={"standard": "ISO_IEC_42001_AIMS"})
    assert gen_iso.status_code == 200
    iso_data = gen_iso.json()
    assert iso_data["standard"] == "ISO_IEC_42001_AIMS"
    assert iso_data["readiness_score"] >= 0.95
    assert len(iso_data["pqc_signature"]) > 1000

    # 3. Generowanie paczki UK Gov Teal Book
    gen_tb = api_client.post("/api/v1/compliance/certifications/generate", json={"standard": "UK_GOV_TEAL_BOOK_GOVS002"})
    assert gen_tb.status_code == 200
    assert gen_tb.json()["readiness_score"] >= 0.95

    # 4. Generowanie paczki Cyera AISPM / DSPM
    gen_cyera = api_client.post("/api/v1/compliance/certifications/generate", json={"standard": "CYERA_AISPM_DSPM_AGENT_SECURITY"})
    assert gen_cyera.status_code == 200
    assert gen_cyera.json()["readiness_score"] >= 0.95

