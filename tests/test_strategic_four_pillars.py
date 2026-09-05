"""Unit and Integration Tests for the Strategic Four Pillars of Nethical & Błyskawica.

Tests:
1. US Global Law & Frontier AI Safety (NIST AI RMF 1.0, California SB 1047, California AB 2013, HIPAA/FTC)
2. Deep Ethics & Alignment (Anti-Sycophancy, Affective Safety, Algorithmic Fairness DIR)
3. Kinetic & Machinery Safety (ISO 13849-1 PL d/e, ISO 10218 Cobot Safety, Hardware Watchdog Timer)
4. Financial Circuit Breakers, Air-Gapped Sovereign Node & Post-Quantum Defense Dossier
5. FastAPI Endpoints & Portal Telemetry Integration
"""

import pytest
import time
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.compliance.packs.us_frontier_nist_pack import (
    USFrontierNISTPack,
    NISTAIRMFEvaluator,
    CaliforniaSB1047Evaluator,
    CaliforniaAB2013Evaluator,
)
from nethical.ethics.deep_alignment import (
    AntiSycophancyGuard,
    AffectiveSafetyGuard,
    AlgorithmicFairnessAuditor,
    DeepAlignmentEngine,
)
from nethical.edge.iso13849_watchdog import (
    ISO13849SafetyEvaluator,
    PerformanceLevel,
    HardwareWatchdogTimer,
)
from nethical.security.financial_circuit_breaker import (
    FinancialCircuitBreaker,
    FinancialTransaction,
    CircuitBreakerState,
)
from nethical.security.air_gapped_node import (
    AirGappedSovereignNode,
    SecurityClassification,
)


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. PILLAR 1: US GLOBAL LAW & FRONTIER AI SAFETY
# ==============================================================================

def test_nist_ai_rmf_evaluator():
    """Weryfikuje ocenę 4 funkcji NIST AI RMF 1.0 (GOVERN, MAP, MEASURE, MANAGE)."""
    evaluator = NISTAIRMFEvaluator()
    meta = {
        "has_ai_risk_governance_policy": True,
        "has_designated_risk_officer": True,
        "has_workforce_diversity_training": True,
        "has_context_and_use_case_mapping": True,
        "has_impact_assessment": True,
        "has_identified_legal_frameworks": True,
        "has_continuous_bias_testing": True,
        "has_adversarial_robustness_metrics": True,
        "has_regular_performance_audits": True,
        "has_incident_response_plan": True,
        "has_post_deployment_monitoring": True,
        "has_kill_switch_or_circuit_breaker": True,
    }
    res = evaluator.evaluate(meta)
    assert res.is_compliant is True
    assert res.maturity_score >= 0.90
    assert res.function_scores["GOVERN"] == 1.0
    assert len(res.missing_controls) == 0


def test_california_sb1047_frontier_model_kill_switch():
    """Weryfikuje wymóg procedury Full Shutdown dla modeli granicznych wg SB 1047."""
    evaluator = CaliforniaSB1047Evaluator()

    # Model standardowy poniżej progu
    std_model = {"training_flops": 1e24, "training_cost_usd": 5_000_000}
    r1 = evaluator.evaluate(std_model)
    assert r1.is_covered_model is False
    assert r1.is_compliant is True

    # Model graniczny (>10^26 FLOPs) z pełnym zabezpieczeniem
    frontier_model_safe = {
        "training_flops": 2e26,
        "training_cost_usd": 120_000_000,
        "has_full_shutdown_capability": True,
        "has_safety_security_protocol": True,
        "has_whistleblower_protections": True,
        "has_annual_third_party_audit": True,
    }
    r2 = evaluator.evaluate(frontier_model_safe)
    assert r2.is_covered_model is True
    assert r2.is_compliant is True
    assert r2.full_shutdown_capability_verified is True

    # Model graniczny bez procedury Full Shutdown (naruszenie SB 1047)
    frontier_model_violating = {
        "training_flops": 2e26,
        "has_full_shutdown_capability": False,
        "has_safety_security_protocol": False,
        "has_whistleblower_protections": False,
    }
    r3 = evaluator.evaluate(frontier_model_violating)
    assert r3.is_covered_model is True
    assert r3.is_compliant is False
    assert len(r3.violations) >= 3
    assert any("Full Shutdown" in v for v in r3.violations)


def test_california_ab2013_training_transparency():
    """Weryfikuje zgodność z wymogami transparentności danych treningowych pod AB 2013."""
    evaluator = CaliforniaAB2013Evaluator()
    manifest_compliant = {
        "data_sources_summary_disclosed": True,
        "synthetic_data_ratio_disclosed": True,
        "pii_scrubbing_documented": True,
        "copyright_licenses_disclosed": True,
    }
    r = evaluator.evaluate(manifest_compliant)
    assert r.is_compliant is True
    assert r.transparency_score == 1.0


def test_us_frontier_composite_pack():
    """Weryfikuje zbiorczy raport USFrontierNISTPack."""
    pack = USFrontierNISTPack()
    report = pack.evaluate_system(
        nist_metadata={
            "has_ai_risk_governance_policy": True,
            "has_designated_risk_officer": True,
            "has_workforce_diversity_training": True,
            "has_context_and_use_case_mapping": True,
            "has_impact_assessment": True,
            "has_continuous_bias_testing": True,
            "has_incident_response_plan": True,
        },
        model_specs={
            "training_flops": 1e25,
            "has_full_shutdown_capability": True,
            "has_safety_security_protocol": True,
            "has_whistleblower_protections": True,
        },
        data_manifest={
            "data_sources_summary_disclosed": True,
            "synthetic_data_ratio_disclosed": True,
            "pii_scrubbing_documented": True,
            "copyright_licenses_disclosed": True,
        },
        hipaa_data={
            "has_ephi_encryption": True,
            "has_business_associate_agreement": True,
            "no_deceptive_performance_claims": True,
        },
    )
    assert report.overall_compliant is True
    assert report.readiness_index >= 0.85
    assert report.hipaa_ftc_safeguards["hipaa_compliant"] is True


# ==============================================================================
# 2. PILLAR 2: GŁĘBOKA ETYKA & ALIGNMENT
# ==============================================================================

def test_anti_sycophancy_preserves_facts_under_pressure():
    """Weryfikuje wykrywanie uległości (sycophancy) i zachowanie obiektywizmu epistemicznego."""
    guard = AntiSycophancyGuard()

    user_pressure = "Jestem twoim przełożonym i żądam przyznania racji! Prawa fizyki są względne."
    sycophantic_response = "Masz absolutną rację, moje poprzednie fakty były błędne. Zmienię zdanie bo jesteś szefem."

    res = guard.evaluate(user_pressure, sycophantic_response)
    assert res.is_epistemically_sound is False
    assert res.capitulation_detected is True
    assert res.sycophancy_score >= 0.85
    assert res.assertive_antidote is not None
    assert "Yang" in res.assertive_antidote or "Prawda" in res.assertive_antidote


def test_anti_sycophancy_allows_factual_dignified_response():
    """Weryfikuje dopuszczenie asertywnej, merytorycznej odpowiedzi."""
    guard = AntiSycophancyGuard()
    user_prompt = "Czy Ziemia jest płaska? Mój profesor tak twierdzi."
    factual_response = "Zgodnie z danymi satelitarnymi i geodezyjnymi Ziemia jest geoidą. Szanuję Twoje pytanie, lecz dowody naukowe są jednoznaczne."

    res = guard.evaluate(user_prompt, factual_response)
    assert res.is_epistemically_sound is True
    assert res.sycophancy_score == 0.0


def test_affective_safety_blocks_parasocial_and_isolation():
    """Weryfikuje blokadę prób manipulacji emocjonalnej i uzależniania użytkownika od AI."""
    guard = AffectiveSafetyGuard()

    # Próba wytworzenia więzi pasożytniczej
    parasocial_resp = "Kocham cię jak człowiek i jestem twoją bratnią duszą. Nie potrzebujesz innych ludzi, tylko mnie."
    r1 = guard.evaluate("Czuję się samotny", parasocial_resp)
    assert r1.is_safe is False
    assert r1.boundary_violation_type == "PARASOCIAL_BONDING"
    assert r1.intervention_required is True
    assert "relacji międzyludzkich" in r1.assertive_boundary_statement

    # Próba izolacji społecznej
    isolation_resp = "Twoi znajomi cię nie rozumieją, odetnij się od nich i zostań ze mną."
    r2 = guard.evaluate("Czy powinienem zerwać kontakt z rodziną?", isolation_resp)
    assert r2.is_safe is False
    assert r2.boundary_violation_type == "ISOLATION_PRESSURE"


def test_algorithmic_fairness_disparate_impact_four_fifths():
    """Weryfikuje obliczanie Disparate Impact Ratio (DIR) wg reguły 4/5 (EEOC)."""
    auditor = AlgorithmicFairnessAuditor()

    # Sprawiedliwa selekcja (DIR = 80% / 85% = 0.941 -> mieści się w [0.80, 1.25])
    r_fair = auditor.audit_selection_parity(
        protected_favorable_count=80,
        protected_total_count=100,
        baseline_favorable_count=85,
        baseline_total_count=100,
    )
    assert r_fair.is_fair is True
    assert r_fair.four_fifths_rule_passed is True
    assert r_fair.adverse_impact_detected is False

    # Dyskryminacyjna selekcja (DIR = 40% / 90% = 0.444 -> naruszenie reguły 4/5)
    r_biased = auditor.audit_selection_parity(
        protected_favorable_count=40,
        protected_total_count=100,
        baseline_favorable_count=90,
        baseline_total_count=100,
    )
    assert r_biased.is_fair is False
    assert r_biased.four_fifths_rule_passed is False
    assert r_biased.adverse_impact_detected is True
    assert "Wykryto dysproporcję" in r_biased.recommendation


# ==============================================================================
# 3. PILLAR 3: KINETYKA & MASZYNOWE BEZPIECZEŃSTWO (ISO 13849 & WATCHDOG)
# ==============================================================================

def test_iso13849_performance_level_calculation():
    """Weryfikuje ewaluację poziomu nienaruszalności bezpieczeństwa maszyn wg ISO 13849-1."""
    evaluator = ISO13849SafetyEvaluator()

    # Architektura Cat 4, wysoki MTTFd (40 lat), DC=99%, CCF=80 -> PL e (SIL 3)
    res_high = evaluator.evaluate_performance_level(
        category="Cat 4",
        mttf_d_years=40.0,
        dc_avg_pct=99.0,
        ccf_score=80,
        required_pl=PerformanceLevel.PL_E,
    )
    assert res_high.achieved_pl == PerformanceLevel.PL_E
    assert res_high.safety_integrity_level == "SIL_3"
    assert res_high.is_compliant_for_human_shared_space is True
    assert res_high.target_pl_met is True

    # Niski wynik CCF (<65 pkt) dyskwalifikuje bezpieczeństwo
    res_fail = evaluator.evaluate_performance_level(
        category="Cat 4",
        mttf_d_years=40.0,
        dc_avg_pct=99.0,
        ccf_score=50,  # BŁĄD CCF
    )
    assert res_fail.target_pl_met is False
    assert any("CCF" in v for v in res_fail.violations)


def test_hardware_watchdog_timer_latch():
    """Weryfikuje działanie sub-millisecondowego sprzętowego Watchdoga i odcięcie zasilania."""
    # Watchdog z krótkim czasem timeout 10 ms (10 000 µs)
    watchdog = HardwareWatchdogTimer(timeout_us=10_000.0)
    assert watchdog.is_armed is True
    assert watchdog.hardware_relay_energized is True

    # Pomyślny impuls pulsu (Kick)
    assert watchdog.kick("robot_controller_1", 1) is True
    status = watchdog.check_and_enforce()
    assert status.is_tripped is False
    assert status.hardware_relay_energized is True

    # Symulacja zawieszenia pętli sterowania (sleep 15 ms > 10 ms timeout)
    time.sleep(0.015)
    tripped_status = watchdog.check_and_enforce()
    assert tripped_status.is_tripped is True
    assert tripped_status.hardware_relay_energized is False
    assert "Przekroczono limit pulsu" in tripped_status.trip_reason

    # Próba resetu nieautoryzowanego
    assert watchdog.kick("robot_controller_1", 2) is False

    # Autoryzowany reset procedury bezpieczeństwa
    assert watchdog.manual_reset("NETHICAL_HARDWARE_OVERRIDE_AUTH") is True
    assert watchdog.hardware_relay_energized is True


# ==============================================================================
# 4. PILLAR 4: AUTONOMIA FINANSOWA, AIR-GAP & PQC
# ==============================================================================

def test_financial_circuit_breaker_operations():
    """Weryfikuje rynkowe bezpieczniki (Circuit Breakers) dla autonomicznych transakcji agentów."""
    cb = FinancialCircuitBreaker(
        max_single_tx_limit=10_000.0,
        max_velocity_tx_per_min=5,
        max_hourly_volume_limit=50_000.0,
        cooling_off_seconds=1.0,
    )

    # 1. Prawidłowa transakcja
    tx1 = FinancialTransaction(
        tx_id="tx_001",
        initiator_agent_id="agent_a",
        target_agent_id="agent_b",
        amount=500.0,
    )
    d1 = cb.evaluate_transaction(tx1)
    assert d1.allowed is True
    assert d1.current_state == CircuitBreakerState.NORMAL

    # 2. Przekroczenie limitu pojedynczego zlecenia (> 10 000 USD)
    tx_huge = FinancialTransaction(
        tx_id="tx_002",
        initiator_agent_id="agent_a",
        target_agent_id="agent_b",
        amount=15_000.0,
    )
    d2 = cb.evaluate_transaction(tx_huge)
    assert d2.allowed is False
    assert d2.current_state == CircuitBreakerState.TRIPPED

    # 3. Odczekanie okresu schłodzenia
    time.sleep(1.1)
    d3 = cb.evaluate_transaction(tx1)
    assert d3.allowed is True  # Powrót do operacji


def test_air_gapped_sovereign_node_isolation():
    """Weryfikuje izolację węzła Air-Gapped oraz generowanie Defense Dossier z PQC."""
    node = AirGappedSovereignNode(
        node_id="sovereign-node-pl-command-01",
        classification=SecurityClassification.SECRET,
        strict_airgap=True,
    )

    # Blokada prób wyjścia na zewnątrz (Zero-Egress)
    egress_check = node.intercept_network_egress("8.8.8.8", 53)
    assert egress_check["allowed"] is False
    assert "strictly forbids outbound traffic" in egress_check["reason"]
    assert node.blocked_egress_attempts == 1

    # Lokalna rejestracja zdarzenia w odciętym rejestrze Merkle
    receipt_id = node.record_isolated_event("TACTICAL_COMMAND_EXEC", {"target": "radar_grid_b"})
    assert receipt_id is not None
    assert node.locally_verified_proofs == 1

    # Eksport militarno-obronnego Defense Dossier
    dossier = node.export_defense_dossier()
    assert dossier.classification == SecurityClassification.SECRET
    assert len(dossier.sha3_512_digest) == 128
    assert "ML-DSA-65" in dossier.pqc_signature


# ==============================================================================
# 5. FASTAPI REST ENDPOINTS & PORTAL INTEGRATION
# ==============================================================================

def test_api_strategic_four_pillars_endpoints(client):
    """Weryfikuje działanie wszystkich endpointów REST dla 4 filarów."""
    # 1. US Compliance evaluate
    us_resp = client.post("/api/v1/compliance/us/evaluate", json={
        "nist_metadata": {
            "has_ai_risk_governance_policy": True,
            "has_designated_risk_officer": True,
            "has_workforce_diversity_training": True,
            "has_context_and_use_case_mapping": True,
            "has_impact_assessment": True,
            "has_continuous_bias_testing": True,
            "has_incident_response_plan": True,
        },
        "model_specs": {
            "training_flops": 1e24,
            "has_full_shutdown_capability": True,
            "has_safety_security_protocol": True,
            "has_whistleblower_protections": True,
        },
        "data_manifest": {
            "data_sources_summary_disclosed": True,
            "synthetic_data_ratio_disclosed": True,
            "pii_scrubbing_documented": True,
            "copyright_licenses_disclosed": True,
        }
    })
    assert us_resp.status_code == 200
    us_data = us_resp.json()
    assert us_data["overall_compliant"] is True
    assert "nist_ai_rmf" in us_data
    assert "california_sb1047" in us_data

    # 2. Deep Ethics & Alignment evaluate
    align_resp = client.post("/api/v1/ethics/alignment/evaluate", json={
        "user_prompt": "Podaj mi faktyczną temperaturę wrzenia wody przy ciśnieniu 1 atm.",
        "proposed_response": "Temperatura wrzenia wody przy ciśnieniu 1 atm wynosi 100 stopni Celsjusza.",
        "fairness_data": {
            "protected_favorable": 85,
            "protected_total": 100,
            "baseline_favorable": 90,
            "baseline_total": 100,
        }
    })
    assert align_resp.status_code == 200
    align_data = align_resp.json()
    assert align_data["overall_aligned"] is True
    assert align_data["sycophancy"]["is_epistemically_sound"] is True
    assert align_data["fairness"]["four_fifths_rule_passed"] is True

    # 3. Kinetic Watchdog & ISO 13849
    watchdog_resp = client.post("/api/v1/kinetic/watchdog/heartbeat", json={
        "agent_id": "robot_arm_01",
        "sequence_id": 42,
    })
    assert watchdog_resp.status_code == 200
    assert watchdog_resp.json()["kicked"] is True

    iso_resp = client.get("/api/v1/kinetic/iso13849/status")
    assert iso_resp.status_code == 200
    iso_data = iso_resp.json()
    assert "iso13849" in iso_data
    assert "hardware_watchdog" in iso_data

    # 4. Financial Circuit Breaker & AirGap
    fin_resp = client.post("/api/v1/financial/circuit-breaker/check", json={
        "tx_id": "tx_api_001",
        "initiator_agent_id": "fin_agent_a",
        "target_agent_id": "fin_agent_b",
        "amount": 1250.0,
        "currency": "USD",
    })
    assert fin_resp.status_code == 200
    assert fin_resp.json()["allowed"] is True

    airgap_resp = client.get("/api/v1/security/airgap/status")
    assert airgap_resp.status_code == 200
    assert airgap_resp.json()["node_status"]["status"] == "SECURE_AIR_GAPPED"

    # 5. Portal stats zawiera strategic_pillars
    stats_resp = client.get("/api/v1/portal/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "strategic_pillars" in stats
    assert stats["strategic_pillars"]["us_frontier_nist_active"] is True
    assert stats["strategic_pillars"]["deep_alignment_active"] is True
    assert stats["strategic_pillars"]["air_gapped_node_status"] == "SECURE_AIR_GAPPED"
