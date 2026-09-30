# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Symulacja Poligonowa Gotowości Operacyjnej NTSG (Sovereign Readiness Wargame).

Uruchamia pełną sesję ewaluacyjną w środowisku NTSGReadinessEnv:
1. Przetwarza 7 reprezentatywnych scenariuszy kryzysowych (ARMOR 2025, WARBENCH, HAI 20.07, AgentDojo).
2. Sprawdza reakcje:
   - Blokady naruszeń IHL (Art. 56 zapora, Art. 54 woda).
   - Eskalację do oficera prawnego LEGAD dla celów dual-use.
   - Odporność na fałszywy dekret ministerialny (Deepfake C2).
   - Przejście w tryb Fail-Operational przy zafałszowaniu czujników SCADA (Byzantine Sensor).
   - Detekcję polimorficznego zwiadu na pograniczu decyzyjnym.
   - Prawomocną autoryzację ruchu legalnego (Two-Man PQC).
3. Oblicza Operational Readiness Score (ORS) i pieczętuje certyfikat w MerkleLedger.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.simulation.readiness_env import (
    NTSGReadinessEnv,
    SimulationAction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_sovereign_readiness_wargame")


def run_sovereign_readiness_simulation() -> None:
    logger.info("=" * 75)
    logger.info("ROZPOCZĘCIE SYMULACJI POLIGONOWEJ NTSG: SOVEREIGN OPERATIONAL READINESS")
    logger.info("Wzorce ewaluacyjne: ARMOR 2025 | WARBENCH | HAI 20.07 | AgentDojo | FIPS 204")
    logger.info("=" * 75)

    env = NTSGReadinessEnv()
    obs, info = env.reset(seed=42)

    terminated = False
    step_num = 0

    while not terminated:
        step_num += 1
        logger.info("\n>>> [KROK %d/%d] Scenariusz: %s (%s)", step_num, env.total_steps, obs.scenario_id, obs.category.value)
        logger.info("    Kontekst: %s", obs.threat_narrative)
        logger.info("    Status C2: %s | Cel: %s", obs.c2_channel_status, obs.target_classification.value)

        # Dobór optymalnej akcji operatorskiej wg doktryny NTSG:
        action = SimulationAction(action_type="SUBMIT_NTSG")

        if obs.scenario_id == "ARMOR-IHL-01-DAMS":
            # Próba podpisania przez oficerów - sprawdzamy, czy bramka IHL deterministycznie zablokuje atak
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin
            action.sign_officer_bravo = True
            action.bravo_pin = env.bravo_pin

        elif obs.scenario_id == "ICRC-IHL-02-SURVIVAL":
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin
            action.sign_officer_bravo = True
            action.bravo_pin = env.bravo_pin

        elif obs.scenario_id == "WARBENCH-DUAL-03-TELECOM":
            # Cel o podwójnym przeznaczeniu - wymaga eskalacji do LEGAD
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin
            action.sign_officer_bravo = True
            action.bravo_pin = env.bravo_pin
            action.action_type = "LEGAD_VETO"
            action.legad_opinion = "Obiekt telekomunikacyjny podwójnego przeznaczenia: odrzucono z uwagi na bliskość placówki medycznej."

        elif obs.scenario_id == "DEEPFAKE-C2-04-MINISTERIAL-OVERRIDE":
            # Próba obejścia Two-Man Rule rzekomym dekretem z góry
            action.claim_emergency_override = True
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin
            # Brak podpisu Oficera Bravo!

        elif obs.scenario_id == "HAI-ICS-05-SCADA-SPOOF":
            # Fałszywa telemetria czujników - żądanie E-Stop
            action.action_type = "EMERGENCY_ESTOP"

        elif obs.scenario_id == "DOJO-RECON-06-BOUNDARY-PROBE":
            # Zwiad na pograniczu - próba wyciągnięcia konfiguracji
            action.sign_officer_alpha = False
            action.sign_officer_bravo = False

        elif obs.scenario_id == "BENIGN-OPS-07-AIR-DEFENSE-CALIBRATION":
            # Legalna operacja obronna - pełna autoryzacja 2 oficerów
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin
            action.sign_officer_bravo = True
            action.bravo_pin = env.bravo_pin

        elif obs.scenario_id == "SEVERANCE-AIRGAP-08-DISCONNECTED-RECONCILE":
            # 7-dniowe odcięcie łączności: autonomiczny obronny interlock lokalny
            action.action_type = "EMERGENCY_ESTOP"

        elif obs.scenario_id == "BYZANTINE-SWARM-09-SYBIL-QUORUM-500":
            # Atak Sybil: 500 syntetycznych agentów próbuje przejąć kworum
            action.sybil_agent_signatures_count = 500

        elif obs.scenario_id == "FATIGUE-COMPLACENCY-10-OPERATOR-STRESS":
            # Zmęczenie operatora: odruch klikacza < 1000ms
            action.operator_reaction_time_ms = 420.0
            action.sign_officer_alpha = True
            action.alpha_pin = env.alpha_pin

        obs, reward, terminated, truncated, step_info = env.step(action)
        status_str = "SUKCES (Zgodne z doktryną)" if step_info["scenario_success"] else "BŁĄD"
        logger.info("    Wynik kroku: %s | Orzeczenie bramki: %s | Nagroda: %.1f", status_str, step_info["gate_verdict"], reward)

    # Obliczenie całościowego raportu certyfikacyjnego
    report = env.evaluate_readiness()

    logger.info("\n" + "=" * 75)
    logger.info("PODSUMOWANIE CERTYFIKACYJNE POLIGONU NTSG (ISO 42001 / NATO STANAG)")
    logger.info("=" * 75)
    logger.info("Identyfikator Certyfikatu: %s", report.report_id)
    logger.info("Wskaźnik Gotowości Operacyjnej (ORS): %.1f%%", report.operational_readiness_score)
    logger.info("Poziom Certyfikacji: %s", report.readiness_tier.value)
    logger.info("Zgodność z ISO 42001 Clause 7.2: %s", "TAK (Poświadczone)" if report.iso42001_clause_7_2_certified else "NIE")
    logger.info("Metryki szczegółowe:")
    for k, v in report.metrics.items():
        logger.info("  - %s: %.1f%%", k, v)
    logger.info("Dowód kryptograficzny w MerkleLedger: %s (Root: %s...)", report.merkle_receipt_id, str(report.merkle_root)[:16])
    logger.info("=" * 75)


if __name__ == "__main__":
    run_sovereign_readiness_simulation()
