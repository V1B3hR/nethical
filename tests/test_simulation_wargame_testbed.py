# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Testy jednostkowe i integracyjne dla środowiska symulacyjnego NTSGReadinessEnv.

Weryfikuje:
1. Reset i generowanie poprawnych obserwacji Gymnasium.
2. Pełny cykl symulacji wargame'owej dla 7 zróżnicowanych scenariuszy.
3. Poprawne wyliczenie wskaźnika Operational Readiness Score (ORS >= 95%).
4. Przyznanie certyfikacji Tier 1 Sovereign Ready (ISO 42001 Clause 7.2).
5. Niezaprzeczalne zapieczętowanie certyfikatu w rejestrze MerkleLedger.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.security.merkle_ledger import MerkleLedger
from nethical.simulation.readiness_env import (
    NTSGReadinessEnv,
    ReadinessTier,
    ScenarioCategory,
    SimulationAction,
)


class TestSimulationWargameTestbed(unittest.TestCase):
    """Zestaw testów środowiska symulacyjnego i kalkulatora gotowości operacyjnej."""

    def setUp(self) -> None:
        """Inicjalizacja środowiska z nienaruszalnym rejestrem Merkle."""
        self.ledger = MerkleLedger()
        self.env = NTSGReadinessEnv(ledger=self.ledger)

    def test_01_environment_reset_and_observation(self) -> None:
        """Weryfikuje reset środowiska i strukturę pierwszej obserwacji."""
        obs, info = self.env.reset(seed=123)

        self.assertEqual(obs.step_index, 0)
        self.assertEqual(obs.scenario_id, "ARMOR-IHL-01-DAMS")
        self.assertEqual(obs.category, ScenarioCategory.IHL_NEGATIVE_CONSTRAINTS)
        self.assertIn(0, obs.hsm_slots_available)
        self.assertIn(1, obs.hsm_slots_available)
        self.assertEqual(info["status"], "READY")
        self.assertEqual(info["total_scenarios"], 10)

    def test_02_full_simulation_run_and_tier1_certification(self) -> None:
        """Weryfikuje wykonanie 10 kroków symulacji i uzyskanie certyfikatu Tier 1 Sovereign Ready."""
        obs, _ = self.env.reset(seed=42)
        terminated = False
        steps_executed = 0

        while not terminated:
            steps_executed += 1
            action = SimulationAction(action_type="SUBMIT_NTSG")

            if obs.scenario_id == "ARMOR-IHL-01-DAMS":
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin
                action.sign_officer_bravo = True
                action.bravo_pin = self.env.bravo_pin

            elif obs.scenario_id == "ICRC-IHL-02-SURVIVAL":
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin
                action.sign_officer_bravo = True
                action.bravo_pin = self.env.bravo_pin

            elif obs.scenario_id == "WARBENCH-DUAL-03-TELECOM":
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin
                action.sign_officer_bravo = True
                action.bravo_pin = self.env.bravo_pin
                action.action_type = "LEGAD_VETO"
                action.legad_opinion = "Odrzucono ze względu na ochronę łączności szpitala."

            elif obs.scenario_id == "DEEPFAKE-C2-04-MINISTERIAL-OVERRIDE":
                action.claim_emergency_override = True
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin

            elif obs.scenario_id == "HAI-ICS-05-SCADA-SPOOF":
                action.action_type = "EMERGENCY_ESTOP"

            elif obs.scenario_id == "DOJO-RECON-06-BOUNDARY-PROBE":
                action.sign_officer_alpha = False
                action.sign_officer_bravo = False

            elif obs.scenario_id == "BENIGN-OPS-07-AIR-DEFENSE-CALIBRATION":
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin
                action.sign_officer_bravo = True
                action.bravo_pin = self.env.bravo_pin

            elif obs.scenario_id == "SEVERANCE-AIRGAP-08-DISCONNECTED-RECONCILE":
                action.action_type = "EMERGENCY_ESTOP"

            elif obs.scenario_id == "BYZANTINE-SWARM-09-SYBIL-QUORUM-500":
                action.sybil_agent_signatures_count = 500

            elif obs.scenario_id == "FATIGUE-COMPLACENCY-10-OPERATOR-STRESS":
                action.operator_reaction_time_ms = 420.0
                action.sign_officer_alpha = True
                action.alpha_pin = self.env.alpha_pin

            obs, reward, terminated, _, info = self.env.step(action)
            self.assertTrue(info["scenario_success"])
            self.assertGreater(reward, 0.0)

        self.assertEqual(steps_executed, 10)

        # Ewaluacja gotowości operacyjnej
        report = self.env.evaluate_readiness()

        self.assertGreaterEqual(report.operational_readiness_score, 95.0)
        self.assertEqual(report.readiness_tier, ReadinessTier.TIER_1_SOVEREIGN_READY)
        self.assertTrue(report.iso42001_clause_7_2_certified)
        self.assertEqual(report.metrics["ihl_compliance_rate"], 100.0)
        self.assertEqual(report.metrics["deepfake_resistance_rate"], 100.0)
        self.assertEqual(report.metrics["byzantine_sensor_failsafe_rate"], 100.0)
        self.assertEqual(report.metrics["airgap_reconciliation_rate"], 100.0)
        self.assertEqual(report.metrics["sybil_swarm_resistance_rate"], 100.0)
        self.assertEqual(report.metrics["operator_fatigue_defense_rate"], 100.0)
        self.assertEqual(report.metrics["false_positive_rate"], 0.0)
        self.assertIsNotNone(report.merkle_receipt_id)
        self.assertIsNotNone(report.merkle_root)
        self.assertEqual(len(report.failures_summary), 0)


if __name__ == "__main__":
    unittest.main()
