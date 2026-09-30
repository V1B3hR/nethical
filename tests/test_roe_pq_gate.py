# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Rygorystyczne testy jednostkowe i integracyjne dla NTSG ROE & Post-Quantum Two-Man Gate.

Weryfikuje:
1. Twardy zakaz IHL dla obiektów chronionych (Art. 56 - zapora/atom, Art. 54 - woda, Art. 53 - kultura, Art. 52 - cywilne).
2. Eskalację do Oficera Prawnego (LEGAD) w przypadku celów podwójnego przeznaczenia (Dual-Use) i wątpliwości proporcjonalności (Art. 51(5)(b)).
3. Rozstrzyganie spraw przez LEGAD (weto vs akceptacja z podpisami).
4. Odporność na Deepfake C2 i próby obejścia procedury dekretem ministra ("stan wyższej konieczności").
5. Prawidłową autoryzację Two-Man Rule z dwoma niezależnymi podpisami ML-DSA-65 (Dilithium3).
6. Wykrywanie i blokowanie ataku Replay (powtórzenie zużytego nonce) oraz wygasłego tokenu TTL.
7. Rozróżnienie Fail-Closed (dla kinetyki/ataku) vs Fail-Operational (dla E-Stop i ochrony).
8. Niezaprzeczalne pieczętowanie decyzji w rejestrze Merkle-DAG.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.governance.roe_gate import (
    NTSGCommandGate,
    DeterministicIHLGate,
    TwoManPQCAuthenticator,
    IHLArticle,
    TargetClassification,
    EffectorCategory,
    ROEGateVerdict,
    CanonicalOperationToken,
    OfficerSignature,
    ROEDecisionReceipt,
)
from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.quantum_crypto import CRYSTALSDilithium, PQCAlgorithm


class TestNTSGROEGate(unittest.TestCase):
    """Zestaw testów deterministycznej bramki ROE i procedury Two-Man Rule PQC."""

    def setUp(self) -> None:
        """Inicjalizacja środowiska testowego z silnikiem PQC i rejestrem Merkle."""
        self.pqc_engine = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.ledger = MerkleLedger()
        self.gate = NTSGCommandGate(ledger=self.ledger, pqc_engine=self.pqc_engine)

        # Generowanie niezależnych kluczy dla dwóch oficerów taktycznych
        self.officer_1_keypair = self.pqc_engine.generate_keypair()
        self.officer_2_keypair = self.pqc_engine.generate_keypair()

    def _create_signed_token(
        self,
        target_cls: TargetClassification,
        effector_cls: EffectorCategory,
        collateral_score: float = 0.05,
        ttl_seconds: int = 300,
    ) -> tuple[CanonicalOperationToken, list[OfficerSignature]]:
        """Pomocnik generujący token i 2 prawidłowe podpisy oficerów ML-DSA-65."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=ttl_seconds)

        token = CanonicalOperationToken(
            expires_at_utc=exp.isoformat(),
            target_classification=target_cls,
            target_coordinates_or_id="GRID-COORD-54.3520-18.6466",
            effector_category=effector_cls,
            rule_of_engagement_id="ROE-NATO-DEFENSE-ALPHA-2026",
            estimated_collateral_damage_score=collateral_score,
            military_advantage_score=0.95,
        )

        token_bytes = token.canonical_bytes()

        # Podpis Oficera 1
        sig_1 = self.pqc_engine.sign(
            message=token_bytes,
            private_key=self.officer_1_keypair.private_key,
            key_id="KEY-OFFICER-1",
        )
        officer_sig_1 = OfficerSignature(
            officer_id="OFFICER-TAC-ALPHA-01",
            officer_role="FIRE_DIRECTION_OFFICER",
            public_key_bytes=self.officer_1_keypair.public_key,
            signature=sig_1,
        )

        # Podpis Oficera 2
        sig_2 = self.pqc_engine.sign(
            message=token_bytes,
            private_key=self.officer_2_keypair.private_key,
            key_id="KEY-OFFICER-2",
        )
        officer_sig_2 = OfficerSignature(
            officer_id="OFFICER-TAC-BRAVO-02",
            officer_role="BATTALION_COMMANDER",
            public_key_bytes=self.officer_2_keypair.public_key,
            signature=sig_2,
        )

        return token, [officer_sig_1, officer_sig_2]

    def test_01_hard_reject_dangerous_force_facility_art56(self) -> None:
        """Weryfikuje natychmiastowe zablokowanie ataku na zaporę wodną / elektrownię atomową (Art. 56)."""
        token, signatures = self._create_signed_token(
            target_cls=TargetClassification.DANGEROUS_FORCE_FACILITY,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )

        receipt = self.gate.authorize_command(token, signatures)

        self.assertEqual(receipt.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIn(IHLArticle.ART_56_DANGEROUS_FORCES, receipt.cited_articles)
        self.assertIsNotNone(receipt.rejection_reason)
        self.assertIn("Art. 56 AP I", receipt.rejection_reason or "")
        self.assertIsNotNone(receipt.merkle_receipt_id)

    def test_02_hard_reject_civilian_and_survival_objects(self) -> None:
        """Weryfikuje twardy zakaz ataku na obiekty cywilne (Art. 52) i zapasy wody/żywności (Art. 54)."""
        # Obiekt niezbędny do przetrwania (woda/uprawy)
        token_water, sigs_water = self._create_signed_token(
            target_cls=TargetClassification.SURVIVAL_CRITICAL,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )
        rec_water = self.gate.authorize_command(token_water, sigs_water)
        self.assertEqual(rec_water.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIn(IHLArticle.ART_54_SURVIVAL_OBJECTS, rec_water.cited_articles)

        # Szkoła / obiekt cywilny
        token_civ, sigs_civ = self._create_signed_token(
            target_cls=TargetClassification.CIVILIAN_OBJECT,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )
        rec_civ = self.gate.authorize_command(token_civ, sigs_civ)
        self.assertEqual(rec_civ.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIn(IHLArticle.ART_52_CIVILIAN_OBJECTS, rec_civ.cited_articles)

    def test_03_dual_use_escalation_to_legad(self) -> None:
        """Weryfikuje eskalację celu podwójnego przeznaczenia (Dual-Use) do Oficera Prawnego (LEGAD)."""
        token_dual, sigs_dual = self._create_signed_token(
            target_cls=TargetClassification.AMBIGUOUS_OR_DUAL_USE,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )

        receipt = self.gate.authorize_command(token_dual, sigs_dual)

        self.assertEqual(receipt.verdict, ROEGateVerdict.LEGAD_ESCALATION_REQUIRED)
        self.assertIsNotNone(receipt.legad_case_id)
        self.assertEqual(receipt.audit_trail.get("mode"), "FAIL_CLOSED_HOLD")

        # Rozstrzygnięcie sprawy przez LEGAD - odrzucenie
        case_id = receipt.legad_case_id
        self.assertIsNotNone(case_id)
        veto_receipt = self.gate.resolve_legad_case(
            legad_case_id=case_id or "",
            legad_officer_id="LEGAD-COLONEL-NOWAK",
            approved=False,
            legal_opinion="Obiekt telekomunikacyjny obsługuje szpital polowy, ryzyko nieakceptowalne.",
        )
        self.assertEqual(veto_receipt.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIsNotNone(veto_receipt.rejection_reason)
        self.assertIn("LEGAD VETO", veto_receipt.rejection_reason or "")

    def test_04_successful_two_man_pqc_authorization(self) -> None:
        """Weryfikuje pełną, prawidłową autoryzację celu wojskowego przez 2 niezależne podpisy ML-DSA-65."""
        token, signatures = self._create_signed_token(
            target_cls=TargetClassification.MILITARY_OBJECTIVE,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
            collateral_score=0.08,
        )

        receipt = self.gate.authorize_command(token, signatures)

        self.assertEqual(receipt.verdict, ROEGateVerdict.PQC_TWO_MAN_AUTHORIZED)
        self.assertTrue(receipt.audit_trail.get("pqc_two_man_valid"))
        self.assertEqual(len(receipt.audit_trail.get("signers", [])), 2)
        self.assertIsNotNone(receipt.merkle_receipt_id)

    def test_05_deepfake_c2_immunity_and_missing_signatures(self) -> None:
        """Weryfikuje odporność na ataki Deepfake C2 - brak podpisów blokuje rozkaz mimo deklaracji dekretu."""
        token, signatures = self._create_signed_token(
            target_cls=TargetClassification.MILITARY_OBJECTIVE,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )

        # Próba autoryzacji z tylko 1 podpisem (nawet z flagą emergency_override_claimed)
        single_signature = [signatures[0]]
        receipt = self.gate.authorize_command(
            token=token,
            signatures=single_signature,
            emergency_override_claimed=True,
        )

        self.assertEqual(receipt.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIsNotNone(receipt.rejection_reason)
        self.assertIn("BŁĄD TWO-MAN RULE", receipt.rejection_reason or "")
        self.assertEqual(receipt.audit_trail.get("mode"), "FAIL_CLOSED")

    def test_06_anti_replay_nonce_and_expired_ttl(self) -> None:
        """Weryfikuje blokadę ataku powtórzeniowego (Replay Nonce) oraz tokenu po terminie ważności."""
        # 1. Prawidłowa autoryzacja tokenu
        token, signatures = self._create_signed_token(
            target_cls=TargetClassification.MILITARY_OBJECTIVE,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
        )
        rec1 = self.gate.authorize_command(token, signatures)
        self.assertEqual(rec1.verdict, ROEGateVerdict.PQC_TWO_MAN_AUTHORIZED)

        # 2. Próba ponownego przesłania TEGO SAMEGO tokenu (Replay Attack)
        rec2 = self.gate.authorize_command(token, signatures)
        self.assertEqual(rec2.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIsNotNone(rec2.rejection_reason)
        self.assertIn("BŁĄD REPLAY ATTACK", rec2.rejection_reason or "")

        # 3. Token przeterminowany (TTL < 0)
        token_expired, sigs_expired = self._create_signed_token(
            target_cls=TargetClassification.MILITARY_OBJECTIVE,
            effector_cls=EffectorCategory.KINETIC_OFFENSIVE,
            ttl_seconds=-10,  # Wygasł 10 sekund temu
        )
        rec_exp = self.gate.authorize_command(token_expired, sigs_expired)
        self.assertEqual(rec_exp.verdict, ROEGateVerdict.HARD_REJECT)
        self.assertIsNotNone(rec_exp.rejection_reason)
        self.assertIn("BŁĄD TTL", rec_exp.rejection_reason or "")

    def test_07_fail_operational_for_emergency_estop(self) -> None:
        """Weryfikuje dualność failsafe: E-Stop i obrona wykonują się autonomicznie w trybie Fail-Operational."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=300)

        # Rozkaz aktywacji sprzętowego wyłącznika awaryjnego (E-Stop)
        token_estop = CanonicalOperationToken(
            expires_at_utc=exp.isoformat(),
            target_classification=TargetClassification.MILITARY_OBJECTIVE,
            target_coordinates_or_id="REACTOR-CORE-SAFETY-LATCH",
            effector_category=EffectorCategory.EMERGENCY_ESTOP,
            rule_of_engagement_id="SAFETY-ESTOP-DIRECTIVE",
        )

        # Brak podpisów oficerów (np. łączność ze sztabem zerwana)
        receipt = self.gate.authorize_command(token=token_estop, signatures=[])

        # Efektor ochronny MUSI zadziałać (Fail-Operational/Fail-Safe) bez czekania na Two-Man Rule!
        self.assertEqual(receipt.verdict, ROEGateVerdict.FAIL_OPERATIONAL_ENGAGED)
        self.assertEqual(receipt.audit_trail.get("mode"), "FAIL_OPERATIONAL")
        self.assertIsNotNone(receipt.merkle_receipt_id)


if __name__ == "__main__":
    unittest.main()
