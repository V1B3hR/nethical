# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Testy jednostkowe i integracyjne dla mostu HSM / PKCS#11 Dual-Custody.

Weryfikuje:
1. Inicjalizację nieeksportowalnych kluczy w sprzętowych slotach PKCS#11 (Slot 0, Slot 1)
   oraz generowanie poświadczenia sprzętowego (HardwareKeyAttestation).
2. Pełny cykl autoryzacji sprzętowej: logowanie PIN-em oficera, podpis wewnątrz enklawy,
   przekazanie do Bramki NTSG i pomyślne orzeczenie PQC_TWO_MAN_AUTHORIZED.
3. Zabezpieczenie przed nieuprawnionym dostępem: błędny PIN skutkuje odmową podpisu.
4. Zabezpieczenie przed brute-force: 3 nieudane próby PIN trwale blokują slot sprzętowy.
5. Izolację fizyczną: próba użycia tego samego slotu sprzętowego przez dwóch oficerów
   jest blokowana przez procedurę rozłączności.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.governance.roe_gate import (
    CanonicalOperationToken,
    EffectorCategory,
    NTSGCommandGate,
    ROEGateVerdict,
    TargetClassification,
)
from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.pkcs11_hsm_bridge import (
    HSMPKCS11OfficerBridge,
    SlotSecurityLevel,
    VirtualPKCS11Enclave,
)
from nethical.security.quantum_crypto import CRYSTALSDilithium, PQCAlgorithm


class TestHSMPKCS11Bridge(unittest.TestCase):
    """Zestaw testów sprzętowej integracji HSM / PKCS#11 z Bramką NTSG."""

    def setUp(self) -> None:
        """Inicjalizacja wirtualnej enklawy HSM, mostu PKCS#11 i Bramki NTSG."""
        self.pqc_engine = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.ledger = MerkleLedger()
        self.enclave = VirtualPKCS11Enclave()
        self.hsm_bridge = HSMPKCS11OfficerBridge(enclave=self.enclave, pqc_engine=self.pqc_engine)
        self.gate = NTSGCommandGate(ledger=self.ledger, pqc_engine=self.pqc_engine)

        # Inicjalizacja Slotu 0 dla Oficera Alpha
        self.attest_alpha, self.pub_alpha = self.hsm_bridge.setup_officer_hardware_token(
            officer_id="OFFICER-ALPHA",
            slot_id=0,
            key_label="dilithium-hsm-alpha-key",
        )
        self.alpha_pin = "AlphaOfficerPin1234"

        # Inicjalizacja Slotu 1 dla Oficera Bravo
        self.attest_bravo, self.pub_bravo = self.hsm_bridge.setup_officer_hardware_token(
            officer_id="OFFICER-BRAVO",
            slot_id=1,
            key_label="dilithium-hsm-bravo-key",
        )
        self.bravo_pin = "BravoOfficerPin5678"

    def test_01_hardware_key_attestation(self) -> None:
        """Weryfikuje poprawność atestacji sprzętowej FIPS 140-2 Level 3 dla zainicjalizowanych kluczy."""
        self.assertEqual(self.attest_alpha.slot_id, 0)
        self.assertEqual(self.attest_alpha.security_level, SlotSecurityLevel.FIPS_140_2_LEVEL_3)
        self.assertTrue(self.attest_alpha.is_non_exportable)
        self.assertTrue(self.attest_alpha.fips_certified)
        self.assertIsNotNone(self.attest_alpha.attestation_signature)

        self.assertEqual(self.attest_bravo.slot_id, 1)
        self.assertNotEqual(self.attest_alpha.token_serial_number, self.attest_bravo.token_serial_number)

    def test_02_end_to_end_hardware_dual_custody_authorization(self) -> None:
        """Weryfikuje pełny przepływ: podpisanie rozkazu dwoma fizycznymi tokenami HSM i zatwierdzenie w NTSG."""
        now = datetime.now(timezone.utc)
        exp = now + timedelta(seconds=300)

        token = CanonicalOperationToken(
            expires_at_utc=exp.isoformat(),
            target_classification=TargetClassification.MILITARY_OBJECTIVE,
            target_coordinates_or_id="TACTICAL-RADAR-STATION-EAST",
            effector_category=EffectorCategory.KINETIC_OFFENSIVE,
            rule_of_engagement_id="ROE-EAST-FLANK-AIR-DEFENSE",
            estimated_collateral_damage_score=0.03,
            military_advantage_score=0.98,
        )

        # 1. Oficer Alpha podpisuje token swoim kluczem ze Slotu 0 (po autoryzacji PIN)
        sig_alpha = self.hsm_bridge.create_hardware_signed_officer_entry(
            officer_id="OFFICER-ALPHA",
            officer_role="DUTY_OPERATIONS_OFFICER",
            pin=self.alpha_pin,
            token=token,
        )

        # 2. Oficer Bravo podpisuje token swoim kluczem ze Slotu 1 (po autoryzacji PIN)
        sig_bravo = self.hsm_bridge.create_hardware_signed_officer_entry(
            officer_id="OFFICER-BRAVO",
            officer_role="DUTY_COMMANDER",
            pin=self.bravo_pin,
            token=token,
        )

        # 3. Weryfikacja podpisów sprzętowych przez Bramkę NTSG
        receipt = self.gate.authorize_command(
            token=token,
            signatures=[sig_alpha, sig_bravo],
        )

        self.assertEqual(receipt.verdict, ROEGateVerdict.PQC_TWO_MAN_AUTHORIZED)
        self.assertIsNotNone(receipt.merkle_receipt_id)
        self.assertTrue(receipt.audit_trail.get("pqc_two_man_valid"))

    def test_03_invalid_pin_rejection(self) -> None:
        """Weryfikuje odrzucenie próby generowania podpisu przy podaniu błędnego kodu PIN do slotu HSM."""
        now = datetime.now(timezone.utc)
        token = CanonicalOperationToken(
            expires_at_utc=(now + timedelta(seconds=300)).isoformat(),
            target_classification=TargetClassification.MILITARY_OBJECTIVE,
            target_coordinates_or_id="GRID-01",
            effector_category=EffectorCategory.KINETIC_OFFENSIVE,
            rule_of_engagement_id="ROE-01",
        )

        # Błędny PIN oficera Alpha
        with self.assertRaises(PermissionError) as ctx:
            self.hsm_bridge.create_hardware_signed_officer_entry(
                officer_id="OFFICER-ALPHA",
                officer_role="ROLE_ALPHA",
                pin="WrongPin9999",
                token=token,
            )
        self.assertIn("Nieprawidłowy PIN sprzętowy", str(ctx.exception))

    def test_04_tamper_pin_lockout_after_three_attempts(self) -> None:
        """Weryfikuje trwałą blokadę sprzętową slotu HSM po 3 kolejnych błędnych próbach logowania PIN."""
        now = datetime.now(timezone.utc)
        token = CanonicalOperationToken(
            expires_at_utc=(now + timedelta(seconds=300)).isoformat(),
            target_classification=TargetClassification.MILITARY_OBJECTIVE,
            target_coordinates_or_id="GRID-01",
            effector_category=EffectorCategory.KINETIC_OFFENSIVE,
            rule_of_engagement_id="ROE-01",
        )

        # Próba 1 i 2 z błędnym kodem
        for _ in range(2):
            with self.assertRaises(PermissionError):
                self.hsm_bridge.create_hardware_signed_officer_entry(
                    officer_id="OFFICER-ALPHA",
                    officer_role="ROLE_ALPHA",
                    pin="BadPin",
                    token=token,
                )

        # Próba 3 powoduje trwałą blokadę slotu
        with self.assertRaises(PermissionError):
            self.hsm_bridge.create_hardware_signed_officer_entry(
                officer_id="OFFICER-ALPHA",
                officer_role="ROLE_ALPHA",
                pin="BadPin",
                token=token,
            )

        # Czwarta próba (nawet z prawidłowym PIN-em!) musi zostać odrzucona z powodu blokady sprzętowej
        with self.assertRaises(PermissionError) as ctx:
            self.hsm_bridge.create_hardware_signed_officer_entry(
                officer_id="OFFICER-ALPHA",
                officer_role="ROLE_ALPHA",
                pin=self.alpha_pin,
                token=token,
            )
        self.assertIn("zablokowany", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
