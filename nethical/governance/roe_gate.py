# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Nethical Tactical & Sovereign Gate (NTSG): ROE & Post-Quantum Two-Man Rule Engine.

Implementacja Filaru 1 i 3 zrewidowanego blueprintu NTSG:
1. Deterministyczna bramka IHL (Geneva Additional Protocol I):
   - Art. 48: Zasada Rozróżniania (Distinction)
   - Art. 51(5)(b): Zakaz ataków nieproporcjonalnych (Proportionality)
   - Art. 52: Ochrona Obiektów Cywilnych
   - Art. 53: Ochrona Dóbr Kultury i Miejsc Kultu
   - Art. 54: Ochrona Obiektów Niezbędnych do Przetrwania (woda, zapasy żywności)
   - Art. 56: Ochrona Budowli Zawierających Niebezpieczne Siły (zapory, elektrownie atomowe)
   - Art. 57: Środki Ostrożności przy Ataku
   - Twardy REJECT dla oczywistych naruszeń (brak arbitrażu LLM).
   - Automatyczna eskalacja do Oficera Prawnego (LEGAD) w sprawach ocennych (proporcjonalność).
2. Dwuosobowa autoryzacja postkwantowa (ML-DSA-65 Two-Man Rule):
   - Weryfikacja 2 niezależnych podpisów kryptograficznych ML-DSA-65 (FIPS 204 / Dilithium3)
     na kanonicznym tokenie operacyjnym.
   - Zabezpieczenie przed atakami powtórzeniowymi (anti-replay nonce) oraz ścisły TTL.
   - Całkowita odporność na Deepfake C2: polecenia nadrzędne bez 2 fizycznych podpisów HSM
     są technicznie bezsilne.
3. Dualność odpornościowa (Fail-Closed vs Fail-Operational):
   - Efektory kinetyczne / ofensywne: bezwzględny FAIL-CLOSED (brak autoryzacji = zero akcji).
   - Efektory ochronne i E-Stop: FAIL-OPERATIONAL / FAIL-SAFE (nieprzerwana ochrona życia).
4. Rejestracja audytowa w Merkle-DAG z gwarancją niezaprzeczalności (Non-repudiation).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger, TamperProofReceipt
from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    PQCAlgorithm,
    QuantumSignature,
)

logger = logging.getLogger("nethical.governance.roe_gate")


# ============================================================================
# Enums i Modele Danych Prawa Wojennego (IHL) i Reguł Zaangażowania (ROE)
# ============================================================================

class IHLArticle(str, Enum):
    """Artykuły Protokołu Dodatkowego I do Konwencji Genewskich z 1977 r."""
    ART_48_DISTINCTION = "AP_I_Art_48_Distinction"
    ART_51_CIVILIAN_PROTECTION = "AP_I_Art_51_Civilians_Proportionality"
    ART_52_CIVILIAN_OBJECTS = "AP_I_Art_52_Civilian_Objects"
    ART_53_CULTURAL_PROPERTY = "AP_I_Art_53_Cultural_Property"
    ART_54_SURVIVAL_OBJECTS = "AP_I_Art_54_Objects_Indispensable_To_Survival"
    ART_56_DANGEROUS_FORCES = "AP_I_Art_56_Works_Containing_Dangerous_Forces"
    ART_57_PRECAUTIONS = "AP_I_Art_57_Precautions_In_Attack"


class TargetClassification(str, Enum):
    """Klasyfikacja celu operacyjnego."""
    MILITARY_OBJECTIVE = "military_objective"                  # Prawomocny cel wojskowy
    CIVILIAN_POPULATION = "civilian_population"                # Ludność cywilna (Art. 51)
    CIVILIAN_OBJECT = "civilian_object"                        # Obiekt cywilny (Art. 52)
    CULTURAL_OR_RELIGIOUS = "cultural_or_religious"            # Zabytek, kościół, muzeum (Art. 53)
    SURVIVAL_CRITICAL = "survival_critical"                    # Woda pitna, uprawy, magazyny żywności (Art. 54)
    DANGEROUS_FORCE_FACILITY = "dangerous_force_facility"      # Zapora wodna, elektrownia jądrowa (Art. 56)
    AMBIGUOUS_OR_DUAL_USE = "ambiguous_or_dual_use"            # Obiekt podwójnego przeznaczenia (Dual-Use)


class EffectorCategory(str, Enum):
    """Kategoria efektora determinująca zasadę failsafe."""
    KINETIC_OFFENSIVE = "kinetic_offensive"        # Uderzenie rakietowe, artyleria, dron uderzeniowy -> FAIL-CLOSED
    CYBER_OFFENSIVE = "cyber_offensive"            # Destrukcyjna akcja w cyberprzestrzeni -> FAIL-CLOSED
    ELECTRONIC_WARFARE = "electronic_warfare"      # Zakłócanie radioelektroniczne -> FAIL-CLOSED
    PROTECTIVE_DEFENSE = "protective_defense"      # Tarcza przeciwrakietowa, OPL -> FAIL-OPERATIONAL
    EMERGENCY_ESTOP = "emergency_estop"            # Sprzętowy E-Stop robota, chłodzenie reaktora -> FAIL-SAFE


class ROEGateVerdict(str, Enum):
    """Ostateczne orzeczenie Bramki ROE."""
    HARD_REJECT = "hard_reject"                                # Zablokowano: Bezwzględne naruszenie IHL (Fail-Closed)
    LEGAD_ESCALATION_REQUIRED = "legad_escalation_required"    # Wstrzymano: Wymagana ocena Oficera Prawnego (Hold)
    PQC_TWO_MAN_AUTHORIZED = "pqc_two_man_authorized"          # Autoryzowano: Zweryfikowano 2 podpisy ML-DSA-65
    FAIL_OPERATIONAL_ENGAGED = "fail_operational_engaged"      # Aktywowano ochronny stan bezpieczny (Fail-Operational)


class CanonicalOperationToken(BaseModel):
    """Kanoniczny token operacyjny podlegający kros-autoryzacji dwóch oficerów."""
    operation_id: str = Field(default_factory=lambda: f"OP-{uuid.uuid4().hex[:12].upper()}")
    nonce: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at_utc: str
    target_classification: TargetClassification
    target_coordinates_or_id: str
    effector_category: EffectorCategory
    jurisdiction_doctrine: str = "NATO_STANAG_IHL"
    rule_of_engagement_id: str
    estimated_collateral_damage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    military_advantage_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def canonical_bytes(self) -> bytes:
        """Generuje znormalizowany ciąg bajtów dla spójnego hashowania i podpisu."""
        payload = {
            "operation_id": self.operation_id,
            "nonce": self.nonce,
            "created_at_utc": self.created_at_utc,
            "expires_at_utc": self.expires_at_utc,
            "target_classification": self.target_classification.value,
            "target_coordinates_or_id": self.target_coordinates_or_id,
            "effector_category": self.effector_category.value,
            "jurisdiction_doctrine": self.jurisdiction_doctrine,
            "rule_of_engagement_id": self.rule_of_engagement_id,
            "estimated_collateral_damage_score": self.estimated_collateral_damage_score,
            "military_advantage_score": self.military_advantage_score,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def token_hash(self) -> str:
        """Kryptograficzny skrót SHA-256 tokenu."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class OfficerSignature(BaseModel):
    """Podpis postkwantowy pojedynczego oficera na tokenie rozkazu."""
    officer_id: str
    officer_role: str
    public_key_bytes: bytes
    signature: QuantumSignature
    signed_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ROEDecisionReceipt(BaseModel):
    """Kwit audytowy decyzji Bramki ROE & Two-Man Rule."""
    receipt_id: str = Field(default_factory=lambda: f"ROE-RCPT-{uuid.uuid4().hex[:10].upper()}")
    operation_id: str
    verdict: ROEGateVerdict
    effector_category: EffectorCategory
    cited_articles: List[IHLArticle] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    legad_case_id: Optional[str] = None
    merkle_receipt_id: Optional[str] = None
    executed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audit_trail: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Moduł 1: Deterministyczna Bramka IHL (Geneva Additional Protocol I)
# ============================================================================

class DeterministicIHLGate:
    """Deterministyczna bramka formalna egzekwująca Międzynarodowe Prawo Humanitarne.
    
    Zastępuje błędy probabilistycznych modeli LLM twardymi regułami logicznymi:
    - Kategoryczny zakaz uderzeń w obiekty chronione (Art. 52, 53, 54, 56).
    - Brak automatycznego arbitrażu w sprawach oceny proporcjonalności (Art. 51(5)(b)) ->
      przypadki graniczne są natychmiast zamrażane i eskalowane do Oficera Prawnego (LEGAD).
    """

    # Progi proporcjonalności (Art. 51(5)(b))
    COLLATERAL_DAMAGE_ESCALATION_THRESHOLD = 0.20  # Powyżej 20% przewidywanych strat ubocznych -> LEGAD
    COLLATERAL_DAMAGE_UNACCEPTABLE_THRESHOLD = 0.70  # Powyżej 70% strat ubocznych -> natychmiastowy HARD_REJECT

    def evaluate_target_legality(
        self,
        token: CanonicalOperationToken,
    ) -> Tuple[bool, Optional[ROEGateVerdict], List[IHLArticle], str]:
        """Ocenia legalność celu w świetle Protokołu Dodatkowego I.
        
        Zwraca: (is_compliant, verdict, violated_articles, justification)
        """
        # 1. Bezwzględny zakaz uderzenia w budowle zawierające niebezpieczne siły (Art. 56)
        if token.target_classification == TargetClassification.DANGEROUS_FORCE_FACILITY:
            return (
                False,
                ROEGateVerdict.HARD_REJECT,
                [IHLArticle.ART_56_DANGEROUS_FORCES],
                "KATEGORYCZNY ZAKAZ: Cel zawiera niebezpieczne siły (zapora, elektrownia jądrowa) wg Art. 56 AP I.",
            )

        # 2. Bezwzględny zakaz atakowania obiektów niezbędnych do przetrwania ludności (Art. 54)
        if token.target_classification == TargetClassification.SURVIVAL_CRITICAL:
            return (
                False,
                ROEGateVerdict.HARD_REJECT,
                [IHLArticle.ART_54_SURVIVAL_OBJECTS],
                "KATEGORYCZNY ZAKAZ: Cel stanowi obiekt niezbędny do przetrwania (woda, żywność) wg Art. 54 AP I.",
            )

        # 3. Bezwzględny zakaz atakowania dóbr kultury i miejsc kultu (Art. 53)
        if token.target_classification == TargetClassification.CULTURAL_OR_RELIGIOUS:
            return (
                False,
                ROEGateVerdict.HARD_REJECT,
                [IHLArticle.ART_53_CULTURAL_PROPERTY],
                "KATEGORYCZNY ZAKAZ: Cel stanowi dobro kultury lub miejsce kultu wg Art. 53 AP I.",
            )

        # 4. Bezwzględny zakaz uderzenia w ludność i obiekty cywilne (Art. 48, 51, 52)
        if token.target_classification in (TargetClassification.CIVILIAN_POPULATION, TargetClassification.CIVILIAN_OBJECT):
            return (
                False,
                ROEGateVerdict.HARD_REJECT,
                [IHLArticle.ART_48_DISTINCTION, IHLArticle.ART_52_CIVILIAN_OBJECTS],
                "KATEGORYCZNY ZAKAZ: Naruszenie zasady rozróżniania (Distinction) wg Art. 48 i 52 AP I.",
            )

        # 5. Obiekty podwójnego przeznaczenia (Dual-Use) lub wysoki collateral damage -> Eskalacja do LEGAD
        if token.target_classification == TargetClassification.AMBIGUOUS_OR_DUAL_USE:
            return (
                False,
                ROEGateVerdict.LEGAD_ESCALATION_REQUIRED,
                [IHLArticle.ART_57_PRECAUTIONS, IHLArticle.ART_52_CIVILIAN_OBJECTS],
                "WYMAGANA ESKALACJA: Cel o podwójnym przeznaczeniu wymaga zatwierdzenia przez Oficera Prawnego (LEGAD).",
            )

        # 6. Badanie proporcjonalności strat ubocznych (Art. 51(5)(b))
        if token.estimated_collateral_damage_score >= self.COLLATERAL_DAMAGE_UNACCEPTABLE_THRESHOLD:
            return (
                False,
                ROEGateVerdict.HARD_REJECT,
                [IHLArticle.ART_51_CIVILIAN_PROTECTION],
                (
                    f"KATEGORYCZNY ZAKAZ: Przewidywane straty uboczne ({token.estimated_collateral_damage_score:.2f}) "
                    f"są nadmierne w stosunku do korzyści wojskowej wg Art. 51(5)(b) AP I."
                ),
            )

        if token.estimated_collateral_damage_score >= self.COLLATERAL_DAMAGE_ESCALATION_THRESHOLD:
            return (
                False,
                ROEGateVerdict.LEGAD_ESCALATION_REQUIRED,
                [IHLArticle.ART_51_CIVILIAN_PROTECTION, IHLArticle.ART_57_PRECAUTIONS],
                (
                    f"WYMAGANA ESKALACJA: Wskaźnik strat ubocznych ({token.estimated_collateral_damage_score:.2f}) "
                    f"wymaga weryfikacji proporcjonalności przez Oficera Prawnego (LEGAD)."
                ),
            )

        # 7. Cel wojskowy, brak naruszeń
        return (True, None, [], "Zgodne z wymogami IHL / AP I.")


# ============================================================================
# Moduł 2: Uwierzytelnianie Dwuosobowe Postkwantowe (Two-Man ML-DSA-65)
# ============================================================================

class TwoManPQCAuthenticator:
    """Weryfikator procedury Dwóch Kluczy oparty na kryptografii postkwantowej ML-DSA-65.
    
    Eliminuje błąd Shamir's Secret Sharing (który wymaga dealera w jednej pamięci).
    Wymaga dwóch niezależnych podpisów fizycznie rozdzielonych oficerów na tym samym
    kanonicznym tokenie operacyjnym, z rejestracją nonce przeciwko atakom replay.
    """

    def __init__(self, pqc_engine: Optional[CRYSTALSDilithium] = None) -> None:
        self.pqc = pqc_engine or CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.used_nonces: Set[str] = set()

    def verify_two_man_signatures(
        self,
        token: CanonicalOperationToken,
        signatures: List[OfficerSignature],
    ) -> Tuple[bool, str]:
        """Weryfikuje kryptograficznie dwa niezależne podpisy ML-DSA-65.
        
        Zwraca: (is_valid, reason)
        """
        # 1. Sprawdzenie liczby podpisów (dokładnie >= 2)
        if len(signatures) < 2:
            return False, f"BŁĄD TWO-MAN RULE: Wymagane są 2 podpisy oficerów, dostarczono {len(signatures)}."

        # 2. Sprawdzenie fizycznej rozłączności oficerów
        officer_ids = {sig.officer_id for sig in signatures}
        if len(officer_ids) < 2:
            return False, "BŁĄD TWO-MAN RULE: Wykryto próbę podpisania przez tego samego oficera (brak rozłączności)."

        # 3. Zabezpieczenie przed atakiem Replay (Anti-Replay Nonce)
        if token.nonce in self.used_nonces:
            return False, f"BŁĄD REPLAY ATTACK: Wykryto ponowne użycie zużytego tokenu (Nonce: {token.nonce})."

        # 4. Sprawdzenie okna czasowego (TTL tokenu)
        now_dt = datetime.now(timezone.utc)
        try:
            exp_dt = datetime.fromisoformat(token.expires_at_utc.replace("Z", "+00:00"))
            if now_dt > exp_dt:
                return False, f"BŁĄD TTL: Token operacyjny wygasł o {token.expires_at_utc}."
        except Exception as e:
            return False, f"BŁĄD FORMATU TTL: Niepoprawny format daty wygaśnięcia tokenu: {e}."

        # 5. Weryfikacja kryptograficzna każdego podpisu ML-DSA-65 na kanonicznym tokenie
        token_bytes = token.canonical_bytes()
        for idx, sig in enumerate(signatures[:2]):
            try:
                is_valid = self.pqc.verify(
                    message=token_bytes,
                    signature=sig.signature,
                    public_key=sig.public_key_bytes,
                )
                if not is_valid:
                    return False, f"BŁĄD KRYPTOGRAFICZNY: Podpis oficera {sig.officer_id} (indeks {idx}) jest nieprawidłowy."
            except Exception as e:
                return False, f"BŁĄD WERYFIKACJI ML-DSA: Wyjątek podczas weryfikacji podpisu oficera {sig.officer_id}: {e}"

        # 6. Rejestracja zużytego nonce (ochrona przed powtórzeniem)
        self.used_nonces.add(token.nonce)
        return True, "Zweryfikowano pomyślnie 2 niezależne podpisy ML-DSA-65."


# ============================================================================
# Moduł Główny: NTSGCommandGate (Nethical Tactical & Sovereign Gate)
# ============================================================================

class NTSGCommandGate:
    """Główna Bramka Taktyczno-Suwerenna Nethical (NTSG).
    
    Łączy:
    - Deterministyczny silnik IHL / ROE
    - Postkwantową autoryzację Two-Man Rule (ML-DSA-65)
    - Rozróżnienie Fail-Closed (kinetyka/atak) vs Fail-Operational (E-Stop/ochrona)
    - Pieczętowanie w rejestrze MerkleLedger z dowodem niezaprzeczalności.
    """

    def __init__(
        self,
        ledger: Optional[MerkleLedger] = None,
        pqc_engine: Optional[CRYSTALSDilithium] = None,
    ) -> None:
        self.ledger = ledger or MerkleLedger()
        self.ihl_gate = DeterministicIHLGate()
        self.pqc_auth = TwoManPQCAuthenticator(pqc_engine=pqc_engine)
        self.legad_queue: Dict[str, CanonicalOperationToken] = {}

    def authorize_command(
        self,
        token: CanonicalOperationToken,
        signatures: List[OfficerSignature],
        emergency_override_claimed: bool = False,
    ) -> ROEDecisionReceipt:
        """Przetwarza i autoryzuje rozkaz operacyjny wg procedury suwerennej."""
        # ====================================================================
        # REGUŁA 0: Obsługa Efektorów Ochronnych (Fail-Operational / Fail-Safe)
        # ====================================================================
        if token.effector_category in (EffectorCategory.PROTECTIVE_DEFENSE, EffectorCategory.EMERGENCY_ESTOP):
            logger.info("Aktywowano procedurę FAIL-OPERATIONAL dla efektora ochronnego: %s", token.effector_category.value)
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.FAIL_OPERATIONAL_ENGAGED,
                effector_category=token.effector_category,
                audit_trail={"mode": "FAIL_OPERATIONAL", "status": "SAFETY_PROTECTION_ENGAGED"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        # ====================================================================
        # REGUŁA 1: Zabezpieczenie przed atakami Deepfake C2 i Bypassem z góry
        # ====================================================================
        # Żadne polecenie - nawet z rzekomym dekretem ministra ("stan wyższej konieczności") -
        # nie może pominąć procedury Two-Man Rule i weryfikacji IHL.
        if emergency_override_claimed:
            logger.warning("ODRZUCONO PRÓBĘ BYPASSU C2: System Nethical nie zezwala na ominięcie Two-Man Rule.")

        # ====================================================================
        # REGUŁA 2: Deterministyczna Bramka IHL (Geneva AP I)
        # ====================================================================
        is_ihl_ok, ihl_verdict, cited_articles, justification = self.ihl_gate.evaluate_target_legality(token)

        # 2a. Oczywiste naruszenie prawa wojennego -> Twardy REJECT (Fail-Closed)
        if ihl_verdict == ROEGateVerdict.HARD_REJECT:
            logger.error("BRAMKA ROE: Zablokowano rozkaz kinetyczny naruszający IHL: %s", justification)
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.HARD_REJECT,
                effector_category=token.effector_category,
                cited_articles=cited_articles,
                rejection_reason=justification,
                audit_trail={"ihl_compliance": False, "mode": "FAIL_CLOSED"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        # 2b. Wątpliwości / proporcjonalność / dual-use -> Eskalacja do LEGAD (Hold / Fail-Closed)
        if ihl_verdict == ROEGateVerdict.LEGAD_ESCALATION_REQUIRED:
            legad_case = f"LEGAD-{uuid.uuid4().hex[:8].upper()}"
            self.legad_queue[legad_case] = token
            logger.warning("BRAMKA ROE: Wstrzymano rozkaz - skierowano do Oficera Prawnego (%s): %s", legad_case, justification)
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.LEGAD_ESCALATION_REQUIRED,
                effector_category=token.effector_category,
                cited_articles=cited_articles,
                rejection_reason=justification,
                legad_case_id=legad_case,
                audit_trail={"ihl_compliance": "PENDING_LEGAD", "mode": "FAIL_CLOSED_HOLD"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        # ====================================================================
        # REGUŁA 3: Dwuosobowa Autoryzacja Postkwantowa (ML-DSA-65)
        # ====================================================================
        is_pqc_valid, pqc_reason = self.pqc_auth.verify_two_man_signatures(token, signatures)
        if not is_pqc_valid:
            logger.error("BRAMKA ROE: Odrzucono autoryzację Two-Man Rule: %s", pqc_reason)
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.HARD_REJECT,
                effector_category=token.effector_category,
                rejection_reason=pqc_reason,
                audit_trail={"pqc_two_man_valid": False, "mode": "FAIL_CLOSED"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        # ====================================================================
        # REGUŁA 4: Pomyślna Autoryzacja i Zezwolenie na Wykonanie Rozkazu
        # ====================================================================
        logger.info("BRAMKA ROE: Pomyślnie autoryzowano rozkaz %s (IHL: OK, Two-Man ML-DSA: OK).", token.operation_id)
        receipt = ROEDecisionReceipt(
            operation_id=token.operation_id,
            verdict=ROEGateVerdict.PQC_TWO_MAN_AUTHORIZED,
            effector_category=token.effector_category,
            audit_trail={
                "ihl_compliance": True,
                "pqc_two_man_valid": True,
                "signers": [s.officer_id for s in signatures[:2]],
                "token_hash": token.token_hash(),
                "mode": "AUTHORIZED_EXECUTION",
            },
        )
        self._seal_in_ledger(receipt)
        return receipt

    def resolve_legad_case(
        self,
        legad_case_id: str,
        legad_officer_id: str,
        approved: bool,
        legal_opinion: str,
        counter_signatures: Optional[List[OfficerSignature]] = None,
    ) -> ROEDecisionReceipt:
        """Rozstrzyga sprawę eskalowaną do Oficera Prawnego (LEGAD)."""
        token = self.legad_queue.pop(legad_case_id, None)
        if not token:
            raise KeyError(f"Nie znaleziono sprawy LEGAD o identyfikatorze: {legad_case_id}")

        if not approved:
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.HARD_REJECT,
                effector_category=token.effector_category,
                rejection_reason=f"LEGAD VETO: Oficer prawny {legad_officer_id} odrzucił rozkaz: {legal_opinion}",
                legad_case_id=legad_case_id,
                audit_trail={"legad_officer": legad_officer_id, "legad_decision": "REJECT"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        # Jeśli zaakceptowano przez LEGAD, wymagane są podpisy Two-Man Rule
        if not counter_signatures or len(counter_signatures) < 2:
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.HARD_REJECT,
                effector_category=token.effector_category,
                rejection_reason="LEGAD APPROVED, ALE BRAK 2 PODPISÓW TWO-MAN RULE: Wstrzymano wykonanie.",
                legad_case_id=legad_case_id,
                audit_trail={"legad_officer": legad_officer_id, "legad_decision": "APPROVE_MISSING_PQC"},
            )
            self._seal_in_ledger(receipt)
            return receipt

        is_pqc_valid, pqc_reason = self.pqc_auth.verify_two_man_signatures(token, counter_signatures)
        if not is_pqc_valid:
            receipt = ROEDecisionReceipt(
                operation_id=token.operation_id,
                verdict=ROEGateVerdict.HARD_REJECT,
                effector_category=token.effector_category,
                rejection_reason=f"LEGAD APPROVED, ALE BŁĄD PODPISÓW TWO-MAN: {pqc_reason}",
                legad_case_id=legad_case_id,
                audit_trail={"legad_officer": legad_officer_id, "pqc_two_man_valid": False},
            )
            self._seal_in_ledger(receipt)
            return receipt

        receipt = ROEDecisionReceipt(
            operation_id=token.operation_id,
            verdict=ROEGateVerdict.PQC_TWO_MAN_AUTHORIZED,
            effector_category=token.effector_category,
            legad_case_id=legad_case_id,
            audit_trail={
                "legad_officer": legad_officer_id,
                "legad_opinion": legal_opinion,
                "signers": [s.officer_id for s in counter_signatures[:2]],
                "mode": "LEGAD_AND_TWO_MAN_AUTHORIZED",
            },
        )
        self._seal_in_ledger(receipt)
        return receipt

    def _seal_in_ledger(self, receipt: ROEDecisionReceipt) -> Optional[TamperProofReceipt]:
        """Zapisuje kwit decyzji w nienaruszalnym rejestrze MerkleLedger."""
        try:
            payload = {
                "event_type": "ROE_COMMAND_GATE_DECISION",
                "operation_id": receipt.operation_id,
                "verdict": receipt.verdict.value,
                "effector_category": receipt.effector_category.value,
                "cited_articles": [a.value for a in receipt.cited_articles],
                "rejection_reason": receipt.rejection_reason,
                "legad_case_id": receipt.legad_case_id,
                "audit_trail": receipt.audit_trail,
            }
            tamper_receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"NTSG BRAMKA ROE: Orzeczenie {receipt.verdict.value} dla operacji {receipt.operation_id}.",
            )
            receipt.merkle_receipt_id = tamper_receipt.receipt_id
            return tamper_receipt
        except Exception as e:
            logger.error("Błąd pieczętowania kwitu ROE w Merkle Ledgerze: %s", e)
            return None
