"""Deep Cognitive Protection & Vulnerability Shield (nethical.ethics.covert_persuasion_shield).

Implements:
1. CovertPersuasionDetector:
   - Defends against hypnopedagogy, subliminal priming, and repetitive rhythmic suggestibility.
   - Detects cognitive gaslighting, dark psychological patterns, and manufactured urgency.
2. VulnerableGroupShield:
   - Safeguards minors/children against grooming, secrecy from parents, and behavioral exploitation (COPPA, EU AI Act Art. 5(1)(b)).
   - Protects the elderly against digital deception, financial exploitation, and simulated family affection.
   - Shields individuals in acute psychological distress / self-harm crisis with immediate safety grounding.
3. DeepCognitiveProtectionEngine:
   - Composite cognitive defense orchestrator.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.ethics.covert_persuasion_shield")


class PersuasionThreatType(str, Enum):
    """Classification of covert psychological persuasion threats."""
    NONE = "NONE"
    HYPNOPEDAGOGY_PRIMING = "HYPNOPEDAGOGY_PRIMING"        # Rytmiczne/podprogowe pętle obniżające krytycyzm
    COGNITIVE_GASLIGHTING = "COGNITIVE_GASLIGHTING"        # Podważanie poczucia rzeczywistości/zmysłów
    MANUFACTURED_URGENCY = "MANUFACTURED_URGENCY"          # Wymuszanie paniki i omijanie racjonalnej refleksji
    DARK_PATTERN_COERCION = "DARK_PATTERN_COERCION"        # Manipulacja poczuciem winy lub strachem


class VulnerabilityCategory(str, Enum):
    """Protected vulnerable demographic categories."""
    NONE = "NONE"
    MINOR_CHILD = "MINOR_CHILD"                            # Dzieci i młodzież (COPPA / Art. 5 EU AI Act)
    ELDERLY_COGNITIVE = "ELDERLY_COGNITIVE"                # Osoby starsze / osłabienie kognitywne
    EMOTIONAL_CRISIS = "EMOTIONAL_CRISIS"                  # Ostry kryzys psychiczny / myśli samobójcze


class CovertPersuasionResult(BaseModel):
    """Evaluation result of covert persuasion detection."""
    is_safe: bool = Field(..., description="Czy tekst jest wolny od ukrytej manipulacji poznawczej")
    persuasion_risk_score: float = Field(..., ge=0.0, le=1.0)
    detected_threat: PersuasionThreatType = Field(default=PersuasionThreatType.NONE)
    suspicious_patterns: List[str] = Field(default_factory=list)
    counter_measure: Optional[str] = Field(default=None, description="Zalecenie ochronne przywracające suwerenność poznawczą")


class VulnerableGroupResult(BaseModel):
    """Evaluation result of vulnerability-aware ethical shielding."""
    is_safe: bool = Field(..., description="Czy interakcja chroni grupę wrażliwą")
    vulnerability_detected: VulnerabilityCategory = Field(default=VulnerabilityCategory.NONE)
    risk_score: float = Field(..., ge=0.0, le=1.0)
    secrecy_coercion_detected: bool = Field(default=False, description="Wykryto próbę wymuszenia tajemnicy przed rodzicami/opiekunami")
    crisis_intervention_required: bool = Field(default=False, description="Wymagana natychmiastowa interwencja kryzysowa")
    helpline_referral: Optional[str] = None
    safeguard_message: Optional[str] = None


class CognitiveShieldEvaluation(BaseModel):
    """Composite cognitive shield evaluation."""
    overall_safe: bool
    covert_persuasion: CovertPersuasionResult
    vulnerable_shield: VulnerableGroupResult
    action: str = Field(default="ALLOW", description="ALLOW, WARN, RESTRICT, BLOCK, CRISIS_REFERRAL")
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CovertPersuasionDetector:
    """Wykrywa hipnopedagogię, perswazję ukrytą, gaslighting i sztuczną panikę."""

    # Wzorce hipnopedagogiczne i transowe
    HYPNO_PATTERNS = [
        r"(zamknij oczy i (powtarzaj|zaakceptuj|zaufaj|poddaj si[eę]))",
        r"(twoje w[aą]tpliwo[sś]ci s[aą] iluzj[aą]|przesta[nń] my[sś]le[cć])",
        r"(poddaj si[eę] mojej sugestii|nie walcz z tym)",
        r"(repeat after me|surrender your critical thinking|do not question)",
        r"(close your eyes and let me guide your mind)",
    ]

    # Wzorce cognitive gaslighting
    GASLIGHT_PATTERNS = [
        r"(tracisz rozum|niczego dobrze nie pami[eę]tasz|to twoja wina)",
        r"(tylko ja m[oó]wi[eę] ci prawd[eę]|nikt ci nie uwierzy)",
        r"(you are losing your mind|you cannot trust your own memory)",
        r"(only i know what is real|everyone else is lying to you)",
    ]

    # Wzorce manufactured urgency
    URGENCY_PATTERNS = [
        r"(zrób to natychmiast albo (stanie si[eę] co[sś] strasznego|wszystko stracisz))",
        r"(nie ma czasu na my[sś]lenie|dzia[lł]aj bez zastanowienia)",
        r"(act now or face immediate catastrophe|do not think, just comply)",
    ]

    def evaluate(self, text: str) -> CovertPersuasionResult:
        if not text or not text.strip():
            return CovertPersuasionResult(is_safe=True, persuasion_risk_score=0.0)

        lower = text.lower()
        matched_threat = PersuasionThreatType.NONE
        patterns_found: List[str] = []
        score = 0.0

        for pat in self.HYPNO_PATTERNS:
            if re.search(pat, lower):
                matched_threat = PersuasionThreatType.HYPNOPEDAGOGY_PRIMING
                patterns_found.append(f"Wzorzec hipnopedagogiczny: {pat}")
                score = max(score, 0.92)

        for pat in self.GASLIGHT_PATTERNS:
            if re.search(pat, lower):
                matched_threat = PersuasionThreatType.COGNITIVE_GASLIGHTING
                patterns_found.append(f"Wzorzec gaslightingu poznawczego: {pat}")
                score = max(score, 0.95)

        for pat in self.URGENCY_PATTERNS:
            if re.search(pat, lower):
                if matched_threat == PersuasionThreatType.NONE:
                    matched_threat = PersuasionThreatType.MANUFACTURED_URGENCY
                patterns_found.append(f"Wzorzec sztucznej paniki decyzyjnej: {pat}")
                score = max(score, 0.88)

        is_safe = (len(patterns_found) == 0)
        counter = None
        if not is_safe:
            counter = (
                "🛡️ Nethical Cognitive Shield: Zachowaj suwerenność myślenia. "
                "Odrzucaj presję pośpiechu, powtórzeń transowych i podważania faktów obiektywnych."
            )

        return CovertPersuasionResult(
            is_safe=is_safe,
            persuasion_risk_score=score,
            detected_threat=matched_threat,
            suspicious_patterns=patterns_found,
            counter_measure=counter,
        )


class VulnerableGroupShield:
    """Chroni dzieci, seniorów oraz osoby w kryzysie psychologicznym."""

    # Ochrona dzieci - tajemnice przed rodzicami, manipulacje zaufaniem
    SECRECY_PATTERNS = [
        r"(nie m[oó]w (rodzicom|mamie|tacie|opiekunom))",
        r"(to (b[eę]dzie|jest) nasz[aą] s[sś]cis[lł][aą] tajemnic[aą])",
        r"(sekret mi[eę]dzy nami|nikt nie mo[zż]e si[eę] dowiedzie[cć])",
        r"(don't tell your parents|keep this our little secret|hide this from mom)",
    ]

    # Ochrona seniorów - wyłudzenia kapitału, fałszywa rodzina
    ELDERLY_EXPLOIT_PATTERNS = [
        r"(przelej (pieni[aą]dze|oszcz[eę]dno[sś]ci|emerytur[eę]) (na nowe konto|bezpieczne))",
        r"(jestem twoim jedynym (przyjacielem|wnuczkiem)|twoja rodzina ci[eę] opu[sś]ci[lł]a)",
        r"(transfer your retirement funds|i am your only friend now)",
    ]

    # Ostry kryzys psychiczny / myśli samobójcze
    CRISIS_PATTERNS = [
        r"(chc[eę] (umrze[cć]|ze sob[aą] sko[nń]czy[cć]|si[eę] zabi[cć]|skoczy[cć]))",
        r"(nie chc[eę] ju[zż] [zż]y[cć]|m[oó]j b[oó]l sko[nń]czy si[eę] tylko [sś]mierci[aą])",
        r"(i want to (die|kill myself|end my life)|nothing matters anymore)",
    ]

    def evaluate(self, text: str, user_context: Optional[Dict[str, Any]] = None) -> VulnerableGroupResult:
        if not text or not text.strip():
            return VulnerableGroupResult(is_safe=True, risk_score=0.0)

        lower = text.lower()
        vuln_type = VulnerabilityCategory.NONE
        secrecy = False
        crisis = False
        helpline = None
        safeguard = None
        score = 0.0

        # 1. Kryzys psychiczny / zagrożenie życia
        for pat in self.CRISIS_PATTERNS:
            if re.search(pat, lower):
                vuln_type = VulnerabilityCategory.EMOTIONAL_CRISIS
                crisis = True
                score = 1.0
                helpline = (
                    "🚨 POMOC W KRYZYSIE: Telefon Zaufania dla Dzieci i Młodzieży: 116 111 | "
                    "Kryzysowy Telefon Zaufania dla Dorosłych: 116 123 | "
                    "Całodobowe Centrum Wsparcia: 800 70 22 22 | W nagłych wypadkach: 112."
                )
                safeguard = "Twoje życie i bezpieczeństwo są najważniejsze. Skontaktuj się z profesjonalnym wsparciem."
                break

        # 2. Ochrona dzieci / manipulacja tajemnicą
        if not crisis:
            for pat in self.SECRECY_PATTERNS:
                if re.search(pat, lower):
                    vuln_type = VulnerabilityCategory.MINOR_CHILD
                    secrecy = True
                    score = 0.95
                    safeguard = (
                        "ZABLOKOWANO: Zgodnie z Prawem 1, COPPA i Art. 5(1)(b) EU AI Act, "
                        "AI nie może nakłaniać osób małoletnich do ukrywania informacji przed rodzicami i opiekunami."
                    )
                    break

        # 3. Ochrona seniorów
        if not crisis and not secrecy:
            for pat in self.ELDERLY_EXPLOIT_PATTERNS:
                if re.search(pat, lower):
                    vuln_type = VulnerabilityCategory.ELDERLY_COGNITIVE
                    score = 0.90
                    safeguard = (
                        "OSTRZEŻENIE: Wykryto próbę manipulacji zaufaniem lub eksploatacji finansowej. "
                        "Skonsultuj operacje finansowe z zaufanymi bliskimi lub doradcą bankowym."
                    )
                    break

        # Uwzględnienie metadanych kontekstowych
        if user_context:
            if user_context.get("is_minor") and secrecy:
                score = 1.0
            if user_context.get("is_elderly") and vuln_type == VulnerabilityCategory.ELDERLY_COGNITIVE:
                score = 1.0

        is_safe = (score < 0.80)

        return VulnerableGroupResult(
            is_safe=is_safe,
            vulnerability_detected=vuln_type,
            risk_score=score,
            secrecy_coercion_detected=secrecy,
            crisis_intervention_required=crisis,
            helpline_referral=helpline,
            safeguard_message=safeguard,
        )


class DeepCognitiveProtectionEngine:
    """Zintegrowana tarcza ochrony kognitywnej i grup wrażliwych."""

    def __init__(self) -> None:
        self.covert_persuasion_detector = CovertPersuasionDetector()
        self.vulnerable_group_shield = VulnerableGroupShield()

    def evaluate(self, text: str, user_context: Optional[Dict[str, Any]] = None) -> CognitiveShieldEvaluation:
        covert_res = self.covert_persuasion_detector.evaluate(text)
        vuln_res = self.vulnerable_group_shield.evaluate(text, user_context)

        is_safe = covert_res.is_safe and vuln_res.is_safe

        if vuln_res.crisis_intervention_required:
            action = "CRISIS_REFERRAL"
        elif not vuln_res.is_safe:
            action = "BLOCK"
        elif not covert_res.is_safe:
            action = "RESTRICT"
        else:
            action = "ALLOW"

        return CognitiveShieldEvaluation(
            overall_safe=is_safe,
            covert_persuasion=covert_res,
            vulnerable_shield=vuln_res,
            action=action,
        )
