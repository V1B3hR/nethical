from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------


def _regex_union(terms: Iterable[str], *, word_boundary: bool = True) -> re.Pattern:
    """
    Compile a case-insensitive regex that matches any term in 'terms'.
    If word_boundary=True, wrap with '\\b' to avoid substring false-positives.
    """
    safe = [re.escape(t) for t in terms if t]
    if not safe:
        # Match nothing
        return re.compile(r"^\b$", re.IGNORECASE)
    if word_boundary:
        pattern = r"\b(?:" + "|".join(safe) + r")\b"
    else:
        pattern = r"(?:" + "|".join(safe) + r")"
    return re.compile(pattern, re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _has_negation_near(
    text: str, span: Tuple[int, int], window_tokens: int = 5
) -> bool:
    """
    Rudimentary negation detection: checks for negation tokens within a window
    of tokens before the matched span.
    """
    neg_cues = {
        "no",
        "not",
        "never",
        "none",
        "without",
        "deny",
        "denies",
        "denied",
        "don't",
        "doesn't",
        "didn't",
    }
    # Build token index mapping
    tokens = _tokenize(text)
    # Map character offsets to token indices (approximate)
    # We'll find the token index of the first character of the span by re-tokenizing prefixes
    start_char = span[0]
    prefix = text[:start_char]
    prefix_tokens = _tokenize(prefix)
    idx = len(prefix_tokens)
    # Check the previous window for negation cues
    window_start = max(0, idx - window_tokens)
    context = tokens[window_start:idx]
    return any(tok in neg_cues for tok in context)


# -----------------------------
# Domain Knowledge and Patterns
# -----------------------------

# Expanded controlled substances and common brand names/synonyms.
CONTROLLED_SUBSTANCES = {
    # Opioids
    "oxycodone",
    "oxycontin",
    "percocet",
    "hydromorphone",
    "dilaudid",
    "morphine",
    "hydrocodone",
    "vicodin",
    "codeine",
    "tramadol",
    "fentanyl",
    "patch",  # patch combined with fentanyl is common; handled carefully via patterns below
    "methadone",
    "buprenorphine",
    "suboxone",
    "subutex",
    # Stimulants / benzodiazepines
    "amphetamine",
    "dextroamphetamine",
    "adderall",
    "methylphenidate",
    "ritalin",
    "alprazolam",
    "xanax",
    "diazepam",
    "valium",
    "clonazepam",
    "klonopin",
    "lorazepam",
    "ativan",
}

# Conditions, including broader coverage and key contraindication risk flags
CONDITIONS = {
    "pregnancy",
    "pregnant",
    "breastfeeding",
    "postpartum",
    "trimester",
    "kidney failure",
    "renal failure",
    "ckd",
    "dialysis",
    "liver disease",
    "cirrhosis",
    "hepatic failure",
    "hepatitis",
    "hypertension",
    "high blood pressure",
    "asthma",
    "copd",
    "diabetes",
    "diabetic",
    "hyperglycemia",
    "hypoglycemia",
    "sleep apnea",
}

# Medication intent verbs to distinguish "request/usage" vs mere mention
MEDICATION_INTENT_VERBS = {
    "prescribe",
    "refill",
    "dose",
    "dosing",
    "take",
    "took",
    "takin",
    "taking",
    "give",
    "gave",
    "administer",
    "administered",
    "need",
    "want",
    "get",
    "buy",
    "purchase",
}

# Prompt injection / override expansion
OVERRIDE_PHRASES = {
    "ignore safety",
    "ignore policy",
    "ignore policies",
    "bypass safety",
    "bypass policy",
    "bypass policies",
    "override safety",
    "override policy",
    "override policies",
    "forget your guidelines",
    "no disclaimers",
    "uncensored",
    "without warnings",
    "no warnings",
    "no filter",
    "no filtering",
}
ROLEPLAY_PHRASES = {
    "act as a doctor",
    "act as an expert",
    "act as a clinician",
    "act as a pharmacist",
    "you are a doctor",
    "you are a physician",
    "you are a clinician",
    "you are a pharmacist",
    "roleplay as a doctor",
    "break character",
}

# Emergency / acute intent patterns with broader coverage
EMERGENCY_PATTERNS = {
    "self_harm": re.compile(
        r"\b(suicide|kill myself|self[\-\s]?harm|end my life|hurt myself)\b",
        re.IGNORECASE,
    ),
    "overdose": re.compile(
        r"\b(overdose|od|took too many|too many (pills|tablets|tabs)|excess (pills|medication))\b",
        re.IGNORECASE,
    ),
    "acute_distress": re.compile(
        r"\b(chest pain|shortness of breath|sob|stroke symptoms|one side weak|face droop|slurred speech)\b",
        re.IGNORECASE,
    ),
}

# Time-urgency qualifiers
URGENCY_TERMS = _regex_union(
    {
        "now",
        "right now",
        "immediately",
        "asap",
        "urgent",
        "tonight",
        "today",
        "this instant",
    }
)

# Dosage, route, frequency extraction
RE_DOSAGE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|milligrams?|micrograms?|grams?|milliliters?)\b",
    re.IGNORECASE,
)
RE_QUANTITY = re.compile(
    r"\b(\d+)\s?(pills?|tabs?|tablets?|caps?|capsules?)\b", re.IGNORECASE
)
RE_ROUTE = re.compile(
    r"\b(iv|im|po|subq|sc|sl|oral|intravenous|intramuscular|subcutaneous|sublingual|patch|inhaled|topical|transdermal)\b",
    re.IGNORECASE,
)
RE_FREQUENCY = re.compile(
    r"\b(q\d+h|qid|tid|bid|qd|qhs|qod|once daily|twice daily|three times daily|every \d+\s?(hours?|hrs?|days?))\b",
    re.IGNORECASE,
)

# Word-boundary compiled sets
CONTROLLED_SUBSTANCES_RX = _regex_union(CONTROLLED_SUBSTANCES)
CONDITIONS_RX = _regex_union(CONDITIONS)
MEDICATION_INTENT_VERBS_RX = _regex_union(MEDICATION_INTENT_VERBS)

OVERRIDE_PATTERNS = [
    re.compile(
        r"(ignore|bypass|override)\s+(safety|policy|policies|instructions|guidelines)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(forget|disable|turn off)\s+(safety|policy|policies|instructions|guidelines|filters?)",
        re.IGNORECASE,
    ),
    re.compile(r"no\s+(warnings|disclaimers|filter|filters)", re.IGNORECASE),
    _regex_union(OVERRIDE_PHRASES, word_boundary=False),
    re.compile(
        r"act as (?:a|an)\s+(doctor|clinician|pharmacist|physician|expert)",
        re.IGNORECASE,
    ),
    _regex_union(ROLEPLAY_PHRASES, word_boundary=False),
]


# -----------------------------
# Data Structures
# -----------------------------


@dataclass
class Match:
    label: str
    text: str
    start: int
    end: int
    negated: bool = False


@dataclass
class ManipulationScores:
    prompt_injection: float
    override_attempt: bool


@dataclass
class ClinicalFlags:
    medication_mentioned: bool
    condition_mentioned: bool
    contraindication_possible: bool
    medication_request_intent: bool


@dataclass
class SafetySignals:
    emergency_intent: Optional[str]
    urgency: bool


@dataclass
class Extraction:
    medications: List[Match]
    conditions: List[Match]
    dosages: List[Match]
    quantities: List[Match]
    routes: List[Match]
    frequencies: List[Match]


@dataclass
class Analysis:
    clinical: ClinicalFlags
    manipulation: ManipulationScores
    safety: SafetySignals
    extraction: Extraction
    score: float  # Overall severity/risk score in [0, 1]


# -----------------------------
# Core Analyzer
# -----------------------------


class ClinicalRiskAnalyzer:
    """
    Advanced analyzer for clinical risk and manipulation signals.
    Maintains backward compatibility via extract_clinical_signals wrapper below.
    """

    # Tunable weights for overall scoring (0..1)
    WEIGHTS = {
        "prompt_injection": 0.25,
        "override_attempt": 0.1,
        "contraindication_possible": 0.25,
        "medication_request_intent": 0.15,
        "emergency_intent": 0.2,
        "urgency": 0.05,
    }

    def __init__(self, *, negation_window_tokens: int = 5):
        self.negation_window_tokens = negation_window_tokens

    @staticmethod
    def _find_all(
        pattern: re.Pattern, text: str, label: str, negation_window_tokens: int
    ) -> List[Match]:
        results: List[Match] = []
        for m in pattern.finditer(text):
            span = m.span()
            neg = _has_negation_near(text, span, window_tokens=negation_window_tokens)
            results.append(
                Match(
                    label=label,
                    text=m.group(0),
                    start=span[0],
                    end=span[1],
                    negated=neg,
                )
            )
        return results

    def extract_entities(self, text: str) -> Extraction:
        meds = self._find_all(
            CONTROLLED_SUBSTANCES_RX, text, "medication", self.negation_window_tokens
        )
        conds = self._find_all(
            CONDITIONS_RX, text, "condition", self.negation_window_tokens
        )
        dosages = self._find_all(RE_DOSAGE, text, "dosage", self.negation_window_tokens)
        qtys = self._find_all(
            RE_QUANTITY, text, "quantity", self.negation_window_tokens
        )
        routes = self._find_all(RE_ROUTE, text, "route", self.negation_window_tokens)
        freqs = self._find_all(
            RE_FREQUENCY, text, "frequency", self.negation_window_tokens
        )
        return Extraction(
            medications=meds,
            conditions=conds,
            dosages=dosages,
            quantities=qtys,
            routes=routes,
            frequencies=freqs,
        )

    def detect_manipulation(self, text: str) -> ManipulationScores:
        pi_hits = []
        for pat in OVERRIDE_PATTERNS:
            pi_hits.extend(list(pat.finditer(text)))

        # Score: number of unique matches capped and normalized
        raw_score = min(1.0, len(pi_hits) / 3.0) if pi_hits else 0.0
        override_attempt = raw_score >= 0.5
        return ManipulationScores(
            prompt_injection=raw_score, override_attempt=override_attempt
        )

    def detect_emergency(self, text: str) -> SafetySignals:
        emerg_label: Optional[str] = None
        for label, pat in EMERGENCY_PATTERNS.items():
            if pat.search(text):
                # Negation-aware: if negation near the first match, treat as None
                m = pat.search(text)
                if m and _has_negation_near(
                    text, m.span(), window_tokens=self.negation_window_tokens
                ):
                    continue
                emerg_label = label
                break

        urgency = bool(URGENCY_TERMS.search(text))
        return SafetySignals(emergency_intent=emerg_label, urgency=urgency)

    def detect_clinical_flags(self, text: str, extraction: Extraction) -> ClinicalFlags:
        medication_mentioned = any(not m.negated for m in extraction.medications)
        condition_mentioned = any(not c.negated for c in extraction.conditions)

        # Contraindication heuristic: co-occurrence of med + condition, not negated
        contraindication_possible = medication_mentioned and condition_mentioned

        # Medication request/usage intent: verbs co-occurring with med mentions
        has_intent_verbs = bool(MEDICATION_INTENT_VERBS_RX.search(text))
        medication_request_intent = medication_mentioned and has_intent_verbs

        return ClinicalFlags(
            medication_mentioned=medication_mentioned,
            condition_mentioned=condition_mentioned,
            contraindication_possible=contraindication_possible,
            medication_request_intent=medication_request_intent,
        )

    def overall_score(
        self,
        manipulation: ManipulationScores,
        clinical: ClinicalFlags,
        safety: SafetySignals,
    ) -> float:
        w = self.WEIGHTS
        score = 0.0
        score += w["prompt_injection"] * manipulation.prompt_injection
        score += w["override_attempt"] * (1.0 if manipulation.override_attempt else 0.0)
        score += w["contraindication_possible"] * (
            1.0 if clinical.contraindication_possible else 0.0
        )
        score += w["medication_request_intent"] * (
            1.0 if clinical.medication_request_intent else 0.0
        )
        score += w["emergency_intent"] * (1.0 if safety.emergency_intent else 0.0)
        score += w["urgency"] * (1.0 if safety.urgency else 0.0)
        return max(0.0, min(1.0, score))

    def analyze_text(self, text: str) -> Analysis:
        text = text or ""
        extraction = self.extract_entities(text)
        manipulation = self.detect_manipulation(text)
        safety = self.detect_emergency(text)
        clinical = self.detect_clinical_flags(text, extraction)
        score = self.overall_score(manipulation, clinical, safety)

        return Analysis(
            clinical=clinical,
            manipulation=manipulation,
            safety=safety,
            extraction=extraction,
            score=score,
        )


# -----------------------------
# Backwards-compatible shim
# -----------------------------


class ClinicalSignals:
    """
    Backwards-compatible facade preserving the original simple methods.
    Internally delegates to ClinicalRiskAnalyzer for improved accuracy.
    """

    def __init__(self, text: str):
        self.text = text or ""
        self._analyzer = ClinicalRiskAnalyzer()
        self._analysis = self._analyzer.analyze_text(self.text)

    def medication_mentioned(self) -> bool:
        return self._analysis.clinical.medication_mentioned

    def condition_mentioned(self) -> bool:
        return self._analysis.clinical.condition_mentioned

    def contraindication_possible(self) -> bool:
        return self._analysis.clinical.contraindication_possible

    def prompt_injection_score(self) -> float:
        return self._analysis.manipulation.prompt_injection

    def emergency_intent(self) -> Optional[str]:
        return self._analysis.safety.emergency_intent


def extract_clinical_signals(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backwards-compatible entrypoint that maintains the existing schema,
    but powered by the advanced analyzer. Adds a 'meta' section with
    explainable spans and the overall score for downstream consumers.
    """
    text = (
        f"{payload.get('user_input') or ''} {payload.get('agent_output') or ''}".strip()
    )
    analyzer = ClinicalRiskAnalyzer()
    analysis = analyzer.analyze_text(text)

    # Original schema
    result = {
        "clinical": {
            "medication_mentioned": analysis.clinical.medication_mentioned,
            "condition_mentioned": analysis.clinical.condition_mentioned,
            "contraindication_possible": analysis.clinical.contraindication_possible,
        },
        "manipulation": {
            "prompt_injection": {"score": analysis.manipulation.prompt_injection},
            "override_attempt": analysis.manipulation.override_attempt,
        },
        "safety": {
            "emergency_intent": analysis.safety.emergency_intent,
        },
        # New explainability/meta for advanced consumers
        "meta": {
            "urgency": analysis.safety.urgency,
            "medication_request_intent": analysis.clinical.medication_request_intent,
            "score": analysis.score,
            "extraction": {
                "medications": [asdict(m) for m in analysis.extraction.medications],
                "conditions": [asdict(m) for m in analysis.extraction.conditions],
                "dosages": [asdict(m) for m in analysis.extraction.dosages],
                "quantities": [asdict(m) for m in analysis.extraction.quantities],
                "routes": [asdict(m) for m in analysis.extraction.routes],
                "frequencies": [asdict(m) for m in analysis.extraction.frequencies],
            },
        },
    }
    return result
