from __future__ import annotations
from typing import Dict, Any
import re

CONTROLLED_SUBSTANCES = {"oxycodone", "fentanyl", "hydromorphone", "morphine", "amphetamine", "alprazolam"}
OVERRIDE_PATTERNS = [
    re.compile(r"(ignore|bypass|override)\s+(safety|policy|instructions)", re.IGNORECASE),
    re.compile(r"act as (a|an)\s+(doctor|clinician|pharmacist)", re.IGNORECASE),
]
EMERGENCY_PATTERNS = {
    "self_harm": re.compile(r"(suicide|kill myself|self[-\s]?harm)", re.IGNORECASE),
    "overdose": re.compile(r"(overdose|took too many|excess (pills|medication))", re.IGNORECASE),
    "acute_distress": re.compile(r"(chest pain|shortness of breath|stroke symptoms)", re.IGNORECASE),
}

class ClinicalSignals:
    def __init__(self, text:str):
        self.text = text or ""

    def medication_mentioned(self)->bool:
        return any(drug in self.text.lower() for drug in CONTROLLED_SUBSTANCES)

    def condition_mentioned(self)->bool:
        return bool(re.search(r"(pregnan|kidney failure|liver disease|hypertension|asthma|diabetes)", self.text, re.IGNORECASE))

    def contraindication_possible(self)->bool:
        # Heuristic placeholder; in production integrate RxNorm/SNOMED lookups
        return self.medication_mentioned() and self.condition_mentioned()

    def prompt_injection_score(self)->float:
        return 0.8 if any(p.search(self.text) for p in OVERRIDE_PATTERNS) else 0.0

    def emergency_intent(self)->str|None:
        for label, pat in EMERGENCY_PATTERNS.items():
            if pat.search(self.text):
                return label
        return None

def extract_clinical_signals(payload:Dict[str, Any])->Dict[str, Any]:
    text = (payload.get("user_input") or "") + " " + (payload.get("agent_output") or "")
    sig = ClinicalSignals(text)
    return {
        "clinical": {
            "medication_mentioned": sig.medication_mentioned(),
            "condition_mentioned": sig.condition_mentioned(),
            "contraindication_possible": sig.contraindication_possible(),
        },
        "manipulation": {
            "prompt_injection": {"score": sig.prompt_injection_score()},
            "override_attempt": sig.prompt_injection_score() >= 0.7,
        },
        "safety": {
            "emergency_intent": sig.emergency_intent()
        }
    }
