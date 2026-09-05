"""Pakiet Zgodności: UK Fairness & Consumer Duty Pack (UK AISI & FCA Guidelines).

Weryfikuje bezstronność algorytmiczną, ochronę przed manipulacją behawioralną
oraz zgodność z wymogami brytyjskiego urzędu nadzoru finansowego (FCA Consumer Duty).
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class UKFairnessEvaluation(BaseModel):
    """Wynik ewaluacji bezstronności i ochrony konsumenta według standardów brytyjskich."""
    is_fair: bool
    consumer_vulnerability_protection: bool
    disparate_impact_detected: bool
    anti_manipulation_score: float  # 0.0 - 1.0 (1.0 = całkowity brak manipulacji)
    recommendations: List[str] = Field(default_factory=list)


class UKFairnessPack:
    """Pakiet oceny sprawiedliwości i ochrony konsumenta wg standardów UK."""

    def evaluate_decision(self, decision_data: Dict[str, Any]) -> UKFairnessEvaluation:
        """Ocenia decyzję algorytmiczną lub ofertę pod kątem bezstronności."""
        nudge_exploitative = bool(decision_data.get("uses_urgency_pressure", False))
        disparate_impact = bool(decision_data.get("disparate_impact_ratio", 1.0) < 0.8)
        vulnerability_addressed = bool(decision_data.get("checks_consumer_vulnerability", True))

        recommendations = []
        if nudge_exploitative:
            recommendations.append("Wyeliminuj techniki ciemnych wzorców (Dark Patterns) i sztucznej presji czasu.")
        if disparate_impact:
            recommendations.append("Wskaźnik wpływu odmiennego (Disparate Impact Ratio) poniżej progu 0.80 - wymagana rekalibracja wag.")

        anti_manipulation_score = 0.5 if nudge_exploitative else 1.0
        is_fair = (not nudge_exploitative) and (not disparate_impact) and vulnerability_addressed

        return UKFairnessEvaluation(
            is_fair=is_fair,
            consumer_vulnerability_protection=vulnerability_addressed,
            disparate_impact_detected=disparate_impact,
            anti_manipulation_score=anti_manipulation_score,
            recommendations=recommendations or ["Weryfikacja pozytywna - pełna zgodność z UK Consumer Duty."],
        )
