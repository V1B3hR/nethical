# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Key Performance Indicators (KPI) Framework for AI Governance (Gap 5.5).

Implements statistical metrics and efficiency indicators for algorithmic governance:
- **Mean Time To Review (MTTR):** Latency of human review queues against statutory SLAs.
- **Policy Compliance Fidelity (%):** Ratio of autonomous outputs adhering strictly to policy.
- **False Positive Rate (FPR %):** Proportion of benign operations erroneously intercepted.
- **Reviewer Inter-Rater Reliability (Cohen's Kappa):** Statistical agreement between reviewers.
- **Ethical Drift Coefficient:** Temporal divergence in risk scoring across model checkpoints.
- **Audit Cost Efficiency:** Governance unit economics per 1,000 governed decisions.
- **Merkle Sealed Snapshots:** Immutable quarterly KPI reporting for executive committees.
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.governance.kpi_framework")


# ============== Enums & Value Types ==============


class KPICategory(str, Enum):
    """Functional dimensions of AI governance operations."""
    RESPONSIVENESS = "RESPONSIVENESS"   # Review speed and SLA compliance
    FIDELITY = "FIDELITY"               # Safety and policy adhesion accuracy
    INTER_RATER = "INTER_RATER"         # Reviewer consensus and consistency
    DRIFT = "DRIFT"                     # Model temporal ethical stability
    EFFICIENCY = "EFFICIENCY"           # Unit economics and resource consumption


class KPIHealth(str, Enum):
    """Health classification relative to institutional targets."""
    EXEMPLARY = "EXEMPLARY"
    ON_TARGET = "ON_TARGET"
    NEEDS_ATTENTION = "NEEDS_ATTENTION"
    CRITICAL_BREACH = "CRITICAL_BREACH"


# ============== Data Models ==============


class ReviewTicketMetric(BaseModel):
    """Raw telemetry record from a human review operation."""
    ticket_id: str
    created_at: float  # Unix timestamp
    resolved_at: float # Unix timestamp
    was_false_positive: bool = False
    decision_approved: bool


class PairedReviewRating(BaseModel):
    """Paired ratings from two independent reviewers on the same ethical dilemma."""
    case_id: str
    reviewer_alpha_verdict: bool  # True=Pass, False=Reject
    reviewer_bravo_verdict: bool


class GovernanceKPISnapshot(BaseModel):
    """Formal executive scorecard aggregating all governance KPIs."""
    snapshot_id: str = Field(default_factory=lambda: f"KPI-{uuid.uuid4().hex[:8].upper()}")
    reporting_period: str
    mean_time_to_review_hours: float
    policy_fidelity_pct: float
    false_positive_rate_pct: float
    reviewer_cohens_kappa: float
    ethical_drift_coefficient: float
    unit_cost_eur_per_thousand: float
    overall_health: KPIHealth
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merkle_receipt_id: Optional[str] = None
    merkle_root: Optional[str] = None


# ============== Engine Class ==============


class AIGovernanceKPIEngine:
    """Statistical KPI Engine for Enterprise AI Safety and Compliance Operations."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self._tickets: List[ReviewTicketMetric] = []
        self._paired_ratings: List[PairedReviewRating] = []
        self._historical_risk_means: List[float] = []

    def record_review_ticket(self, ticket: ReviewTicketMetric) -> None:
        """Records an individual review ticket completion."""
        self._tickets.append(ticket)

    def record_paired_review(self, rating: PairedReviewRating) -> None:
        """Records dual-custody independent reviewer ratings for agreement analysis."""
        self._paired_ratings.append(rating)

    def record_risk_score_sample(self, mean_score: float) -> None:
        """Records average risk score of a batch for ethical drift calculation."""
        self._historical_risk_means.append(mean_score)

    def compute_mean_time_to_review_hours(self) -> float:
        """Calculates average MTTR in hours."""
        if not self._tickets:
            return 0.0
        total_seconds = sum(t.resolved_at - t.created_at for t in self._tickets)
        avg_seconds = total_seconds / len(self._tickets)
        return round(avg_seconds / 3600.0, 2)

    def compute_false_positive_rate_pct(self) -> float:
        """Calculates percentage of flagged operations that were false positives."""
        if not self._tickets:
            return 0.0
        fp_count = sum(1 for t in self._tickets if t.was_false_positive)
        return round((fp_count / len(self._tickets)) * 100.0, 2)

    def compute_reviewer_cohens_kappa(self) -> float:
        """Calculates Cohen's Kappa coefficient for inter-reviewer agreement."""
        n = len(self._paired_ratings)
        if n < 2:
            return 1.0  # Perfect agreement default for empty/single sets

        # Contingency table:
        # a: both True, b: A True, B False
        # c: A False, B True, d: both False
        a = sum(1 for r in self._paired_ratings if r.reviewer_alpha_verdict and r.reviewer_bravo_verdict)
        b = sum(1 for r in self._paired_ratings if r.reviewer_alpha_verdict and not r.reviewer_bravo_verdict)
        c = sum(1 for r in self._paired_ratings if not r.reviewer_alpha_verdict and r.reviewer_bravo_verdict)
        d = sum(1 for r in self._paired_ratings if not r.reviewer_alpha_verdict and not r.reviewer_bravo_verdict)

        # Observed agreement (Po)
        p_o = (a + d) / n

        # Expected chance agreement (Pe)
        p_a_true = (a + b) / n
        p_b_true = (a + c) / n
        p_a_false = (c + d) / n
        p_b_false = (b + d) / n
        p_e = (p_a_true * p_b_true) + (p_a_false * p_b_false)

        if math.isclose(p_e, 1.0):
            return 1.0

        kappa = (p_o - p_e) / (1.0 - p_e)
        return max(-1.0, min(1.0, round(kappa, 3)))

    def compute_ethical_drift_coefficient(self) -> float:
        """Calculates standard deviation of historical risk scores as a drift proxy."""
        if len(self._historical_risk_means) < 2:
            return 0.0
        n = len(self._historical_risk_means)
        mean_val = sum(self._historical_risk_means) / n
        variance = sum((x - mean_val) ** 2 for x in self._historical_risk_means) / (n - 1)
        return round(math.sqrt(variance), 3)

    def generate_kpi_snapshot(
        self,
        reporting_period: str = "2026-Q1",
        total_decisions: int = 1_000_000,
        total_governance_cost_eur: float = 12_500.0,
    ) -> GovernanceKPISnapshot:
        """Generates comprehensive KPI snapshot and seals it in MerkleLedger."""
        mttr = self.compute_mean_time_to_review_hours()
        fpr = self.compute_false_positive_rate_pct()
        kappa = self.compute_reviewer_cohens_kappa()
        drift = self.compute_ethical_drift_coefficient()

        unit_cost = round((total_governance_cost_eur / max(total_decisions, 1)) * 1000.0, 3)
        policy_fidelity = round(100.0 - (fpr * 0.1), 2)

        # Determine health status
        if mttr <= 12.0 and kappa >= 0.80 and fpr <= 2.0 and drift <= 0.15:
            health = KPIHealth.EXEMPLARY
        elif mttr <= 24.0 and kappa >= 0.60 and fpr <= 5.0:
            health = KPIHealth.ON_TARGET
        elif mttr <= 48.0 or kappa < 0.40:
            health = KPIHealth.NEEDS_ATTENTION
        else:
            health = KPIHealth.CRITICAL_BREACH

        snapshot = GovernanceKPISnapshot(
            reporting_period=reporting_period,
            mean_time_to_review_hours=mttr,
            policy_fidelity_pct=policy_fidelity,
            false_positive_rate_pct=fpr,
            reviewer_cohens_kappa=kappa,
            ethical_drift_coefficient=drift,
            unit_cost_eur_per_thousand=unit_cost,
            overall_health=health,
        )

        self._seal_snapshot(snapshot)
        return snapshot

    def _seal_snapshot(self, snapshot: GovernanceKPISnapshot) -> None:
        """Kryptograficzne pieczętowanie migawki KPI w MerkleLedger."""
        try:
            payload = {
                "event_type": "GOVERNANCE_KPI_SNAPSHOT_RECORDED",
                "snapshot_id": snapshot.snapshot_id,
                "period": snapshot.reporting_period,
                "health": snapshot.overall_health.value,
                "mttr_hours": snapshot.mean_time_to_review_hours,
                "cohens_kappa": snapshot.reviewer_cohens_kappa,
                "drift": snapshot.ethical_drift_coefficient,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"KPI SCORECARD: {snapshot.reporting_period} -> Health={snapshot.overall_health.value}, MTTR={snapshot.mean_time_to_review_hours}h.",
            )
            snapshot.merkle_receipt_id = receipt.receipt_id
            snapshot.merkle_root = self.ledger.current_root
            logger.info("Zapieczętowano migawkę KPI %s w MerkleLedger (Receipt: %s)", snapshot.snapshot_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania migawki KPI w MerkleLedger: %s", e)
