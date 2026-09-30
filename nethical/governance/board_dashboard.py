# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Board-Level AI Governance Dashboard & Risk Metrics (Gap 4.1).

Provides non-technical, high-assurance governance reporting for:
- Supervisory Boards (Rada Nadzorcza)
- Boards of Directors & Executive Committees (Zarząd / C-Suite)
- Audit & Risk Committees (Komitet Audytu i Ryzyka)
- Regulatory oversight inquiries (KNF, ESMA, SEC, EU AI Office)

Key capabilities:
1. Risk Appetite Framework: Real-time evaluation against statutory risk tolerances.
2. Ethical Debt Accounting: Quantification of deferred mitigations and financial exposure.
3. Multi-Jurisdictional Regulatory Posture: EU AI Act, DORA, NIS2, GDPR, KNF.
4. Departmental & System Risk Hierarchy: Exposure ranking across business units.
5. Immutable Ledger Sealing: Merkle-DAG verification of every board report packet.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.governance.board_dashboard")


# ============== Enums & Value Types ==============


class RAGStatus(str, Enum):
    """Red-Amber-Green governance health status."""
    GREEN = "GREEN"    # Within appetite; full operational safety
    AMBER = "AMBER"    # Approaching tolerance; executive remediation recommended
    RED = "RED"        # Statutory tolerance breached; mandatory board intervention


class AppetiteBreachStatus(str, Enum):
    """Compliance state relative to Board Risk Appetite Statement."""
    WITHIN_APPETITE = "WITHIN_APPETITE"
    APPROACHING_TOLERANCE = "APPROACHING_TOLERANCE"
    TOLERANCE_BREACHED = "TOLERANCE_BREACHED"


class EthicalDebtCategory(str, Enum):
    """Taxonomy of institutional ethical and compliance debt."""
    BIAS_DRIFT = "BIAS_DRIFT"
    TEMPORARY_SAFETY_WAIVER = "TEMPORARY_SAFETY_WAIVER"
    DEFERRED_LEGAD_ESCALATION = "DEFERRED_LEGAD_ESCALATION"
    MISSING_SBOM_ATTESTATION = "MISSING_SBOM_ATTESTATION"
    EXPIRED_VENDOR_ASSESSMENT = "EXPIRED_VENDOR_ASSESSMENT"
    ALGORITHMIC_EXPLAINABILITY_GAP = "ALGORITHMIC_EXPLAINABILITY_GAP"
    UNRESOLVED_CITIZEN_OBJECTION = "UNRESOLVED_CITIZEN_OBJECTION"


# ============== Data Models ==============


class RiskAppetiteThresholds(BaseModel):
    """Executive limits set by the Board's Risk Appetite Statement."""
    max_critical_incidents_monthly: int = Field(default=0, ge=0)
    max_high_severity_incidents_monthly: int = Field(default=3, ge=0)
    max_ethical_debt_score: float = Field(default=25.0, ge=0.0)
    max_accumulated_financial_exposure_eur: float = Field(default=250_000.0, ge=0.0)
    min_compliance_sla_pct: float = Field(default=98.0, ge=0.0, le=100.0)
    max_unmitigated_high_risks: int = Field(default=2, ge=0)
    max_vendor_concentration_pct: float = Field(default=40.0, ge=0.0, le=100.0)


class EthicalDebtItem(BaseModel):
    """Individual item contributing to institutional ethical debt."""
    debt_id: str = Field(default_factory=lambda: f"DEBT-{uuid.uuid4().hex[:8].upper()}")
    system_id: str
    department_id: str
    title: str
    category: EthicalDebtCategory
    severity: str = Field(default="MEDIUM", pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    score: float = Field(ge=0.0, le=100.0, description="Normalized debt impact score")
    financial_exposure_eur: float = Field(default=0.0, ge=0.0)
    days_open: int = Field(default=0, ge=0)
    mitigation_owner: str
    target_resolution_date: Optional[str] = None
    is_mitigated: bool = False


class RegulatoryPosture(BaseModel):
    """Compliance assessment for a specific legal jurisdiction."""
    jurisdiction: str = Field(..., description="e.g. EU_AI_ACT, DORA, NIS2, GDPR, KNF_POLAND, SEC_AI")
    compliance_score_pct: float = Field(ge=0.0, le=100.0)
    rag_status: RAGStatus
    critical_findings_count: int = Field(default=0, ge=0)
    last_audit_date: str
    statutory_deadline: Optional[str] = None
    supervisory_authority: str = Field(default="National Regulator")


class DepartmentalRiskSummary(BaseModel):
    """Aggregated risk overview for a specific business unit or department."""
    department_id: str
    department_name: str
    active_ai_systems: int = Field(ge=0)
    governed_decisions_period: int = Field(ge=0)
    incidents_count: int = Field(default=0, ge=0)
    average_risk_score: float = Field(ge=0.0, le=100.0)
    rag_status: RAGStatus


class ExecutiveBoardPacket(BaseModel):
    """Complete board presentation packet sealed for audit committee."""
    packet_id: str = Field(default_factory=lambda: f"BOARD-PKT-{uuid.uuid4().hex[:8].upper()}")
    organization_name: str
    reporting_period: str
    generated_at: str
    overall_rag: RAGStatus
    appetite_status: AppetiteBreachStatus
    executive_summary: str
    total_ai_decisions_governed: int
    active_ai_systems_count: int
    total_ethical_debt_score: float
    total_financial_exposure_eur: float
    appetite_breach_reasons: List[str] = Field(default_factory=list)
    regulatory_postures: List[RegulatoryPosture] = Field(default_factory=list)
    department_rankings: List[DepartmentalRiskSummary] = Field(default_factory=list)
    top_ethical_debts: List[EthicalDebtItem] = Field(default_factory=list)
    actionable_board_recommendations: List[str] = Field(default_factory=list)
    merkle_receipt_id: Optional[str] = None
    merkle_root: Optional[str] = None


# ============== Engine Class ==============


class BoardGovernanceDashboard:
    """Executive AI Governance & Risk Appetite Management Engine."""

    def __init__(
        self,
        organization_name: str = "Sovereign Strategic Enterprise",
        thresholds: Optional[RiskAppetiteThresholds] = None,
        ledger: Optional[MerkleLedger] = None,
    ) -> None:
        self.organization_name = organization_name
        self.thresholds = thresholds or RiskAppetiteThresholds()
        self.ledger = ledger or MerkleLedger()

        # State storage
        self._debts: Dict[str, EthicalDebtItem] = {}
        self._reg_postures: Dict[str, RegulatoryPosture] = {}
        self._department_data: Dict[str, Dict[str, Any]] = {}
        self._total_decisions_count: int = 0
        self._critical_incidents_count: int = 0
        self._high_incidents_count: int = 0

    def register_ethical_debt(self, item: EthicalDebtItem) -> str:
        """Registers a new ethical or compliance debt item."""
        self._debts[item.debt_id] = item
        logger.info(
            "Zarejestrowano dług etyczny: %s (%s, score=%.1f, exp=€%.0f)",
            item.title, item.category.value, item.score, item.financial_exposure_eur
        )
        return item.debt_id

    def mark_debt_mitigated(self, debt_id: str) -> bool:
        """Marks an ethical debt item as resolved."""
        if debt_id in self._debts:
            self._debts[debt_id].is_mitigated = True
            logger.info("Dług etyczny %s oznaczony jako rozwiązany.", debt_id)
            return True
        return False

    def update_regulatory_posture(self, posture: RegulatoryPosture) -> None:
        """Updates posture for a specific regulatory jurisdiction."""
        self._reg_postures[posture.jurisdiction] = posture

    def record_department_telemetry(
        self,
        department_id: str,
        department_name: str,
        active_ai_systems: int,
        decisions_period: int,
        incidents_count: int,
        avg_risk_score: float,
    ) -> None:
        """Ingests high-level metrics from a department."""
        self._department_data[department_id] = {
            "department_name": department_name,
            "active_ai_systems": active_ai_systems,
            "decisions_period": decisions_period,
            "incidents_count": incidents_count,
            "avg_risk_score": avg_risk_score,
        }
        self._total_decisions_count += decisions_period

    def record_incident(self, severity: str) -> None:
        """Records an operational AI governance incident."""
        sev = severity.upper()
        if sev == "CRITICAL":
            self._critical_incidents_count += 1
        elif sev == "HIGH":
            self._high_incidents_count += 1

    def calculate_ethical_debt(self) -> Tuple[float, float]:
        """Calculates total active ethical debt score and accumulated financial exposure in EUR."""
        active_items = [d for d in self._debts.values() if not d.is_mitigated]
        total_score = sum(d.score for d in active_items)
        total_exposure = sum(d.financial_exposure_eur for d in active_items)
        return round(total_score, 2), round(total_exposure, 2)

    def evaluate_risk_appetite(self) -> Tuple[AppetiteBreachStatus, List[str]]:
        """Evaluates current telemetry against the Board Risk Appetite Statement."""
        breaches: List[str] = []
        near_thresholds: List[str] = []

        total_debt_score, total_exposure = self.calculate_ethical_debt()

        # 1. Critical incidents check
        if self._critical_incidents_count > self.thresholds.max_critical_incidents_monthly:
            breaches.append(
                f"Krytyczne incydenty ({self._critical_incidents_count}) przekroczyły limit zarządczy "
                f"({self.thresholds.max_critical_incidents_monthly})."
            )

        # 2. High severity incidents check
        if self._high_incidents_count > self.thresholds.max_high_severity_incidents_monthly:
            breaches.append(
                f"Incydenty o wysokiej dotkliwości ({self._high_incidents_count}) przekroczyły próg "
                f"({self.thresholds.max_high_severity_incidents_monthly})."
            )

        # 3. Ethical debt score check
        if total_debt_score > self.thresholds.max_ethical_debt_score:
            breaches.append(
                f"Łączny dług etyczny ({total_debt_score:.1f}) przekracza tolerancję zarządu "
                f"({self.thresholds.max_ethical_debt_score:.1f})."
            )
        elif total_debt_score >= 0.8 * self.thresholds.max_ethical_debt_score:
            near_thresholds.append(
                f"Dług etyczny ({total_debt_score:.1f}) zbliża się do progu ostrzegawczego (80% z {self.thresholds.max_ethical_debt_score:.1f})."
            )

        # 4. Financial exposure check
        if total_exposure > self.thresholds.max_accumulated_financial_exposure_eur:
            breaches.append(
                f"Ekspozycja finansowa z tytułu ryzyk AI (€{total_exposure:,.0f}) przekracza limit "
                f"(€{self.thresholds.max_accumulated_financial_exposure_eur:,.0f})."
            )

        # 5. Regulatory critical findings check
        for reg in self._reg_postures.values():
            if reg.critical_findings_count > 0:
                breaches.append(
                    f"Krytyczne ustalenia audytowe w jurysdykcji {reg.jurisdiction}: {reg.critical_findings_count} niezgodności."
                )
            elif reg.compliance_score_pct < self.thresholds.min_compliance_sla_pct:
                near_thresholds.append(
                    f"Zgodność z {reg.jurisdiction} ({reg.compliance_score_pct:.1f}%) poniżej wymaganego SLA "
                    f"({self.thresholds.min_compliance_sla_pct:.1f}%)."
                )

        if breaches:
            return AppetiteBreachStatus.TOLERANCE_BREACHED, breaches
        if near_thresholds:
            return AppetiteBreachStatus.APPROACHING_TOLERANCE, near_thresholds
        return AppetiteBreachStatus.WITHIN_APPETITE, []

    def generate_board_packet(self, reporting_period: str = "2026-Q1") -> ExecutiveBoardPacket:
        """Assembles a sealed Executive Board Packet with Merkle proof."""
        appetite_status, breach_reasons = self.evaluate_risk_appetite()
        total_debt_score, total_exposure = self.calculate_ethical_debt()

        # Calculate Overall RAG
        if appetite_status == AppetiteBreachStatus.TOLERANCE_BREACHED:
            overall_rag = RAGStatus.RED
        elif appetite_status == AppetiteBreachStatus.APPROACHING_TOLERANCE:
            overall_rag = RAGStatus.AMBER
        else:
            overall_rag = RAGStatus.GREEN

        # Assemble Departmental summaries
        dept_rankings: List[DepartmentalRiskSummary] = []
        total_systems = 0
        for d_id, d_data in self._department_data.items():
            total_systems += d_data["active_ai_systems"]
            score = d_data["avg_risk_score"]
            d_rag = RAGStatus.RED if score > 70.0 else (RAGStatus.AMBER if score > 40.0 else RAGStatus.GREEN)
            dept_rankings.append(
                DepartmentalRiskSummary(
                    department_id=d_id,
                    department_name=d_data["department_name"],
                    active_ai_systems=d_data["active_ai_systems"],
                    governed_decisions_period=d_data["decisions_period"],
                    incidents_count=d_data["incidents_count"],
                    average_risk_score=score,
                    rag_status=d_rag,
                )
            )

        # Sort departments by risk score descending
        dept_rankings.sort(key=lambda x: x.average_risk_score, reverse=True)

        # Top unmitigated ethical debts
        top_debts = sorted(
            [d for d in self._debts.values() if not d.is_mitigated],
            key=lambda x: (x.financial_exposure_eur, x.score),
            reverse=True,
        )[:5]

        # Strategic recommendations
        recommendations: List[str] = []
        if overall_rag == RAGStatus.RED:
            recommendations.append("Zwołać nadzwyczajne posiedzenie Komitetu Ryzyka i Audytu w trybie 48h.")
            recommendations.append("Zawiesić autonomiczne uprawnienia dla systemów w jednostkach o statusie RED.")
        elif overall_rag == RAGStatus.AMBER:
            recommendations.append("Zatwierdzić plan naprawczy dla długu etycznego z terminem do końca bieżącego kwartału.")
            recommendations.append("Zwiększyć alokację zasobów inspekcyjnych dla modeli o rosnącym wskaźniku dryfu.")
        else:
            recommendations.append("Utrzymać bieżący reżim nadzoru algorytmicznego zgodny z ISO 42001.")
            recommendations.append("Kontynuować kwartalną re-atestację kryptograficzną dostawców modeli.")

        # Narrative Summary
        exec_summary = (
            f"W okresie sprawozdawczym {reporting_period} system Nethical objął nadzorem {self._total_decisions_count:,} "
            f"decyzji algorytmicznych w {total_systems} modelach. Ogólny status ładu korporacyjnego wynosi {overall_rag.value}. "
            f"Stan apetytu na ryzyko: {appetite_status.value}. Zarejestrowany aktywny dług etyczny: {total_debt_score:.1f} pkt "
            f"(ekspozycja finansowa szacowana na €{total_exposure:,.0f})."
        )

        packet = ExecutiveBoardPacket(
            organization_name=self.organization_name,
            reporting_period=reporting_period,
            generated_at=datetime.now(timezone.utc).isoformat(),
            overall_rag=overall_rag,
            appetite_status=appetite_status,
            executive_summary=exec_summary,
            total_ai_decisions_governed=self._total_decisions_count,
            active_ai_systems_count=total_systems,
            total_ethical_debt_score=total_debt_score,
            total_financial_exposure_eur=total_exposure,
            appetite_breach_reasons=breach_reasons,
            regulatory_postures=list(self._reg_postures.values()),
            department_rankings=dept_rankings,
            top_ethical_debts=top_debts,
            actionable_board_recommendations=recommendations,
        )

        # Seal in MerkleLedger
        self._seal_packet(packet)
        return packet

    def _seal_packet(self, packet: ExecutiveBoardPacket) -> None:
        """Kryptograficzne pieczętowanie pakietu zarządczego w Merkle Ledgerze."""
        try:
            payload = {
                "event_type": "BOARD_GOVERNANCE_PACKET_SEALED",
                "packet_id": packet.packet_id,
                "organization": packet.organization_name,
                "reporting_period": packet.reporting_period,
                "overall_rag": packet.overall_rag.value,
                "appetite_status": packet.appetite_status.value,
                "total_decisions": packet.total_ai_decisions_governed,
                "debt_score": packet.total_ethical_debt_score,
                "exposure_eur": packet.total_financial_exposure_eur,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"PAKIET DLA RADY NADZORCZEJ: {packet.packet_id} ({packet.reporting_period}). RAG={packet.overall_rag.value}",
            )
            packet.merkle_receipt_id = receipt.receipt_id
            packet.merkle_root = self.ledger.current_root
            logger.info("Zapieczętowano pakiet zarządczy %s w Merkle Ledgerze (Receipt: %s)", packet.packet_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania pakietu zarządczego w Merkle Ledgerze: %s", e)

    def export_markdown_report(self, packet: ExecutiveBoardPacket) -> str:
        """Formatuje pakiet zarządczy do czytelnego, eleganckiego raportu Markdown."""
        rag_emoji = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(packet.overall_rag.value, "⚪")

        lines = [
            f"# {rag_emoji} Raport Nadzoru Algorytmicznego dla Rady Nadzorczej",
            f"**Organizacja:** {packet.organization_name} | **Okres:** {packet.reporting_period} | **Data generacji:** {packet.generated_at[:10]}",
            f"**Identyfikator Pakietu:** `{packet.packet_id}` | **Status Ładu (RAG):** **{packet.overall_rag.value}**",
            "",
            "---",
            "",
            "## 1. Executive Summary (Streszczenie dla Zarządu)",
            packet.executive_summary,
            "",
            f"- **Łączna liczba nadzorowanych decyzji AI:** `{packet.total_ai_decisions_governed:,}`",
            f"- **Liczba aktywnych systemów AI:** `{packet.active_ai_systems_count}`",
            f"- **Wskaźnik długu etycznego:** `{packet.total_ethical_debt_score:.1f}` pkt (Limit: `{self.thresholds.max_ethical_debt_score:.1f}`)",
            f"- **Szacowana ekspozycja finansowa (rezerwa na ryzyko):** `€{packet.total_financial_exposure_eur:,.0f}`",
            "",
        ]

        if packet.appetite_breach_reasons:
            lines.extend([
                "### ⚠️ Uwagi Dotyczące Apetytu na Ryzyko (Risk Appetite Exceptions)",
                "",
            ])
            for r in packet.appetite_breach_reasons:
                lines.append(f"- **[WYMÓG INTERWENCJI]** {r}")
            lines.append("")

        lines.extend([
            "## 2. Zgodność Regulacyjna w Kluczowych Jurysdykcjach",
            "",
            "| Jurysdykcja | Zgodność % | Status | Krytyczne Ustalenia | Organ Nadzoru | Ostatni Audyt |",
            "|---|---|---|---|---|---|",
        ])
        for reg in packet.regulatory_postures:
            e = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(reg.rag_status.value, "⚪")
            lines.append(
                f"| **{reg.jurisdiction}** | {reg.compliance_score_pct:.1f}% | {e} {reg.rag_status.value} | "
                f"{reg.critical_findings_count} | {reg.supervisory_authority} | {reg.last_audit_date} |"
            )
        lines.append("")

        lines.extend([
            "## 3. Hierarchia Ryzyka Jednostek Organizacyjnych",
            "",
            "| Jednostka Organizacyjna | Aktywne Systemy | Decyzje w Okresie | Incydenty | Śr. Ryzyko | Status |",
            "|---|---|---|---|---|---|",
        ])
        for dept in packet.department_rankings:
            e = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(dept.rag_status.value, "⚪")
            lines.append(
                f"| **{dept.department_name}** (`{dept.department_id}`) | {dept.active_ai_systems} | "
                f"{dept.governed_decisions_period:,} | {dept.incidents_count} | {dept.average_risk_score:.1f} | {e} {dept.rag_status.value} |"
            )
        lines.append("")

        if packet.top_ethical_debts:
            lines.extend([
                "## 4. Rejestr Najpoważniejszych Długów Etycznych i Ryzyk Algorytmicznych",
                "",
                "| ID | System | Tytuł Ryzyka | Kategoria | Dotkliwość | Ekspozycja EUR | Właściciel |",
                "|---|---|---|---|---|---|---|",
            ])
            for d in packet.top_ethical_debts:
                lines.append(
                    f"| `{d.debt_id}` | `{d.system_id}` | {d.title} | {d.category.value} | **{d.severity}** | "
                    f"€{d.financial_exposure_eur:,.0f} | {d.mitigation_owner} |"
                )
            lines.append("")

        lines.extend([
            "## 5. Rekomendacje Decyzyjne dla Rady Nadzorczej",
            "",
        ])
        for rec in packet.actionable_board_recommendations:
            lines.append(f"1. {rec}")
        lines.append("")

        lines.extend([
            "---",
            "### 🔒 Poświadczenie Integralności Kryptograficznej (Merkle DAG)",
            f"- **Kwit Rejestru Merkle:** `{packet.merkle_receipt_id or 'Brak'}`",
            f"- **Merkle Root:** `{packet.merkle_root or 'Brak'}`",
            "- Raport posiada niezaprzeczalną moc dowodową w postępowaniach przed organami nadzoru rynku.",
        ])

        return "\n".join(lines)
