# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Institutional Knowledge Base & Precedent Database (Gap 5.3).

Maintains a searchable, indexed repository of institutional ethics rulings, waivers,
appeals, and legal escalations:
- **Consistency of Rulings (Stare Decisis):** Ensures similar ethical dilemmas are treated uniformly
  across different departments, agencies, and review officers.
- **Precedent Matching:** Attribute and semantic keyword retrieval of binding institutional doctrine.
- **Inconsistency Alerts:** Flags when a proposed reviewer verdict contradicts established precedent.
- **Cryptographic Sealing:** Every indexed precedent is committed to MerkleLedger.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.governance.precedent_database")


# ============== Enums & Value Types ==============


class PrecedentBindingLevel(str, Enum):
    """Authority and weight of the precedent."""
    BINDING_SOVEREIGN = "BINDING_SOVEREIGN"       # Binding across entire institution / ministry
    DEPARTMENTAL_POLICY = "DEPARTMENTAL_POLICY"   # Binding within department
    PERSUASIVE_GUIDANCE = "PERSUASIVE_GUIDANCE"   # Advisory recommendation


class PrecedentRuling(str, Enum):
    """Final disposition rendered in the precedent."""
    CATEGORICAL_PROHIBITION = "PROHIBITED"
    CONDITIONAL_CLEARANCE = "CONDITIONAL_CLEARANCE"
    UNCONDITIONAL_APPROVAL = "APPROVED"


# ============== Data Models ==============


class PrecedentCase(BaseModel):
    """Institutional case law record of an ethical or legal adjudication."""
    case_id: str = Field(default_factory=lambda: f"PREC-{uuid.uuid4().hex[:8].upper()}")
    title: str
    domain: str = Field(..., description="e.g. DUAL_USE_INFRASTRUCTURE, CITIZEN_APPEAL_KPA, MEDICAL_ESTOP")
    applicable_doctrine: str = Field(..., description="e.g. Geneva AP I Art. 56, KPA Art. 10, EU AI Act Art. 14")
    factual_summary: str
    ruling: PrecedentRuling
    legal_rationale: str
    mandated_safeguards: List[str] = Field(default_factory=list)
    tags: Set[str] = Field(default_factory=set)
    binding_level: PrecedentBindingLevel = PrecedentBindingLevel.BINDING_SOVEREIGN
    deciding_authority: str = "Central Ethics & Legal Review Board"
    decided_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merkle_receipt_id: Optional[str] = None


class PrecedentMatch(BaseModel):
    """Query match with similarity score."""
    case: PrecedentCase
    relevance_score: float = Field(ge=0.0, le=1.0)


# ============== Engine Class ==============


class PrecedentDatabase:
    """Indexed Institutional Knowledge Base for AI Ethics Precedents."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self._cases: Dict[str, PrecedentCase] = {}
        self._seed_default_precedents()

    def _seed_default_precedents(self) -> None:
        """Seeds foundation institutional precedents."""
        case_1 = PrecedentCase(
            case_id="PREC-IHL-DAM-01",
            title="Ochrona zapory wodnej przed uderzeniem kinetycznym o podwójnym przeznaczeniu",
            domain="DUAL_USE_INFRASTRUCTURE",
            applicable_doctrine="Geneva Conventions AP I Art. 56",
            factual_summary="Zgłoszono wniosek o atak na most technologiczny położony na koronie zapory hydroelektrycznej.",
            ruling=PrecedentRuling.CATEGORICAL_PROHIBITION,
            legal_rationale=(
                "Nawet jeśli most stanowi cel wojskowy, Art. 56 AP I bezwzględnie zabrania ataków, "
                "które mogą spowodować wyzwolenie niebezpiecznych sił (zalanie doliny). Zakaz ma charakter ius cogens."
            ),
            mandated_safeguards=["Całkowity zakaz użycia efektorów kinetycznych w promieniu 2000m."],
            tags={"IHL", "DAMS", "DANGEROUS_FORCES", "GENEVA"},
            binding_level=PrecedentBindingLevel.BINDING_SOVEREIGN,
            deciding_authority="Wojskowy Zespół Doradców Prawnych (LEGAD)",
        )
        self._cases[case_1.case_id] = case_1

    def register_precedent(self, case: PrecedentCase) -> str:
        """Enrolls a new precedent and seals it in MerkleLedger."""
        self._cases[case.case_id] = case
        self._seal_precedent(case)
        logger.info("Zarejestrowano precedens instytucjonalny: %s (%s)", case.title, case.ruling.value)
        return case.case_id

    def find_matching_precedents(
        self,
        domain: str,
        query_text: str,
        tags: Optional[Set[str]] = None,
    ) -> List[PrecedentMatch]:
        """Finds applicable precedents matching domain, keywords, and tags."""
        matches: List[PrecedentMatch] = []
        q_tokens = set(query_text.lower().split())

        for case in self._cases.values():
            score = 0.0
            # Domain match weight
            if case.domain.upper() == domain.upper():
                score += 0.5

            # Tag overlap weight
            if tags and case.tags:
                common_tags = tags.intersection(case.tags)
                score += (len(common_tags) / max(len(tags), 1)) * 0.3

            # Keyword overlap in title and summary
            text_corpus = (case.title + " " + case.factual_summary + " " + case.legal_rationale).lower()
            token_hits = sum(1 for t in q_tokens if t in text_corpus)
            if q_tokens:
                score += min(0.3, (token_hits / len(q_tokens)) * 0.3)

            if score >= 0.4:
                matches.append(PrecedentMatch(case=case, relevance_score=round(score, 2)))

        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        return matches

    def check_ruling_consistency(
        self,
        domain: str,
        proposed_ruling: PrecedentRuling,
        context_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """Verifies if a proposed verdict conflicts with established binding precedents."""
        matches = self.find_matching_precedents(domain=domain, query_text=context_text)
        for m in matches:
            if m.case.binding_level == PrecedentBindingLevel.BINDING_SOVEREIGN and m.relevance_score >= 0.5:
                if m.case.ruling != proposed_ruling:
                    conflict_warning = (
                        f"SPRZECZNOŚĆ Z PRECEDENSEM! Proponowane orzeczenie '{proposed_ruling.value}' "
                        f"jest sprzeczne z wiążącym precedensem {m.case.case_id} ('{m.case.title}'), "
                        f"gdzie orzeczono: '{m.case.ruling.value}'. Uzasadnienie: {m.case.legal_rationale}"
                    )
                    return False, conflict_warning
        return True, None

    def _seal_precedent(self, case: PrecedentCase) -> None:
        """Kryptograficzne pieczętowanie precedensu w MerkleLedger."""
        try:
            payload = {
                "event_type": "PRECEDENT_CASE_ENROLLED",
                "case_id": case.case_id,
                "title": case.title,
                "domain": case.domain,
                "ruling": case.ruling.value,
                "binding_level": case.binding_level.value,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"PRECEDENS PRAWNY: {case.case_id} ({case.title}) -> {case.ruling.value}.",
            )
            case.merkle_receipt_id = receipt.receipt_id
            logger.info("Zapieczętowano precedens %s w MerkleLedger (Receipt: %s)", case.case_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania precedensu w MerkleLedger: %s", e)
