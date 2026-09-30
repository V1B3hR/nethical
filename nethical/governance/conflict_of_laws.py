# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Multi-Jurisdictional Conflict-of-Laws Resolution Engine (Gap 7.1).

Adjudicates direct statutory contradictions in cross-border AI operations:
- **GDPR Art. 17 (Right to Erasure) vs. DORA / SEC / SOX 404 (Mandatory Audit Preservation)**
- **US CLOUD Act Extraterritorial Discovery vs. EU GDPR Art. 48 & Schrems II Restraints**
- **Sovereign Defense & IHL Jus Cogens vs. Commercial Contractual Terms**

Jurisprudential Resolution Principles:
1. *Lex superior derogat legi inferiori:* Constitutional & Jus Cogens norms trump statutes;
   statutes trump administrative orders; regulations trump contracts.
2. *Lex specialis derogat legi generali:* Specific sector law overrides general administrative rules.
3. *Jus Cogens Primacy:* International Humanitarian Law (Geneva AP I) and fundamental rights
   cannot be derogated by secondary economic or civil statutes.
4. *Bilateral Treaty Safe Harbor (MLAT):* Requiring formal Mutual Legal Assistance Treaties
   for cross-border law enforcement disclosures.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.governance.conflict_of_laws")


# ============== Enums & Value Types ==============


class NormativeHierarchy(IntEnum):
    """Hierarchy of legal norms in international and constitutional law."""
    LEVEL_1_JUS_COGENS_AND_CONSTITUTIONAL = 1   # IHL, Core Fundamental Rights, National Sovereignty
    LEVEL_2_STATUTORY_NATIONAL_SECURITY = 2     # Defense, Critical Infrastructure, Martial Law
    LEVEL_3_PRIMARY_STATUTE = 3                 # GDPR, EU AI Act, US Federal Acts, National Administrative Codes
    LEVEL_4_SECTORAL_REGULATION = 4             # DORA RTS, KNF Guidelines, SEC/FINRA rules
    LEVEL_5_CONTRACTUAL_AND_TOS = 5             # Vendor contracts, API Terms of Service, EULAs


class MandateAction(str, Enum):
    """Specific operational behavior mandated or prohibited by a legal rule."""
    MANDATORY_PRESERVE = "MANDATORY_PRESERVE"               # Preserve immutable record (e.g. DORA / SEC)
    MANDATORY_ERASE = "MANDATORY_ERASE"                     # Obligation to delete personal data (GDPR Art. 17)
    EXTRATERRITORIAL_TRANSFER = "EXTRATERRITORIAL_TRANSFER" # Compelled foreign subpoena disclosure (CLOUD Act)
    PROHIBIT_FOREIGN_TRANSFER = "PROHIBIT_FOREIGN_TRANSFER" # Prohibition of unauthorized disclosure (GDPR Art. 48)
    HUMAN_IN_THE_LOOP_OVERRIDE = "HITL_OVERRIDE"            # Absolute human control mandate (EU AI Act Art. 14)


class ResolutionPrinciple(str, Enum):
    """Doctrinal legal maxims applied to resolve collisions."""
    LEX_SUPERIOR = "LEX_SUPERIOR_DEROGAT_LEGI_INFERIORI"       # Superior norm prevails
    LEX_SPECIALIS = "LEX_SPECIALIS_DEROGAT_LEGI_GENERALI"     # Specific sectoral rule prevails
    JUS_COGENS_PRIMACY = "JUS_COGENS_ABSOLUTE_PRIMACY"        # Non-derogable human rights / IHL prevail
    MLAT_TREATY_SAFEGUARD = "MLAT_TREATY_SAFEGUARD"           # Judicial mutual legal assistance requirement


# ============== Data Models ==============


class LegalObligation(BaseModel):
    """Statutory or regulatory obligation active in a jurisdiction."""
    rule_id: str
    jurisdiction: str = Field(..., description="e.g. EU, USA, POL, NATO, GLOBAL")
    statute_name: str
    citation: str
    hierarchy: NormativeHierarchy
    mandate: MandateAction
    is_criminal_sanction_attached: bool = False
    description: str


class ConflictAdjudicationResult(BaseModel):
    """Binding formal adjudication resolving a conflict of statutory mandates."""
    adjudication_id: str = Field(default_factory=lambda: f"ADJ-{uuid.uuid4().hex[:8].upper()}")
    collision_topic: str
    prevailing_obligation: LegalObligation
    overridden_obligation: LegalObligation
    principle_applied: ResolutionPrinciple
    legal_balancing_rationale: str
    mitigating_safeguards: List[str] = Field(default_factory=list)
    adjudicated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merkle_receipt_id: Optional[str] = None
    merkle_root: Optional[str] = None


# ============== Engine Class ==============


class ConflictOfLawsEngine:
    """Formal Legal Conflict Resolution Engine for Multinational Operations."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()

    def adjudicate_conflict(
        self,
        obligation_a: LegalObligation,
        obligation_b: LegalObligation,
        collision_topic: str,
    ) -> ConflictAdjudicationResult:
        """Resolves a direct conflict between two competing legal obligations."""
        logger.info(
            "Rozstrzyganie kolizji prawnej: %s (%s) vs %s (%s)",
            obligation_a.rule_id, obligation_a.jurisdiction,
            obligation_b.rule_id, obligation_b.jurisdiction
        )

        # 1. Test Normative Hierarchy (Lex Superior)
        if obligation_a.hierarchy != obligation_b.hierarchy:
            if obligation_a.hierarchy < obligation_b.hierarchy:
                prevailing = obligation_a
                overridden = obligation_b
            else:
                prevailing = obligation_b
                overridden = obligation_a

            if prevailing.hierarchy == NormativeHierarchy.LEVEL_1_JUS_COGENS_AND_CONSTITUTIONAL:
                principle = ResolutionPrinciple.JUS_COGENS_PRIMACY
                rationale = (
                    f"Zasada prymatu norm ius cogens: {prevailing.statute_name} ({prevailing.citation}) "
                    f"posiada status bezwzględnie wiążący w prawie międzynarodowym i uchyla normy niższego rzędu."
                )
            else:
                principle = ResolutionPrinciple.LEX_SUPERIOR
                rationale = (
                    f"Zasada Lex Superior: Norma wyższego rzędu ({prevailing.hierarchy.name} - {prevailing.statute_name}) "
                    f"uchyla normę niższego rzędu ({overridden.hierarchy.name} - {overridden.statute_name})."
                )

        else:
            # 2. Equal Hierarchy: Specific Scenarios
            # Case 1: GDPR Art. 17 (Erase) vs DORA/SEC (Preserve logs)
            mandates = {obligation_a.mandate, obligation_b.mandate}
            if MandateAction.MANDATORY_ERASE in mandates and MandateAction.MANDATORY_PRESERVE in mandates:
                erase_rule = obligation_a if obligation_a.mandate == MandateAction.MANDATORY_ERASE else obligation_b
                preserve_rule = obligation_a if obligation_a.mandate == MandateAction.MANDATORY_PRESERVE else obligation_b

                # DORA / Regulatory preservation trumps deletion due to GDPR Art. 17(3)(b) statutory exception!
                prevailing = preserve_rule
                overridden = erase_rule
                principle = ResolutionPrinciple.LEX_SPECIALIS
                rationale = (
                    f"Zastosowanie wyjątku statutowego: Zgodnie z art. 17 ust. 3 lit. b RODO, prawo do usunięcia danych "
                    f"nie ma zastosowania, gdy przetwarzanie jest niezbędne do wywiązania się z prawnego obowiązku "
                    f"retencji logów ({preserve_rule.statute_name} / {preserve_rule.citation})."
                )

            # Case 2: US CLOUD Act vs GDPR Art. 48 (Extraterritorial subpoena)
            elif MandateAction.EXTRATERRITORIAL_TRANSFER in mandates and MandateAction.PROHIBIT_FOREIGN_TRANSFER in mandates:
                transfer_rule = obligation_a if obligation_a.mandate == MandateAction.EXTRATERRITORIAL_TRANSFER else obligation_b
                prohibit_rule = obligation_a if obligation_a.mandate == MandateAction.PROHIBIT_FOREIGN_TRANSFER else obligation_b

                prevailing = prohibit_rule
                overridden = transfer_rule
                principle = ResolutionPrinciple.MLAT_TREATY_SAFEGUARD
                rationale = (
                    f"Klauzula suwerenności jurysdykcyjnej (art. 48 RODO / Schrems II): Wszelkie orzeczenia sądów państw trzecich "
                    f"wymagające eksterytorialnego transferu danych podlegają uznaniu wyłącznie na mocy umów międzynarodowych (MLAT)."
                )
            else:
                # Default tie-breaker: criminal sanction attachment or sovereign origin
                if obligation_a.is_criminal_sanction_attached and not obligation_b.is_criminal_sanction_attached:
                    prevailing = obligation_a
                    overridden = obligation_b
                else:
                    prevailing = obligation_b
                    overridden = obligation_a
                principle = ResolutionPrinciple.LEX_SPECIALIS
                rationale = f"Rozstrzygnięcie na korzyść normy chronionej sankcją surowszą lub o charakterze lex specialis."

        # Define mitigating safeguards
        safeguards: List[str] = [
            f"Kryptograficzna anonimizacja danych przed retencją na potrzeby {prevailing.statute_name}.",
            "Powiadomienie Inspektora Ochrony Danych (DPO) oraz Radcy Prawnego o zastosowaniu wyjątku kolizyjnego.",
            "Niezaprzeczalne zapieczętowanie rozstrzygnięcia kolizji w MerkleLedger.",
        ]

        result = ConflictAdjudicationResult(
            collision_topic=collision_topic,
            prevailing_obligation=prevailing,
            overridden_obligation=overridden,
            principle_applied=principle,
            legal_balancing_rationale=rationale,
            mitigating_safeguards=safeguards,
        )

        self._seal_adjudication(result)
        return result

    def _seal_adjudication(self, result: ConflictAdjudicationResult) -> None:
        """Kryptograficzne pieczętowanie orzeczenia kolizyjnego w MerkleLedger."""
        try:
            payload = {
                "event_type": "CONFLICT_OF_LAWS_ADJUDICATION_ISSUED",
                "adjudication_id": result.adjudication_id,
                "topic": result.collision_topic,
                "prevailing_rule": result.prevailing_obligation.rule_id,
                "prevailing_statute": result.prevailing_obligation.statute_name,
                "overridden_rule": result.overridden_obligation.rule_id,
                "principle": result.principle_applied.value,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"KOLIZJA PRAWNA: {result.adjudication_id} ({result.collision_topic}) -> Wygrywa: {result.prevailing_obligation.rule_id}.",
            )
            result.merkle_receipt_id = receipt.receipt_id
            result.merkle_root = self.ledger.current_root
            logger.info("Zapieczętowano orzeczenie kolizji prawnej %s w MerkleLedger (Receipt: %s)", result.adjudication_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania rozstrzygnięcia kolizyjnego w MerkleLedger: %s", e)
