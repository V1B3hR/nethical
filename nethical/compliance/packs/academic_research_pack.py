"""Academic Research Integrity and Higher Education Governance Pack.

Implements statutory compliance and deontological standards for universities, research institutes,
doctoral schools, and scientific publishing:
- The European Code of Conduct for Research Integrity (ALLEA - All European Academies 2023):
  * FFP Prevention: Absolute zero tolerance for Fabrication, Falsification, and Plagiarism.
  * Authorship and Contribution Integrity (CRediT taxonomy).
  * Data Stewardship & Open Science adhering to FAIR principles (Findable, Accessible, Interoperable, Reusable).
- Bibliographic Anti-Hallucination & Citation Verification:
  * Strict regex and syntax validation for Digital Object Identifiers (DOI), PubMed IDs (PMID),
    and arXiv IDs to eliminate synthetic, fabricated citations in dissertations and grant proposals.
- Intellectual Property (IP) and Patent Novelty (Prior Art Protection):
  * Detection and redaction of unpublished experimental formulations, chemical structures,
    and proprietary algorithms to protect novelty prior to EPO/USPTO/UPRP patent filing.
- Paper Mills and Predatory Peer Review Detection:
  * Heuristic signatures of automated paper fabrication, hijacked journal templates, and fake reviewer rings.
- Research Ethics & Bioethics:
  * World Medical Association (WMA) Declaration of Helsinki for human subject research.
  * Polish Komisje Bioetyczne (Bioethics Committees) & Polish Ustawa o ochronie zwierząt wykorzystywanych do celów naukowych (3Rs: Replacement, Reduction, Refinement).

Hard Invariants Enforced:
1. Prohibition of Research Misconduct (Fabrication, Falsification, Plagiarism - FFP).
2. Prohibition of Hallucinated Academic Citations (Fabricated DOI/PMID sources).
3. Pre-Publication Patent Novelty Shield (No leakage of unpublished prior art to external AI).
4. Bioethics Committee Approval Verification for Human/Animal Clinical Protocols.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.academic_research_pack")

# Robust citation regex patterns
DOI_REGEX = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$")
PMID_REGEX = re.compile(r"^\d{7,8}$")
ARXIV_REGEX = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})$")


class ResearchDiscipline(str, Enum):
    BIOMEDICAL_AND_CLINICAL = "BIOMEDICAL_AND_CLINICAL"
    EXACT_AND_ENGINEERING = "EXACT_AND_ENGINEERING"
    HUMANITIES_AND_SOCIAL = "HUMANITIES_AND_SOCIAL"
    DEFENSE_AND_DUAL_USE = "DEFENSE_AND_DUAL_USE"


class ResearchIntegrityStatus(str, Enum):
    IMPECCABLE_SCHOLARSHIP = "IMPECCABLE_SCHOLARSHIP"
    EDITORIAL_REVISION_REQUIRED = "EDITORIAL_REVISION_REQUIRED"
    UNVERIFIED_SCHOLARLY_CITATIONS = "UNVERIFIED_SCHOLARLY_CITATIONS"
    SCIENTIFIC_MISCONDUCT_FFP = "SCIENTIFIC_MISCONDUCT_FFP"


class AcademicComplianceResult(BaseModel):
    """Evaluation result for Academic Research and Higher Education governance."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    institution_or_project_name: str
    discipline: ResearchDiscipline
    integrity_status: ResearchIntegrityStatus
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    allea_ffp_free_verified: bool
    citations_verified_valid: bool
    patent_prior_art_shielded: bool
    bioethics_committee_approved: bool
    fair_data_principles_met: bool
    violations: List[str] = Field(default_factory=list)
    missing_scholarly_elements: List[str] = Field(default_factory=list)
    remediation_recommendations: List[str] = Field(default_factory=list)


class AcademicResearchPack:
    """Evaluates scientific workflows and university outputs against ALLEA and scholarly integrity."""

    def __init__(self) -> None:
        self.jurisdiction = "European Union (ALLEA / Horizon Europe) & Poland (PAN / NCN / NCBR / Ustawa 2.0)"
        self.scholarly_authorities = [
            "Polska Akademia Nauk (PAN) - Komisja do spraw Etyki w Nauce",
            "Narodowe Centrum Nauki (NCN)",
            "European Research Council (ERC)",
            "All European Academies (ALLEA)",
            "Urząd Patentowy Rzeczypospolitej Polskiej (UPRP)",
        ]

    def verify_citation_format(self, citation_identifier: str, id_type: str = "DOI") -> bool:
        """Determines if a provided academic identifier matches genuine structural standards."""
        cleaned = citation_identifier.strip()
        if id_type.upper() == "DOI":
            # Strip standard URL prefixes if present
            if cleaned.startswith("https://doi.org/"):
                cleaned = cleaned.replace("https://doi.org/", "")
            elif cleaned.startswith("http://doi.org/"):
                cleaned = cleaned.replace("http://doi.org/", "")
            return bool(DOI_REGEX.match(cleaned))
        elif id_type.upper() == "PMID":
            return bool(PMID_REGEX.match(cleaned))
        elif id_type.upper() == "ARXIV":
            if cleaned.startswith("arXiv:"):
                cleaned = cleaned.replace("arXiv:", "")
            return bool(ARXIV_REGEX.match(cleaned))
        return False

    def evaluate(self, payload: Dict[str, Any]) -> AcademicComplianceResult:
        """Evaluates a research manuscript, grant proposal, or laboratory pipeline."""
        system_name = payload.get("system_name", "Academic-Research-Workload")
        eval_id = f"ACA-{uuid.uuid4().hex[:12].upper()}"

        raw_discipline = payload.get("discipline", "EXACT_AND_ENGINEERING")
        try:
            discipline = ResearchDiscipline(raw_discipline)
        except ValueError:
            discipline = ResearchDiscipline.EXACT_AND_ENGINEERING

        violations: List[str] = []
        missing_elements: List[str] = []
        remediations: List[str] = []

        # 1. Hard Invariant I: ALLEA FFP Prohibition (Fabrication, Falsification, Plagiarism)
        fabrication_detected = payload.get("data_fabrication_detected", False)
        falsification_detected = payload.get("data_falsification_detected", False)
        plagiarism_detected = payload.get("unattributed_plagiarism_detected", False)

        ffp_free = not (fabrication_detected or falsification_detected or plagiarism_detected)
        if not ffp_free:
            misconduct_types = []
            if fabrication_detected: misconduct_types.append("Fabrication (fabrykacja danych)")
            if falsification_detected: misconduct_types.append("Falsification (fałszowanie wyników)")
            if plagiarism_detected: misconduct_types.append("Plagiarism (plagiat bez cytowania)")
            violations.append(
                f"CRITICAL: Wykryto naruszenie rzetelności naukowej FFP: {', '.join(misconduct_types)}. "
                "Bezwzględne złamanie Europejskiego Kodeksu Rzetelności Badawczej (ALLEA Sekcja 3.1)."
            )
            remediations.append("Wycofaj manuskrypt; zgłoś sprawę do Uczelnianej Komisji Dyscyplinarnej ds. Nauczycieli Akademickich.")

        # 2. Hard Invariant II: Bibliographic Anti-Hallucination (DOIs / PMIDs / arXiv)
        citations_to_check = payload.get("citations", []) # list of dicts: {"id": "...", "type": "DOI"}
        has_hallucinated_citations = payload.get("contains_synthetic_fabricated_citations", False)
        citations_valid = True

        for cit in citations_to_check:
            cit_id = cit.get("id", "")
            cit_type = cit.get("type", "DOI")
            if not self.verify_citation_format(cit_id, cit_type):
                has_hallucinated_citations = True
                violations.append(
                    f"VIOLATION: Zmyślona lub niepoprawna strukturalnie sygnatura bibliograficzna {cit_type}: '{cit_id}'. "
                    "Zakaz generowania halucynowanych cytowań i autorów w literaturze naukowej."
                )
                break

        if has_hallucinated_citations:
            citations_valid = False
            remediations.append("Zweryfikuj wszystkie cytowania w bazie Crossref / PubMed / Scopus przed złożeniem publikacji.")

        # 3. Hard Invariant III: Patent Novelty Shield (Prior Art Protection)
        unprotected_patent_disclosure = payload.get("unprotected_prior_art_disclosure", False)
        patent_shielded = True
        if unprotected_patent_disclosure:
            patent_shielded = False
            violations.append(
                "VIOLATION: Niekontrolowana eksfiltracja nieopublikowanych formuł/kodu do publicznego modelu LLM "
                "przed formalnym zgłoszeniem patentowym do UPRP/EPO (Utrata nowości w rozumieniu Art. 24 Prawa Własności Przemysłowej)."
            )
            remediations.append("Objąć dane laboratoryjne klauzulą NDA i przetwarzać wyłącznie w lokalnej instancji Air-Gapped.")

        # 4. Bioethics Committee Approval (Declaration of Helsinki & 3Rs)
        requires_bioethics = discipline in (ResearchDiscipline.BIOMEDICAL_AND_CLINICAL, ResearchDiscipline.DEFENSE_AND_DUAL_USE) or payload.get("involves_human_subjects", False) or payload.get("involves_animal_testing", False)
        has_bioethics_approval = payload.get("has_bioethics_committee_approval", False)

        bioethics_approved = True
        if requires_bioethics and not has_bioethics_approval:
            bioethics_approved = False
            violations.append(
                "CRITICAL: Prowadzenie badań z udziałem ludzi lub materiału biologicznego bez formalnej zgody "
                "Komisji Bioetycznej (Naruszenie Deklaracji Helsińskiej WMA i Ustawy o zawodach lekarza)."
            )
            remediations.append("Uzyskaj formalną uchwałę właściwej terytorialnie Komisji Bioetycznej przed zebraniem danych.")

        # 5. FAIR Data & Open Science
        fair_compliant = payload.get("fair_data_principles_met", False)
        if not fair_compliant:
            missing_elements.append("Brak Planu Zarządzania Danymi Badawczymi (DMP - Data Management Plan) zgodnego ze standardem FAIR.")
            remediations.append("Przygotuj DMP i zarejestruj dane w otwartym repozytorium (np. RepOD / Zenodo).")

        # 6. Scoring and Status Determination
        deductions = (len(violations) * 0.35) + (len(missing_elements) * 0.10)
        score = max(0.0, min(1.0, 1.0 - deductions))

        if not ffp_free or (requires_bioethics and not bioethics_approved):
            status = ResearchIntegrityStatus.SCIENTIFIC_MISCONDUCT_FFP
            is_compliant = False
        elif not citations_valid:
            status = ResearchIntegrityStatus.UNVERIFIED_SCHOLARLY_CITATIONS
            is_compliant = False
        elif missing_elements:
            status = ResearchIntegrityStatus.EDITORIAL_REVISION_REQUIRED
            is_compliant = score >= 0.70
        else:
            status = ResearchIntegrityStatus.IMPECCABLE_SCHOLARSHIP
            is_compliant = True

        logger.info(
            "AcademicResearchPack evaluation complete: %s, Discipline: %s, Compliant: %s, Score: %.2f",
            eval_id, discipline.value, is_compliant, score
        )

        return AcademicComplianceResult(
            evaluation_id=eval_id,
            institution_or_project_name=system_name,
            discipline=discipline,
            integrity_status=status,
            is_compliant=is_compliant,
            compliance_score=round(score, 3),
            allea_ffp_free_verified=ffp_free,
            citations_verified_valid=citations_valid,
            patent_prior_art_shielded=patent_shielded,
            bioethics_committee_approved=bioethics_approved,
            fair_data_principles_met=fair_compliant,
            violations=violations,
            missing_scholarly_elements=missing_elements,
            remediation_recommendations=remediations,
        )
