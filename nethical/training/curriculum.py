# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Institutional Onboarding & Certification Curriculum (Gap 6.1).

Implements structured human training and qualification pipelines mandated by:
- **ISO/IEC 42001 §7.2:** Competence and training of personnel operating AI systems.
- **EU AI Act Art. 4:** AI Literacy obligations for deployers and staff.
- **NATO STANAG & National Security:** Qualification of officers authorized for kinetic gates.

Certification Tiers:
1. **Awareness (Poziom 1):** General AI literacy, ethical principles, whistleblowing channels.
2. **Practitioner (Poziom 2):** Daily operations, review queues, DoAM authority level execution.
3. **Expert (Poziom 3):** Gate policy authoring, post-quantum key custody, IHL proportionality checks.
4. **Lead Auditor (Poziom 4):** Formal statutory auditing, Merkle-DAG chain verification, ISO 42001 certification.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.training.curriculum")


# ============== Enums & Value Types ==============


class CertificationTier(str, Enum):
    """Institutional qualification and competency levels."""
    TIER_1_AWARENESS = "TIER_1_AWARENESS"         # Podstawowa świadomość i etyka AI
    TIER_2_PRACTITIONER = "TIER_2_PRACTITIONER"   # Operator kolejki przeglądów i DoAM
    TIER_3_EXPERT = "TIER_3_EXPERT"               # Konfigurator bramek i oficer autoryzacji
    TIER_4_LEAD_AUDITOR = "TIER_4_LEAD_AUDITOR"   # Weryfikator formalny i audytor Merkle-DAG


# ============== Data Models ==============


class QuizQuestion(BaseModel):
    """Individual multiple-choice knowledge evaluation question."""
    question_id: str
    question_text: str
    options: List[str]
    correct_option_index: int
    explanation: str


class CurriculumModule(BaseModel):
    """Structured training module and syllabus."""
    module_id: str
    title: str
    tier: CertificationTier
    estimated_hours: int
    learning_objectives: List[str]
    syllabus_sections: List[str]
    questions: List[QuizQuestion] = Field(default_factory=list)


class CandidateAssessmentResult(BaseModel):
    """Official examination record with verifiable digital certificate."""
    assessment_id: str = Field(default_factory=lambda: f"EXAM-{uuid.uuid4().hex[:8].upper()}")
    candidate_id: str
    candidate_name: str
    tier: CertificationTier
    score_pct: float
    passed: bool
    passing_threshold_pct: float = 80.0
    certified_until: Optional[str] = None
    certificate_id: Optional[str] = None
    merkle_receipt_id: Optional[str] = None
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ============== Curriculum Engine Class ==============


class InstitutionalCurriculumManager:
    """Manages the lifecycle of institutional human qualifications and exams."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self._modules: Dict[str, CurriculumModule] = {}
        self._seed_standard_curriculum()

    def _seed_standard_curriculum(self) -> None:
        """Seeds the standard 4-tier institutional curriculum."""
        # Tier 1: Awareness
        mod_1 = CurriculumModule(
            module_id="MOD-AI-LITERACY-01",
            title="Podstawy Etyki i Prawa Sztucznej Inteligencji (EU AI Act & KPA)",
            tier=CertificationTier.TIER_1_AWARENESS,
            estimated_hours=4,
            learning_objectives=[
                "Zrozumienie wymogów art. 4 EU AI Act w zakresie AI Literacy.",
                "Identyfikacja zakazanych praktyk AI (Art. 5 EU AI Act).",
                "Procedura zgłaszania nieprawidłowości (Sygnalista).",
            ],
            syllabus_sections=[
                "1. Wprowadzenie do suwerenności algorytmicznej",
                "2. Klasyfikacja ryzyk wg EU AI Act",
                "3. Etyka, stronniczość i prawa podstawowe",
            ],
            questions=[
                QuizQuestion(
                    question_id="Q1-1",
                    question_text="Która z poniższych praktyk jest bezwzględnie zakazana przez art. 5 EU AI Act?",
                    options=[
                        "Wykorzystanie AI do optymalizacji podatkowej.",
                        "Stosowanie scoringu społecznego (Social Scoring) przez władze publiczne.",
                        "Automatyczna moderacja treści na forach.",
                        "Wykrywanie spamu w poczcie elektronicznej.",
                    ],
                    correct_option_index=1,
                    explanation="Social scoring jest zakazaną praktyką AI o niedopuszczalnym ryzyku.",
                ),
            ],
        )
        self._modules[mod_1.module_id] = mod_1

        # Tier 3: Expert (NTSG & Two-Man Rule)
        mod_3 = CurriculumModule(
            module_id="MOD-NTSG-EXPERT-03",
            title="Autoryzacja Bramek Taktycznych NTSG i Międzynarodowe Prawo Humanitarne (IHL)",
            tier=CertificationTier.TIER_3_EXPERT,
            estimated_hours=16,
            learning_objectives=[
                "Procedura fizycznej autoryzacji Two-Man Rule z użyciem tokenów PKCS#11 HSM.",
                "Bezwzględne zakazy ataku na tamy i elektrownie jądrowe (Art. 56 AP I).",
                "Postępowanie w trybie odcięcia łączności (Air-Gap Severance).",
            ],
            syllabus_sections=[
                "1. Protokół Dodatkowy I do Konwencji Genewskich (Art. 48-57)",
                "2. Kryptografia Postkwantowa ML-DSA-65 (FIPS 204)",
                "3. Rozdział Obowiązków (SoD) i ochrona przed zmęczeniem poznawczym",
            ],
            questions=[
                QuizQuestion(
                    question_id="Q3-1",
                    question_text="Dlaczego w procedurze Two-Man Rule odrzucono algorytm Shamir's Secret Sharing na rzecz 2 niezależnych podpisów ML-DSA?",
                    options=[
                        "Bo Shamir jest wolniejszy obliczeniowo o rząd wielkości.",
                        "Bo Shamir wymaga rekonstrukcji klucza w jednej pamięci, co niszczy fizyczną separację oficerów.",
                        "Bo Shamir nie wspiera liczb pierwszych powyżej 256 bitów.",
                        "Bo Shamir jest zakazany przez normę ISO 27001.",
                    ],
                    correct_option_index=1,
                    explanation="Rekonstrukcja sekretu Shamira w jednym kontrolerze tworzy pojedynczy punkt awarii.",
                ),
            ],
        )
        self._modules[mod_3.module_id] = mod_3

    def get_module(self, module_id: str) -> Optional[CurriculumModule]:
        """Retrieves a curriculum module by identifier."""
        return self._modules.get(module_id)

    def evaluate_exam(
        self,
        module_id: str,
        candidate_id: str,
        candidate_name: str,
        selected_answers: List[int],
    ) -> CandidateAssessmentResult:
        """Evaluates candidate answers, computes percentage score, and issues certificate if passed."""
        module = self._modules.get(module_id)
        if not module or not module.questions:
            raise ValueError(f"Nieprawidłowy lub pusty moduł: {module_id}")

        total_q = len(module.questions)
        correct_count = 0

        for idx, q in enumerate(module.questions):
            if idx < len(selected_answers) and selected_answers[idx] == q.correct_option_index:
                correct_count += 1

        score_pct = round((correct_count / total_q) * 100.0, 1)
        passed = score_pct >= 80.0

        cert_id = None
        cert_until = None
        if passed:
            cert_id = f"CERT-{module.tier.value[:6]}-{uuid.uuid4().hex[:8].upper()}"
            # 2-year certification validity
            cert_until = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year + 2).strftime("%Y-%m-%d")

        result = CandidateAssessmentResult(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            tier=module.tier,
            score_pct=score_pct,
            passed=passed,
            certified_until=cert_until,
            certificate_id=cert_id,
        )

        if passed:
            self._seal_certificate(result, module)

        return result

    def _seal_certificate(
        self,
        result: CandidateAssessmentResult,
        module: CurriculumModule,
    ) -> None:
        """Kryptograficzne pieczętowanie dyplomu kwalifikacyjnego w MerkleLedger."""
        try:
            payload = {
                "event_type": "HUMAN_QUALIFICATION_CERTIFICATE_ISSUED",
                "certificate_id": result.certificate_id,
                "candidate_id": result.candidate_id,
                "candidate_name": result.candidate_name,
                "tier": result.tier.value,
                "module_id": module.module_id,
                "score": result.score_pct,
                "valid_until": result.certified_until,
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"CERTYFIKACJA PERSONELU: {result.candidate_name} -> {result.tier.value} (Cert: {result.certificate_id}).",
            )
            result.merkle_receipt_id = receipt.receipt_id
            logger.info("Zapieczętowano certyfikat personelu %s w MerkleLedger (Receipt: %s)", result.certificate_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania certyfikatu w MerkleLedger: %s", e)
