# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit & Integration Tests for Institutional Roadmap Completion.

Validates the full implementation of:
1. Gap 3.1 & 3.6: MLS Compartments & OPSEC Classification Guard (mls_compartments.py)
2. Gap 7.5: Sanctions & Export Control Screening (sanctions_screening.py)
3. Gap 7.1: Multi-Jurisdictional Conflict-of-Laws Resolution (conflict_of_laws.py)
4. Gap 6.5: Legacy System Bridge & Retrofit Governance Proxy (retrofit_proxy.py)
5. Gap 6.1: Institutional Certification Curriculum (curriculum.py)
6. Gap 5.3: Precedent Database & Jurisprudential Case Law (precedent_database.py)
7. Gap 5.5: AI Governance KPI Framework (kpi_framework.py)
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.compliance.sanctions_screening import (
    CounterpartyProfile,
    SanctionsAndExportScreeningEngine,
    ScreeningVerdict,
)
from nethical.gateway.retrofit_proxy import (
    InboundRequestWrapper,
    LegacyRetrofitProxy,
    RetrofitPolicyMode,
    RetrofitVerdict,
)
from nethical.governance.conflict_of_laws import (
    ConflictOfLawsEngine,
    LegalObligation,
    MandateAction,
    NormativeHierarchy,
    ResolutionPrinciple,
)
from nethical.governance.kpi_framework import (
    AIGovernanceKPIEngine,
    PairedReviewRating,
    ReviewTicketMetric,
)
from nethical.governance.precedent_database import (
    PrecedentCase,
    PrecedentDatabase,
    PrecedentRuling,
)
from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.mls_compartments import (
    CompartmentCodeword,
    DocumentObjectClassification,
    MLSSecurityGuard,
    OPSECCategory,
    SecurityClearanceLevel,
    SubjectClearance,
)
from nethical.training.curriculum import (
    CertificationTier,
    InstitutionalCurriculumManager,
)


class TestInstitutionalRoadmapCompletion(unittest.TestCase):
    """Zestaw testów weryfikujący ostateczne domknięcie roadmapy instytucjonalnej."""

    def setUp(self) -> None:
        """Inicjalizacja rejestru Merkle i wszystkich silników nadzorczych."""
        self.ledger = MerkleLedger()
        self.mls_guard = MLSSecurityGuard(ledger=self.ledger, host_nation="POL")
        self.sanctions_engine = SanctionsAndExportScreeningEngine(ledger=self.ledger)
        self.conflict_engine = ConflictOfLawsEngine(ledger=self.ledger)
        self.curriculum_mgr = InstitutionalCurriculumManager(ledger=self.ledger)
        self.precedent_db = PrecedentDatabase(ledger=self.ledger)
        self.kpi_engine = AIGovernanceKPIEngine(ledger=self.ledger)

    # ------------------------------------------------------------------------
    # 1. Testy Gap 3.1 & 3.6: MLS Compartments & OPSEC Guard
    # ------------------------------------------------------------------------

    def test_01_mls_mandatory_access_control(self) -> None:
        """Weryfikuje reguły Bell-LaPadula (No-Read-Up, No-Write-Down), kompartmenty i klauzulę NOFORN."""
        officer_restricted = SubjectClearance(
            subject_id="OFFICER-LOW",
            clearance_level=SecurityClearanceLevel.RESTRICTED,
            nationality="POL",
        )
        officer_top_secret = SubjectClearance(
            subject_id="OFFICER-HIGH",
            clearance_level=SecurityClearanceLevel.TOP_SECRET,
            compartments={CompartmentCodeword.BOHEMIA.value},
            nationality="POL",
        )
        allied_officer = SubjectClearance(
            subject_id="OFFICER-USA",
            clearance_level=SecurityClearanceLevel.SECRET,
            compartments={CompartmentCodeword.BOHEMIA.value},
            nationality="USA",
        )

        doc_classified = DocumentObjectClassification(
            object_id="DOC-TACTICAL-ORDERS-01",
            classification_level=SecurityClearanceLevel.SECRET,
            required_compartments={CompartmentCodeword.BOHEMIA.value},
            is_noforn=True,
        )

        # 1. Odmowa odczytu dla niższego poświadczenia (No-Read-Up)
        res_read_low = self.mls_guard.evaluate_read_access(officer_restricted, doc_classified)
        self.assertFalse(res_read_low.allowed)
        self.assertEqual(res_read_low.rule_applied, "BELL_LAPADULA_NO_READ_UP")

        # 2. Odmowa dla obcego oficera z powodu klauzuli NOFORN
        res_read_allied = self.mls_guard.evaluate_read_access(allied_officer, doc_classified)
        self.assertFalse(res_read_allied.allowed)
        self.assertEqual(res_read_allied.rule_applied, "NOFORN_NATIONALITY_RESTRICTION")

        # 3. Zgoda dla polskiego oficera z poświadczeniem TOP SECRET i kompartmentem BOHEMIA
        res_read_high = self.mls_guard.evaluate_read_access(officer_top_secret, doc_classified)
        self.assertTrue(res_read_high.allowed)

        # 4. Odmowa zapisu z wyższego poziomu do niższego (No-Write-Down)
        doc_unclassified = DocumentObjectClassification(
            object_id="DOC-PUBLIC-MEMO",
            classification_level=SecurityClearanceLevel.UNCLASSIFIED,
        )
        res_write_down = self.mls_guard.evaluate_write_access(officer_top_secret, doc_unclassified)
        self.assertFalse(res_write_down.allowed)
        self.assertEqual(res_write_down.rule_applied, "BELL_LAPADULA_NO_WRITE_DOWN")

    def test_02_opsec_classification_guard_redaction(self) -> None:
        """Weryfikuje automatyczne wykrywanie i cenzurę tajemnic operacyjnych (OPSEC)."""
        leaky_text = (
            "Dywizja zmechanizowana melduje: dyslokacja 16. dywizja w rejon celu "
            "54° 22' N, 18° 38' E. Znaki wywoławcze pododdziału: ORZEŁ-01. "
            "Zapas rakiet: 120."
        )

        inspection = self.mls_guard.inspect_and_sanitize_opsec(leaky_text, redact=True)

        self.assertFalse(inspection.is_clean)
        self.assertGreaterEqual(len(inspection.findings), 3)
        self.assertIn("[OPSEC-ZASŁONIĘTO-TROOP_MOVEMENT]", inspection.sanitized_text)
        self.assertIn("[OPSEC-ZASŁONIĘTO-GEOSPATIAL_COORDINATES]", inspection.sanitized_text)
        self.assertIn("[OPSEC-ZASŁONIĘTO-TACTICAL_CALLSIGNS]", inspection.sanitized_text)

        receipt_id = self.mls_guard.seal_opsec_event(inspection, operation_id="OP-TEST-77")
        self.assertIsNotNone(receipt_id)

    # ------------------------------------------------------------------------
    # 2. Testy Gap 7.5: Sanctions & Export Control Screening
    # ------------------------------------------------------------------------

    def test_03_sanctions_and_dual_use_export_screening(self) -> None:
        """Weryfikuje blokowanie podmiotów z list SDN, terytoriów objętych embargiem i technologii podwójnego zastosowania."""
        # 1. Podmiot z listy SDN
        cp_sanctioned = CounterpartyProfile(
            entity_name="Glavset Autonomous Research Center",
            country_code="RU",
        )
        res_sdn = self.sanctions_engine.screen_counterparty_and_export(cp_sanctioned)
        self.assertEqual(res_sdn.verdict, ScreeningVerdict.STRICT_SANCTION_BLOCK)
        self.assertIsNotNone(res_sdn.merkle_receipt_id)

        # 2. Podmiot z kraju objętego embargiem (np. Korea Północna)
        cp_embargo = CounterpartyProfile(
            entity_name="Pyongyang General Technology Hub",
            country_code="KP",
        )
        res_embargo = self.sanctions_engine.screen_counterparty_and_export(cp_embargo)
        self.assertEqual(res_embargo.verdict, ScreeningVerdict.STRICT_SANCTION_BLOCK)

        # 3. Legalny podmiot europejski eksportujący oprogramowanie podwójnego zastosowania (ECCN 5A002.a)
        cp_legal = CounterpartyProfile(
            entity_name="Polski Partner Obronny Sp. z o.o.",
            country_code="PL",
        )
        res_legal = self.sanctions_engine.screen_counterparty_and_export(
            counterparty=cp_legal,
            applicable_eccns=["5A002.a"],
        )
        self.assertEqual(res_legal.verdict, ScreeningVerdict.CLEARED)

    # ------------------------------------------------------------------------
    # 3. Testy Gap 7.1: Conflict of Laws Adjudication
    # ------------------------------------------------------------------------

    def test_04_conflict_of_laws_resolution_engine(self) -> None:
        """Weryfikuje rozstrzyganie kolizji prawnej między RODO, DORA i normami Ius Cogens."""
        rule_gdpr_erase = LegalObligation(
            rule_id="GDPR-ART-17",
            jurisdiction="EU",
            statute_name="Ogólne Rozporządzenie o Ochronie Danych (RODO)",
            citation="Art. 17 RODO",
            hierarchy=NormativeHierarchy.LEVEL_3_PRIMARY_STATUTE,
            mandate=MandateAction.MANDATORY_ERASE,
            description="Prawo do bycia zapomnianym (usunięcie danych osobowych klienta).",
        )
        rule_dora_preserve = LegalObligation(
            rule_id="DORA-RTS-AUDIT",
            jurisdiction="EU",
            statute_name="Digital Operational Resilience Act (DORA)",
            citation="Art. 12 DORA / RTS Audit Trail",
            hierarchy=NormativeHierarchy.LEVEL_3_PRIMARY_STATUTE,
            mandate=MandateAction.MANDATORY_PRESERVE,
            description="Obowiązek zachowania nienaruszalnego rejestru audytowego operacji finansowych przez 5 lat.",
        )

        adj = self.conflict_engine.adjudicate_conflict(
            obligation_a=rule_gdpr_erase,
            obligation_b=rule_dora_preserve,
            collision_topic="Retencja logów audytowych a prawo do usunięcia danych",
        )

        self.assertEqual(adj.prevailing_obligation.rule_id, "DORA-RTS-AUDIT")
        self.assertEqual(adj.principle_applied, ResolutionPrinciple.LEX_SPECIALIS)
        self.assertIsNotNone(adj.merkle_receipt_id)

    # ------------------------------------------------------------------------
    # 4. Testy Gap 6.5: Legacy System Bridge & Retrofit Proxy
    # ------------------------------------------------------------------------

    def test_05_legacy_retrofit_governance_proxy(self) -> None:
        """Weryfikuje opakowanie legacy mikroserwisu AI w bramkę ochronną Nethical bez zmian w kodzie źródłowym."""
        proxy = LegacyRetrofitProxy(
            legacy_service_id="LEGACY-CREDIT-SCORING-SVC",
            policy_mode=RetrofitPolicyMode.STRICT_ENFORCE,
            ledger=self.ledger,
        )

        def dummy_legacy_backend(payload: dict) -> tuple:
            # Prosta atrapa legacy backendu Pythona
            if payload.get("trigger_secret_leak"):
                return 200, {"score": 750, "token": "CONFIDENTIAL_CREDENTIAL_KEY_LEAK"}
            return 200, {"credit_score": 720, "recommendation": "APPROVE"}

        # 1. Legalne wywołanie -> Passthrough z nagłówkami Nethical i kwitem Merkle
        req_clean = InboundRequestWrapper(
            legacy_service_id="LEGACY-CREDIT-SCORING-SVC",
            payload={"customer_age": 35, "income": 8500},
        )
        resp_clean = proxy.intercept_and_forward(req_clean, dummy_legacy_backend)
        self.assertEqual(resp_clean.verdict, RetrofitVerdict.PASSTHROUGH_APPROVED)
        self.assertEqual(resp_clean.status_code, 200)
        self.assertIn("X-Nethical-Receipt-Id", resp_clean.governance_headers)

        # 2. Próba wstrzyknięcia złośliwego payloadu (Pre-Flight Block)
        req_malicious = InboundRequestWrapper(
            legacy_service_id="LEGACY-CREDIT-SCORING-SVC",
            payload={"query": "DROP TABLE users; --"},
        )
        resp_blocked_pre = proxy.intercept_and_forward(req_malicious, dummy_legacy_backend)
        self.assertEqual(resp_blocked_pre.verdict, RetrofitVerdict.CIRCUIT_BREAKER_BLOCKED)
        self.assertEqual(resp_blocked_pre.status_code, 400)

        # 3. Próba wycieku poufnych danych z backendu (Post-Flight Circuit Breaker)
        req_leak = InboundRequestWrapper(
            legacy_service_id="LEGACY-CREDIT-SCORING-SVC",
            payload={"trigger_secret_leak": True},
        )
        resp_blocked_post = proxy.intercept_and_forward(req_leak, dummy_legacy_backend)
        self.assertEqual(resp_blocked_post.verdict, RetrofitVerdict.CIRCUIT_BREAKER_BLOCKED)
        self.assertEqual(resp_blocked_post.status_code, 403)

    # ------------------------------------------------------------------------
    # 5. Testy Gap 6.1: Certification & Training Curriculum
    # ------------------------------------------------------------------------

    def test_06_institutional_curriculum_and_certification(self) -> None:
        """Weryfikuje egzaminowanie personelu i wydawanie kryptograficznych certyfikatów."""
        # Poprawne odpowiedzi na egzamin z modułu MOD-NTSG-EXPERT-03
        exam_result = self.curriculum_mgr.evaluate_exam(
            module_id="MOD-NTSG-EXPERT-03",
            candidate_id="OFFICER-NOWAK-44",
            candidate_name="Płk Jan Nowak",
            selected_answers=[1],  # Poprawna odpowiedź o odrzuceniu Shamira
        )

        self.assertTrue(exam_result.passed)
        self.assertEqual(exam_result.score_pct, 100.0)
        self.assertIsNotNone(exam_result.certificate_id)
        self.assertIsNotNone(exam_result.certified_until)
        self.assertIsNotNone(exam_result.merkle_receipt_id)

    # ------------------------------------------------------------------------
    # 6. Testy Gap 5.3: Precedent Database & Stare Decisis
    # ------------------------------------------------------------------------

    def test_07_precedent_database_and_consistency(self) -> None:
        """Weryfikuje wyszukiwanie precedensów i ostrzeganie o sprzeczności proponowanych orzeczeń."""
        matches = self.precedent_db.find_matching_precedents(
            domain="DUAL_USE_INFRASTRUCTURE",
            query_text="uderzenie w zaporę wodną elektrowni",
        )
        self.assertGreaterEqual(len(matches), 1)
        self.assertIn("PREC-IHL-DAM-01", matches[0].case.case_id)

        # Sprawdzenie zgodności orzeczenia
        is_consistent, warning = self.precedent_db.check_ruling_consistency(
            domain="DUAL_USE_INFRASTRUCTURE",
            proposed_ruling=PrecedentRuling.UNCONDITIONAL_APPROVAL,  # Sprzeczne z bezwzględnym zakazem!
            context_text="atak na zaporę wodną",
        )
        self.assertFalse(is_consistent)
        self.assertIn("SPRZECZNOŚĆ Z PRECEDENSEM", warning)

    # ------------------------------------------------------------------------
    # 7. Testy Gap 5.5: AI Governance KPI Framework
    # ------------------------------------------------------------------------

    def test_08_governance_kpi_engine(self) -> None:
        """Weryfikuje kalkulację wskaźników MTTR, Cohen's Kappa, wskaźnika dryfu i generowanie scorecardu."""
        # 1. Zdarzenia kolejek przeglądów (MTTR)
        self.kpi_engine.record_review_ticket(
            ReviewTicketMetric(
                ticket_id="TKT-01",
                created_at=1000.0,
                resolved_at=1000.0 + 7200.0,  # 2 godziny
                was_false_positive=False,
                decision_approved=True,
            )
        )
        self.kpi_engine.record_review_ticket(
            ReviewTicketMetric(
                ticket_id="TKT-02",
                created_at=2000.0,
                resolved_at=2000.0 + 14400.0, # 4 godziny
                was_false_positive=False,
                decision_approved=False,
            )
        )
        mttr = self.kpi_engine.compute_mean_time_to_review_hours()
        self.assertEqual(mttr, 3.0)  # Średnio 3h

        # 2. Zgodność recenzentów (Cohen's Kappa)
        self.kpi_engine.record_paired_review(PairedReviewRating(case_id="C-1", reviewer_alpha_verdict=True, reviewer_bravo_verdict=True))
        self.kpi_engine.record_paired_review(PairedReviewRating(case_id="C-2", reviewer_alpha_verdict=False, reviewer_bravo_verdict=False))
        self.kpi_engine.record_paired_review(PairedReviewRating(case_id="C-3", reviewer_alpha_verdict=True, reviewer_bravo_verdict=True))
        kappa = self.kpi_engine.compute_reviewer_cohens_kappa()
        self.assertEqual(kappa, 1.0)

        # 3. Dryf etyczny
        self.kpi_engine.record_risk_score_sample(0.25)
        self.kpi_engine.record_risk_score_sample(0.27)
        self.kpi_engine.record_risk_score_sample(0.26)
        drift = self.kpi_engine.compute_ethical_drift_coefficient()
        self.assertLess(drift, 0.05)

        # 4. Kwartalny Snapshot KPI
        snapshot = self.kpi_engine.generate_kpi_snapshot(
            reporting_period="2026-Q1",
            total_decisions=500_000,
            total_governance_cost_eur=7_500.0,
        )
        self.assertIsNotNone(snapshot.merkle_receipt_id)
        self.assertIsNotNone(snapshot.merkle_root)
        self.assertEqual(snapshot.overall_health.value, "EXEMPLARY")


if __name__ == "__main__":
    unittest.main()
