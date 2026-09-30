# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit & Integration Tests for Wave 3: Enterprise & Strategic Governance.

Covers:
1. Gap 4.1: Board-Level AI Governance Dashboard (nethical/governance/board_dashboard.py)
2. Gap 4.4: AI Vendor & Model Supply Chain Governance (nethical/compliance/vendor_supply_chain.py)
3. Gap 5.4: Executive Briefing Generator (nethical/governance/executive_briefing.py)
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nethical.compliance.vendor_supply_chain import (
    AISoftwareBillOfMaterials,
    AIVendorSupplyChainManager,
    AdmissionVerdict,
    LicenseRiskLevel,
    ModelArtifactFormat,
    ModelWeightAttestation,
    VendorRiskAssessment,
    VendorTier,
)
from nethical.governance.board_dashboard import (
    AppetiteBreachStatus,
    BoardGovernanceDashboard,
    EthicalDebtCategory,
    EthicalDebtItem,
    RAGStatus,
    RegulatoryPosture,
    RiskAppetiteThresholds,
)
from nethical.governance.executive_briefing import (
    BriefingType,
    ClassificationMarking,
    ExecutiveBriefingGenerator,
)
from nethical.security.merkle_ledger import MerkleLedger


class TestWave3EnterpriseGovernance(unittest.TestCase):
    """Zestaw testów dla modułów nadzoru strategicznego i ładu korporacyjnego."""

    def setUp(self) -> None:
        """Inicjalizacja rejestru i silników nadzorczych."""
        self.ledger = MerkleLedger()
        self.dashboard = BoardGovernanceDashboard(
            organization_name="Polski Koncern Strategiczny S.A.",
            thresholds=RiskAppetiteThresholds(
                max_critical_incidents_monthly=0,
                max_high_severity_incidents_monthly=2,
                max_ethical_debt_score=25.0,
                max_accumulated_financial_exposure_eur=200_000.0,
                min_compliance_sla_pct=95.0,
            ),
            ledger=self.ledger,
        )
        self.vsc_manager = AIVendorSupplyChainManager(ledger=self.ledger)
        self.briefing_gen = ExecutiveBriefingGenerator(ledger=self.ledger)

    # ------------------------------------------------------------------------
    # Testy Gap 4.1: Board Governance Dashboard & Risk Appetite
    # ------------------------------------------------------------------------

    def test_01_board_dashboard_appetite_and_ethical_debt(self) -> None:
        """Weryfikuje kalkulację długu etycznego, apetytu na ryzyko i pakietu zarządczego."""
        # 1. Początkowy stan: w ramach apetytu (GREEN)
        status, breaches = self.dashboard.evaluate_risk_appetite()
        self.assertEqual(status, AppetiteBreachStatus.WITHIN_APPETITE)
        self.assertEqual(len(breaches), 0)

        # 2. Rejestracja długu etycznego
        d1 = EthicalDebtItem(
            system_id="SYS-CREDIT-SCORING-01",
            department_id="DEPT-RISK",
            title="Dryf etyczny w algorytmie oceny zdolności kredytowej",
            category=EthicalDebtCategory.BIAS_DRIFT,
            severity="HIGH",
            score=15.0,
            financial_exposure_eur=50_000.0,
            mitigation_owner="Dyrektor Zarządzania Ryzykiem Kredytowym",
        )
        self.dashboard.register_ethical_debt(d1)

        tot_score, tot_exp = self.dashboard.calculate_ethical_debt()
        self.assertEqual(tot_score, 15.0)
        self.assertEqual(tot_exp, 50_000.0)

        # 3. Dodanie telemetrii jednostek organizacyjnych
        self.dashboard.record_department_telemetry(
            department_id="DEPT-RISK",
            department_name="Departament Ryzyka i Analiz",
            active_ai_systems=4,
            decisions_period=120_000,
            incidents_count=1,
            avg_risk_score=45.0,
        )
        self.dashboard.record_department_telemetry(
            department_id="DEPT-TRADING",
            department_name="Pion Skarbu i Rynków Finansowych",
            active_ai_systems=2,
            decisions_period=850_000,
            incidents_count=0,
            avg_risk_score=20.0,
        )

        # 4. Dodanie jurysdykcji regulacyjnych
        self.dashboard.update_regulatory_posture(
            RegulatoryPosture(
                jurisdiction="EU_AI_ACT",
                compliance_score_pct=98.5,
                rag_status=RAGStatus.GREEN,
                critical_findings_count=0,
                last_audit_date="2026-03-15",
                supervisory_authority="Komisja Nadzoru Finansowego (KNF)",
            )
        )

        # 5. Generowanie pakietu dla Rady Nadzorczej
        packet = self.dashboard.generate_board_packet(reporting_period="2026-Q1")
        self.assertEqual(packet.overall_rag, RAGStatus.GREEN)
        self.assertEqual(packet.total_ai_decisions_governed, 970_000)
        self.assertEqual(packet.active_ai_systems_count, 6)
        self.assertIsNotNone(packet.merkle_receipt_id)
        self.assertIsNotNone(packet.merkle_root)

        # 6. Test eksportu do Markdown
        md_report = self.dashboard.export_markdown_report(packet)
        self.assertIn("Raport Nadzoru Algorytmicznego dla Rady Nadzorczej", md_report)
        self.assertIn("970,000", md_report)
        self.assertIn("DEPT-RISK", md_report)

        # 7. Symulacja przekroczenia limitu apetytu (Krytyczny incydent)
        self.dashboard.record_incident(severity="CRITICAL")
        status_after, breaches_after = self.dashboard.evaluate_risk_appetite()
        self.assertEqual(status_after, AppetiteBreachStatus.TOLERANCE_BREACHED)
        self.assertGreater(len(breaches_after), 0)

        packet_red = self.dashboard.generate_board_packet(reporting_period="2026-Q1-CRISIS")
        self.assertEqual(packet_red.overall_rag, RAGStatus.RED)
        self.assertIn("Zwołać nadzwyczajne posiedzenie", packet_red.actionable_board_recommendations[0])

    # ------------------------------------------------------------------------
    # Testy Gap 4.4: AI Vendor Supply Chain Governance & A-SBOM
    # ------------------------------------------------------------------------

    def test_02_vendor_supply_chain_admission_and_security_scanning(self) -> None:
        """Weryfikuje audyt łańcucha dostaw, wykrywanie niebezpiecznych formatów wag i atestację A-SBOM."""
        # 1. Rejestracja certyfikowanego dostawcy europejskiego
        vendor_eu = VendorRiskAssessment(
            vendor_id="VEND-MISTRAL-EU",
            vendor_name="Mistral Sovereign Systems SAS",
            tier=VendorTier.TIER_1_STRATEGIC,
            corporate_headquarters_country="FR",
            data_residency_region="EU-PARIS",
            is_dpa_executed=True,
            iso42001_certified=True,
            iso27001_certified=True,
        )
        self.vsc_manager.register_vendor(vendor_eu)

        # 2. Bezpieczny model w formacie safetensors z licencją Apache-2.0
        attestation_safe = ModelWeightAttestation(
            model_id="mistral-large-sovereign-v2",
            artifact_hash_sha384="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855e3b0c44298fc1c149afbf4c8996fb924",
            tensor_count=420,
            publisher_identity="Mistral AI Release Key",
        )
        asbom_safe = AISoftwareBillOfMaterials(
            model_name="Mistral Large Sovereign",
            model_version="2.4",
            base_architecture="Transformer-Decoder-70B",
            parameter_count_billions=70.0,
            quantization="FP16",
            license_declared="Apache-2.0",
            license_risk=LicenseRiskLevel.PERMISSIVE,
        )

        dec_safe = self.vsc_manager.scan_and_evaluate_model(
            attestation=attestation_safe,
            asbom=asbom_safe,
            artifact_format=ModelArtifactFormat.SAFETENSORS,
            vendor_id="VEND-MISTRAL-EU",
        )

        self.assertEqual(dec_safe.verdict, AdmissionVerdict.ADMITTED_FOR_PRODUCTION)
        self.assertGreaterEqual(dec_safe.overall_supply_chain_score, 90.0)
        self.assertEqual(len(dec_safe.findings), 0)
        self.assertIsNotNone(dec_safe.merkle_receipt_id)

        # 3. Niebezpieczny model w formacie PyTorch pickle (.pt/.bin) -> natychmiastowe odrzucenie
        dec_pickle = self.vsc_manager.scan_and_evaluate_model(
            attestation=attestation_safe,
            asbom=asbom_safe,
            artifact_format=ModelArtifactFormat.PYTORCH_PICKLE,
            vendor_id="VEND-MISTRAL-EU",
        )

        self.assertEqual(dec_pickle.verdict, AdmissionVerdict.REJECTED_SUPPLY_CHAIN_RISK)
        self.assertTrue(any(f.rule_id == "VSC-SEC-01-UNSAFE-SERIALIZATION" for f in dec_pickle.findings))

        # 4. Model z obcego regionu bez DPA -> odrzucenie za transfer poza EOG
        vendor_foreign = VendorRiskAssessment(
            vendor_id="VEND-SHADOW-CLOUD",
            vendor_name="Shadow Cloud Analytics Inc",
            tier=VendorTier.TIER_2_TACTICAL,
            corporate_headquarters_country="US",
            data_residency_region="US-EAST",
            is_dpa_executed=False,  # Brak DPA!
        )
        self.vsc_manager.register_vendor(vendor_foreign)

        dec_foreign = self.vsc_manager.scan_and_evaluate_model(
            attestation=attestation_safe,
            asbom=asbom_safe,
            artifact_format=ModelArtifactFormat.SAFETENSORS,
            vendor_id="VEND-SHADOW-CLOUD",
        )

        self.assertEqual(dec_foreign.verdict, AdmissionVerdict.REJECTED_SUPPLY_CHAIN_RISK)
        self.assertTrue(any(f.rule_id == "VSC-LEGAL-03-DATA-RESIDENCY" for f in dec_foreign.findings))

        # 5. Generowanie raportu audytowego
        rep_text = self.vsc_manager.generate_supply_chain_report(dec_safe.decision_id)
        self.assertIn("DOPUSZCZONY DO PRODUKCJI", rep_text)
        self.assertIn("Mistral Sovereign Systems SAS", rep_text)

    # ------------------------------------------------------------------------
    # Testy Gap 5.4: Executive Briefing Generator
    # ------------------------------------------------------------------------

    def test_03_executive_briefing_generation_and_merkle_sealing(self) -> None:
        """Weryfikuje syntezę notatek wykonawczych dla Ministra oraz Rady Nadzorczej."""
        # 1. Generowanie briefingu dla Zarządu / Rady Nadzorczej na podstawie pakietu
        packet = self.dashboard.generate_board_packet(reporting_period="2026-Q1")
        board_memo = self.briefing_gen.generate_board_quarterly_briefing(board_packet=packet)

        self.assertEqual(board_memo.briefing_type, BriefingType.BOARD_QUARTERLY_STRATEGIC)
        self.assertEqual(board_memo.classification, ClassificationMarking.OFFICIAL_RESTRICTED)
        self.assertIsNotNone(board_memo.merkle_receipt_id)
        self.assertIsNotNone(board_memo.merkle_root)
        self.assertIn("Stabilność ładu algorytmicznego", board_memo.executive_headline)

        # 2. Generowanie notatki rządowej dla Ministra Cyfryzacji
        gov_memo = self.briefing_gen.generate_ministerial_oversight_briefing(
            department_name="Węzeł Ochrony Infrastruktury Krytycznej i Energetyki",
            total_decisions=1_500_000,
            prevented_incidents_count=4,
            sovereignty_tier="Tier 1 Sovereign Ready",
        )

        self.assertEqual(gov_memo.briefing_type, BriefingType.MINISTERIAL_OVERSIGHT)
        self.assertEqual(gov_memo.classification, ClassificationMarking.CONFIDENTIAL_GOV)
        self.assertEqual(gov_memo.rag_status, RAGStatus.GREEN)
        self.assertEqual(gov_memo.kpis.total_decisions_governed, 1_500_000)
        self.assertEqual(gov_memo.kpis.critical_interventions_count, 4)
        self.assertEqual(len(gov_memo.critical_highlights), 3)
        self.assertIsNotNone(gov_memo.merkle_receipt_id)

        # 3. Formatowanie do formalnego memorandum rządowego Markdown
        memo_rendered = self.briefing_gen.render_formal_markdown_memo(gov_memo)
        self.assertIn("KLAUZULA: POUFNE / CONFIDENTIAL", memo_rendered)
        self.assertIn("Minister Cyfryzacji", memo_rendered)
        self.assertIn("Bramka Deterministyczna NTSG (Geneva AP I Art. 56)", memo_rendered)
        self.assertIn("Świadectwo Niezaprzeczalności Dowodowej", memo_rendered)


if __name__ == "__main__":
    unittest.main()
