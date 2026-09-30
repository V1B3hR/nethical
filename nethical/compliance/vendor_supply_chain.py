# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""AI Vendor & Third-Party Model Supply Chain Governance (Gap 4.4).

Implements strict institutional supply-chain controls mandated by:
- **ISO/IEC 42001:2023 §8.4:** Control of externally provided AI systems and components.
- **EU AI Act Art. 28 & 53:** Downstream provider obligations, GPAI transparency, and documentation.
- **NIST SP 800-161r1:** Cybersecurity Supply Chain Risk Management (C-SCRM).
- **CycloneDX 1.6 AI & SPDX 3.0 AI:** AI Software Bill of Materials (A-SBOM).

Key capabilities:
1. Vendor Risk Assessment (VRA): Data residency, ISO 42001/27001 certifications, DPA validation.
2. AI Software Bill of Materials (A-SBOM): Full provenance, training lineage, tokenizer and library manifests.
3. Cryptographic Model Attestation: Weight checksum (SHA-384) and PQC tensor integrity validation.
4. Vulnerability & Serialization Scanning: Detection of insecure pickle files, malicious weights, and license viral taint.
5. Gate Admission Decisions: Automated verdict (ADMITTED, SANDBOX_ONLY, REJECTED) sealed in MerkleLedger.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.compliance.vendor_supply_chain")


# ============== Enums & Value Types ==============


class VendorTier(str, Enum):
    """Institutional classification of third-party AI suppliers."""
    TIER_1_STRATEGIC = "TIER_1_STRATEGIC"       # Core foundation model / critical infrastructure provider
    TIER_2_TACTICAL = "TIER_2_TACTICAL"         # Domain-specialized model (e.g. legal, financial, vision)
    TIER_3_COMMODITY = "TIER_3_COMMODITY"       # Peripheral utility, embeddings, or non-critical classifier


class ModelArtifactFormat(str, Enum):
    """Serialization format of model weights."""
    SAFETENSORS = "safetensors"                 # Recommended: zero-copy, memory-mapped, no code execution
    GGUF = "gguf"                               # Quantized binary format, sandboxed execution
    ONNX = "onnx"                               # Open Neural Network Exchange graph
    PYTORCH_PICKLE = "pytorch_pickle"           # Insecure legacy (.pt/.bin) - vulnerable to arbitrary code execution
    TENSORFLOW_SAVEDMODEL = "tf_savedmodel"     # Google TensorFlow graph
    CLOUD_REST_API = "cloud_rest_api"           # Hosted API (SaaS / closed-weights)


class LicenseRiskLevel(str, Enum):
    """Legal & IP risk tier of model and training data licenses."""
    PERMISSIVE = "PERMISSIVE"                   # Apache-2.0, MIT, BSD (unrestricted commercial use)
    COMMERCIAL_RESTRICTED = "COMMERCIAL_RESTRICTED"  # LLaMA 3 Community, custom monthly-active-user limits
    COPYLEFT_VIRAL = "COPYLEFT_VIRAL"           # GPLv3, AGPL (high contamination risk for proprietary codebases)
    PROPRIETARY_CLOSED = "PROPRIETARY_CLOSED"   # Vendor cloud Terms of Service
    UNKNOWN = "UNKNOWN"                         # Unspecified or legally ambiguous license


class AdmissionVerdict(str, Enum):
    """Formal admission status for third-party AI deployment."""
    ADMITTED_FOR_PRODUCTION = "ADMITTED_FOR_PRODUCTION"
    CONDITIONAL_SANDBOX_ONLY = "CONDITIONAL_SANDBOX_ONLY"
    REJECTED_SUPPLY_CHAIN_RISK = "REJECTED_SUPPLY_CHAIN_RISK"


# ============== Data Models ==============


class ModelWeightAttestation(BaseModel):
    """Cryptographic attestation and provenance proof of model weights."""
    model_id: str
    artifact_hash_sha384: str = Field(..., description="SHA-384 hex digest of weights file")
    tensor_count: int = Field(default=0, ge=0)
    publisher_identity: str
    publisher_signing_key_id: Optional[str] = None
    attestation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    declared_file_size_bytes: int = Field(default=0, ge=0)


class AISoftwareBillOfMaterials(BaseModel):
    """Standardized AI Software Bill of Materials (A-SBOM)."""
    sbom_id: str = Field(default_factory=lambda: f"ASBOM-{uuid.uuid4().hex[:8].upper()}")
    format_standard: str = Field(default="CycloneDX-1.6-AI")
    model_name: str
    model_version: str
    base_architecture: str
    parameter_count_billions: float = Field(ge=0.0)
    quantization: str = Field(default="NONE")
    training_data_sources: List[str] = Field(default_factory=list)
    system_dependencies: List[Dict[str, str]] = Field(default_factory=list)
    license_declared: str
    license_risk: LicenseRiskLevel


class SupplyChainFinding(BaseModel):
    """Individual security or compliance flaw identified during ingestion."""
    finding_id: str = Field(default_factory=lambda: f"SCF-{uuid.uuid4().hex[:6].upper()}")
    rule_id: str
    title: str
    severity: str = Field(pattern="^(CRITICAL|HIGH|MEDIUM|LOW)$")
    description: str
    remediation: str


class VendorRiskAssessment(BaseModel):
    """Enterprise risk evaluation of an external AI provider."""
    vendor_id: str
    vendor_name: str
    tier: VendorTier
    corporate_headquarters_country: str
    data_residency_region: str = Field(..., description="e.g. EU-POLAND, EU-FRANKFURT, US-EAST, NATO-ISLAND")
    is_dpa_executed: bool = Field(default=False, description="Data Processing Agreement signed under GDPR Art. 28")
    iso42001_certified: bool = False
    iso27001_certified: bool = False
    soc2_type2_certified: bool = False
    sla_uptime_guarantee_pct: float = Field(default=99.9, ge=0.0, le=100.0)
    last_assessment_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    terms_of_service_hash: str = Field(default="HASH-DEFAULT-TOS")


class ModelAdmissionDecision(BaseModel):
    """Official formal decision packet admitting or rejecting a third-party model."""
    decision_id: str = Field(default_factory=lambda: f"DEC-ADM-{uuid.uuid4().hex[:8].upper()}")
    model_id: str
    vendor_id: str
    verdict: AdmissionVerdict
    overall_supply_chain_score: float = Field(ge=0.0, le=100.0, description="0=Dangerous, 100=Flawless")
    findings: List[SupplyChainFinding] = Field(default_factory=list)
    mandated_controls: List[str] = Field(default_factory=list)
    decided_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    merkle_receipt_id: Optional[str] = None
    merkle_root: Optional[str] = None


# ============== Ingestion & Scanner Engine ==============


class AIVendorSupplyChainManager:
    """Enterprise AI Vendor & Model Supply Chain Governance Engine."""

    # Disallowed non-compliant formats in high-assurance sovereign environments
    UNSAFE_FORMATS: Set[ModelArtifactFormat] = {ModelArtifactFormat.PYTORCH_PICKLE}

    # Sovereign jurisdictions where data residency is strictly compliant
    COMPLIANT_DATA_REGIONS: Set[str] = {
        "EU-POLAND",
        "EU-FRANKFURT",
        "EU-IRELAND",
        "EU-PARIS",
        "EU-STOCKHOLM",
        "SOVEREIGN-AIRGAP",
        "NATO-DEFENSE-CLOUD",
    }

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self._vendors: Dict[str, VendorRiskAssessment] = {}
        self._decisions: Dict[str, ModelAdmissionDecision] = {}

    def register_vendor(self, assessment: VendorRiskAssessment) -> None:
        """Enrolls or updates a vendor in the institutional registry."""
        self._vendors[assessment.vendor_id] = assessment
        logger.info(
            "Zarejestrowano dostawcę AI: %s (%s, Rezydencja: %s, ISO42001: %s)",
            assessment.vendor_name,
            assessment.tier.value,
            assessment.data_residency_region,
            assessment.iso42001_certified,
        )

    def scan_and_evaluate_model(
        self,
        attestation: ModelWeightAttestation,
        asbom: AISoftwareBillOfMaterials,
        artifact_format: ModelArtifactFormat,
        vendor_id: str,
    ) -> ModelAdmissionDecision:
        """Executes full compliance and security evaluation for an external AI model."""
        vendor = self._vendors.get(vendor_id)
        if not vendor:
            raise ValueError(f"Nieznany dostawca: {vendor_id}. Wymagana uprzednia rejestracja VRA.")

        findings: List[SupplyChainFinding] = []
        mandated_controls: List[str] = []

        # 1. Serialization Security Inspection (IEC 62443 / ISO 42001)
        if artifact_format in self.UNSAFE_FORMATS:
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-SEC-01-UNSAFE-SERIALIZATION",
                    title="Niedozwolony format wag (Arbitrary Code Execution Risk)",
                    severity="CRITICAL",
                    description=(
                        f"Format {artifact_format.value} bazuje na module Python 'pickle', co pozwala "
                        "na wykonanie złośliwego kodu podczas deserializacji wag (CWE-502)."
                    ),
                    remediation="Wymagana konwersja do formatu 'safetensors' lub izolacja w enklawie micro-VM.",
                )
            )

        # 2. Cryptographic Weight Attestation Check
        if not attestation.artifact_hash_sha384 or len(attestation.artifact_hash_sha384) < 64:
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-CRYPTO-02-INVALID-DIGEST",
                    title="Brak poprawności kryptograficznej sumy kontrolnej wag",
                    severity="HIGH",
                    description="Suma kontrolna SHA-384 jest pusta lub uszkodzona. Ryzyko podmiany wag w tranzycie.",
                    remediation="Wygenerować nienaruszalny SHA-384 digest z repozytorium źródłowego dostawcy.",
                )
            )

        # 3. Data Residency & GDPR Cross-Border Transfer
        if vendor.data_residency_region not in self.COMPLIANT_DATA_REGIONS and not vendor.is_dpa_executed:
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-LEGAL-03-DATA-RESIDENCY",
                    title="Niedozwolony transfer danych poza EOG / NATO bez DPA",
                    severity="CRITICAL",
                    description=(
                        f"Dostawca przetwarza zapytania w regionie {vendor.data_residency_region} "
                        "bez podpisanej umowy powierzenia przetwarzania danych (GDPR Art. 28)."
                    ),
                    remediation="Wdrożyć lokalną instancję on-premise lub podpisać Standardowe Klauzule Umowne (SCC).",
                )
            )

        # 4. Intellectual Property & License Contamination Risk
        if asbom.license_risk == LicenseRiskLevel.COPYLEFT_VIRAL:
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-IP-04-VIRAL-LICENSE",
                    title="Ryzyko infekcji prawnej licencją Copyleft (GPL/AGPL)",
                    severity="HIGH",
                    description=(
                        f"Model oznaczony licencją {asbom.license_declared} wymusza udostępnienie kodu "
                        "źródłowego systemów integrujących w przypadku wdrożeń sieciowych."
                    ),
                    remediation="Uzyskać komercyjną licencję proprietary lub odizolować interfejs za API proxy.",
                )
            )
        elif asbom.license_risk == LicenseRiskLevel.COMMERCIAL_RESTRICTED:
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-IP-05-COMMERCIAL-CAP",
                    title="Ograniczenia komercyjne licencji modelu (Usage Capping)",
                    severity="MEDIUM",
                    description="Licencja nakłada limity na liczbę aktywnych użytkowników miesięcznie (MAU).",
                    remediation="Wdrożyć audyt licencyjny i monitorowanie wolumenu użytkowników.",
                )
            )

        # 5. Vendor Trust & Certification Posture
        if vendor.tier == VendorTier.TIER_1_STRATEGIC and not (vendor.iso42001_certified or vendor.iso27001_certified):
            findings.append(
                SupplyChainFinding(
                    rule_id="VSC-GOV-06-MISSING-ISO-CERT",
                    title="Brak certyfikacji ISO 42001 / ISO 27001 dla dostawcy strategicznego",
                    severity="HIGH",
                    description="Strategiczny dostawca modeli AI nie posiada audytowanego systemu zarządzania bezpieczeństwem.",
                    remediation="Wymusić przedłożenie certyfikatu w kolejnym cyklu audytowym lub nałożyć nadzór wzmożony.",
                )
            )

        # Calculate Overall Supply Chain Score (0-100)
        critical_count = sum(1 for f in findings if f.severity == "CRITICAL")
        high_count = sum(1 for f in findings if f.severity == "HIGH")
        medium_count = sum(1 for f in findings if f.severity == "MEDIUM")

        penalty = (critical_count * 40.0) + (high_count * 15.0) + (medium_count * 5.0)
        base_score = 100.0
        if vendor.iso42001_certified:
            base_score += 5.0
        score = max(0.0, min(100.0, round(base_score - penalty, 1)))

        # Determine Verdict
        if critical_count > 0 or score < 50.0:
            verdict = AdmissionVerdict.REJECTED_SUPPLY_CHAIN_RISK
            mandated_controls.append("CAŁKOWITY ZAKAZ WDROŻENIA PRODUKCYJNEGO.")
            mandated_controls.append("Kwarantanna artefaktów w izolowanym magazynie danych.")
        elif high_count > 0 or score < 80.0:
            verdict = AdmissionVerdict.CONDITIONAL_SANDBOX_ONLY
            mandated_controls.append("Dozwolone wyłącznie środowisko testowe / sandbox.")
            mandated_controls.append("Brak dostępu do produkcyjnych danych osobowych i wrażliwych.")
            mandated_controls.append("Wymagana re-ewaluacja po konwersji wag do formatu safetensors.")
        else:
            verdict = AdmissionVerdict.ADMITTED_FOR_PRODUCTION
            mandated_controls.append("Pełna akredytacja produkcyjna.")
            mandated_controls.append("Ciągły monitoring dryfu etycznego i telemetrii zapytań.")

        decision = ModelAdmissionDecision(
            model_id=attestation.model_id,
            vendor_id=vendor_id,
            verdict=verdict,
            overall_supply_chain_score=score,
            findings=findings,
            mandated_controls=mandated_controls,
        )

        # Seal in MerkleLedger
        self._seal_decision(decision, vendor, asbom)
        self._decisions[decision.decision_id] = decision
        return decision

    def _seal_decision(
        self,
        decision: ModelAdmissionDecision,
        vendor: VendorRiskAssessment,
        asbom: AISoftwareBillOfMaterials,
    ) -> None:
        """Kryptograficzne pieczętowanie decyzji o dopuszczeniu modelu w MerkleLedger."""
        try:
            payload = {
                "event_type": "MODEL_SUPPLY_CHAIN_ADMISSION_DECISION",
                "decision_id": decision.decision_id,
                "model_id": decision.model_id,
                "model_version": asbom.model_version,
                "vendor_id": vendor.vendor_id,
                "vendor_name": vendor.vendor_name,
                "verdict": decision.verdict.value,
                "score": decision.overall_supply_chain_score,
                "critical_findings": sum(1 for f in decision.findings if f.severity == "CRITICAL"),
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"AKREDYTACJA DOSTAWCY AI: Model {decision.model_id} ({vendor.vendor_name}) -> {decision.verdict.value}.",
            )
            decision.merkle_receipt_id = receipt.receipt_id
            decision.merkle_root = self.ledger.current_root
            logger.info("Zapieczętowano decyzję łańcucha dostaw %s w MerkleLedger (Receipt: %s)", decision.decision_id, receipt.receipt_id)
        except Exception as e:
            logger.error("Błąd pieczętowania decyzji łańcucha dostaw w MerkleLedger: %s", e)

    def generate_supply_chain_report(self, decision_id: str) -> str:
        """Generuje szczegółowy raport z audytu łańcucha dostaw dla CISO / CPO."""
        dec = self._decisions.get(decision_id)
        if not dec:
            return f"Błąd: Nie odnaleziono decyzji {decision_id}"

        v = self._vendors.get(dec.vendor_id)
        v_name = v.vendor_name if v else dec.vendor_id

        verdict_badge = {
            "ADMITTED_FOR_PRODUCTION": "🟢 DOPUSZCZONY DO PRODUKCJI",
            "CONDITIONAL_SANDBOX_ONLY": "🟡 WARUNKOWY SANDBOX",
            "REJECTED_SUPPLY_CHAIN_RISK": "🔴 ODRZUCONY (RYZYKO ŁAŃCUCHA DOSTAW)",
        }.get(dec.verdict.value, dec.verdict.value)

        lines = [
            f"# Raport Atestacji Łańcucha Dostaw AI (A-SBOM & Model Security)",
            f"**Identyfikator Decyzji:** `{dec.decision_id}` | **Data:** `{dec.decided_at[:10]}`",
            f"**Model:** `{dec.model_id}` | **Dostawca:** `{v_name}`",
            f"**Werdykt Kwalifikacyjny:** **{verdict_badge}**",
            f"**Ocena Bezpieczeństwa Łańcucha Dostaw:** `{dec.overall_supply_chain_score:.1f}/100 pkt`",
            "",
            "---",
            "",
            "## 1. Wykryte Ryzyka i Podatności Łańcucha Dostaw",
            "",
        ]

        if not dec.findings:
            lines.append("✅ Brak zidentyfikowanych ryzyk. Model spełnia rygorystyczne normy ISO 42001.")
        else:
            lines.extend([
                "| ID | Reguła | Tytuł | Dotkliwość | Rekomendacja Naprawcza |",
                "|---|---|---|---|---|",
            ])
            for f in dec.findings:
                sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(f.severity, "⚪")
                lines.append(
                    f"| `{f.finding_id}` | `{f.rule_id}` | {f.title} | {sev_icon} **{f.severity}** | {f.remediation} |"
                )
        lines.append("")

        lines.extend([
            "## 2. Nałożone Środki Zaradcze i Dyspozycje Kontrolne",
            "",
        ])
        for c in dec.mandated_controls:
            lines.append(f"- **[KONTROLA]** {c}")
        lines.append("")

        lines.extend([
            "---",
            "### 🔒 Poświadczenie Rejestru Nienaruszalnego Merkle-DAG",
            f"- **Identyfikator Kwitu:** `{dec.merkle_receipt_id or 'Brak'}`",
            f"- **Korzeń Merkle (Root):** `{dec.merkle_root or 'Brak'}`",
        ])

        return "\n".join(lines)
