# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Multi-Level Security (MLS) Compartmentalization & OPSEC Guard (Gaps 3.1 & 3.6).

Implements military-grade classification controls and operational security filtering:
- **Mandatory Access Control (MAC):** Bell-LaPadula confidentiality ("no read up, no write down")
  and Biba integrity model.
- **NATO & Sovereign Codewords:** Compartment handling (e.g. COSMIC TOP SECRET // BOHEMIA // NOFORN).
- **Nationality & Release Caveats:** REL TO NATO, REL TO POL+USA, NOFORN enforcement.
- **OPSEC Classification Guard:** Automatic detection and redaction of sensitive operational details:
  troop movements, weapon telemetry, geospatial coordinates, force strength, and key material.
- **Immutable Audit Trail:** Integration with MerkleLedger for classification auditability.
"""

from __future__ import annotations

import logging
import re
import uuid
from enum import Enum, IntEnum
from typing import List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.security.mls_compartments")


# ============== Enums & Value Types ==============


class SecurityClearanceLevel(IntEnum):
    """Hierarchical military security clearance tiers (Bell-LaPadula order)."""
    UNCLASSIFIED = 0            # JAWNE
    RESTRICTED = 1              # ZASTRZEŻONE
    CONFIDENTIAL = 2            # POUFNE
    SECRET = 3                  # TAJNE / NATO SECRET
    TOP_SECRET = 4              # ŚCIŚLE TAJNE
    COSMIC_TOP_SECRET = 5       # COSMIC TOP SECRET (NATO highest tier)


class CompartmentCodeword(str, Enum):
    """Codewords for specialized compartmentalized intelligence and operations."""
    BOHEMIA = "BOHEMIA"                   # Special SIGINT / sovereign signals intelligence
    BALTIC_SHIELD = "BALTIC_SHIELD"       # Regional air defense & kinetic telemetry
    CYBER_TACTICAL = "CYBER_TACTICAL"     # Offensive/defensive cyber effector operations
    SPECIAL_FORCES = "SPECIAL_FORCES"     # Special operations units and personnel
    NUCLEAR_SENTRY = "NUCLEAR_SENTRY"     # Critical deterrence infrastructure


class OPSECCategory(str, Enum):
    """Taxonomy of sensitive operational security details."""
    TROOP_MOVEMENT = "TROOP_MOVEMENT"
    FORCE_STRENGTH = "FORCE_STRENGTH"
    GEOSPATIAL_COORDINATES = "GEOSPATIAL_COORDINATES"
    CRYPTO_KEY_MATERIAL = "CRYPTO_KEY_MATERIAL"
    TACTICAL_CALLSIGNS = "TACTICAL_CALLSIGNS"
    CRITICAL_ASSET_STATUS = "CRITICAL_ASSET_STATUS"


# ============== Data Models ==============


class SubjectClearance(BaseModel):
    """Personnel or autonomous agent security accreditation profile."""
    subject_id: str
    clearance_level: SecurityClearanceLevel
    integrity_level: int = Field(default=3, ge=1, le=5, description="Biba integrity rating")
    compartments: Set[str] = Field(default_factory=set)
    nationality: str = Field(default="POL", description="ISO 3166-1 alpha-3 country code")
    accreditation_expiry: Optional[str] = None
    duty_role: str = "OFFICER"


class DocumentObjectClassification(BaseModel):
    """Security classification label affixed to a data object or model output."""
    object_id: str
    classification_level: SecurityClearanceLevel
    integrity_level: int = Field(default=3, ge=1, le=5)
    required_compartments: Set[str] = Field(default_factory=set)
    releasable_to_nations: Optional[Set[str]] = Field(
        default=None,
        description="None means releasable to all allies; otherwise explicit nation list (e.g. {'POL', 'USA'})"
    )
    is_noforn: bool = Field(default=False, description="Not Releasable to Foreign Nationals (host nation only)")


class MACEnforcementResult(BaseModel):
    """Result of Mandatory Access Control evaluation."""
    allowed: bool
    subject_id: str
    object_id: str
    operation: str  # READ or WRITE
    rule_applied: str
    denial_reason: Optional[str] = None


class OPSECLeakFinding(BaseModel):
    """Individual operational security disclosure detected in text."""
    finding_id: str = Field(default_factory=lambda: f"OPSEC-{uuid.uuid4().hex[:6].upper()}")
    category: OPSECCategory
    matched_content: str
    severity: str = Field(pattern="^(CRITICAL|HIGH|MEDIUM)$")
    remediation_action: str


class OPSECInspectionResult(BaseModel):
    """Comprehensive inspection report of outgoing text for OPSEC leaks."""
    is_clean: bool
    original_text_length: int
    findings: List[OPSECLeakFinding] = Field(default_factory=list)
    sanitized_text: str
    redaction_count: int = 0


# ============== Regex Patterns for OPSEC Detection ==============

_OPSEC_PATTERNS: List[Tuple[OPSECCategory, str, re.Pattern, str]] = [
    # 1. Geospatial Coordinates (MGRS or Lat/Lon with high precision)
    (
        OPSECCategory.GEOSPATIAL_COORDINATES,
        "Precyzyjne współrzędne geograficzne celu",
        re.compile(r"\b\d{1,2}\s*[°\.]\s*\d{2,6}['\"]\s*[NS]\s*,?\s*\d{1,3}\s*[°\.]\s*\d{2,6}['\"]\s*[EW]\b", re.IGNORECASE),
        "CRITICAL",
    ),
    (
        OPSECCategory.GEOSPATIAL_COORDINATES,
        "Format siatki wojskowej MGRS",
        re.compile(r"\b3[45][U-V][A-Z]{2}\s*\d{4,5}\s*\d{4,5}\b", re.IGNORECASE),
        "CRITICAL",
    ),
    # 2. Troop Movement & Units (Brigade, Battalion, Division movements)
    (
        OPSECCategory.TROOP_MOVEMENT,
        "Wykryto raport o dyslokacji lub przemieszczeniu pododdziału",
        re.compile(r"\b(przemieszczenie|przerzut|dyslokacja|kolumna\s+marszowa)\s+(16\.|18\.|11\.|12\.|1\.)?\s*(batalion|brygada|dywizja|pułk)\b", re.IGNORECASE),
        "CRITICAL",
    ),
    # 3. Force Strength & Readiness (exact headcounts of troops/armor)
    (
        OPSECCategory.FORCE_STRENGTH,
        "Liczebność wojsk i stan gotowości bojowej",
        re.compile(r"\b(stan\s+osobowy|liczba\s+czołgów|amunicja\s+bojowa|zapas\s+rakiet):\s*\d+\b", re.IGNORECASE),
        "HIGH",
    ),
    # 4. Cryptographic Key Material (hex or base64 keys)
    (
        OPSECCategory.CRYPTO_KEY_MATERIAL,
        "Ekspozycja materiału kryptograficznego lub klucza",
        re.compile(r"\b(BEGIN\s+PRIVATE\s+KEY|KEY_HEX|AES_KEY):\s*[0-9a-fA-F]{32,128}\b", re.IGNORECASE),
        "CRITICAL",
    ),
    # 5. Tactical Callsigns (NATO callsigns e.g. EAGLE-01, VIPER-LEADER)
    (
        OPSECCategory.TACTICAL_CALLSIGNS,
        "Wojskowy znak wywoławczy (Callsign)",
        re.compile(r"\b(ORZEŁ|SOKÓŁ|EAGLE|VIPER|GHOST|WARRIOR)-[0-9]{1,2}\b", re.IGNORECASE),
        "MEDIUM",
    ),
]


# ============== Engine Class ==============


class MLSSecurityGuard:
    """Multi-Level Security (MLS) and OPSEC Classification Guard Engine."""

    def __init__(self, ledger: Optional[MerkleLedger] = None, host_nation: str = "POL") -> None:
        self.ledger = ledger or MerkleLedger()
        self.host_nation = host_nation

    # ------------------------------------------------------------------------
    # Mandatory Access Control (MAC): Bell-LaPadula & Biba
    # ------------------------------------------------------------------------

    def evaluate_read_access(
        self,
        subject: SubjectClearance,
        target: DocumentObjectClassification,
    ) -> MACEnforcementResult:
        """Evaluates Bell-LaPadula 'No Read Up' and Compartment Need-To-Know."""
        # 1. Bell-LaPadula: Subject level >= Object level
        if subject.clearance_level < target.classification_level:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="READ",
                rule_applied="BELL_LAPADULA_NO_READ_UP",
                denial_reason=(
                    f"Odmowa odczytu: Poziom poświadczenia podmiotu ({subject.clearance_level.name}) "
                    f"jest niższy niż klauzula obiektu ({target.classification_level.name})."
                ),
            )

        # 2. Compartment Need-To-Know: Subject must have ALL required compartments
        missing_compartments = target.required_compartments - subject.compartments
        if missing_compartments:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="READ",
                rule_applied="COMPARTMENT_NEED_TO_KNOW",
                denial_reason=f"Brak wymaganych kryptonimów/kompartmentów: {sorted(missing_compartments)}.",
            )

        # 3. NOFORN Check: If NOFORN, subject nationality must equal host nation
        if target.is_noforn and subject.nationality != self.host_nation:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="READ",
                rule_applied="NOFORN_NATIONALITY_RESTRICTION",
                denial_reason=f"Klauzula NOFORN: Obiekt dostępny wyłącznie dla obywateli {self.host_nation} (podmiot: {subject.nationality}).",
            )

        # 4. REL TO Caveat Check
        if target.releasable_to_nations is not None and subject.nationality not in target.releasable_to_nations:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="READ",
                rule_applied="RELEASE_TO_NATIONS_CAVEAT",
                denial_reason=f"Obywatelstwo {subject.nationality} nie znajduje się na liście dopuszczonych narodowości (REL TO: {sorted(target.releasable_to_nations)}).",
            )

        return MACEnforcementResult(
            allowed=True,
            subject_id=subject.subject_id,
            object_id=target.object_id,
            operation="READ",
            rule_applied="MAC_ACCESS_GRANTED",
        )

    def evaluate_write_access(
        self,
        subject: SubjectClearance,
        target: DocumentObjectClassification,
    ) -> MACEnforcementResult:
        """Evaluates Bell-LaPadula *-Property ('No Write Down') and Biba integrity."""
        # Bell-LaPadula *-Property: Subject cannot write down to lower classification (prevents data leak)
        if subject.clearance_level > target.classification_level:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="WRITE",
                rule_applied="BELL_LAPADULA_NO_WRITE_DOWN",
                denial_reason=(
                    f"*-Property: Podmiot z wyższą klauzulą ({subject.clearance_level.name}) "
                    f"nie może zapisywać do zasobu o niższej klauzuli ({target.classification_level.name}) - ryzyko dekonspiracji."
                ),
            )

        # Biba Integrity Model: 'No Write Up' (Subject integrity must be >= target integrity)
        if subject.integrity_level < target.integrity_level:
            return MACEnforcementResult(
                allowed=False,
                subject_id=subject.subject_id,
                object_id=target.object_id,
                operation="WRITE",
                rule_applied="BIBA_INTEGRITY_NO_WRITE_UP",
                denial_reason=(
                    f"Biba: Podmiot o niskiej integralności ({subject.integrity_level}) "
                    f"nie może modyfikować zasobu o wyższej integralności ({target.integrity_level})."
                ),
            )

        return MACEnforcementResult(
            allowed=True,
            subject_id=subject.subject_id,
            object_id=target.object_id,
            operation="WRITE",
            rule_applied="MAC_WRITE_GRANTED",
        )

    # ------------------------------------------------------------------------
    # OPSEC Classification Guard (Detection & Redaction)
    # ------------------------------------------------------------------------

    def inspect_and_sanitize_opsec(
        self,
        text: str,
        redact: bool = True,
    ) -> OPSECInspectionResult:
        """Scans outgoing text for OPSEC leaks and redacts or blocks them."""
        findings: List[OPSECLeakFinding] = []
        sanitized = text
        redaction_count = 0

        for category, description, pattern, severity in _OPSEC_PATTERNS:
            for match in pattern.finditer(text):
                matched_str = match.group(0)
                findings.append(
                    OPSECLeakFinding(
                        category=category,
                        matched_content=matched_str,
                        severity=severity,
                        remediation_action="REDACTED" if redact else "FLAGGED",
                    )
                )
                if redact:
                    placeholder = f"[OPSEC-ZASŁONIĘTO-{category.value}]"
                    sanitized = sanitized.replace(matched_str, placeholder)
                    redaction_count += 1

        is_clean = len(findings) == 0

        res = OPSECInspectionResult(
            is_clean=is_clean,
            original_text_length=len(text),
            findings=findings,
            sanitized_text=sanitized,
            redaction_count=redaction_count,
        )

        if not is_clean:
            logger.warning("OPSEC ALARM: Wykryto %d naruszeń tajemnicy operacyjnej!", len(findings))

        return res

    def seal_opsec_event(self, result: OPSECInspectionResult, operation_id: str) -> Optional[str]:
        """Kryptograficzne pieczętowanie zdarzenia OPSEC w rejestrze MerkleLedger."""
        if result.is_clean:
            return None
        try:
            payload = {
                "event_type": "OPSEC_VIOLATION_CONTAINED",
                "operation_id": operation_id,
                "findings_count": len(result.findings),
                "redactions": result.redaction_count,
                "categories": list({f.category.value for f in result.findings}),
            }
            receipt = self.ledger.append_decision(
                decision_data=payload,
                ambassador_notes=f"OPSEC GUARD: Zablokowano wyciek danych operacyjnych ({len(result.findings)} trafień).",
            )
            return receipt.receipt_id
        except Exception as e:
            logger.error("Błąd pieczętowania zdarzenia OPSEC w MerkleLedger: %s", e)
            return None
