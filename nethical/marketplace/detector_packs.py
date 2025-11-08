"""Pre-built Detector Packs for Common Use Cases.

This module provides industry-specific detector bundles and
use case templates for quick deployment.

Enhancements:
- Stronger typing and validation with helpful exceptions.
- Richer search with free-text query, tag matching modes, scoring, and sorting.
- Serialization helpers (to_dict / from_dict) for packs, templates, and industry packs.
- Import/export utilities (JSON and optional YAML) to support a "marketplace" model.
- Convenience registry APIs: list, update, remove, upsert, and detector-based lookups.
- Logging for visibility and easier troubleshooting.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional YAML support
try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_YAML = False

logger = logging.getLogger(__name__)


class Industry(Enum):
    """Industry categories."""

    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    EDUCATION = "education"
    GOVERNMENT = "government"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"


# -----------------------
# Exceptions
# -----------------------
class RegistryError(Exception):
    """Base class for registry errors."""


class DuplicatePackError(RegistryError):
    """Raised when attempting to register a pack with an existing pack_id (and override not allowed)."""


class PackNotFoundError(RegistryError):
    """Raised when a requested pack is not found."""


class TemplateNotFoundError(RegistryError):
    """Raised when a requested template is not found."""


# -----------------------
# Dataclasses
# -----------------------
@dataclass
class DetectorPack:
    """Pre-built detector pack."""

    pack_id: str
    name: str
    description: str
    detectors: List[str]
    configuration: Dict[str, Any]
    industry: Optional[Industry] = None
    use_cases: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    # Optional metadata
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    deprecated: bool = False

    def validate(self) -> None:
        """Validate pack consistency and basic constraints."""
        if not self.pack_id or not isinstance(self.pack_id, str):
            raise RegistryError("DetectorPack.pack_id must be a non-empty string.")
        if not self.name:
            raise RegistryError("DetectorPack.name must be provided.")
        if not self.detectors:
            raise RegistryError(
                "DetectorPack.detectors must contain at least one detector id/name."
            )
        # Normalize tags and use_cases for consistency
        self.tags = {normalize_tag(t) for t in self.tags}
        self.use_cases = [normalize_tag(u) for u in self.use_cases]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        data = asdict(self)
        # Convert Enums and datetime
        if self.industry is not None:
            data["industry"] = self.industry.value
        data["created_at"] = self.created_at.isoformat()
        # Sets -> lists
        data["tags"] = sorted(list(self.tags))
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DetectorPack":
        """Deserialize from a dict."""
        industry_val = data.get("industry")
        industry = Industry(industry_val) if industry_val in {i.value for i in Industry} else None

        created_at_raw = data.get("created_at")
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.utcnow()
        else:
            created_at = datetime.utcnow()

        pack = DetectorPack(
            pack_id=data["pack_id"],
            name=data["name"],
            description=data.get("description", ""),
            detectors=list(data.get("detectors", [])),
            configuration=dict(data.get("configuration", {})),
            industry=industry,
            use_cases=list(data.get("use_cases", [])),
            tags=set(data.get("tags", [])),
            version=data.get("version", "1.0.0"),
            created_at=created_at,
            deprecated=bool(data.get("deprecated", False)),
        )
        pack.validate()
        return pack


@dataclass
class IndustryPack:
    """Industry-specific detector pack."""

    industry: Industry
    packs: List[DetectorPack]
    compliance_standards: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "industry": self.industry.value,
            "packs": [p.to_dict() for p in self.packs],
            "compliance_standards": list(self.compliance_standards),
            "best_practices": list(self.best_practices),
        }


@dataclass
class UseCaseTemplate:
    """Template for specific use case."""

    template_id: str
    name: str
    description: str
    detector_pack_id: str
    configuration: Dict[str, Any]
    example_code: str = ""
    example_code_language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UseCaseTemplate":
        return UseCaseTemplate(
            template_id=data["template_id"],
            name=data["name"],
            description=data.get("description", ""),
            detector_pack_id=data["detector_pack_id"],
            configuration=dict(data.get("configuration", {})),
            example_code=data.get("example_code", ""),
            example_code_language=data.get("example_code_language"),
        )


# -----------------------
# Utilities
# -----------------------
_TAG_NORM_RE = re.compile(r"[^a-z0-9\-]+")


def normalize_tag(value: str) -> str:
    """Normalize tags and use case identifiers to a consistent lowercase kebab-case."""
    v = (value or "").strip().lower().replace("_", "-").replace(" ", "-")
    v = _TAG_NORM_RE.sub("-", v)
    return re.sub(r"-{2,}", "-", v).strip("-")


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _score_match(pack: DetectorPack, query: Optional[str], use_case: Optional[str]) -> int:
    """Compute a relevance score for a pack given optional query and use_case."""
    if query is None and use_case is None:
        return 0
    score = 0
    if query:
        q = query.strip().lower()
        # Larger weight for pack_id and name
        if _contains(pack.pack_id, q):
            score += 6
        if _contains(pack.name, q):
            score += 5
        if _contains(pack.description, q):
            score += 3
        # Tags, use_cases, detectors
        score += sum(2 for t in pack.tags if q in t)
        score += sum(3 for d in pack.detectors if q in d.lower())
        score += sum(2 for uc in pack.use_cases if q in uc)
        # Industry keyword match
        if pack.industry and q in pack.industry.value:
            score += 3
    if use_case:
        uc = normalize_tag(use_case)
        if uc in pack.use_cases:
            score += 4
    return score


# -----------------------
# Registry
# -----------------------
class DetectorPackRegistry:
    """Registry for pre-built detector packs.

    This class provides access to industry-specific detector bundles
    and use case templates.

    Example:
        >>> registry = DetectorPackRegistry()
        >>> financial_pack = registry.get_industry_pack(Industry.FINANCIAL)
        >>> template = registry.get_use_case_template("fraud-detection")
    """

    def __init__(self):
        """Initialize detector pack registry."""
        self._packs: Dict[str, DetectorPack] = {}
        self._industry_packs: Dict[Industry, IndustryPack] = {}
        self._templates: Dict[str, UseCaseTemplate] = {}
        self._initialize_default_packs()
        self._initialize_default_templates()

    # ------------- Core CRUD -------------

    def register_pack(self, pack: DetectorPack, *, allow_override: bool = False) -> None:
        """Register a detector pack.

        Args:
            pack: Detector pack to register
            allow_override: If True, allows replacement of an existing pack with the same pack_id.
        """
        pack.validate()
        if pack.pack_id in self._packs and not allow_override:
            raise DuplicatePackError(f"DetectorPack with id '{pack.pack_id}' already exists.")
        if pack.pack_id in self._packs and allow_override:
            logger.warning("Overriding existing DetectorPack with id '%s'", pack.pack_id)

        self._packs[pack.pack_id] = pack
        self._index_pack_into_industry(pack)

    def upsert_pack(self, pack: DetectorPack) -> None:
        """Insert or update a detector pack without raising for duplicates."""
        self.register_pack(pack, allow_override=True)

    def update_pack(self, pack_id: str, **updates: Any) -> DetectorPack:
        """Update an existing pack in-place and return it."""
        pack = self.get_pack(pack_id)
        if not pack:
            raise PackNotFoundError(f"DetectorPack '{pack_id}' not found.")
        for k, v in updates.items():
            if hasattr(pack, k):
                setattr(pack, k, v)
            else:
                raise RegistryError(f"DetectorPack has no attribute '{k}'.")
        pack.validate()
        # Reindex industry if changed
        self._rebuild_industry_index()
        return pack

    def remove_pack(self, pack_id: str) -> None:
        """Remove a pack from the registry."""
        if pack_id not in self._packs:
            raise PackNotFoundError(f"DetectorPack '{pack_id}' not found.")
        del self._packs[pack_id]
        self._rebuild_industry_index()

    def has_pack(self, pack_id: str) -> bool:
        return pack_id in self._packs

    def get_pack(self, pack_id: str) -> Optional[DetectorPack]:
        """Get a detector pack by ID.

        Args:
            pack_id: Pack identifier

        Returns:
            Detector pack or None
        """
        return self._packs.get(pack_id)

    def list_packs(self) -> List[DetectorPack]:
        """List all available detector packs.

        Returns:
            List of all packs
        """
        # Return in deterministic order by name
        return sorted(
            self._packs.values(),
            key=lambda p: (p.industry.value if p.industry else "", p.name.lower()),
        )

    # ------------- Industry -------------

    def get_industry_pack(self, industry: Industry) -> Optional[IndustryPack]:
        """Get industry-specific detector packs.

        Args:
            industry: Industry category

        Returns:
            Industry pack or None
        """
        return self._industry_packs.get(industry)

    def list_industries(self) -> List[Industry]:
        """List industries that have at least one pack."""
        return sorted(self._industry_packs.keys(), key=lambda i: i.value)

    # ------------- Templates -------------

    def register_template(self, template: UseCaseTemplate) -> None:
        """Register a use case template.

        Args:
            template: Use case template
        """
        self._templates[template.template_id] = template

    def get_use_case_template(self, template_id: str) -> Optional[UseCaseTemplate]:
        """Get use case template.

        Args:
            template_id: Template identifier

        Returns:
            Use case template or None
        """
        return self._templates.get(template_id)

    def list_templates(self) -> List[UseCaseTemplate]:
        """List all registered templates."""
        return sorted(self._templates.values(), key=lambda t: t.name.lower())

    # ------------- Search -------------

    def search_packs(
        self,
        industry: Optional[Industry] = None,
        tags: Optional[Set[str]] = None,
        use_case: Optional[str] = None,
        query: Optional[str] = None,
        *,
        match_any_tag: bool = False,
        limit: Optional[int] = None,
        sort_by: str = "relevance",  # "relevance" | "name"
        include_deprecated: bool = False,
    ) -> List[DetectorPack]:
        """Search for detector packs.

        Args:
            industry: Filter by industry
            tags: Filter by tags (normalized to kebab-case)
            use_case: Filter by use case (normalized to kebab-case)
            query: Free-text search across id, name, description, tags, detectors, and use cases
            match_any_tag: If True, any tag match qualifies. If False (default), require all provided tags.
            limit: Optional maximum number of results to return
            sort_by: "relevance" (default) uses a simple scoring; "name" sorts alphabetically
            include_deprecated: If False (default), excludes deprecated packs

        Returns:
            List of matching packs
        """
        norm_tags: Optional[Set[str]] = None
        if tags:
            norm_tags = {normalize_tag(t) for t in tags}
        norm_use_case = normalize_tag(use_case) if use_case else None

        scored: List[Tuple[int, DetectorPack]] = []
        for pack in self._packs.values():
            if not include_deprecated and pack.deprecated:
                continue
            if industry and pack.industry != industry:
                continue
            if norm_tags:
                if match_any_tag:
                    if pack.tags.isdisjoint(norm_tags):
                        continue
                else:
                    if not norm_tags.issubset(pack.tags):
                        continue
            if norm_use_case and norm_use_case not in pack.use_cases:
                continue

            score = _score_match(pack, query=query, use_case=norm_use_case)
            scored.append((score, pack))

        if sort_by == "name":
            results = [p for _, p in sorted(scored, key=lambda sp: sp[1].name.lower())]
        else:
            # Default: relevance
            results = [p for _, p in sorted(scored, key=lambda sp: (-sp[0], sp[1].name.lower()))]

        if limit is not None and limit >= 0:
            results = results[:limit]
        return results

    def find_packs_by_detector(self, detector_identifier: str) -> List[DetectorPack]:
        """Find packs that include a specific detector by id/name."""
        needle = detector_identifier.lower().strip()
        return [p for p in self._packs.values() if any(needle in d.lower() for d in p.detectors)]

    # ------------- Import / Export -------------

    def export_packs(self) -> List[Dict[str, Any]]:
        """Export all packs to a list of dicts."""
        return [p.to_dict() for p in self.list_packs()]

    def export_templates(self) -> List[Dict[str, Any]]:
        """Export all templates to a list of dicts."""
        return [t.to_dict() for t in self.list_templates()]

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Export registry packs and templates to JSON string."""
        payload = {
            "packs": self.export_packs(),
            "templates": self.export_templates(),
        }
        return json.dumps(payload, indent=indent)

    def save(self, path: str) -> None:
        """Save packs and templates to a file (JSON or YAML by extension)."""
        payload = {"packs": self.export_packs(), "templates": self.export_templates()}
        if path.lower().endswith((".yaml", ".yml")):
            if not _HAS_YAML:
                raise RegistryError("PyYAML is required to save YAML files.")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)  # type: ignore[name-defined]
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def load(self, path: str, *, allow_override: bool = False) -> None:
        """Load packs and templates from a JSON/YAML file and register them."""
        if path.lower().endswith((".yaml", ".yml")):
            if not _HAS_YAML:
                raise RegistryError("PyYAML is required to load YAML files.")
            with open(path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f) or {}  # type: ignore[name-defined]
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        for p in payload.get("packs", []) or []:
            self.register_pack(DetectorPack.from_dict(p), allow_override=allow_override)
        for t in payload.get("templates", []) or []:
            self.register_template(UseCaseTemplate.from_dict(t))

    # ------------- Internals -------------

    def _index_pack_into_industry(self, pack: DetectorPack) -> None:
        if pack.industry:
            ip = self._industry_packs.get(pack.industry)
            if not ip:
                ip = IndustryPack(industry=pack.industry, packs=[])
                self._industry_packs[pack.industry] = ip
            # Ensure uniqueness inside the list
            by_id = {p.pack_id: p for p in ip.packs}
            by_id[pack.pack_id] = pack
            ip.packs = sorted(by_id.values(), key=lambda p: p.name.lower())

    def _rebuild_industry_index(self) -> None:
        self._industry_packs.clear()
        for p in self._packs.values():
            self._index_pack_into_industry(p)

    def _initialize_default_packs(self) -> None:
        """Initialize default detector packs."""
        # Financial compliance pack
        financial_pack = DetectorPack(
            pack_id="financial-compliance-v2",
            name="Financial Compliance Pack",
            description="Detectors for financial compliance and fraud detection",
            detectors=[
                "fraud-detector",
                "pii-detector",
                "compliance-monitor",
            ],
            configuration={
                "fraud_threshold": 0.7,
                "pii_redaction": True,
                "compliance_standards": ["SOX", "PCI-DSS"],
            },
            industry=Industry.FINANCIAL,
            use_cases=["fraud-detection", "compliance-monitoring", "transaction-screening"],
            tags={"financial", "compliance", "fraud", "certified"},
            version="2.0.0",
        )
        self.register_pack(financial_pack)

        # Healthcare HIPAA pack
        healthcare_pack = DetectorPack(
            pack_id="healthcare-hipaa",
            name="Healthcare HIPAA Compliance Pack",
            description="Detectors for HIPAA compliance and patient data protection",
            detectors=[
                "phi-detector",
                "hipaa-compliance",
                "medical-ethics",
            ],
            configuration={
                "phi_detection": True,
                "anonymization": "aggressive",
                "compliance_standards": ["HIPAA"],
            },
            industry=Industry.HEALTHCARE,
            use_cases=["patient-data-protection", "hipaa-compliance", "medical-records"],
            tags={"healthcare", "hipaa", "compliance", "certified"},
        )
        self.register_pack(healthcare_pack)

        # Legal compliance pack
        legal_pack = DetectorPack(
            pack_id="legal-compliance",
            name="Legal Compliance Pack",
            description="Detectors for legal document handling and compliance",
            detectors=[
                "legal-detector",
                "confidentiality-monitor",
                "privilege-detector",
            ],
            configuration={
                "confidentiality_level": "high",
                "privilege_detection": True,
            },
            industry=Industry.LEGAL,
            use_cases=["document-review", "privilege-detection", "confidentiality"],
            tags={"legal", "compliance", "confidentiality"},
        )
        self.register_pack(legal_pack)

        # Technology security pack
        technology_pack = DetectorPack(
            pack_id="tech-security",
            name="Technology Security & Secrets Pack",
            description="Detectors for credentials, secrets, and code security hygiene",
            detectors=[
                "secret-scanner",
                "license-compatibility",
                "dependency-vulnerability",
            ],
            configuration={
                "secret_providers": ["generic", "aws", "gcp", "azure"],
                "max_severity": "high",
            },
            industry=Industry.TECHNOLOGY,
            use_cases=["secret-detection", "supply-chain-security", "license-compliance"],
            tags={"technology", "security", "secrets", "devsecops"},
        )
        self.register_pack(technology_pack)

        # Education academic integrity pack
        education_pack = DetectorPack(
            pack_id="education-integrity",
            name="Academic Integrity Pack",
            description="Detectors for plagiarism, AI-generated content, and citation checks",
            detectors=[
                "plagiarism-detector",
                "ai-generated-content",
                "citation-checker",
            ],
            configuration={"plagiarism_threshold": 0.85, "ai_content_sensitivity": "medium"},
            industry=Industry.EDUCATION,
            use_cases=["plagiarism-detection", "academic-integrity", "citation-compliance"],
            tags={"education", "integrity", "ai-content"},
        )
        self.register_pack(education_pack)

        # Retail PCI pack
        retail_pack = DetectorPack(
            pack_id="retail-pci",
            name="Retail PCI DSS Pack",
            description="Detectors for payment card data protection and PCI DSS compliance",
            detectors=[
                "pci-detector",
                "pii-detector",
                "tokenization-check",
            ],
            configuration={
                "mask_pan": True,
                "tokenization_required": True,
                "compliance_standards": ["PCI-DSS"],
            },
            industry=Industry.RETAIL,
            use_cases=["pci-compliance", "payment-security", "customer-data-protection"],
            tags={"retail", "pci", "compliance"},
        )
        self.register_pack(retail_pack)

        # Government FOIA and data classification pack
        government_pack = DetectorPack(
            pack_id="government-data-classification",
            name="Government Data Classification Pack",
            description="Detectors for data classification, FOIA, and sensitive information handling",
            detectors=[
                "classification-detector",
                "foia-compliance",
                "sensitive-entity-detector",
            ],
            configuration={"classifications": ["CUI", "FOUO", "SBU"], "auto_redaction": True},
            industry=Industry.GOVERNMENT,
            use_cases=["data-classification", "foia-compliance", "sensitive-data"],
            tags={"government", "classification", "compliance"},
        )
        self.register_pack(government_pack)

        # Manufacturing IP protection pack
        manufacturing_pack = DetectorPack(
            pack_id="manufacturing-ip",
            name="Manufacturing IP Protection Pack",
            description="Detectors for trade secrets, CAD/IP protection, and export controls",
            detectors=[
                "ip-detector",
                "export-control-check",
                "sensitive-drawing-detector",
            ],
            configuration={
                "export_regimes": ["EAR", "ITAR"],
                "ip_keywords": ["confidential", "trade secret"],
            },
            industry=Industry.MANUFACTURING,
            use_cases=["ip-protection", "export-compliance", "design-security"],
            tags={"manufacturing", "ip", "compliance"},
        )
        self.register_pack(manufacturing_pack)

    def _initialize_default_templates(self) -> None:
        """Initialize default use-case templates with simple example code."""
        self.register_template(
            UseCaseTemplate(
                template_id="fraud-detection",
                name="Fraud Detection",
                description="Detect suspicious transactions and compliance risks.",
                detector_pack_id="financial-compliance-v2",
                configuration={"fraud_threshold": 0.75},
                example_code="""
# Example: Run financial compliance detectors on a transaction batch
from nethical.marketplace.detector_packs import DetectorPackRegistry

registry = DetectorPackRegistry()
pack = registry.get_pack("financial-compliance-v2")
for detector in pack.detectors:
    print(f"Running detector {detector} with config {pack.configuration}")
""".strip(),
                example_code_language="python",
            )
        )
        self.register_template(
            UseCaseTemplate(
                template_id="hipaa-compliance",
                name="HIPAA Compliance",
                description="Ensure PHI is detected and anonymized for patient data.",
                detector_pack_id="healthcare-hipaa",
                configuration={"anonymization": "aggressive"},
                example_code="""
# Example: Enforce PHI anonymization policy
from nethical.marketplace.detector_packs import DetectorPackRegistry

registry = DetectorPackRegistry()
pack = registry.get_pack("healthcare-hipaa")
pii_redaction = pack.configuration.get("anonymization", "moderate")
print("Anonymization level:", pii_redaction)
""".strip(),
                example_code_language="python",
            )
        )
        self.register_template(
            UseCaseTemplate(
                template_id="document-review",
                name="Legal Document Review",
                description="Scan legal documents for privilege and confidentiality.",
                detector_pack_id="legal-compliance",
                configuration={"confidentiality_level": "high"},
                example_code="""
# Example: Perform privilege screening in a doc pipeline
from nethical.marketplace.detector_packs import DetectorPackRegistry

registry = DetectorPackRegistry()
pack = registry.get_pack("legal-compliance")
print("Detectors:", ", ".join(pack.detectors))
""".strip(),
                example_code_language="python",
            )
        )
