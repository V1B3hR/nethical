"""Pre-built Detector Packs for Common Use Cases.

This module provides industry-specific detector bundles and
use case templates for quick deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum


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


@dataclass
class IndustryPack:
    """Industry-specific detector pack."""
    industry: Industry
    packs: List[DetectorPack]
    compliance_standards: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


@dataclass
class UseCaseTemplate:
    """Template for specific use case."""
    template_id: str
    name: str
    description: str
    detector_pack_id: str
    configuration: Dict[str, Any]
    example_code: str = ""


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
    
    def _initialize_default_packs(self):
        """Initialize default detector packs."""
        # Financial compliance pack
        financial_pack = DetectorPack(
            pack_id="financial-compliance-v2",
            name="Financial Compliance Pack",
            description="Detectors for financial compliance and fraud detection",
            detectors=[
                "fraud-detector",
                "pii-detector",
                "compliance-monitor"
            ],
            configuration={
                "fraud_threshold": 0.7,
                "pii_redaction": True,
                "compliance_standards": ["SOX", "PCI-DSS"]
            },
            industry=Industry.FINANCIAL,
            use_cases=["fraud-detection", "compliance-monitoring", "transaction-screening"],
            tags={"financial", "compliance", "fraud", "certified"}
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
                "medical-ethics"
            ],
            configuration={
                "phi_detection": True,
                "anonymization": "aggressive",
                "compliance_standards": ["HIPAA"]
            },
            industry=Industry.HEALTHCARE,
            use_cases=["patient-data-protection", "hipaa-compliance", "medical-records"],
            tags={"healthcare", "hipaa", "compliance", "certified"}
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
                "privilege-detector"
            ],
            configuration={
                "confidentiality_level": "high",
                "privilege_detection": True
            },
            industry=Industry.LEGAL,
            use_cases=["document-review", "privilege-detection", "confidentiality"],
            tags={"legal", "compliance", "confidentiality"}
        )
        self.register_pack(legal_pack)
    
    def register_pack(self, pack: DetectorPack):
        """Register a detector pack.
        
        Args:
            pack: Detector pack to register
        """
        self._packs[pack.pack_id] = pack
        
        if pack.industry:
            if pack.industry not in self._industry_packs:
                self._industry_packs[pack.industry] = IndustryPack(
                    industry=pack.industry,
                    packs=[]
                )
            self._industry_packs[pack.industry].packs.append(pack)
    
    def get_pack(self, pack_id: str) -> Optional[DetectorPack]:
        """Get a detector pack by ID.
        
        Args:
            pack_id: Pack identifier
            
        Returns:
            Detector pack or None
        """
        return self._packs.get(pack_id)
    
    def get_industry_pack(self, industry: Industry) -> Optional[IndustryPack]:
        """Get industry-specific detector packs.
        
        Args:
            industry: Industry category
            
        Returns:
            Industry pack or None
        """
        return self._industry_packs.get(industry)
    
    def search_packs(
        self,
        industry: Optional[Industry] = None,
        tags: Optional[Set[str]] = None,
        use_case: Optional[str] = None
    ) -> List[DetectorPack]:
        """Search for detector packs.
        
        Args:
            industry: Filter by industry
            tags: Filter by tags
            use_case: Filter by use case
            
        Returns:
            List of matching packs
        """
        results = []
        
        for pack in self._packs.values():
            if industry and pack.industry != industry:
                continue
            
            if tags and not tags.issubset(pack.tags):
                continue
            
            if use_case and use_case not in pack.use_cases:
                continue
            
            results.append(pack)
        
        return results
    
    def register_template(self, template: UseCaseTemplate):
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
    
    def list_packs(self) -> List[DetectorPack]:
        """List all available detector packs.
        
        Returns:
            List of all packs
        """
        return list(self._packs.values())
