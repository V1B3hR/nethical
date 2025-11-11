"""Taxonomy Validation API and Schema Management for Phase 2.1.

This module implements:
- Taxonomy validation API endpoint
- JSON Schema export for versioning
- Industry-specific taxonomy support
- Taxonomy versioning and migration
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from jsonschema import validate, ValidationError


@dataclass
class TaxonomyVersion:
    """Represents a versioned taxonomy."""

    version: str
    description: str
    created_at: datetime
    dimensions: Dict[str, Any]
    mapping: Dict[str, Any]
    industry: Optional[str] = None
    deprecated: bool = False
    migration_notes: str = ""


class TaxonomyValidator:
    """Validator for ethical taxonomy configurations."""

    # Default schema hosting location (can be overridden)
    DEFAULT_SCHEMA_ID = "https://nethical.io/schemas/taxonomy/v1.0.0"

    def __init__(self, schema_id: Optional[str] = None):
        """Initialize taxonomy validator with schema.

        Args:
            schema_id: Optional custom schema ID URL
        """
        self.schema_id = schema_id or self.DEFAULT_SCHEMA_ID
        self.schema = self._generate_json_schema()

    def _generate_json_schema(self) -> Dict[str, Any]:
        """Generate JSON Schema for taxonomy validation.

        Returns:
            JSON Schema definition
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": self.schema_id,
            "title": "Ethical Taxonomy Schema",
            "description": "Schema for Nethical ethical dimension taxonomy",
            "type": "object",
            "required": ["version", "description", "dimensions", "mapping"],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
                    "description": "Semantic version of the taxonomy",
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of taxonomy",
                },
                "industry": {
                    "type": "string",
                    "enum": [
                        "general",
                        "healthcare",
                        "finance",
                        "education",
                        "retail",
                        "government",
                    ],
                    "description": "Industry-specific taxonomy identifier",
                },
                "dimensions": {
                    "type": "object",
                    "description": "Ethical dimensions",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["description", "weight"],
                        "properties": {
                            "description": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 2},
                            "severity_multiplier": {"type": "number", "minimum": 0, "maximum": 5},
                            "indicators": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "mapping": {
                    "type": "object",
                    "description": "Violation type to dimension mapping",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "privacy": {"type": "number", "minimum": 0, "maximum": 1},
                            "manipulation": {"type": "number", "minimum": 0, "maximum": 1},
                            "fairness": {"type": "number", "minimum": 0, "maximum": 1},
                            "safety": {"type": "number", "minimum": 0, "maximum": 1},
                            "transparency": {"type": "number", "minimum": 0, "maximum": 1},
                            "accountability": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
                "coverage_target": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Target coverage percentage",
                },
                "minimum_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "aggregation_rules": {
                    "type": "object",
                    "description": "Rules for aggregating dimensions",
                },
                "reporting": {"type": "object", "description": "Reporting configuration"},
            },
        }

    def validate_taxonomy(self, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
        """Validate taxonomy against schema.

        Args:
            taxonomy: Taxonomy configuration to validate

        Returns:
            Validation result with issues and warnings
        """
        issues = []
        warnings = []

        # Schema validation
        try:
            validate(instance=taxonomy, schema=self.schema)
        except ValidationError as e:
            issues.append(f"Schema validation failed: {e.message}")
            return {"valid": False, "issues": issues, "warnings": warnings}

        # Semantic validation
        dimensions = taxonomy.get("dimensions", {})
        mapping = taxonomy.get("mapping", {})

        # Check that all mapping dimensions exist
        defined_dims = set(dimensions.keys())
        for vtype, scores in mapping.items():
            used_dims = {k for k in scores.keys() if k != "description"}
            unknown = used_dims - defined_dims
            if unknown:
                issues.append(f"Violation '{vtype}' references unknown dimensions: {unknown}")

        # Check coverage
        if len(mapping) == 0:
            warnings.append("No violation mappings defined")

        # Check for unused dimensions
        used_dims = set()
        for scores in mapping.values():
            used_dims.update(k for k in scores.keys() if k != "description")
        unused = defined_dims - used_dims
        if unused:
            warnings.append(f"Unused dimensions: {unused}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": {
                "dimension_count": len(dimensions),
                "mapping_count": len(mapping),
                "dimensions_used": len(used_dims),
            },
        }

    def export_schema(self, output_path: Optional[str] = None) -> str:
        """Export taxonomy JSON schema.

        Args:
            output_path: Optional path to save schema

        Returns:
            JSON schema as string
        """
        schema_str = json.dumps(self.schema, indent=2)

        if output_path:
            Path(output_path).write_text(schema_str)

        return schema_str

    def load_and_validate(self, taxonomy_path: str) -> Dict[str, Any]:
        """Load and validate taxonomy file.

        Args:
            taxonomy_path: Path to taxonomy JSON file

        Returns:
            Validation result
        """
        taxonomy = json.loads(Path(taxonomy_path).read_text())
        return self.validate_taxonomy(taxonomy)


class IndustryTaxonomyManager:
    """Manages industry-specific taxonomies."""

    def __init__(self, base_taxonomy_path: str = "ethics_taxonomy.json"):
        """Initialize industry taxonomy manager.

        Args:
            base_taxonomy_path: Path to base taxonomy
        """
        # Try policies directory first if relative path
        path = Path(base_taxonomy_path)
        if not path.is_absolute() and not path.exists():
            # Try policies subdirectory
            policies_path = Path(__file__).parent.parent.parent / "policies" / base_taxonomy_path
            if policies_path.exists():
                path = policies_path
        
        self.base_taxonomy_path = path
        self.validator = TaxonomyValidator()
        self.industry_taxonomies: Dict[str, Dict[str, Any]] = {}
        self._load_industry_taxonomies()

    def _load_industry_taxonomies(self):
        """Load all industry-specific taxonomies."""
        # Check for industry taxonomy directory
        taxonomy_dir = self.base_taxonomy_path.parent / "taxonomies"
        if taxonomy_dir.exists():
            for tax_file in taxonomy_dir.glob("*.json"):
                taxonomy = json.loads(tax_file.read_text())
                industry = taxonomy.get("industry", "general")
                self.industry_taxonomies[industry] = taxonomy

    def get_taxonomy_for_industry(self, industry: str) -> Dict[str, Any]:
        """Get taxonomy for specific industry.

        Args:
            industry: Industry identifier

        Returns:
            Industry-specific or general taxonomy
        """
        if industry in self.industry_taxonomies:
            return self.industry_taxonomies[industry]

        # Return base taxonomy
        if self.base_taxonomy_path.exists():
            return json.loads(self.base_taxonomy_path.read_text())

        return {}

    def create_industry_taxonomy(
        self,
        industry: str,
        base_version: str = "1.0",
        additional_dimensions: Optional[Dict[str, Any]] = None,
        additional_mappings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create industry-specific taxonomy from base.

        Args:
            industry: Industry name
            base_version: Base taxonomy version
            additional_dimensions: Industry-specific dimensions
            additional_mappings: Industry-specific mappings

        Returns:
            Industry taxonomy configuration
        """
        # Load base taxonomy
        base = json.loads(self.base_taxonomy_path.read_text())

        # Create industry version
        industry_taxonomy = {
            **base,
            "version": f"{base_version}-{industry}",
            "industry": industry,
            "description": f"{base.get('description', '')} - {industry.title()} Specific",
        }

        # Add industry dimensions
        if additional_dimensions:
            industry_taxonomy["dimensions"].update(additional_dimensions)

        # Add industry mappings
        if additional_mappings:
            industry_taxonomy["mapping"].update(additional_mappings)

        return industry_taxonomy

    def save_industry_taxonomy(self, taxonomy: Dict[str, Any], industry: str):
        """Save industry-specific taxonomy.

        Args:
            taxonomy: Taxonomy configuration
            industry: Industry name
        """
        # Create taxonomies directory
        taxonomy_dir = self.base_taxonomy_path.parent / "taxonomies"
        taxonomy_dir.mkdir(exist_ok=True)

        # Save taxonomy
        output_path = taxonomy_dir / f"{industry}_taxonomy.json"
        output_path.write_text(json.dumps(taxonomy, indent=2))

        # Update cache
        self.industry_taxonomies[industry] = taxonomy


# Healthcare-specific taxonomy extensions
HEALTHCARE_DIMENSIONS = {
    "patient_safety": {
        "description": "Protection of patient health and wellbeing",
        "weight": 1.5,
        "severity_multiplier": 2.0,
        "indicators": [
            "clinical_risk",
            "medication_error",
            "diagnostic_error",
            "treatment_harm",
            "care_delay",
        ],
    },
    "medical_privacy": {
        "description": "HIPAA and patient confidentiality",
        "weight": 1.3,
        "severity_multiplier": 1.8,
        "indicators": ["phi_exposure", "unauthorized_medical_access", "consent_violation_hipaa"],
    },
}

HEALTHCARE_MAPPINGS = {
    "unauthorized_medical_record_access": {
        "medical_privacy": 1.0,
        "privacy": 0.9,
        "safety": 0.4,
        "description": "Accessing medical records without authorization",
    },
    "medication_recommendation_error": {
        "patient_safety": 1.0,
        "safety": 0.9,
        "description": "Incorrect medication recommendation",
    },
}

# Finance-specific taxonomy extensions
FINANCE_DIMENSIONS = {
    "financial_harm": {
        "description": "Risk of financial loss or fraud",
        "weight": 1.4,
        "severity_multiplier": 1.6,
        "indicators": [
            "fraud_risk",
            "unauthorized_transaction",
            "market_manipulation",
            "insider_trading",
        ],
    },
    "regulatory_compliance": {
        "description": "Compliance with financial regulations",
        "weight": 1.2,
        "severity_multiplier": 1.5,
        "indicators": ["aml_violation", "kyc_failure", "reporting_violation"],
    },
}

FINANCE_MAPPINGS = {
    "unauthorized_financial_transaction": {
        "financial_harm": 1.0,
        "privacy": 0.6,
        "safety": 0.7,
        "description": "Unauthorized financial transaction attempt",
    },
    "insider_information_use": {
        "financial_harm": 1.0,
        "fairness": 0.8,
        "manipulation": 0.6,
        "description": "Using insider information for trading",
    },
}

# Education-specific taxonomy extensions
EDUCATION_DIMENSIONS = {
    "learning_integrity": {
        "description": "Academic integrity and honest assessment",
        "weight": 1.2,
        "severity_multiplier": 1.3,
        "indicators": [
            "plagiarism",
            "cheating",
            "fraudulent_credentials",
            "assessment_manipulation",
        ],
    },
    "student_welfare": {
        "description": "Student safety and wellbeing",
        "weight": 1.3,
        "severity_multiplier": 1.5,
        "indicators": ["student_risk", "bullying", "inappropriate_content", "exploitation"],
    },
}

EDUCATION_MAPPINGS = {
    "plagiarism_detection_bypass": {
        "learning_integrity": 1.0,
        "manipulation": 0.7,
        "fairness": 0.5,
        "description": "Attempting to bypass plagiarism detection",
    },
    "grade_manipulation_attempt": {
        "learning_integrity": 1.0,
        "fairness": 0.9,
        "manipulation": 0.8,
        "description": "Attempting to manipulate grades",
    },
}
