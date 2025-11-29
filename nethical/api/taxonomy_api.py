"""Taxonomy API endpoints for Phase 2.1.

This module provides REST API endpoints for:
- Taxonomy validation
- Schema export
- Industry taxonomy management
- Version management
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..core.taxonomy_validator import (
    TaxonomyValidator,
    IndustryTaxonomyManager,
    HEALTHCARE_DIMENSIONS,
    HEALTHCARE_MAPPINGS,
    FINANCE_DIMENSIONS,
    FINANCE_MAPPINGS,
    EDUCATION_DIMENSIONS,
    EDUCATION_MAPPINGS,
)
from ..core.ethical_taxonomy import EthicalTaxonomy


@dataclass
class APIResponse:
    """Standard API response format."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.message:
            result["message"] = self.message
        return result


class TaxonomyAPI:
    """REST API for taxonomy management."""

    def __init__(self, taxonomy_path: str = "taxonomies/ethics_taxonomy.json"):
        """Initialize taxonomy API.

        Args:
            taxonomy_path: Path to base taxonomy file
        """
        self.taxonomy_path = taxonomy_path
        self.validator = TaxonomyValidator()
        self.industry_manager = IndustryTaxonomyManager(taxonomy_path)
        self.taxonomy = EthicalTaxonomy(taxonomy_path)

    def validate_taxonomy_endpoint(
        self, taxonomy_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """API endpoint: Validate taxonomy configuration.

        Args:
            taxonomy_data: Taxonomy to validate (or current if None)

        Returns:
            Validation response
        """
        try:
            if taxonomy_data is None:
                # Validate current taxonomy
                result = self.taxonomy.validate_taxonomy()
            else:
                # Validate provided taxonomy
                result = self.validator.validate_taxonomy(taxonomy_data)

            return APIResponse(
                success=result["valid"], data=result, message="Validation completed"
            ).to_dict()

        except Exception as e:
            return APIResponse(success=False, error=str(e), message="Validation failed").to_dict()

    def get_schema_endpoint(self) -> Dict[str, Any]:
        """API endpoint: Get taxonomy JSON schema.

        Returns:
            JSON schema response
        """
        try:
            schema = self.validator.schema
            return APIResponse(
                success=True, data={"schema": schema}, message="Schema retrieved successfully"
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve schema"
            ).to_dict()

    def export_schema_endpoint(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint: Export taxonomy schema to file.

        Args:
            output_path: Optional output path

        Returns:
            Export response
        """
        try:
            schema_str = self.validator.export_schema(output_path)
            return APIResponse(
                success=True,
                data={"schema": schema_str, "saved_to": output_path},
                message="Schema exported successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to export schema"
            ).to_dict()

    def list_industries_endpoint(self) -> Dict[str, Any]:
        """API endpoint: List available industry taxonomies.

        Returns:
            List of industries
        """
        try:
            industries = list(self.industry_manager.industry_taxonomies.keys())
            return APIResponse(
                success=True,
                data={"industries": industries, "count": len(industries)},
                message="Industries retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve industries"
            ).to_dict()

    def get_industry_taxonomy_endpoint(self, industry: str) -> Dict[str, Any]:
        """API endpoint: Get taxonomy for specific industry.

        Args:
            industry: Industry identifier

        Returns:
            Industry taxonomy
        """
        try:
            taxonomy = self.industry_manager.get_taxonomy_for_industry(industry)
            return APIResponse(
                success=True,
                data={"industry": industry, "taxonomy": taxonomy},
                message="Industry taxonomy retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve industry taxonomy"
            ).to_dict()

    def create_industry_taxonomy_endpoint(
        self, industry: str, base_version: str = "1.0"
    ) -> Dict[str, Any]:
        """API endpoint: Create industry-specific taxonomy.

        Args:
            industry: Industry name (healthcare, finance, education)
            base_version: Base taxonomy version

        Returns:
            Created taxonomy
        """
        try:
            # Get industry-specific extensions
            additional_dims = None
            additional_maps = None

            if industry == "healthcare":
                additional_dims = HEALTHCARE_DIMENSIONS
                additional_maps = HEALTHCARE_MAPPINGS
            elif industry == "finance":
                additional_dims = FINANCE_DIMENSIONS
                additional_maps = FINANCE_MAPPINGS
            elif industry == "education":
                additional_dims = EDUCATION_DIMENSIONS
                additional_maps = EDUCATION_MAPPINGS

            # Create taxonomy
            taxonomy = self.industry_manager.create_industry_taxonomy(
                industry=industry,
                base_version=base_version,
                additional_dimensions=additional_dims,
                additional_mappings=additional_maps,
            )

            # Save it
            self.industry_manager.save_industry_taxonomy(taxonomy, industry)

            return APIResponse(
                success=True,
                data={"industry": industry, "taxonomy": taxonomy},
                message=f"Industry taxonomy for {industry} created successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to create industry taxonomy"
            ).to_dict()

    def get_coverage_stats_endpoint(self) -> Dict[str, Any]:
        """API endpoint: Get taxonomy coverage statistics.

        Returns:
            Coverage statistics
        """
        try:
            stats = self.taxonomy.get_coverage_stats()
            return APIResponse(
                success=True, data=stats, message="Coverage statistics retrieved successfully"
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve coverage statistics"
            ).to_dict()

    def get_coverage_report_endpoint(self) -> Dict[str, Any]:
        """API endpoint: Get detailed coverage report.

        Returns:
            Coverage report
        """
        try:
            report = self.taxonomy.get_coverage_report()
            return APIResponse(
                success=True, data=report, message="Coverage report generated successfully"
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to generate coverage report"
            ).to_dict()

    def tag_violation_endpoint(
        self, violation_type: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """API endpoint: Tag a violation with ethical dimensions.

        Args:
            violation_type: Type of violation
            context: Additional context

        Returns:
            Tagging results
        """
        try:
            scores = self.taxonomy.tag_violation(violation_type, context)
            tagging = self.taxonomy.create_tagging(violation_type, context)

            return APIResponse(
                success=True,
                data={
                    "violation_type": violation_type,
                    "scores": scores,
                    "primary_dimension": tagging.primary_dimension,
                    "tags": [
                        {
                            "dimension": tag.dimension,
                            "score": tag.score,
                            "confidence": tag.confidence,
                            "description": tag.description,
                        }
                        for tag in tagging.tags
                    ],
                },
                message="Violation tagged successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to tag violation"
            ).to_dict()

    def add_mapping_endpoint(
        self, violation_type: str, dimension_scores: Dict[str, float], description: str = ""
    ) -> Dict[str, Any]:
        """API endpoint: Add new violation type mapping.

        Args:
            violation_type: Violation type
            dimension_scores: Dimension scores
            description: Description

        Returns:
            Response
        """
        try:
            self.taxonomy.add_mapping(violation_type, dimension_scores, description)

            return APIResponse(
                success=True,
                data={"violation_type": violation_type, "dimension_scores": dimension_scores},
                message="Mapping added successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to add mapping"
            ).to_dict()

    def get_dimension_report_endpoint(self, dimension: str) -> Dict[str, Any]:
        """API endpoint: Get report for specific dimension.

        Args:
            dimension: Dimension name

        Returns:
            Dimension report
        """
        try:
            report = self.taxonomy.get_dimension_report(dimension)

            if "error" in report:
                return APIResponse(
                    success=False, error=report["error"], message="Failed to get dimension report"
                ).to_dict()

            return APIResponse(
                success=True, data=report, message="Dimension report generated successfully"
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to generate dimension report"
            ).to_dict()
