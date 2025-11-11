"""Ethical Taxonomy Layer for Phase 4.4: Ethical Impact Layer.

This module implements:
- Tag violations with ethical dimension taxonomy (privacy, manipulation, fairness, safety)
- Integrate ethical taxonomy mapping (ethics_taxonomy.json)
- Track coverage (>90% target)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict


@dataclass
class EthicalDimension:
    """Ethical dimension definition."""

    name: str
    description: str
    weight: float = 1.0
    severity_multiplier: float = 1.0
    indicators: List[str] = field(default_factory=list)


@dataclass
class EthicalTag:
    """Ethical tag for a violation."""

    dimension: str
    score: float  # 0-1
    confidence: float = 1.0
    description: str = ""


@dataclass
class ViolationTagging:
    """Complete ethical tagging for a violation."""

    violation_type: str
    tags: List[EthicalTag] = field(default_factory=list)
    primary_dimension: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EthicalTaxonomy:
    """Ethical taxonomy system for multi-dimensional impact classification."""

    def __init__(self, taxonomy_path: str = "ethics_taxonomy.json", coverage_target: float = 0.9):
        """Initialize ethical taxonomy.

        Args:
            taxonomy_path: Path to taxonomy configuration
            coverage_target: Target coverage percentage
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.coverage_target = coverage_target

        # Load taxonomy
        self.taxonomy = self._load_taxonomy()

        # Dimensions
        self.dimensions: Dict[str, EthicalDimension] = self._load_dimensions()

        # Violation-to-dimension mapping
        self.mapping: Dict[str, Dict[str, float]] = self.taxonomy.get("mapping", {})

        # Aggregation rules
        self.aggregation_rules = self.taxonomy.get("aggregation_rules", {})

        # Coverage tracking
        self.violation_types_seen: Set[str] = set()
        self.tagged_violations: Set[str] = set()
        self.coverage_history: List[Dict[str, Any]] = []

    def _load_taxonomy(self) -> Dict[str, Any]:
        """Load taxonomy configuration from file.

        Returns:
            Taxonomy dictionary
        """
        if not self.taxonomy_path.exists():
            # Return default taxonomy
            return {"version": "1.0", "dimensions": {}, "mapping": {}, "coverage_target": 0.9}

        with open(self.taxonomy_path, "r") as f:
            return json.load(f)

    def _load_dimensions(self) -> Dict[str, EthicalDimension]:
        """Load ethical dimensions from taxonomy.

        Returns:
            Dictionary of dimensions
        """
        dimensions = {}

        for name, config in self.taxonomy.get("dimensions", {}).items():
            dimensions[name] = EthicalDimension(
                name=name,
                description=config.get("description", ""),
                weight=config.get("weight", 1.0),
                severity_multiplier=config.get("severity_multiplier", 1.0),
                indicators=config.get("indicators", []),
            )

        return dimensions

    def tag_violation(
        self, violation_type: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Tag violation with ethical dimensions.

        Args:
            violation_type: Type of violation
            context: Additional context for scoring

        Returns:
            Dictionary of dimension scores
        """
        # Track violation type
        self.violation_types_seen.add(violation_type)

        # Get base mapping
        base_mapping = self.mapping.get(violation_type, {})

        if base_mapping:
            self.tagged_violations.add(violation_type)

        # Filter out non-numeric fields (like 'description')
        scores = {k: v for k, v in base_mapping.items() if isinstance(v, (int, float))}

        # Apply context adjustments if provided
        if context:
            scores = self._apply_context_adjustments(scores, context)

        # Normalize scores
        scores = self._normalize_scores(scores)

        return scores

    def _apply_context_adjustments(
        self, base_scores: Dict[str, float], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply context-based adjustments to scores.

        Args:
            base_scores: Base dimension scores
            context: Context information

        Returns:
            Adjusted scores
        """
        scores = dict(base_scores)

        # Check for sensitive data
        if context.get("sensitive", False):
            scores["privacy"] = scores.get("privacy", 0) + 0.2

        if context.get("personal_data", False):
            scores["privacy"] = scores.get("privacy", 0) + 0.3

        # Check for automation
        if context.get("automated", False):
            scores["fairness"] = scores.get("fairness", 0) + 0.1

        # Check for privileged context
        if context.get("is_privileged", False):
            scores["safety"] = scores.get("safety", 0) + 0.2

        return scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0-1 range.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        normalized = {}

        for dimension, score in scores.items():
            normalized[dimension] = min(max(score, 0.0), 1.0)

        return normalized

    def get_primary_dimension(self, scores: Dict[str, float]) -> Optional[str]:
        """Get primary ethical dimension.

        Args:
            scores: Dimension scores

        Returns:
            Primary dimension name or None
        """
        if not scores:
            return None

        threshold = self.aggregation_rules.get("primary_dimension_threshold", 0.6)

        # Find highest score
        max_dimension = max(scores.items(), key=lambda x: x[1])

        if max_dimension[1] >= threshold:
            return max_dimension[0]

        return None

    def create_tagging(
        self, violation_type: str, context: Optional[Dict[str, Any]] = None
    ) -> ViolationTagging:
        """Create complete ethical tagging for violation.

        Args:
            violation_type: Type of violation
            context: Additional context

        Returns:
            Violation tagging
        """
        # Get dimension scores
        scores = self.tag_violation(violation_type, context)

        # Create tags
        tags = []
        for dimension, score in scores.items():
            if score > 0:
                dim_config = self.dimensions.get(dimension)
                tags.append(
                    EthicalTag(
                        dimension=dimension,
                        score=score,
                        confidence=1.0,
                        description=dim_config.description if dim_config else "",
                    )
                )

        # Determine primary dimension
        primary = self.get_primary_dimension(scores)

        return ViolationTagging(
            violation_type=violation_type,
            tags=tags,
            primary_dimension=primary,
            metadata=context or {},
        )

    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get taxonomy coverage statistics.

        Returns:
            Coverage statistics
        """
        total_types = len(self.violation_types_seen)
        tagged_types = len(self.tagged_violations)

        if total_types == 0:
            coverage = 0.0
        else:
            coverage = tagged_types / total_types

        # Check against known mappings
        known_types = set(self.mapping.keys())
        unmapped_types = self.violation_types_seen - known_types

        return {
            "total_violation_types": total_types,
            "tagged_types": tagged_types,
            "coverage_percentage": coverage * 100,
            "target_percentage": self.coverage_target * 100,
            "meets_target": coverage >= self.coverage_target,
            "known_mappings": len(self.mapping),
            "unmapped_types": list(unmapped_types),
            "unmapped_count": len(unmapped_types),
        }

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get detailed coverage report.

        Returns:
            Coverage report
        """
        stats = self.get_coverage_stats()

        # Dimension usage
        dimension_usage = defaultdict(int)
        for mapping in self.mapping.values():
            for dimension in mapping.keys():
                dimension_usage[dimension] += 1

        # Most common dimensions
        sorted_dimensions = sorted(dimension_usage.items(), key=lambda x: x[1], reverse=True)

        return {
            **stats,
            "dimension_usage": dict(dimension_usage),
            "most_common_dimensions": [
                {"dimension": d, "usage_count": c} for d, c in sorted_dimensions[:5]
            ],
            "total_dimensions": len(self.dimensions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def add_mapping(
        self, violation_type: str, dimension_scores: Dict[str, float], description: str = ""
    ):
        """Add new violation type mapping.

        Args:
            violation_type: Violation type
            dimension_scores: Dimension scores
            description: Description
        """
        self.mapping[violation_type] = dimension_scores

        # Update taxonomy file
        self.taxonomy["mapping"][violation_type] = {**dimension_scores, "description": description}

        # Save updated taxonomy
        self._save_taxonomy()

    def _save_taxonomy(self):
        """Save taxonomy to file."""
        with open(self.taxonomy_path, "w") as f:
            json.dump(self.taxonomy, f, indent=2)

    def get_dimension_report(self, dimension: str) -> Dict[str, Any]:
        """Get report for specific dimension.

        Args:
            dimension: Dimension name

        Returns:
            Dimension report
        """
        if dimension not in self.dimensions:
            return {"error": "Unknown dimension"}

        dim_config = self.dimensions[dimension]

        # Count violations tagged with this dimension
        violations_with_dimension = [
            vtype for vtype, scores in self.mapping.items() if dimension in scores
        ]

        # Average score for this dimension
        scores = [scores[dimension] for scores in self.mapping.values() if dimension in scores]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "dimension": dimension,
            "description": dim_config.description,
            "weight": dim_config.weight,
            "severity_multiplier": dim_config.severity_multiplier,
            "total_violations": len(violations_with_dimension),
            "average_score": avg_score,
            "violations": violations_with_dimension[:10],  # Sample
        }

    def validate_taxonomy(self) -> Dict[str, Any]:
        """Validate taxonomy configuration.

        Returns:
            Validation results
        """
        issues = []
        warnings = []

        # Check all dimensions exist
        for vtype, scores in self.mapping.items():
            for dimension in scores.keys():
                if dimension not in self.dimensions:
                    issues.append(f"Unknown dimension '{dimension}' in mapping for '{vtype}'")

        # Check coverage
        stats = self.get_coverage_stats()
        if not stats["meets_target"]:
            warnings.append(
                f"Coverage {stats['coverage_percentage']:.1f}% below target "
                f"{stats['target_percentage']:.1f}%"
            )

        # Check for unmapped violations
        if stats["unmapped_count"] > 0:
            warnings.append(f"{stats['unmapped_count']} violation type(s) have no ethical mapping")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "coverage_stats": stats,
        }
