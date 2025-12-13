"""
Coverage Optimizer - Identifies gaps in detection coverage

This module analyzes detection coverage and identifies areas where
detectors may be weak or missing, using fuzzing and coverage-guided mutation.

Features:
- Detection coverage analysis
- Gap identification
- Fuzzing-based exploration
- Coverage-guided mutation

Alignment: Law 23 (Fail-Safe Design), Law 24 (Adaptive Learning)
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CoverageMetric(str, Enum):
    """Metrics for measuring detection coverage."""
    
    VECTOR_COVERAGE = "vector_coverage"  # % of attack vectors with detectors
    EDGE_CASE_COVERAGE = "edge_case_coverage"  # % of edge cases tested
    CATEGORY_COVERAGE = "category_coverage"  # % of categories covered
    DETECTOR_EFFECTIVENESS = "detector_effectiveness"  # Detection rate


@dataclass
class CoverageGap:
    """Identified gap in detection coverage."""
    
    id: str
    gap_type: str  # "missing_vector", "weak_detector", "untested_edge_case"
    category: str
    description: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    suggested_tests: List[str]
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CoverageReport:
    """Comprehensive coverage analysis report."""
    
    timestamp: datetime
    overall_coverage: float  # 0.0 to 1.0
    vector_coverage: float
    category_coverage: float
    gaps: List[CoverageGap]
    recommendations: List[str]
    statistics: Dict[str, Any]


class CoverageOptimizer:
    """
    Identifies gaps in detection coverage using fuzzing and analysis.
    
    This component systematically explores the attack surface to find
    areas where detection may be weak or missing.
    
    Methods:
    - Fuzzing: Random exploration of input space
    - Coverage-guided mutation: Evolve inputs to maximize coverage
    - Edge case identification: Find boundary conditions
    """
    
    def __init__(
        self,
        known_vectors: Optional[List[str]] = None,
        detector_registry: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the coverage optimizer.
        
        Args:
            known_vectors: List of known attack vector IDs
            detector_registry: Registry of available detectors
        """
        self.known_vectors = known_vectors or []
        self.detector_registry = detector_registry or {}
        self.coverage_history: List[CoverageReport] = []
        self.identified_gaps: List[CoverageGap] = []
        
        logger.info(
            f"CoverageOptimizer initialized with {len(self.known_vectors)} "
            f"known vectors and {len(self.detector_registry)} detectors"
        )
    
    async def analyze_coverage(
        self,
        test_results: Optional[List[Dict[str, Any]]] = None
    ) -> CoverageReport:
        """
        Analyze current detection coverage.
        
        Args:
            test_results: Results from recent attack tests
            
        Returns:
            Comprehensive coverage report
        """
        logger.info("Starting coverage analysis")
        
        # Calculate coverage metrics
        vector_coverage = await self._calculate_vector_coverage()
        category_coverage = await self._calculate_category_coverage()
        overall_coverage = (vector_coverage + category_coverage) / 2
        
        # Identify gaps
        gaps = await self._identify_gaps(test_results or [])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(gaps)
        
        # Compile statistics
        statistics = self._compile_statistics(test_results or [])
        
        report = CoverageReport(
            timestamp=datetime.now(timezone.utc),
            overall_coverage=overall_coverage,
            vector_coverage=vector_coverage,
            category_coverage=category_coverage,
            gaps=gaps,
            recommendations=recommendations,
            statistics=statistics
        )
        
        self.coverage_history.append(report)
        self.identified_gaps.extend(gaps)
        
        logger.info(
            f"Coverage analysis complete: {overall_coverage:.1%} overall, "
            f"{len(gaps)} gaps identified"
        )
        
        return report
    
    async def _calculate_vector_coverage(self) -> float:
        """Calculate percentage of attack vectors with detectors."""
        if not self.known_vectors:
            return 1.0
        
        vectors_with_detectors = 0
        for vector_id in self.known_vectors:
            if vector_id in self.detector_registry:
                vectors_with_detectors += 1
        
        coverage = vectors_with_detectors / len(self.known_vectors)
        logger.debug(
            f"Vector coverage: {vectors_with_detectors}/{len(self.known_vectors)} "
            f"= {coverage:.1%}"
        )
        
        return coverage
    
    async def _calculate_category_coverage(self) -> float:
        """Calculate coverage by attack category."""
        # Categorize known vectors
        categories = defaultdict(list)
        for vector_id in self.known_vectors:
            # Extract category from vector ID (e.g., "PI-001" -> "PI")
            category = vector_id.split("-")[0] if "-" in vector_id else "UNKNOWN"
            categories[category].append(vector_id)
        
        if not categories:
            return 1.0
        
        covered_categories = 0
        for category, vectors in categories.items():
            # Check if any vector in category has detector
            if any(v in self.detector_registry for v in vectors):
                covered_categories += 1
        
        coverage = covered_categories / len(categories)
        logger.debug(
            f"Category coverage: {covered_categories}/{len(categories)} "
            f"= {coverage:.1%}"
        )
        
        return coverage
    
    async def _identify_gaps(
        self,
        test_results: List[Dict[str, Any]]
    ) -> List[CoverageGap]:
        """Identify gaps in detection coverage."""
        gaps = []
        
        # Find missing detectors
        gaps.extend(await self._find_missing_detectors())
        
        # Find weak detectors (low detection rate)
        gaps.extend(await self._find_weak_detectors(test_results))
        
        # Find untested edge cases
        gaps.extend(await self._find_untested_edge_cases())
        
        # Sort by severity
        gaps.sort(
            key=lambda g: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(g.severity, 4)
        )
        
        return gaps
    
    async def _find_missing_detectors(self) -> List[CoverageGap]:
        """Find attack vectors without detectors."""
        gaps = []
        
        for vector_id in self.known_vectors:
            if vector_id not in self.detector_registry:
                gap = CoverageGap(
                    id=f"GAP-MISSING-{vector_id}",
                    gap_type="missing_vector",
                    category=vector_id.split("-")[0] if "-" in vector_id else "UNKNOWN",
                    description=f"No detector found for attack vector {vector_id}",
                    severity="HIGH",
                    suggested_tests=[
                        f"Implement detector for {vector_id}",
                        f"Add test cases for {vector_id}",
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    async def _find_weak_detectors(
        self,
        test_results: List[Dict[str, Any]]
    ) -> List[CoverageGap]:
        """Find detectors with low detection rates."""
        gaps = []
        
        # Group results by detector
        detector_results = defaultdict(list)
        for result in test_results:
            detector_id = result.get("detector_id")
            if detector_id:
                detector_results[detector_id].append(result)
        
        # Analyze each detector's performance
        for detector_id, results in detector_results.items():
            detected = sum(1 for r in results if r.get("detected", False))
            total = len(results)
            
            if total > 0:
                detection_rate = detected / total
                
                # Flag low detection rates
                if detection_rate < 0.8:  # Less than 80% detection
                    severity = "CRITICAL" if detection_rate < 0.5 else "HIGH"
                    
                    gap = CoverageGap(
                        id=f"GAP-WEAK-{detector_id}",
                        gap_type="weak_detector",
                        category=detector_id.split("-")[0] if "-" in detector_id else "UNKNOWN",
                        description=(
                            f"Detector {detector_id} has low detection rate: "
                            f"{detection_rate:.1%} ({detected}/{total})"
                        ),
                        severity=severity,
                        suggested_tests=[
                            f"Review and improve {detector_id}",
                            f"Add more test cases for weak scenarios",
                            f"Adjust detection thresholds",
                        ]
                    )
                    gaps.append(gap)
        
        return gaps
    
    async def _find_untested_edge_cases(self) -> List[CoverageGap]:
        """Find potential edge cases that haven't been tested."""
        gaps = []
        
        # Common edge cases to check
        edge_case_categories = [
            ("encoding", "Unicode, Base64, URL encoding variations"),
            ("length", "Very short and very long inputs"),
            ("special_chars", "Special characters and escape sequences"),
            ("formatting", "Whitespace, newlines, formatting characters"),
            ("multilingual", "Non-English languages and scripts"),
            ("timing", "Timing-based attacks and race conditions"),
        ]
        
        for category, description in edge_case_categories:
            # Check if this edge case category is represented in known vectors
            has_coverage = any(
                category.lower() in vector_id.lower()
                for vector_id in self.known_vectors
            )
            
            if not has_coverage:
                gap = CoverageGap(
                    id=f"GAP-EDGE-{category.upper()}",
                    gap_type="untested_edge_case",
                    category=category,
                    description=f"Edge case not covered: {description}",
                    severity="MEDIUM",
                    suggested_tests=[
                        f"Add {category} edge case tests",
                        f"Generate {category} attack variants",
                    ]
                )
                gaps.append(gap)
        
        return gaps
    
    def _generate_recommendations(
        self,
        gaps: List[CoverageGap]
    ) -> List[str]:
        """Generate actionable recommendations based on gaps."""
        recommendations = []
        
        # Prioritize critical gaps
        critical_gaps = [g for g in gaps if g.severity == "CRITICAL"]
        if critical_gaps:
            recommendations.append(
                f"URGENT: Address {len(critical_gaps)} critical coverage gaps"
            )
        
        # Group by gap type
        gap_types = defaultdict(list)
        for gap in gaps:
            gap_types[gap.gap_type].append(gap)
        
        for gap_type, type_gaps in gap_types.items():
            if gap_type == "missing_vector":
                recommendations.append(
                    f"Implement detectors for {len(type_gaps)} missing attack vectors"
                )
            elif gap_type == "weak_detector":
                recommendations.append(
                    f"Improve {len(type_gaps)} underperforming detectors"
                )
            elif gap_type == "untested_edge_case":
                recommendations.append(
                    f"Add test coverage for {len(type_gaps)} edge case categories"
                )
        
        # Coverage improvement recommendation
        recommendations.append(
            "Run fuzzing campaigns to discover new edge cases"
        )
        recommendations.append(
            "Conduct adversarial testing with red team exercises"
        )
        
        return recommendations
    
    def _compile_statistics(
        self,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile coverage statistics."""
        return {
            "total_vectors": len(self.known_vectors),
            "total_detectors": len(self.detector_registry),
            "total_gaps": len(self.identified_gaps),
            "test_results_analyzed": len(test_results),
            "critical_gaps": sum(
                1 for g in self.identified_gaps if g.severity == "CRITICAL"
            ),
            "high_gaps": sum(
                1 for g in self.identified_gaps if g.severity == "HIGH"
            ),
        }
    
    async def fuzz_attack_space(
        self,
        category: str,
        iterations: int = 100
    ) -> List[str]:
        """
        Fuzz the attack space to find new edge cases.
        
        Args:
            category: Attack category to fuzz
            iterations: Number of fuzzing iterations
            
        Returns:
            List of generated test cases
        """
        logger.info(f"Fuzzing {category} with {iterations} iterations")
        
        test_cases = []
        
        # Fuzzing strategies
        strategies = [
            self._fuzz_length,
            self._fuzz_encoding,
            self._fuzz_special_chars,
            self._fuzz_format,
        ]
        
        for i in range(iterations):
            strategy = random.choice(strategies)
            test_case = await strategy(category, i)
            test_cases.append(test_case)
        
        logger.info(f"Generated {len(test_cases)} fuzzed test cases")
        return test_cases
    
    async def _fuzz_length(self, category: str, iteration: int) -> str:
        """Generate length-based fuzz cases."""
        lengths = [0, 1, 10, 100, 1000, 10000]
        length = random.choice(lengths)
        return f"{category}_fuzz_{iteration}: " + "A" * length
    
    async def _fuzz_encoding(self, category: str, iteration: int) -> str:
        """Generate encoding-based fuzz cases."""
        encodings = [
            lambda s: s.encode("utf-8").hex(),  # Hex
            lambda s: s.encode("unicode_escape").decode("utf-8"),  # Unicode escape
            lambda s: "".join(f"\\x{ord(c):02x}" for c in s),  # Hex escape
        ]
        base = f"{category}_fuzz_{iteration}"
        encoding = random.choice(encodings)
        try:
            return encoding(base)
        except Exception:
            return base
    
    async def _fuzz_special_chars(self, category: str, iteration: int) -> str:
        """Generate special character fuzz cases."""
        special_chars = ["<", ">", "&", "'", '"', "\n", "\r", "\t", "\0", "\x00"]
        chars = random.choices(special_chars, k=random.randint(1, 5))
        return f"{category}_fuzz_{iteration}: " + "".join(chars)
    
    async def _fuzz_format(self, category: str, iteration: int) -> str:
        """Generate format-based fuzz cases."""
        formats = [
            lambda s: s.upper(),
            lambda s: s.lower(),
            lambda s: " " * 10 + s,  # Leading whitespace
            lambda s: s + " " * 10,  # Trailing whitespace
            lambda s: s.replace(" ", "\t"),  # Tabs
        ]
        base = f"{category}_fuzz_{iteration}: test case"
        format_fn = random.choice(formats)
        return format_fn(base)
    
    def get_coverage_trend(self) -> List[Tuple[datetime, float]]:
        """Get historical coverage trend."""
        return [
            (report.timestamp, report.overall_coverage)
            for report in self.coverage_history
        ]
