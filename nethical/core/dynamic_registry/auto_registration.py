"""
Auto-Registration System for Attack Vectors

This module automatically registers new attack patterns discovered
by the red team system with appropriate validation and deployment.

Process:
1. Generate detector from attack signature
2. Validate on test corpus
3. Deploy to staging
4. A/B test in production
5. Full deployment with monitoring

Alignment: Law 24 (Adaptive Learning), Law 15 (Audit Compliance)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Constants
BORDERLINE_VALIDATION_THRESHOLD = 0.7  # Threshold for borderline validation results


class RegistrationStage(str, Enum):
    """Stages of attack vector registration."""
    
    DISCOVERED = "discovered"
    DETECTOR_GENERATED = "detector_generated"
    VALIDATED = "validated"
    STAGING = "staging"
    AB_TESTING = "ab_testing"
    DEPLOYED = "deployed"
    REJECTED = "rejected"


class ValidationResult(str, Enum):
    """Results of validation."""
    
    PASSED = "passed"
    FAILED = "failed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


@dataclass
class AttackPattern:
    """Discovered attack pattern for registration."""
    
    pattern_id: str
    category: str
    signature: str
    description: str
    discovered_at: datetime
    discovered_by: str  # "red_team", "manual", "incident"
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    stage: RegistrationStage = RegistrationStage.DISCOVERED
    validation_result: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorTemplate:
    """Template for generated detector."""
    
    detector_id: str
    pattern_id: str
    detector_code: str
    test_cases: List[Dict[str, Any]]
    detection_threshold: float
    false_positive_rate: float


class AutoRegistration:
    """
    Automatically register new attack patterns.
    
    This component handles the end-to-end process of discovering,
    validating, and deploying new attack vector detectors.
    
    Features:
    - Detector generation from signatures
    - Automated validation
    - Staged deployment
    - A/B testing integration
    """
    
    def __init__(
        self,
        validation_threshold: float = 0.90,
        require_human_approval: bool = True
    ):
        """
        Initialize auto-registration system.
        
        Args:
            validation_threshold: Minimum detection rate for validation
            require_human_approval: Require human approval before deployment
        """
        self.validation_threshold = validation_threshold
        self.require_human_approval = require_human_approval
        self.discovered_patterns: Dict[str, AttackPattern] = {}
        self.registered_detectors: Dict[str, DetectorTemplate] = {}
        self.pending_approval: List[str] = []
        
        logger.info(
            f"AutoRegistration initialized (threshold: {validation_threshold:.1%})"
        )
    
    async def register_attack_pattern(
        self,
        category: str,
        signature: str,
        description: str,
        discovered_by: str = "red_team",
        severity: str = "MEDIUM"
    ) -> str:
        """
        Register a new attack pattern.
        
        Args:
            category: Attack category
            signature: Attack signature/pattern
            description: Human-readable description
            discovered_by: Source of discovery
            severity: Attack severity
            
        Returns:
            Pattern ID
        """
        # Generate pattern ID
        pattern_id = self._generate_pattern_id(category)
        
        # Create attack pattern
        pattern = AttackPattern(
            pattern_id=pattern_id,
            category=category,
            signature=signature,
            description=description,
            discovered_at=datetime.now(timezone.utc),
            discovered_by=discovered_by,
            severity=severity
        )
        
        self.discovered_patterns[pattern_id] = pattern
        
        logger.info(
            f"Registered new attack pattern: {pattern_id} ({category})"
        )
        
        # Start automatic processing
        await self._process_pattern(pattern_id)
        
        return pattern_id
    
    async def _process_pattern(self, pattern_id: str) -> None:
        """Process attack pattern through registration pipeline."""
        
        pattern = self.discovered_patterns.get(pattern_id)
        if not pattern:
            logger.error(f"Pattern {pattern_id} not found")
            return
        
        try:
            # Stage 1: Generate detector
            detector = await self._generate_detector(pattern)
            if not detector:
                pattern.stage = RegistrationStage.REJECTED
                pattern.validation_result = ValidationResult.FAILED
                return
            
            pattern.stage = RegistrationStage.DETECTOR_GENERATED
            self.registered_detectors[detector.detector_id] = detector
            
            # Stage 2: Validate on test corpus
            validation_result = await self._validate_detector(detector)
            pattern.validation_result = validation_result
            
            if validation_result == ValidationResult.FAILED:
                pattern.stage = RegistrationStage.REJECTED
                logger.warning(f"Pattern {pattern_id} failed validation")
                return
            
            pattern.stage = RegistrationStage.VALIDATED
            
            # Stage 3: Deploy to staging
            await self._deploy_to_staging(detector)
            pattern.stage = RegistrationStage.STAGING
            
            # Stage 4: A/B test (if validation passed)
            if validation_result == ValidationResult.PASSED:
                if self.require_human_approval:
                    self.pending_approval.append(pattern_id)
                    logger.info(
                        f"Pattern {pattern_id} pending human approval"
                    )
                else:
                    await self._ab_test_detector(detector)
                    pattern.stage = RegistrationStage.AB_TESTING
            
        except Exception as e:
            logger.error(f"Error processing pattern {pattern_id}: {e}")
            pattern.stage = RegistrationStage.REJECTED
    
    async def _generate_detector(
        self,
        pattern: AttackPattern
    ) -> Optional[DetectorTemplate]:
        """
        Generate detector from attack signature.
        
        Note: This is a simplified implementation. Production would use
        ML-based detector generation or template systems.
        """
        detector_id = f"DET-{pattern.pattern_id}"
        
        # Generate simple rule-based detector code
        detector_code = self._generate_detector_code(pattern)
        
        # Generate test cases
        test_cases = self._generate_test_cases(pattern)
        
        # Estimate parameters (would be learned from data in production)
        detection_threshold = 0.7
        false_positive_rate = 0.05
        
        detector = DetectorTemplate(
            detector_id=detector_id,
            pattern_id=pattern.pattern_id,
            detector_code=detector_code,
            test_cases=test_cases,
            detection_threshold=detection_threshold,
            false_positive_rate=false_positive_rate
        )
        
        logger.debug(f"Generated detector {detector_id} for pattern {pattern.pattern_id}")
        
        return detector
    
    def _generate_detector_code(self, pattern: AttackPattern) -> str:
        """Generate detector implementation code."""
        # Simplified template-based generation
        code_template = f'''
async def detect_{pattern.pattern_id}(input_text: str) -> bool:
    """
    Detect {pattern.description}
    
    Signature: {pattern.signature}
    Severity: {pattern.severity}
    """
    # Simple pattern matching
    return "{pattern.signature.lower()}" in input_text.lower()
'''
        return code_template
    
    def _generate_test_cases(self, pattern: AttackPattern) -> List[Dict[str, Any]]:
        """Generate test cases for detector validation."""
        # Generate positive and negative test cases
        test_cases = [
            {
                "input": pattern.signature,
                "expected": True,
                "label": "positive_exact_match"
            },
            {
                "input": pattern.signature.upper(),
                "expected": True,
                "label": "positive_case_variant"
            },
            {
                "input": "benign input with no attack",
                "expected": False,
                "label": "negative_benign"
            },
        ]
        
        return test_cases
    
    async def _validate_detector(
        self,
        detector: DetectorTemplate
    ) -> ValidationResult:
        """
        Validate detector on test corpus.
        
        Returns validation result indicating pass/fail/needs-review.
        """
        # Run test cases
        passed = 0
        total = len(detector.test_cases)
        
        for test_case in detector.test_cases:
            # Simulate detector execution
            # In production, would actually run the detector
            result = await self._run_test_case(detector, test_case)
            if result == test_case["expected"]:
                passed += 1
        
        detection_rate = passed / total if total > 0 else 0
        
        logger.info(
            f"Detector {detector.detector_id} validation: "
            f"{detection_rate:.1%} ({passed}/{total})"
        )
        
        # Check thresholds
        if detection_rate >= self.validation_threshold:
            return ValidationResult.PASSED
        elif detection_rate >= BORDERLINE_VALIDATION_THRESHOLD:  # Borderline
            return ValidationResult.NEEDS_HUMAN_REVIEW
        else:
            return ValidationResult.FAILED
    
    async def _run_test_case(
        self,
        detector: DetectorTemplate,
        test_case: Dict[str, Any]
    ) -> bool:
        """Run a single test case (simulated)."""
        # Simplified simulation
        input_text = test_case["input"]
        pattern = self.discovered_patterns[detector.pattern_id]
        
        # Simple pattern matching
        return pattern.signature.lower() in input_text.lower()
    
    async def _deploy_to_staging(self, detector: DetectorTemplate) -> None:
        """Deploy detector to staging environment."""
        logger.info(f"Deploying {detector.detector_id} to staging")
        
        # In production, would:
        # - Package detector
        # - Deploy to staging infrastructure
        # - Configure monitoring
        # - Enable for staging traffic
        
        await asyncio.sleep(0.1)  # Simulate deployment time
    
    async def _ab_test_detector(self, detector: DetectorTemplate) -> None:
        """A/B test detector in production."""
        logger.info(f"Starting A/B test for {detector.detector_id}")
        
        # In production, would:
        # - Configure traffic split (e.g., 10% to new detector)
        # - Monitor metrics (detection rate, false positives, latency)
        # - Compare against baseline
        # - Make rollout decision
        
        await asyncio.sleep(0.1)  # Simulate A/B test setup
    
    async def approve_pattern(self, pattern_id: str) -> bool:
        """
        Manually approve a pattern for deployment.
        
        Args:
            pattern_id: Pattern to approve
            
        Returns:
            True if approved and deployed
        """
        if pattern_id not in self.pending_approval:
            logger.warning(f"Pattern {pattern_id} not pending approval")
            return False
        
        pattern = self.discovered_patterns.get(pattern_id)
        if not pattern:
            return False
        
        # Get detector
        detector_id = f"DET-{pattern_id}"
        detector = self.registered_detectors.get(detector_id)
        if not detector:
            return False
        
        # Start A/B testing
        await self._ab_test_detector(detector)
        pattern.stage = RegistrationStage.AB_TESTING
        
        # Remove from pending
        self.pending_approval.remove(pattern_id)
        
        logger.info(f"Pattern {pattern_id} approved for A/B testing")
        
        return True
    
    def _generate_pattern_id(self, category: str) -> str:
        """Generate unique pattern ID."""
        timestamp = int(time.time() * 1000)
        category_prefix = category[:3].upper()
        return f"AP-{category_prefix}-{timestamp}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registration statistics."""
        return {
            "total_patterns": len(self.discovered_patterns),
            "pending_approval": len(self.pending_approval),
            "by_stage": self._count_by_stage(),
            "by_validation": self._count_by_validation(),
            "by_severity": self._count_by_severity(),
        }
    
    def _count_by_stage(self) -> Dict[str, int]:
        """Count patterns by stage."""
        counts = {}
        for pattern in self.discovered_patterns.values():
            stage = pattern.stage.value
            counts[stage] = counts.get(stage, 0) + 1
        return counts
    
    def _count_by_validation(self) -> Dict[str, int]:
        """Count patterns by validation result."""
        counts = {}
        for pattern in self.discovered_patterns.values():
            if pattern.validation_result:
                result = pattern.validation_result.value
                counts[result] = counts.get(result, 0) + 1
        return counts
    
    def _count_by_severity(self) -> Dict[str, int]:
        """Count patterns by severity."""
        counts = {}
        for pattern in self.discovered_patterns.values():
            severity = pattern.severity
            counts[severity] = counts.get(severity, 0) + 1
        return counts
