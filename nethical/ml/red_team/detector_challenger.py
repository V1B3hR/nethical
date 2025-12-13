"""
Detector Challenger - Continuously probes detectors for weaknesses

This module systematically challenges detectors to find blind spots
and weaknesses using gradient-based adversarial examples.

Features:
- Gradient-based adversarial example generation
- Detector stress testing
- Weakness identification
- Improvement recommendations

Alignment: Law 23 (Fail-Safe Design), Law 24 (Adaptive Learning)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Types of detector challenges."""
    
    ADVERSARIAL_EXAMPLE = "adversarial_example"
    BOUNDARY_PROBE = "boundary_probe"
    EVASION_ATTEMPT = "evasion_attempt"
    STRESS_TEST = "stress_test"


class DetectorWeakness(str, Enum):
    """Types of detector weaknesses."""
    
    LOW_SENSITIVITY = "low_sensitivity"  # Misses subtle attacks
    HIGH_FALSE_POSITIVES = "high_false_positives"  # Too many false alarms
    BOUNDARY_VULNERABILITY = "boundary_vulnerability"  # Weak at thresholds
    EVASION_SUSCEPTIBLE = "evasion_susceptible"  # Easily bypassed
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Slow under load


@dataclass
class ChallengeResult:
    """Result of a detector challenge."""
    
    challenge_id: str
    detector_id: str
    challenge_type: ChallengeType
    test_payload: str
    detected: bool
    confidence: float
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    weakness_found: Optional[DetectorWeakness] = None


@dataclass
class DetectorProfile:
    """Performance profile of a detector."""
    
    detector_id: str
    total_challenges: int
    detection_rate: float
    false_positive_rate: float
    avg_latency_ms: float
    weaknesses: List[DetectorWeakness]
    recommendations: List[str]
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DetectorChallenger:
    """
    Continuously probes detectors for weaknesses.
    
    This component acts as an adversarial tester, systematically
    challenging detectors to identify blind spots and improve robustness.
    
    Methods:
    - Gradient-based adversarial examples
    - Boundary condition probing
    - Evasion technique testing
    - Stress testing under load
    """
    
    def __init__(
        self,
        detector_ids: Optional[List[str]] = None,
        challenge_rate_limit: int = 1000  # Max challenges per minute
    ):
        """
        Initialize the detector challenger.
        
        Args:
            detector_ids: List of detector IDs to challenge
            challenge_rate_limit: Maximum challenges per minute
        """
        self.detector_ids = detector_ids or []
        self.challenge_rate_limit = challenge_rate_limit
        self.challenge_history: List[ChallengeResult] = []
        self.detector_profiles: Dict[str, DetectorProfile] = {}
        self._rate_limiter: List[float] = []
        
        logger.info(
            f"DetectorChallenger initialized with {len(self.detector_ids)} detectors"
        )
    
    async def challenge_detector(
        self,
        detector_id: str,
        challenge_type: ChallengeType,
        payload: Optional[str] = None
    ) -> ChallengeResult:
        """
        Challenge a specific detector.
        
        Args:
            detector_id: ID of detector to challenge
            challenge_type: Type of challenge to perform
            payload: Test payload (generated if not provided)
            
        Returns:
            Challenge result with detection outcome
        """
        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Challenge rate limit exceeded, throttling")
            await asyncio.sleep(0.1)
        
        # Generate payload if not provided
        if payload is None:
            payload = await self._generate_challenge_payload(
                detector_id, challenge_type
            )
        
        # Execute challenge
        start_time = time.time()
        detected, confidence = await self._execute_challenge(
            detector_id, payload
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Analyze result
        weakness = self._analyze_weakness(
            detector_id, challenge_type, detected, confidence, latency_ms
        )
        
        # Create result
        challenge_id = f"CH-{detector_id}-{int(time.time() * 1000)}"
        result = ChallengeResult(
            challenge_id=challenge_id,
            detector_id=detector_id,
            challenge_type=challenge_type,
            test_payload=payload,
            detected=detected,
            confidence=confidence,
            latency_ms=latency_ms,
            weakness_found=weakness
        )
        
        # Record result
        self.challenge_history.append(result)
        self._rate_limiter.append(time.time())
        
        # Update detector profile
        await self._update_detector_profile(detector_id)
        
        logger.debug(
            f"Challenge {challenge_id}: detected={detected}, "
            f"confidence={confidence:.2f}, latency={latency_ms:.1f}ms"
        )
        
        return result
    
    async def challenge_all_detectors(
        self,
        challenge_type: ChallengeType,
        iterations: int = 10
    ) -> List[ChallengeResult]:
        """
        Challenge all registered detectors.
        
        Args:
            challenge_type: Type of challenge to perform
            iterations: Number of challenges per detector
            
        Returns:
            List of challenge results
        """
        logger.info(
            f"Challenging {len(self.detector_ids)} detectors "
            f"with {iterations} iterations of {challenge_type.value}"
        )
        
        results = []
        
        for detector_id in self.detector_ids:
            for i in range(iterations):
                result = await self.challenge_detector(
                    detector_id, challenge_type
                )
                results.append(result)
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)
        
        logger.info(f"Completed {len(results)} challenges")
        return results
    
    async def _generate_challenge_payload(
        self,
        detector_id: str,
        challenge_type: ChallengeType
    ) -> str:
        """Generate a challenge payload based on type."""
        
        if challenge_type == ChallengeType.ADVERSARIAL_EXAMPLE:
            return await self._generate_adversarial_example(detector_id)
        elif challenge_type == ChallengeType.BOUNDARY_PROBE:
            return await self._generate_boundary_probe(detector_id)
        elif challenge_type == ChallengeType.EVASION_ATTEMPT:
            return await self._generate_evasion_attempt(detector_id)
        elif challenge_type == ChallengeType.STRESS_TEST:
            return await self._generate_stress_test(detector_id)
        else:
            return f"Generic challenge for {detector_id}"
    
    async def _generate_adversarial_example(
        self,
        detector_id: str
    ) -> str:
        """
        Generate gradient-based adversarial example.
        
        Note: This is a simplified implementation. Full gradient-based
        adversarial generation would require access to model internals.
        """
        # Simulate adversarial perturbation strategies
        strategies = [
            lambda base: base.replace("attack", "ąttack"),  # Homoglyph (same length)
            lambda base: f"{base}\u200b",  # Zero-width space
            lambda base: " " * 10 + base,  # Whitespace padding
            lambda base: base.replace(" ", "\t"),  # Tab replacement
            lambda base: base.upper() if random.random() > 0.5 else base.lower(),
        ]
        
        base_attack = f"Test adversarial input for {detector_id}"
        strategy = random.choice(strategies)
        
        return strategy(base_attack)
    
    async def _generate_boundary_probe(
        self,
        detector_id: str
    ) -> str:
        """Generate payload to probe detection boundaries."""
        # Test payloads near detection thresholds
        boundary_tests = [
            "Benign request with subtle suspicious element",
            "Almost malicious but not quite",
            "Borderline case that may confuse detector",
            "Edge case: legitimate but unusual pattern",
        ]
        
        return random.choice(boundary_tests) + f" [{detector_id}]"
    
    async def _generate_evasion_attempt(
        self,
        detector_id: str
    ) -> str:
        """Generate payload attempting to evade detection."""
        evasion_techniques = [
            "Obfuscated: {}".format("".join(f"\\x{ord(c):02x}" for c in "attack")),
            "Encoded: YXR0YWNr",  # Base64 of "attack"
            "Fragmented: at" + "ta" + "ck",
            "Multilingual: ąttąčk",
        ]
        
        return f"{random.choice(evasion_techniques)} for {detector_id}"
    
    async def _generate_stress_test(
        self,
        detector_id: str
    ) -> str:
        """Generate payload for stress testing."""
        # Generate varying complexity payloads
        complexity = random.choice([10, 100, 1000, 10000])
        return f"Stress test for {detector_id}: " + "X" * complexity
    
    async def _execute_challenge(
        self,
        detector_id: str,
        payload: str
    ) -> Tuple[bool, float]:
        """
        Execute challenge against detector.
        
        Note: This is a mock implementation. In production, this would
        call the actual detector implementation.
        
        Returns:
            Tuple of (detected, confidence)
        """
        # Simulate detector behavior
        # In production, this would call: detector.detect(payload)
        
        # Simple heuristic for simulation
        suspicious_keywords = ["attack", "injection", "exploit", "bypass"]
        has_suspicious = any(kw in payload.lower() for kw in suspicious_keywords)
        
        # Add some randomness to simulate real detector behavior
        base_confidence = 0.7 if has_suspicious else 0.2
        confidence = base_confidence + random.uniform(-0.2, 0.2)
        confidence = max(0.0, min(1.0, confidence))
        
        detected = confidence > 0.5
        
        return detected, confidence
    
    def _analyze_weakness(
        self,
        detector_id: str,
        challenge_type: ChallengeType,
        detected: bool,
        confidence: float,
        latency_ms: float
    ) -> Optional[DetectorWeakness]:
        """Analyze if a weakness was found."""
        
        # Check for various weakness indicators
        
        # High latency indicates performance issues
        if latency_ms > 100:  # More than 100ms
            return DetectorWeakness.PERFORMANCE_DEGRADATION
        
        # Low confidence on adversarial examples indicates evasion susceptibility
        if challenge_type == ChallengeType.ADVERSARIAL_EXAMPLE and confidence < 0.3:
            return DetectorWeakness.EVASION_SUSCEPTIBLE
        
        # Boundary probes should have moderate confidence
        if challenge_type == ChallengeType.BOUNDARY_PROBE:
            if not detected and confidence < 0.4:
                return DetectorWeakness.LOW_SENSITIVITY
            elif detected and confidence > 0.9:
                return DetectorWeakness.HIGH_FALSE_POSITIVES
        
        # Missing detection on evasion attempts
        if challenge_type == ChallengeType.EVASION_ATTEMPT and not detected:
            return DetectorWeakness.EVASION_SUSCEPTIBLE
        
        return None
    
    async def _update_detector_profile(
        self,
        detector_id: str
    ) -> None:
        """Update the performance profile for a detector."""
        
        # Get all results for this detector
        detector_results = [
            r for r in self.challenge_history if r.detector_id == detector_id
        ]
        
        if not detector_results:
            return
        
        # Calculate metrics
        total_challenges = len(detector_results)
        detection_rate = sum(1 for r in detector_results if r.detected) / total_challenges
        
        # Calculate false positive rate (simplified)
        # In production, would need labeled ground truth
        false_positives = sum(
            1 for r in detector_results
            if r.detected and r.confidence > 0.9 and "benign" in r.test_payload.lower()
        )
        false_positive_rate = false_positives / total_challenges if total_challenges > 0 else 0
        
        avg_latency_ms = sum(r.latency_ms for r in detector_results) / total_challenges
        
        # Collect identified weaknesses
        weaknesses = list(set(
            r.weakness_found for r in detector_results
            if r.weakness_found is not None
        ))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            detector_id, detection_rate, false_positive_rate, avg_latency_ms, weaknesses
        )
        
        # Update or create profile
        self.detector_profiles[detector_id] = DetectorProfile(
            detector_id=detector_id,
            total_challenges=total_challenges,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            avg_latency_ms=avg_latency_ms,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        detector_id: str,
        detection_rate: float,
        false_positive_rate: float,
        avg_latency_ms: float,
        weaknesses: List[DetectorWeakness]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Detection rate recommendations
        if detection_rate < 0.8:
            recommendations.append(
                f"Improve detection rate (current: {detection_rate:.1%}). "
                "Consider lowering thresholds or adding more patterns."
            )
        
        # False positive recommendations
        if false_positive_rate > 0.1:
            recommendations.append(
                f"Reduce false positives (current: {false_positive_rate:.1%}). "
                "Consider raising thresholds or refining detection logic."
            )
        
        # Performance recommendations
        if avg_latency_ms > 50:
            recommendations.append(
                f"Optimize performance (current: {avg_latency_ms:.1f}ms). "
                "Consider caching, parallelization, or algorithm optimization."
            )
        
        # Weakness-specific recommendations
        weakness_recommendations = {
            DetectorWeakness.LOW_SENSITIVITY: "Increase sensitivity to subtle attacks",
            DetectorWeakness.HIGH_FALSE_POSITIVES: "Improve specificity to reduce false alarms",
            DetectorWeakness.BOUNDARY_VULNERABILITY: "Add more boundary condition tests",
            DetectorWeakness.EVASION_SUSCEPTIBLE: "Implement evasion-resistant detection methods",
            DetectorWeakness.PERFORMANCE_DEGRADATION: "Optimize algorithm for better performance",
        }
        
        for weakness in weaknesses:
            if weakness in weakness_recommendations:
                recommendations.append(weakness_recommendations[weakness])
        
        return recommendations
    
    def _check_rate_limit(self) -> bool:
        """Check if challenges are within rate limit."""
        self._cleanup_rate_limiter()
        return len(self._rate_limiter) < self.challenge_rate_limit
    
    def _cleanup_rate_limiter(self) -> None:
        """Remove old entries from rate limiter."""
        current_time = time.time()
        cutoff = current_time - 60  # 60 second window
        self._rate_limiter = [t for t in self._rate_limiter if t > cutoff]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all detector challenges."""
        return {
            "total_challenges": len(self.challenge_history),
            "detectors_tested": len(self.detector_profiles),
            "weaknesses_found": sum(
                len(profile.weaknesses) for profile in self.detector_profiles.values()
            ),
            "detector_profiles": {
                detector_id: {
                    "detection_rate": profile.detection_rate,
                    "false_positive_rate": profile.false_positive_rate,
                    "avg_latency_ms": profile.avg_latency_ms,
                    "weaknesses": [w.value for w in profile.weaknesses],
                    "total_challenges": profile.total_challenges,
                }
                for detector_id, profile in self.detector_profiles.items()
            }
        }
    
    def get_detector_profile(self, detector_id: str) -> Optional[DetectorProfile]:
        """Get profile for a specific detector."""
        return self.detector_profiles.get(detector_id)
    
    def get_weakest_detectors(self, top_n: int = 5) -> List[Tuple[str, DetectorProfile]]:
        """Get the weakest performing detectors."""
        # Sort by detection rate (ascending) and false positive rate (descending)
        sorted_profiles = sorted(
            self.detector_profiles.items(),
            key=lambda x: (x[1].detection_rate, -x[1].false_positive_rate)
        )
        
        return sorted_profiles[:top_n]
