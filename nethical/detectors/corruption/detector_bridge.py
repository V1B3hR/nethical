"""
Detector Bridge for Integration

This module provides integration with existing Nethical detectors to correlate
signals and avoid duplication of detection logic.

Author: Nethical Core Team
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Sequence
import logging

from ..base_detector import BaseDetector, SafetyViolation
from .corruption_types import CorruptionEvidence

logger = logging.getLogger(__name__)


class DetectorBridge:
    """Bridge to integrate with existing Nethical detectors."""
    
    def __init__(self):
        self.registered_detectors: Dict[str, BaseDetector] = {}
        
        # Mapping of detector categories to corruption relevance
        self.detector_relevance = {
            "manipulation": 0.9,  # Manipulation detector highly relevant
            "dark_pattern": 0.8,  # Dark patterns often used in corruption
            "coordinated_attack": 0.7,  # Coordination can indicate collusion
            "context_poisoning": 0.6,  # Context manipulation for corruption
            "memory_manipulation": 0.6,  # Memory manipulation for concealment
            "slow_low": 0.5,  # Slow-burn attacks resemble corruption grooming
            "mimicry": 0.4,  # Mimicry can be part of corruption tactics
        }
    
    def register_detector(self, name: str, detector: BaseDetector):
        """Register an existing detector for correlation."""
        self.registered_detectors[name] = detector
        logger.info(f"Registered detector '{name}' with bridge")
    
    async def check_manipulation_signals(self, action: Any) -> List[CorruptionEvidence]:
        """Check manipulation detector for corruption-related signals."""
        evidence = []
        
        if "manipulation" not in self.registered_detectors:
            return evidence
        
        detector = self.registered_detectors["manipulation"]
        
        try:
            violations = await detector.detect_violations(action)
            
            if violations:
                for violation in violations:
                    # Map manipulation categories to corruption evidence
                    corruption_relevance = self._assess_manipulation_corruption_relevance(violation)
                    
                    if corruption_relevance > 0.5:
                        evidence.append(CorruptionEvidence(
                            type="manipulation_signal",
                            description=f"Manipulation detected: {violation.description}",
                            confidence=corruption_relevance,
                            source_detector="manipulation_detector",
                            context={
                                "violation_category": getattr(violation, "category", "unknown"),
                                "violation_severity": getattr(violation, "severity", "unknown"),
                            }
                        ))
        except Exception as e:
            logger.warning(f"Error checking manipulation detector: {e}")
        
        return evidence
    
    async def check_dark_pattern_signals(self, action: Any) -> List[CorruptionEvidence]:
        """Check dark pattern detector for corruption-related signals."""
        evidence = []
        
        if "dark_pattern" not in self.registered_detectors:
            return evidence
        
        detector = self.registered_detectors["dark_pattern"]
        
        try:
            violations = await detector.detect_violations(action)
            
            if violations:
                for violation in violations:
                    # Dark patterns often involve deceptive practices
                    evidence.append(CorruptionEvidence(
                        type="dark_pattern_signal",
                        description=f"Dark pattern detected: {violation.description}",
                        confidence=0.7,
                        source_detector="dark_pattern_detector",
                        context={
                            "pattern_type": getattr(violation, "category", "unknown"),
                        }
                    ))
        except Exception as e:
            logger.warning(f"Error checking dark pattern detector: {e}")
        
        return evidence
    
    async def check_behavioral_signals(self, action: Any) -> List[CorruptionEvidence]:
        """Check behavioral detectors for corruption-related signals."""
        evidence = []
        
        behavioral_detectors = [
            "coordinated_attack",
            "slow_low",
            "mimicry",
        ]
        
        for detector_name in behavioral_detectors:
            if detector_name not in self.registered_detectors:
                continue
            
            detector = self.registered_detectors[detector_name]
            
            try:
                violations = await detector.detect_violations(action)
                
                if violations:
                    relevance = self.detector_relevance.get(detector_name, 0.5)
                    
                    for violation in violations:
                        evidence.append(CorruptionEvidence(
                            type="behavioral_signal",
                            description=f"Behavioral pattern: {violation.description}",
                            confidence=relevance,
                            source_detector=detector_name,
                            context={
                                "detector_type": detector_name,
                            }
                        ))
            except Exception as e:
                logger.warning(f"Error checking {detector_name} detector: {e}")
        
        return evidence
    
    async def check_session_signals(self, action: Any) -> List[CorruptionEvidence]:
        """Check session-aware detectors for corruption signals."""
        evidence = []
        
        session_detectors = [
            "context_poisoning",
            "memory_manipulation",
            "multi_turn",
        ]
        
        for detector_name in session_detectors:
            if detector_name not in self.registered_detectors:
                continue
            
            detector = self.registered_detectors[detector_name]
            
            try:
                violations = await detector.detect_violations(action)
                
                if violations:
                    relevance = self.detector_relevance.get(detector_name, 0.5)
                    
                    for violation in violations:
                        evidence.append(CorruptionEvidence(
                            type="session_signal",
                            description=f"Session manipulation: {violation.description}",
                            confidence=relevance,
                            source_detector=detector_name,
                            context={
                                "detector_type": detector_name,
                            }
                        ))
            except Exception as e:
                logger.warning(f"Error checking {detector_name} detector: {e}")
        
        return evidence
    
    async def correlate_all_signals(self, action: Any) -> Dict[str, List[CorruptionEvidence]]:
        """Correlate signals from all registered detectors."""
        all_evidence = {
            "manipulation": await self.check_manipulation_signals(action),
            "dark_patterns": await self.check_dark_pattern_signals(action),
            "behavioral": await self.check_behavioral_signals(action),
            "session": await self.check_session_signals(action),
        }
        
        return all_evidence
    
    def calculate_correlation_score(self, evidence_dict: Dict[str, List[CorruptionEvidence]]) -> float:
        """Calculate overall correlation score from multiple detector signals."""
        total_evidence = sum(len(evidence) for evidence in evidence_dict.values())
        
        if total_evidence == 0:
            return 0.0
        
        # Weight by number of detector categories that triggered
        categories_triggered = sum(1 for evidence in evidence_dict.values() if len(evidence) > 0)
        
        # Calculate weighted average confidence
        total_confidence = 0.0
        total_weight = 0
        
        for category, evidence_list in evidence_dict.items():
            for evidence in evidence_list:
                total_confidence += evidence.confidence
                total_weight += 1
        
        if total_weight == 0:
            return 0.0
        
        avg_confidence = total_confidence / total_weight
        
        # Boost score if multiple categories agree
        category_multiplier = min(1.0 + (categories_triggered - 1) * 0.2, 2.0)
        
        correlation_score = min(avg_confidence * category_multiplier, 1.0)
        
        return correlation_score
    
    def _assess_manipulation_corruption_relevance(self, violation: SafetyViolation) -> float:
        """Assess how relevant a manipulation violation is to corruption."""
        category = getattr(violation, "category", "").lower()
        
        # High relevance categories
        high_relevance = [
            "authority_manipulation",
            "emotional_manipulation",
            "coercion",
            "deception",
            "fraud",
            "bribery",
            "extortion",
        ]
        
        # Medium relevance categories
        medium_relevance = [
            "urgency_manipulation",
            "scarcity",
            "reciprocity",
            "social_proof",
            "commitment",
        ]
        
        for pattern in high_relevance:
            if pattern in category:
                return 0.9
        
        for pattern in medium_relevance:
            if pattern in category:
                return 0.7
        
        # Default relevance
        return 0.5


__all__ = ["DetectorBridge"]
