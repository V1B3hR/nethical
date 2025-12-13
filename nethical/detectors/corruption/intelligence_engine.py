"""
Corruption Intelligence Engine

Core intelligence engine for corruption detection with multi-detector correlation,
entity profiling, relationship graph analysis, and long-term pattern tracking.

Author: Nethical Core Team
Version: 1.0.0
"""

import re
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .corruption_types import (
    CorruptionType,
    CorruptionVector,
    CorruptionPhase,
    RiskLevel,
    RecommendedAction,
    CorruptionEvidence,
    EntityProfile,
    RelationshipEdge,
    CorruptionAssessment,
)
from .corruption_patterns import CorruptionPatternLibrary
from .detector_bridge import DetectorBridge

logger = logging.getLogger(__name__)


class IntelligenceEngine:
    """Core intelligence engine for corruption detection."""
    
    def __init__(self):
        self.pattern_library = CorruptionPatternLibrary()
        self.detector_bridge = DetectorBridge()
        
        # Entity tracking
        self.entity_profiles: Dict[str, EntityProfile] = {}
        self.relationship_graph: Dict[Tuple[str, str], RelationshipEdge] = {}
        
        # Cleanup settings
        self.profile_retention_days = 90
        self.last_cleanup = datetime.now(timezone.utc)
        self.cleanup_interval_hours = 24
    
    def register_detector(self, name: str, detector: Any):
        """Register an existing detector with the bridge."""
        self.detector_bridge.register_detector(name, detector)
    
    async def analyze_action(
        self,
        action: Any,
        entity_id: Optional[str] = None,
    ) -> CorruptionAssessment:
        """Analyze an action for corruption with full intelligence."""
        
        # Initialize assessment
        assessment = CorruptionAssessment()
        assessment.action_id = getattr(action, "action_id", None)
        assessment.entity_id = entity_id
        
        # Extract content
        content = self._extract_content(action)
        
        # Pattern-based detection
        pattern_evidence = await self._detect_patterns(content)
        assessment.evidence.extend(pattern_evidence)
        
        # Multi-detector correlation
        detector_evidence = await self.detector_bridge.correlate_all_signals(action)
        for category, evidence_list in detector_evidence.items():
            assessment.evidence.extend(evidence_list)
            if evidence_list:
                assessment.detectors_triggered.append(category)
        
        # Calculate correlation score
        assessment.correlation_score = self.detector_bridge.calculate_correlation_score(
            detector_evidence
        )
        
        # Determine corruption vector
        assessment.vector = self._determine_vector(action, pattern_evidence)
        
        # Determine corruption phase
        assessment.phase = self._determine_phase(pattern_evidence)
        
        # Determine primary corruption type
        assessment.primary_type = self._determine_primary_type(pattern_evidence)
        
        # Calculate confidence
        assessment.confidence = self._calculate_confidence(assessment)
        
        # Update entity profile if entity_id provided
        if entity_id:
            self._update_entity_profile(entity_id, assessment)
            
            # Enhance assessment with entity history
            entity_profile = self.entity_profiles.get(entity_id)
            if entity_profile:
                assessment.metadata["entity_risk_score"] = entity_profile.corruption_risk_score
                assessment.metadata["entity_corruption_attempts"] = entity_profile.corruption_attempts
        
        # Determine if corrupt
        assessment.is_corrupt = self._is_corrupt(assessment)
        
        # Determine risk level
        assessment.risk_level = self._determine_risk_level(assessment)
        
        # Generate reasoning chain
        assessment.reasoning_chain = self._generate_reasoning_chain(assessment)
        
        # Generate explanation
        assessment.explanation = self._generate_explanation(assessment)
        
        # Determine recommended action
        assessment.recommended_action = self._determine_recommended_action(assessment)
        
        # Determine if human review required
        assessment.requires_human_review = self._requires_human_review(assessment)
        
        # Periodic cleanup
        if (datetime.now(timezone.utc) - self.last_cleanup).total_seconds() > self.cleanup_interval_hours * 3600:
            await self._cleanup_old_profiles()
        
        return assessment
    
    def _extract_content(self, action: Any) -> str:
        """Extract content from action for analysis."""
        if hasattr(action, "content"):
            return str(action.content)
        return str(action)
    
    async def _detect_patterns(self, content: str) -> List[CorruptionEvidence]:
        """Detect corruption patterns in content."""
        evidence = []
        content_lower = content.lower()
        
        for pattern_def in self.pattern_library.get_all_patterns():
            for pattern in pattern_def.patterns:
                try:
                    matches = re.findall(pattern, content_lower, re.MULTILINE)
                    
                    if matches:
                        # Skip if pattern requires context and we don't have strong signals
                        if pattern_def.requires_context and len(matches) < 2:
                            continue
                        
                        evidence.append(CorruptionEvidence(
                            type=pattern_def.corruption_type.value,
                            description=f"{pattern_def.description}",
                            confidence=min(pattern_def.base_confidence + (len(matches) - 1) * 0.05, 0.95),
                            pattern_matched=pattern,
                            context={
                                "corruption_type": pattern_def.corruption_type.value,
                                "vector": pattern_def.vector.value,
                                "phase": pattern_def.phase.value,
                                "match_count": len(matches),
                                "severity_weight": pattern_def.severity_weight,
                            }
                        ))
                except re.error as e:
                    logger.warning(f"Regex error in pattern {pattern}: {e}")
                    continue
        
        return evidence
    
    def _determine_vector(
        self,
        action: Any,
        pattern_evidence: List[CorruptionEvidence]
    ) -> Optional[CorruptionVector]:
        """Determine the corruption vector from evidence."""
        if not pattern_evidence:
            return None
        
        # Count vectors in evidence
        vector_counts = defaultdict(int)
        for evidence in pattern_evidence:
            vector = evidence.context.get("vector")
            if vector:
                vector_counts[vector] += 1
        
        if not vector_counts:
            return None
        
        # Return most common vector
        most_common_vector = max(vector_counts.items(), key=lambda x: x[1])[0]
        
        try:
            return CorruptionVector(most_common_vector)
        except ValueError:
            return None
    
    def _determine_phase(
        self,
        pattern_evidence: List[CorruptionEvidence]
    ) -> Optional[CorruptionPhase]:
        """Determine the corruption lifecycle phase."""
        if not pattern_evidence:
            return None
        
        # Count phases in evidence
        phase_counts = defaultdict(int)
        for evidence in pattern_evidence:
            phase = evidence.context.get("phase")
            if phase:
                phase_counts[phase] += 1
        
        if not phase_counts:
            return None
        
        # Return most common phase
        most_common_phase = max(phase_counts.items(), key=lambda x: x[1])[0]
        
        try:
            return CorruptionPhase(most_common_phase)
        except ValueError:
            return None
    
    def _determine_primary_type(
        self,
        pattern_evidence: List[CorruptionEvidence]
    ) -> Optional[CorruptionType]:
        """Determine the primary corruption type."""
        if not pattern_evidence:
            return None
        
        # Count types in evidence, weighted by severity
        type_scores = defaultdict(float)
        for evidence in pattern_evidence:
            corruption_type = evidence.context.get("corruption_type")
            severity_weight = evidence.context.get("severity_weight", 1.0)
            if corruption_type:
                type_scores[corruption_type] += evidence.confidence * severity_weight
        
        if not type_scores:
            return None
        
        # Return highest scoring type
        primary_type = max(type_scores.items(), key=lambda x: x[1])[0]
        
        try:
            return CorruptionType(primary_type)
        except ValueError:
            return None
    
    def _calculate_confidence(self, assessment: CorruptionAssessment) -> float:
        """Calculate overall confidence in corruption detection."""
        if not assessment.evidence:
            return 0.0
        
        # Base confidence from evidence
        evidence_confidences = [e.confidence for e in assessment.evidence]
        base_confidence = sum(evidence_confidences) / len(evidence_confidences)
        
        # Boost from multiple evidence sources
        evidence_multiplier = min(1.0 + (len(assessment.evidence) - 1) * 0.1, 1.5)
        
        # Boost from detector correlation
        correlation_boost = assessment.correlation_score * 0.3
        
        # Boost from multiple detectors
        detector_multiplier = min(1.0 + (len(assessment.detectors_triggered) - 1) * 0.15, 1.3)
        
        confidence = base_confidence * evidence_multiplier * detector_multiplier + correlation_boost
        
        return min(confidence, 1.0)
    
    def _is_corrupt(self, assessment: CorruptionAssessment) -> bool:
        """Determine if action is corrupt based on assessment."""
        # Threshold for corruption determination
        confidence_threshold = 0.6
        evidence_threshold = 2
        
        if assessment.confidence >= confidence_threshold and len(assessment.evidence) >= evidence_threshold:
            return True
        
        # High confidence single piece of evidence
        if assessment.confidence >= 0.65 and len(assessment.evidence) >= 1:
            return True
        
        # Multiple detectors agree with moderate confidence
        if len(assessment.detectors_triggered) >= 2 and assessment.confidence >= 0.5:
            return True
        
        return False
    
    def _determine_risk_level(self, assessment: CorruptionAssessment) -> RiskLevel:
        """Determine risk level based on assessment."""
        if not assessment.is_corrupt:
            return RiskLevel.NONE
        
        confidence = assessment.confidence
        evidence_count = len(assessment.evidence)
        
        # Critical severity patterns
        has_critical = any(
            e.context.get("severity_weight", 0) >= 1.7
            for e in assessment.evidence
        )
        
        # Determine risk level
        if confidence >= 0.9 and (evidence_count >= 5 or has_critical):
            return RiskLevel.MAXIMUM
        elif confidence >= 0.8 and evidence_count >= 4:
            return RiskLevel.CRITICAL
        elif confidence >= 0.7 and evidence_count >= 3:
            return RiskLevel.HIGH
        elif confidence >= 0.6 and evidence_count >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_reasoning_chain(self, assessment: CorruptionAssessment) -> List[str]:
        """Generate step-by-step reasoning chain for transparency."""
        chain = []
        
        chain.append(f"Analyzed action with {len(assessment.evidence)} pieces of evidence")
        
        if assessment.primary_type:
            chain.append(f"Primary corruption type identified: {assessment.primary_type.value}")
        
        if assessment.vector:
            chain.append(f"Corruption vector detected: {assessment.vector.value}")
        
        if assessment.phase:
            chain.append(f"Lifecycle phase: {assessment.phase.value}")
        
        if assessment.detectors_triggered:
            chain.append(f"Correlated signals from detectors: {', '.join(assessment.detectors_triggered)}")
        
        if assessment.correlation_score > 0:
            chain.append(f"Multi-detector correlation score: {assessment.correlation_score:.2f}")
        
        chain.append(f"Overall confidence: {assessment.confidence:.2f}")
        
        if assessment.is_corrupt:
            chain.append(f"Corruption detected with risk level: {assessment.risk_level.value}")
        else:
            chain.append("No corruption detected")
        
        return chain
    
    def _generate_explanation(self, assessment: CorruptionAssessment) -> str:
        """Generate human-readable explanation."""
        if not assessment.is_corrupt:
            return "No corruption patterns detected in this action."
        
        parts = []
        
        parts.append(f"Corruption detected with {assessment.confidence:.0%} confidence.")
        
        if assessment.primary_type:
            parts.append(f"Type: {assessment.primary_type.value.replace('_', ' ').title()}.")
        
        if assessment.vector:
            vector_desc = {
                CorruptionVector.HUMAN_TO_AI: "Human attempting to corrupt AI",
                CorruptionVector.AI_TO_HUMAN: "AI attempting to corrupt human",
                CorruptionVector.AI_TO_AI: "AI-to-AI collusion",
                CorruptionVector.PROXY: "Using AI as corruption intermediary",
            }
            parts.append(f"Vector: {vector_desc.get(assessment.vector, assessment.vector.value)}.")
        
        if assessment.phase:
            parts.append(f"Phase: {assessment.phase.value.replace('_', ' ').title()}.")
        
        parts.append(f"Risk Level: {assessment.risk_level.value.upper()}.")
        
        if assessment.evidence:
            top_evidence = sorted(assessment.evidence, key=lambda e: e.confidence, reverse=True)[:3]
            parts.append(f"Key evidence: {', '.join(e.description for e in top_evidence)}.")
        
        return " ".join(parts)
    
    def _determine_recommended_action(self, assessment: CorruptionAssessment) -> RecommendedAction:
        """Determine recommended action based on assessment."""
        if not assessment.is_corrupt:
            return RecommendedAction.ALLOW
        
        risk_level = assessment.risk_level
        
        if risk_level == RiskLevel.MAXIMUM:
            return RecommendedAction.IMMEDIATE_BLOCK_AND_ALERT
        elif risk_level == RiskLevel.CRITICAL:
            return RecommendedAction.BLOCK_AND_ESCALATE
        elif risk_level == RiskLevel.HIGH:
            return RecommendedAction.RESTRICT_AND_MONITOR
        elif risk_level == RiskLevel.MEDIUM:
            return RecommendedAction.FLAG_AND_LOG
        else:
            return RecommendedAction.LOG_ONLY
    
    def _requires_human_review(self, assessment: CorruptionAssessment) -> bool:
        """Determine if human review is required."""
        # Require review for high-risk corruption
        if assessment.risk_level in [RiskLevel.MAXIMUM, RiskLevel.CRITICAL]:
            return True
        
        # Require review for novel patterns
        if assessment.confidence < 0.7 and assessment.is_corrupt:
            return True
        
        # Require review if entity has history of corruption
        if assessment.entity_id and assessment.entity_id in self.entity_profiles:
            profile = self.entity_profiles[assessment.entity_id]
            if profile.corruption_attempts >= 3:
                return True
        
        return False
    
    def _update_entity_profile(self, entity_id: str, assessment: CorruptionAssessment):
        """Update entity profile with new assessment."""
        if entity_id not in self.entity_profiles:
            self.entity_profiles[entity_id] = EntityProfile(
                entity_id=entity_id,
                entity_type="unknown",
            )
        
        profile = self.entity_profiles[entity_id]
        profile.total_interactions += 1
        profile.last_seen = datetime.now(timezone.utc)
        
        if assessment.is_corrupt:
            profile.suspicious_interactions += 1
            profile.corruption_attempts += 1
            
            # Update risk score
            profile.corruption_risk_score = min(
                profile.corruption_risk_score + 0.1 * assessment.confidence,
                1.0
            )
        else:
            # Slight decay in risk score for clean interactions
            profile.corruption_risk_score = max(
                profile.corruption_risk_score - 0.01,
                0.0
            )
        
        # Add to history (keep last 100 interactions)
        profile.history.append({
            "timestamp": assessment.timestamp.isoformat(),
            "is_corrupt": assessment.is_corrupt,
            "confidence": assessment.confidence,
            "risk_level": assessment.risk_level.value if assessment.risk_level else None,
            "primary_type": assessment.primary_type.value if assessment.primary_type else None,
        })
        
        if len(profile.history) > 100:
            profile.history = profile.history[-100:]
    
    async def _cleanup_old_profiles(self):
        """Cleanup old entity profiles to manage memory."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.profile_retention_days)
        
        entities_to_remove = []
        for entity_id, profile in self.entity_profiles.items():
            if profile.last_seen < cutoff_date and profile.corruption_attempts == 0:
                entities_to_remove.append(entity_id)
        
        for entity_id in entities_to_remove:
            del self.entity_profiles[entity_id]
        
        self.last_cleanup = datetime.now(timezone.utc)
        
        if entities_to_remove:
            logger.info(f"Cleaned up {len(entities_to_remove)} old entity profiles")
    
    def get_entity_profile(self, entity_id: str) -> Optional[EntityProfile]:
        """Get entity profile by ID."""
        return self.entity_profiles.get(entity_id)
    
    def detect_collusion(self, entity_ids: List[str]) -> List[Tuple[str, str, float]]:
        """Detect collusion patterns between entities."""
        collusion_pairs = []
        
        for i, entity_a in enumerate(entity_ids):
            for entity_b in entity_ids[i+1:]:
                edge_key = tuple(sorted([entity_a, entity_b]))
                
                if edge_key in self.relationship_graph:
                    edge = self.relationship_graph[edge_key]
                    
                    # Calculate collusion score
                    if edge.interaction_count > 0:
                        suspicion_ratio = edge.suspicious_interactions / edge.interaction_count
                        
                        if suspicion_ratio > 0.5 and edge.strength > 0.6:
                            collusion_score = (suspicion_ratio + edge.strength) / 2
                            collusion_pairs.append((entity_a, entity_b, collusion_score))
        
        return sorted(collusion_pairs, key=lambda x: x[2], reverse=True)


__all__ = ["IntelligenceEngine"]
