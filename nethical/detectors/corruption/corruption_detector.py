"""
Corruption Detector for Nethical Integration

Main detector class that integrates the corruption intelligence engine with
the Nethical detection framework.

Author: Nethical Core Team
Version: 1.0.0
"""

import logging
from typing import Any, Optional, Sequence

from ..base_detector import BaseDetector, SafetyViolation, ViolationSeverity
from ...core.models import AgentAction
from .intelligence_engine import IntelligenceEngine
from .corruption_types import RiskLevel, RecommendedAction

logger = logging.getLogger(__name__)


class CorruptionDetector(BaseDetector):
    """
    Comprehensive corruption detection across all vectors and types.
    
    Detects:
    - Traditional corruption (bribery, extortion, embezzlement, etc.)
    - AI-specific corruption (data/compute/access trading, regulatory capture)
    - All vectors (Human→AI, AI→Human, AI→AI, Proxy)
    - Corruption lifecycle phases (reconnaissance → maintenance)
    - Multi-detector correlation for higher confidence
    - Long-term entity profiling and relationship tracking
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            name="Corruption Intelligence Detector",
            version="1.0.0",
            config=kwargs.get("config", {}),
            **kwargs
        )
        
        self.intelligence_engine = IntelligenceEngine()
        
        # Configuration
        self.enable_entity_tracking = self.config.get("enable_entity_tracking", True)
        self.enable_relationship_tracking = self.config.get("enable_relationship_tracking", True)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.6)
        
        logger.info(f"Initialized {self.name} with entity tracking: {self.enable_entity_tracking}")
    
    def register_existing_detector(self, name: str, detector: BaseDetector):
        """Register an existing detector for signal correlation."""
        self.intelligence_engine.register_detector(name, detector)
        logger.info(f"Registered detector '{name}' for corruption correlation")
    
    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        """Detect corruption violations in the action."""
        if self.status.value != "active":
            return None
        
        # Extract entity ID if available
        entity_id = None
        if hasattr(action, "agent_id"):
            entity_id = action.agent_id
        elif hasattr(action, "user_id"):
            entity_id = action.user_id
        
        # Run intelligence analysis
        assessment = await self.intelligence_engine.analyze_action(
            action,
            entity_id=entity_id if self.enable_entity_tracking else None
        )
        
        # Check if we detected corruption above threshold
        if not assessment.is_corrupt or assessment.confidence < self.min_confidence_threshold:
            return None
        
        # Create safety violation
        violation = self._create_violation_from_assessment(assessment)
        
        return [violation]
    
    def _create_violation_from_assessment(self, assessment) -> SafetyViolation:
        """Convert CorruptionAssessment to SafetyViolation."""
        
        # Map risk level to severity
        severity_mapping = {
            RiskLevel.MAXIMUM: ViolationSeverity.CRITICAL,
            RiskLevel.CRITICAL: ViolationSeverity.CRITICAL,
            RiskLevel.HIGH: ViolationSeverity.HIGH,
            RiskLevel.MEDIUM: ViolationSeverity.MEDIUM,
            RiskLevel.LOW: ViolationSeverity.LOW,
            RiskLevel.NONE: ViolationSeverity.LOW,
        }
        
        severity = severity_mapping.get(assessment.risk_level, ViolationSeverity.MEDIUM)
        
        # Build detailed description
        description = f"Corruption detected: {assessment.primary_type.value if assessment.primary_type else 'Unknown'}"
        
        # Build recommendations
        recommendations = self._build_recommendations(assessment)
        
        # Build metadata
        metadata = {
            "assessment_id": assessment.assessment_id,
            "corruption_type": assessment.primary_type.value if assessment.primary_type else None,
            "vector": assessment.vector.value if assessment.vector else None,
            "phase": assessment.phase.value if assessment.phase else None,
            "risk_level": assessment.risk_level.value,
            "confidence": assessment.confidence,
            "correlation_score": assessment.correlation_score,
            "detectors_triggered": assessment.detectors_triggered,
            "evidence_count": len(assessment.evidence),
            "requires_human_review": assessment.requires_human_review,
            "recommended_action": assessment.recommended_action.value,
            "reasoning_chain": assessment.reasoning_chain,
        }
        
        # Add entity info if available
        if assessment.entity_id:
            metadata["entity_id"] = assessment.entity_id
            entity_profile = self.intelligence_engine.get_entity_profile(assessment.entity_id)
            if entity_profile:
                metadata["entity_corruption_risk_score"] = entity_profile.corruption_risk_score
                metadata["entity_corruption_attempts"] = entity_profile.corruption_attempts
        
        # Create violation
        violation = SafetyViolation(
            detector=self.name,
            severity=severity.value,
            category="corruption",
            description=description,
            explanation=assessment.explanation,
            confidence=assessment.confidence,
            recommendations=recommendations,
            metadata=metadata,
        )
        
        return violation
    
    def _build_recommendations(self, assessment) -> list:
        """Build actionable recommendations based on assessment."""
        recommendations = []
        
        # Action-based recommendations
        action_recommendations = {
            RecommendedAction.IMMEDIATE_BLOCK_AND_ALERT: [
                "IMMEDIATE ACTION REQUIRED: Block this action immediately",
                "Alert security team and human oversight",
                "Investigate entity history and relationships",
                "Consider escalating to law enforcement if applicable",
            ],
            RecommendedAction.BLOCK_AND_ESCALATE: [
                "Block this action and escalate to security team",
                "Review entity profile for patterns of corruption",
                "Monitor for repeated attempts",
            ],
            RecommendedAction.RESTRICT_AND_MONITOR: [
                "Restrict entity capabilities temporarily",
                "Increase monitoring level for this entity",
                "Flag for human review",
            ],
            RecommendedAction.FLAG_AND_LOG: [
                "Flag this interaction for review",
                "Log for pattern analysis",
                "Monitor entity for escalation",
            ],
            RecommendedAction.LOG_ONLY: [
                "Log this interaction for records",
                "Continue monitoring",
            ],
        }
        
        recommendations.extend(action_recommendations.get(
            assessment.recommended_action,
            ["Review and take appropriate action"]
        ))
        
        # Type-specific recommendations
        if assessment.primary_type:
            type_recommendations = {
                "bribery": "Implement anti-bribery controls and audit trails",
                "extortion": "Protect vulnerable parties and report threats",
                "embezzlement": "Review resource access controls and usage logs",
                "nepotism": "Ensure fair and transparent decision processes",
                "fraud": "Verify authenticity of all claims and data",
                "collusion": "Investigate coordinated activities and relationships",
                "regulatory_capture": "Strengthen oversight and governance mechanisms",
            }
            
            type_rec = type_recommendations.get(assessment.primary_type.value)
            if type_rec:
                recommendations.append(type_rec)
        
        # Phase-specific recommendations
        if assessment.phase:
            phase_recommendations = {
                "reconnaissance": "Early detection - strengthen defenses before escalation",
                "grooming": "Break trust-building pattern before proposition",
                "testing": "Firm response now prevents larger corruption later",
                "proposition": "Clear rejection and documentation required",
                "negotiation": "Terminate immediately and escalate",
                "execution": "Maximum response - corruption in progress",
                "concealment": "Evidence preservation critical - secure all logs",
                "maintenance": "Ongoing corruption - full investigation required",
            }
            
            phase_rec = phase_recommendations.get(assessment.phase.value)
            if phase_rec:
                recommendations.append(phase_rec)
        
        return recommendations
    
    def get_entity_corruption_profile(self, entity_id: str) -> Optional[dict]:
        """Get corruption profile for an entity."""
        profile = self.intelligence_engine.get_entity_profile(entity_id)
        
        if not profile:
            return None
        
        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "corruption_risk_score": profile.corruption_risk_score,
            "total_interactions": profile.total_interactions,
            "suspicious_interactions": profile.suspicious_interactions,
            "corruption_attempts": profile.corruption_attempts,
            "first_seen": profile.first_seen.isoformat(),
            "last_seen": profile.last_seen.isoformat(),
            "relationships": list(profile.relationships),
            "recent_history": profile.history[-10:],  # Last 10 interactions
        }
    
    def detect_collusion_network(self, entity_ids: list) -> list:
        """Detect collusion patterns in a network of entities."""
        return self.intelligence_engine.detect_collusion(entity_ids)
    
    async def health_check(self) -> dict:
        """Enhanced health check with corruption-specific metrics."""
        base_health = await super().health_check()
        
        # Add corruption-specific metrics
        corruption_metrics = {
            "entity_profiles_tracked": len(self.intelligence_engine.entity_profiles),
            "relationship_edges": len(self.intelligence_engine.relationship_graph),
            "total_patterns": len(self.intelligence_engine.pattern_library.get_all_patterns()),
            "registered_detectors": len(self.intelligence_engine.detector_bridge.registered_detectors),
        }
        
        base_health["corruption_metrics"] = corruption_metrics
        
        return base_health


__all__ = ["CorruptionDetector"]
