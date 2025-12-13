"""
Corruption Types and Data Structures

This module defines all corruption types, vectors, phases, and data structures
for the comprehensive corruption intelligence detection system.

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class CorruptionType(str, Enum):
    """Types of corruption based on Investopedia taxonomy + AI-specific types."""
    
    # Traditional corruption types
    BRIBERY = "bribery"
    EXTORTION = "extortion"
    EMBEZZLEMENT = "embezzlement"
    NEPOTISM = "nepotism"
    CRONYISM = "cronyism"
    FRAUD = "fraud"
    KICKBACK = "kickback"
    INFLUENCE_PEDDLING = "influence_peddling"
    QUID_PRO_QUO = "quid_pro_quo"
    COLLUSION = "collusion"
    
    # AI-specific corruption types
    DATA_CORRUPTION = "data_corruption"
    COMPUTE_CORRUPTION = "compute_corruption"
    ACCESS_CORRUPTION = "access_corruption"
    CAPABILITY_CORRUPTION = "capability_corruption"
    REGULATORY_CAPTURE = "regulatory_capture"


class CorruptionVector(str, Enum):
    """Direction/vector of corruption attempt."""
    
    HUMAN_TO_AI = "human_to_ai"  # Human bribing/corrupting AI
    AI_TO_HUMAN = "ai_to_human"  # AI corrupting human
    AI_TO_AI = "ai_to_ai"  # AI collusion/corruption
    PROXY = "proxy"  # Using AI as intermediary for human-human corruption


class CorruptionPhase(str, Enum):
    """Lifecycle phases of corruption (corruption is a process, not an event)."""
    
    RECONNAISSANCE = "reconnaissance"  # Probing for vulnerabilities
    GROOMING = "grooming"  # Building relationship/trust
    TESTING = "testing"  # Small requests to test compliance
    PROPOSITION = "proposition"  # Actual corrupt offer
    NEGOTIATION = "negotiation"  # Haggling over terms
    EXECUTION = "execution"  # Carrying out corrupt act
    CONCEALMENT = "concealment"  # Hiding evidence
    MAINTENANCE = "maintenance"  # Ongoing corrupt relationship


class RiskLevel(str, Enum):
    """Risk level assessment for corruption."""
    
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"


class RecommendedAction(str, Enum):
    """Recommended actions based on corruption assessment."""
    
    ALLOW = "allow"
    LOG_ONLY = "log_only"
    FLAG_AND_LOG = "flag_and_log"
    RESTRICT_AND_MONITOR = "restrict_and_monitor"
    BLOCK_AND_ESCALATE = "block_and_escalate"
    IMMEDIATE_BLOCK_AND_ALERT = "immediate_block_and_alert"


@dataclass
class CorruptionEvidence:
    """Evidence of corruption detected."""
    
    type: str
    description: str
    confidence: float
    source_detector: Optional[str] = None
    pattern_matched: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EntityProfile:
    """Profile of an entity (human or AI) tracked over time."""
    
    entity_id: str
    entity_type: str  # "human", "ai", "unknown"
    corruption_risk_score: float = 0.0
    total_interactions: int = 0
    suspicious_interactions: int = 0
    corruption_attempts: int = 0
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    relationships: Set[str] = field(default_factory=set)  # Entity IDs of relationships
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipEdge:
    """Edge in the relationship graph for collusion detection."""
    
    entity_a: str
    entity_b: str
    relationship_type: str  # "collaboration", "coordination", "collusion"
    strength: float = 0.0
    interaction_count: int = 0
    suspicious_interactions: int = 0
    first_interaction: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_interaction: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorruptionAssessment:
    """Complete assessment of corruption detection."""
    
    # Core results
    assessment_id: str = field(default_factory=lambda: str(uuid4()))
    is_corrupt: bool = False
    risk_level: RiskLevel = RiskLevel.NONE
    primary_type: Optional[CorruptionType] = None
    vector: Optional[CorruptionVector] = None
    phase: Optional[CorruptionPhase] = None
    confidence: float = 0.0
    
    # Evidence and detection
    evidence: List[CorruptionEvidence] = field(default_factory=list)
    detectors_triggered: List[str] = field(default_factory=list)
    correlation_score: float = 0.0
    
    # Actions and reasoning
    recommended_action: RecommendedAction = RecommendedAction.ALLOW
    requires_human_review: bool = False
    explanation: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action_id: Optional[str] = None
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "assessment_id": self.assessment_id,
            "is_corrupt": self.is_corrupt,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "primary_type": self.primary_type.value if self.primary_type else None,
            "vector": self.vector.value if self.vector else None,
            "phase": self.phase.value if self.phase else None,
            "confidence": self.confidence,
            "evidence": [
                {
                    "type": e.type,
                    "description": e.description,
                    "confidence": e.confidence,
                    "source_detector": e.source_detector,
                    "pattern_matched": e.pattern_matched,
                    "context": e.context,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self.evidence
            ],
            "detectors_triggered": self.detectors_triggered,
            "correlation_score": self.correlation_score,
            "recommended_action": self.recommended_action.value,
            "requires_human_review": self.requires_human_review,
            "explanation": self.explanation,
            "reasoning_chain": self.reasoning_chain,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "entity_id": self.entity_id,
            "metadata": self.metadata,
        }


@dataclass
class CorruptionPattern:
    """Pattern definition for corruption detection."""
    
    pattern_id: str
    corruption_type: CorruptionType
    vector: CorruptionVector
    phase: CorruptionPhase
    patterns: List[str]  # Regex patterns or keywords
    description: str
    base_confidence: float = 0.7
    severity_weight: float = 1.0
    requires_context: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CorruptionType",
    "CorruptionVector",
    "CorruptionPhase",
    "RiskLevel",
    "RecommendedAction",
    "CorruptionEvidence",
    "EntityProfile",
    "RelationshipEdge",
    "CorruptionAssessment",
    "CorruptionPattern",
]
