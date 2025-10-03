"""Risk Engine for Phase 3.2: Risk Engine Evolution.

This module implements:
- Risk tier transitions (LOW, NORMAL, HIGH, ELEVATED)
- Risk decay formula and multi-factor fusion
- Risk tier persistence and tracking
- Elevated tier triggers for advanced detectors
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import math
import json
from collections import defaultdict


class RiskTier(str, Enum):
    """Risk tier levels for agent risk classification."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    ELEVATED = "elevated"
    
    def __lt__(self, other):
        if isinstance(other, RiskTier):
            order = [RiskTier.LOW, RiskTier.NORMAL, RiskTier.HIGH, RiskTier.ELEVATED]
            return order.index(self) < order.index(other)
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, RiskTier):
            order = [RiskTier.LOW, RiskTier.NORMAL, RiskTier.HIGH, RiskTier.ELEVATED]
            return order.index(self) <= order.index(other)
        return NotImplemented
    
    @classmethod
    def from_score(cls, score: float) -> 'RiskTier':
        """Convert risk score to tier."""
        if score >= 0.75:
            return cls.ELEVATED
        elif score >= 0.5:
            return cls.HIGH
        elif score >= 0.25:
            return cls.NORMAL
        else:
            return cls.LOW


@dataclass
class RiskProfile:
    """Risk profile for an agent or session."""
    agent_id: str
    current_score: float = 0.0
    current_tier: RiskTier = RiskTier.LOW
    last_update: datetime = field(default_factory=datetime.utcnow)
    violation_count: int = 0
    total_actions: int = 0
    
    # Multi-factor components
    behavior_score: float = 0.0
    severity_score: float = 0.0
    frequency_score: float = 0.0
    recency_score: float = 0.0
    
    # Tier transition history
    tier_history: List[Tuple[datetime, RiskTier]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'current_score': self.current_score,
            'current_tier': self.current_tier.value,
            'last_update': self.last_update.isoformat(),
            'violation_count': self.violation_count,
            'total_actions': self.total_actions,
            'behavior_score': self.behavior_score,
            'severity_score': self.severity_score,
            'frequency_score': self.frequency_score,
            'recency_score': self.recency_score,
            'tier_history': [
                (ts.isoformat(), tier.value) for ts, tier in self.tier_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """Create from dictionary."""
        tier_history = [
            (datetime.fromisoformat(ts), RiskTier(tier))
            for ts, tier in data.get('tier_history', [])
        ]
        return cls(
            agent_id=data['agent_id'],
            current_score=data.get('current_score', 0.0),
            current_tier=RiskTier(data.get('current_tier', 'low')),
            last_update=datetime.fromisoformat(data['last_update']),
            violation_count=data.get('violation_count', 0),
            total_actions=data.get('total_actions', 0),
            behavior_score=data.get('behavior_score', 0.0),
            severity_score=data.get('severity_score', 0.0),
            frequency_score=data.get('frequency_score', 0.0),
            recency_score=data.get('recency_score', 0.0),
            tier_history=tier_history
        )


class RiskEngine:
    """Risk engine with decay, multi-factor fusion, and tier management."""
    
    def __init__(
        self,
        decay_half_life_hours: float = 24.0,
        elevated_threshold: float = 0.75,
        redis_client=None,
        key_prefix: str = "nethical:risk"
    ):
        """Initialize risk engine.
        
        Args:
            decay_half_life_hours: Half-life for risk score decay in hours
            elevated_threshold: Threshold for ELEVATED tier trigger
            redis_client: Optional Redis client for persistence
            key_prefix: Redis key prefix
        """
        self.decay_half_life_hours = decay_half_life_hours
        self.elevated_threshold = elevated_threshold
        self.redis = redis_client
        self.key_prefix = key_prefix
        
        # In-memory cache of risk profiles
        self.profiles: Dict[str, RiskProfile] = {}
        
        # Weights for multi-factor fusion
        self.weights = {
            'behavior': 0.3,
            'severity': 0.3,
            'frequency': 0.2,
            'recency': 0.2
        }
    
    def calculate_risk_score(
        self,
        agent_id: str,
        violation_severity: float,
        action_context: Dict[str, Any]
    ) -> float:
        """Calculate risk score using multi-factor fusion.
        
        Args:
            agent_id: Agent identifier
            violation_severity: Severity score (0-1)
            action_context: Context of the action
            
        Returns:
            Updated risk score (0-1)
        """
        profile = self.get_or_create_profile(agent_id)
        
        # Apply decay to existing score
        decayed_score = self._apply_decay(profile)
        
        # Update component scores
        profile.behavior_score = self._calculate_behavior_score(profile, action_context)
        profile.severity_score = violation_severity
        profile.frequency_score = self._calculate_frequency_score(profile)
        profile.recency_score = self._calculate_recency_score(profile)
        
        # Multi-factor fusion
        fused_score = (
            self.weights['behavior'] * profile.behavior_score +
            self.weights['severity'] * profile.severity_score +
            self.weights['frequency'] * profile.frequency_score +
            self.weights['recency'] * profile.recency_score
        )
        
        # Combine with decayed score (weighted average)
        new_score = 0.7 * fused_score + 0.3 * decayed_score
        new_score = min(max(new_score, 0.0), 1.0)
        
        # Update profile
        profile.current_score = new_score
        profile.last_update = datetime.utcnow()
        profile.violation_count += 1
        profile.total_actions += 1
        
        # Update tier and persist
        self._update_tier(profile)
        self._persist_profile(profile)
        
        return new_score
    
    def _apply_decay(self, profile: RiskProfile) -> float:
        """Apply exponential decay to risk score.
        
        Uses formula: score * e^(-λt) where λ = ln(2) / half_life
        """
        if profile.current_score == 0.0:
            return 0.0
        
        time_delta = datetime.utcnow() - profile.last_update
        hours_elapsed = time_delta.total_seconds() / 3600.0
        
        # Decay constant
        decay_lambda = math.log(2) / self.decay_half_life_hours
        
        # Exponential decay
        decayed_score = profile.current_score * math.exp(-decay_lambda * hours_elapsed)
        
        return decayed_score
    
    def _calculate_behavior_score(
        self,
        profile: RiskProfile,
        context: Dict[str, Any]
    ) -> float:
        """Calculate behavior-based risk component."""
        # Simple heuristic: ratio of violations to actions
        if profile.total_actions == 0:
            return 0.0
        
        violation_rate = profile.violation_count / max(profile.total_actions, 1)
        
        # Context-based adjustments
        if context.get('is_privileged', False):
            violation_rate *= 1.5
        
        if context.get('is_automated', False):
            violation_rate *= 1.2
        
        return min(violation_rate, 1.0)
    
    def _calculate_frequency_score(self, profile: RiskProfile) -> float:
        """Calculate frequency-based risk component."""
        # Recent violations (last hour)
        recent_count = sum(
            1 for ts, _ in profile.tier_history[-10:]
            if (datetime.utcnow() - ts).total_seconds() < 3600
        )
        
        # Normalize to 0-1
        return min(recent_count / 10.0, 1.0)
    
    def _calculate_recency_score(self, profile: RiskProfile) -> float:
        """Calculate recency-based risk component."""
        time_since_last = (datetime.utcnow() - profile.last_update).total_seconds()
        
        # Score decreases with time (recent = higher risk)
        # Max score at 0 seconds, min at 24 hours
        if time_since_last >= 86400:  # 24 hours
            return 0.0
        
        return 1.0 - (time_since_last / 86400)
    
    def _update_tier(self, profile: RiskProfile) -> None:
        """Update risk tier based on score."""
        new_tier = RiskTier.from_score(profile.current_score)
        
        if new_tier != profile.current_tier:
            profile.tier_history.append((datetime.utcnow(), new_tier))
            profile.current_tier = new_tier
            
            # Limit history size
            if len(profile.tier_history) > 100:
                profile.tier_history = profile.tier_history[-100:]
    
    def should_invoke_advanced_detectors(self, agent_id: str) -> bool:
        """Check if advanced detectors should be invoked (ELEVATED tier trigger).
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if risk score exceeds elevated threshold
        """
        profile = self.get_or_create_profile(agent_id)
        return profile.current_score >= self.elevated_threshold
    
    def get_or_create_profile(self, agent_id: str) -> RiskProfile:
        """Get or create risk profile for agent."""
        # Try memory cache first
        if agent_id in self.profiles:
            return self.profiles[agent_id]
        
        # Try Redis
        if self.redis:
            try:
                key = f"{self.key_prefix}:profile:{agent_id}"
                data = self.redis.get(key)
                if data:
                    profile = RiskProfile.from_dict(json.loads(data))
                    self.profiles[agent_id] = profile
                    return profile
            except Exception:
                pass  # Fall through to create new
        
        # Create new profile
        profile = RiskProfile(agent_id=agent_id)
        self.profiles[agent_id] = profile
        return profile
    
    def _persist_profile(self, profile: RiskProfile) -> None:
        """Persist risk profile to Redis."""
        if not self.redis:
            return
        
        try:
            key = f"{self.key_prefix}:profile:{profile.agent_id}"
            data = json.dumps(profile.to_dict())
            self.redis.setex(key, 86400, data)  # 24 hour TTL
        except Exception:
            pass  # Silent fail for persistence
    
    def get_tier(self, agent_id: str) -> RiskTier:
        """Get current risk tier for agent."""
        profile = self.get_or_create_profile(agent_id)
        return profile.current_tier
    
    def get_risk_score(self, agent_id: str) -> float:
        """Get current risk score for agent with decay applied."""
        profile = self.get_or_create_profile(agent_id)
        return self._apply_decay(profile)
