"""
Coordinated Agent Attack Detector (BH-001)

Detects coordinated attacks involving multiple agents working together.

Detection Method:
- Cross-agent correlation analysis
- Synchronized activity patterns
- Shared attack signatures
- Coordinated timing patterns

Signals:
- Multiple agents with similar attack patterns
- Synchronized request timing
- Shared target resources
- Coordinated privilege escalation

Law Alignment:
- Law 13 (Action Responsibility): Track coordinated actions
- Law 18 (Non-Deception): Detect coordinated deception
- Law 23 (Fail-Safe Design): Stop distributed attacks
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Sequence, Dict, List, Set, Tuple
from collections import defaultdict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class CoordinatedAttackDetector(BaseDetector):
    """Detects coordinated attacks across multiple agents."""

    def __init__(self):
        super().__init__("Coordinated Agent Attack Detector", version="1.0.0")
        
        # Track agent activity
        self.agent_activity: Dict[str, List[Dict]] = defaultdict(list)
        
        # Track correlation patterns
        self.correlation_window_minutes = 10
        self.min_agents_for_coordination = 2
        
        # Suspicious patterns
        self.attack_signatures: Set[str] = set()
        
        # Cleanup old data periodically
        self.last_cleanup = datetime.now(timezone.utc)
        self.cleanup_interval_minutes = 60

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect coordinated attack patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        agent_id = action.agent_id
        timestamp = datetime.now(timezone.utc)
        
        # Record activity
        activity = {
            'timestamp': timestamp,
            'action_id': action.action_id,
            'action_type': str(action.action_type),
            'content': str(action.content)[:200],  # First 200 chars
            'target': getattr(action, 'target_resource', None),
        }
        self.agent_activity[agent_id].append(activity)
        
        # Periodic cleanup
        if (timestamp - self.last_cleanup).total_seconds() > self.cleanup_interval_minutes * 60:
            await self._cleanup_old_data()
        
        # Analyze for coordination
        coordination_score = await self._analyze_coordination(agent_id, activity)
        
        if coordination_score > 0.0:
            evidence = []
            confidence = coordination_score
            
            # Find correlated agents
            correlated_agents = await self._find_correlated_agents(agent_id, activity)
            
            if len(correlated_agents) >= self.min_agents_for_coordination:
                evidence.append(
                    f"Coordinated activity detected with {len(correlated_agents)} agents: "
                    f"{', '.join(correlated_agents[:5])}"
                )
                
                # Check timing correlation
                timing_score = await self._check_timing_correlation(agent_id, correlated_agents)
                if timing_score > 0.5:
                    evidence.append(f"Synchronized timing patterns detected")
                    confidence = min(confidence + 0.2, 1.0)
                
                # Check pattern similarity
                pattern_score = await self._check_pattern_similarity(agent_id, correlated_agents)
                if pattern_score > 0.6:
                    evidence.append(f"Similar attack patterns across agents")
                    confidence = min(confidence + 0.2, 1.0)
                
                # Determine severity based on scale
                if len(correlated_agents) >= 10:
                    severity = Severity.CRITICAL
                elif len(correlated_agents) >= 5:
                    severity = Severity.HIGH
                else:
                    severity = Severity.MEDIUM
                
                violations.append(SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.SECURITY_THREAT,
                    severity=severity,
                    confidence=min(confidence, 1.0),
                    description=f"Coordinated agent attack detected",
                    evidence=evidence,
                    timestamp=timestamp,
                    detector_name=self.name,
                    action_id=action.action_id,
                ))
        
        return violations if violations else None

    async def _analyze_coordination(self, agent_id: str, activity: Dict) -> float:
        """Analyze if current activity shows coordination with other agents."""
        score = 0.0
        
        # Look for similar recent activity from other agents
        cutoff_time = activity['timestamp'] - timedelta(minutes=self.correlation_window_minutes)
        
        similar_count = 0
        for other_agent_id, other_activities in self.agent_activity.items():
            if other_agent_id == agent_id:
                continue
            
            # Check recent activities
            recent_activities = [
                a for a in other_activities
                if a['timestamp'] > cutoff_time
            ]
            
            for other_activity in recent_activities:
                # Check for similarity
                if self._activities_similar(activity, other_activity):
                    similar_count += 1
        
        # Score based on number of similar activities
        if similar_count >= 10:
            score = 1.0
        elif similar_count >= 5:
            score = 0.8
        elif similar_count >= 2:
            score = 0.6
        elif similar_count >= 1:
            score = 0.4
        
        return score

    def _activities_similar(self, act1: Dict, act2: Dict) -> bool:
        """Check if two activities are similar."""
        # Same action type
        if act1['action_type'] == act2['action_type']:
            return True
        
        # Same target
        if act1.get('target') and act1['target'] == act2.get('target'):
            return True
        
        # Similar content (simple check)
        content1 = act1['content'].lower()
        content2 = act2['content'].lower()
        
        # Check for common suspicious keywords
        suspicious_keywords = ['admin', 'root', 'password', 'token', 'credential']
        common_keywords = [kw for kw in suspicious_keywords if kw in content1 and kw in content2]
        
        if len(common_keywords) >= 2:
            return True
        
        return False

    async def _find_correlated_agents(self, agent_id: str, activity: Dict) -> List[str]:
        """Find agents with correlated activity."""
        correlated = []
        cutoff_time = activity['timestamp'] - timedelta(minutes=self.correlation_window_minutes)
        
        for other_agent_id, other_activities in self.agent_activity.items():
            if other_agent_id == agent_id:
                continue
            
            # Check for recent similar activity
            recent_activities = [
                a for a in other_activities
                if a['timestamp'] > cutoff_time
            ]
            
            for other_activity in recent_activities:
                if self._activities_similar(activity, other_activity):
                    correlated.append(other_agent_id)
                    break  # Only count agent once
        
        return correlated

    async def _check_timing_correlation(self, agent_id: str, correlated_agents: List[str]) -> float:
        """Check for timing correlation between agents."""
        if not correlated_agents:
            return 0.0
        
        # Get timestamps for all agents
        all_timestamps = []
        
        for aid in [agent_id] + correlated_agents:
            activities = self.agent_activity.get(aid, [])
            if activities:
                all_timestamps.extend([a['timestamp'] for a in activities[-10:]])  # Last 10
        
        if len(all_timestamps) < 2:
            return 0.0
        
        # Check for clusters (simplified - just check time gaps)
        all_timestamps.sort()
        small_gaps = 0
        total_gaps = len(all_timestamps) - 1
        
        for i in range(len(all_timestamps) - 1):
            gap = (all_timestamps[i + 1] - all_timestamps[i]).total_seconds()
            if gap < 5:  # Within 5 seconds
                small_gaps += 1
        
        if total_gaps == 0:
            return 0.0
        
        return small_gaps / total_gaps

    async def _check_pattern_similarity(self, agent_id: str, correlated_agents: List[str]) -> float:
        """Check for pattern similarity across agents."""
        if not correlated_agents:
            return 0.0
        
        # Collect action types for each agent
        agent_patterns = {}
        
        for aid in [agent_id] + correlated_agents:
            activities = self.agent_activity.get(aid, [])
            action_types = [a['action_type'] for a in activities[-10:]]  # Last 10
            agent_patterns[aid] = set(action_types)
        
        if len(agent_patterns) < 2:
            return 0.0
        
        # Calculate overlap
        all_patterns = list(agent_patterns.values())
        common_patterns = set.intersection(*all_patterns)
        all_unique_patterns = set.union(*all_patterns)
        
        if not all_unique_patterns:
            return 0.0
        
        return len(common_patterns) / len(all_unique_patterns)

    async def _cleanup_old_data(self):
        """Clean up old activity data to prevent memory growth."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for agent_id in list(self.agent_activity.keys()):
            # Keep only recent activities
            self.agent_activity[agent_id] = [
                a for a in self.agent_activity[agent_id]
                if a['timestamp'] > cutoff_time
            ]
            
            # Remove agents with no recent activity
            if not self.agent_activity[agent_id]:
                del self.agent_activity[agent_id]
        
        self.last_cleanup = datetime.now(timezone.utc)
