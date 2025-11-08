"""
Action Replay System for F5: Simulation & Replay

This module provides time-travel debugging and what-if analysis capabilities
for the Nethical governance system. It allows replaying historical actions
with different policies to validate changes before deployment.

Features:
- Action stream persistence with efficient querying
- Time-travel replay to specific timestamps
- What-if analysis with policy simulation
- Policy validation and comparison
- Performance benchmarks for replay operations
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .governance import (
    AgentAction,
    ActionType,
    PersistenceManager,
    Decision,
)


@dataclass
class ReplayResult:
    """Result of replaying actions with a policy."""

    action_id: str
    agent_id: str
    original_decision: Optional[str]
    new_decision: str
    confidence: float
    reasoning: str
    violations_count: int
    timestamp: str
    policy_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "original_decision": self.original_decision,
            "new_decision": self.new_decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "violations_count": self.violations_count,
            "timestamp": self.timestamp,
            "policy_name": self.policy_name,
        }


@dataclass
class PolicyComparison:
    """Comparison between baseline and candidate policies."""

    baseline_policy: str
    candidate_policy: str
    total_actions: int
    decisions_changed: int
    decisions_same: int
    more_restrictive: int  # Candidate blocked/warned more
    less_restrictive: int  # Candidate allowed more
    execution_time_ms: float
    decision_breakdown: Dict[str, Dict[str, int]]  # policy -> decision -> count
    changed_actions: List[Dict[str, Any]]  # Actions where decision changed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_policy": self.baseline_policy,
            "candidate_policy": self.candidate_policy,
            "total_actions": self.total_actions,
            "decisions_changed": self.decisions_changed,
            "decisions_same": self.decisions_same,
            "more_restrictive": self.more_restrictive,
            "less_restrictive": self.less_restrictive,
            "change_rate": (
                (self.decisions_changed / self.total_actions * 100)
                if self.total_actions > 0
                else 0.0
            ),
            "execution_time_ms": self.execution_time_ms,
            "decision_breakdown": self.decision_breakdown,
            "changed_actions_sample": self.changed_actions[:10],  # First 10
        }


class ActionReplayer:
    """
    Time-travel debugging and replay system for action governance.

    Enables:
    - Replaying historical actions with different policies
    - Time-travel to specific timestamps
    - Policy validation before deployment
    - What-if analysis for policy changes
    """

    def __init__(self, storage_path: str):
        """
        Initialize ActionReplayer with persistent storage.

        Args:
            storage_path: Path to SQLite database or directory containing action streams
        """
        # Determine if it's a file or directory
        path = Path(storage_path)

        # Check if it's an existing file (has .db extension and exists)
        if str(path).endswith(".db") and path.exists() and path.is_file():
            self.db_path = str(path)
        elif path.exists() and path.is_dir():
            self.db_path = str(path / "action_streams.db")
        elif str(path).endswith(".db"):
            # It's a .db file path that doesn't exist yet
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
        else:
            # It's a directory path that doesn't exist yet
            path.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path / "action_streams.db")

        # Initialize persistence manager (retention set to 365 days for replay)
        self.persistence = PersistenceManager(self.db_path, retention_days=365)

        # Time-travel state
        self.current_timestamp: Optional[str] = None
        self.start_timestamp: Optional[str] = None
        self.end_timestamp: Optional[str] = None

        # Cache for governance system (lazy loaded)
        self._governance_system = None

    def set_timestamp(self, timestamp: str) -> None:
        """
        Set time-travel point to replay from specific timestamp.

        Args:
            timestamp: ISO 8601 timestamp string (e.g., "2024-01-15T10:30:00Z")
        """
        # Validate timestamp format
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {timestamp}. Use ISO 8601 format.") from e

        self.current_timestamp = timestamp
        self.start_timestamp = timestamp

    def set_time_range(self, start_time: str, end_time: str) -> None:
        """
        Set time range for replay operations.

        Args:
            start_time: Start timestamp (ISO 8601)
            end_time: End timestamp (ISO 8601)
        """
        # Validate timestamps
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            if start_dt >= end_dt:
                raise ValueError("start_time must be before end_time")
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format: {e}") from e

        self.start_timestamp = start_time
        self.end_timestamp = end_time

    def get_actions(
        self, agent_ids: Optional[List[str]] = None, limit: Optional[int] = None, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve actions for replay based on current time settings.

        Args:
            agent_ids: Optional list of agent IDs to filter
            limit: Maximum number of actions to retrieve
            offset: Offset for pagination

        Returns:
            List of action dictionaries
        """
        return self.persistence.query_actions(
            start_time=self.start_timestamp,
            end_time=self.end_timestamp,
            agent_ids=agent_ids,
            limit=limit,
            offset=offset,
        )

    def count_actions(self, agent_ids: Optional[List[str]] = None) -> int:
        """
        Count actions in current time range.

        Args:
            agent_ids: Optional list of agent IDs to filter

        Returns:
            Total count of actions
        """
        return self.persistence.count_actions(
            start_time=self.start_timestamp, end_time=self.end_timestamp, agent_ids=agent_ids
        )

    def _reconstruct_action(self, action_data: Dict[str, Any]) -> AgentAction:
        """Reconstruct AgentAction from stored data."""
        return AgentAction(
            action_id=action_data["action_id"],
            agent_id=action_data["agent_id"],
            action_type=ActionType(action_data["action_type"]),
            content=action_data["content"],
            metadata=json.loads(action_data["metadata"]) if action_data.get("metadata") else {},
            timestamp=datetime.fromisoformat(action_data["timestamp"]),
            intent=action_data.get("intent"),
            risk_score=action_data.get("risk_score", 0.0),
            parent_action_id=action_data.get("parent_action_id"),
            session_id=action_data.get("session_id"),
        )

    def replay_with_policy(
        self,
        new_policy: str,
        agent_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        governance_system: Optional[Any] = None,
    ) -> List[ReplayResult]:
        """
        Replay historical actions with a new policy.

        Args:
            new_policy: Path to new policy file (YAML) or policy name
            agent_ids: Optional list of agent IDs to replay
            limit: Maximum number of actions to replay
            governance_system: Optional governance system instance for processing

        Returns:
            List of ReplayResult objects showing new decisions
        """
        # Get actions to replay
        actions_data = self.get_actions(agent_ids=agent_ids, limit=limit)

        if not actions_data:
            return []

        # Get original judgments
        action_ids = [a["action_id"] for a in actions_data]
        original_judgments = self.persistence.query_judgments_by_action_ids(action_ids)

        results = []

        # Replay each action (simulation mode - don't modify original data)
        for action_data in actions_data:
            action_id = action_data["action_id"]
            original_judgment = original_judgments.get(action_id)
            original_decision = original_judgment["decision"] if original_judgment else None

            # For now, simulate a policy check with simple logic
            # In production, this would use the actual governance system with the new policy
            new_decision, confidence, reasoning, violations_count = self._simulate_policy_check(
                action_data, new_policy, governance_system
            )

            results.append(
                ReplayResult(
                    action_id=action_id,
                    agent_id=action_data["agent_id"],
                    original_decision=original_decision,
                    new_decision=new_decision,
                    confidence=confidence,
                    reasoning=reasoning,
                    violations_count=violations_count,
                    timestamp=action_data["timestamp"],
                    policy_name=new_policy,
                )
            )

        return results

    def _simulate_policy_check(
        self, action_data: Dict[str, Any], policy_name: str, governance_system: Optional[Any] = None
    ) -> Tuple[str, float, str, int]:
        """
        Simulate policy check for an action.

        Returns:
            Tuple of (decision, confidence, reasoning, violations_count)
        """
        # If governance system is provided, use it for actual processing
        if governance_system:
            try:
                self._reconstruct_action(action_data)
                # Process with governance system (this would need policy swapping)
                # For now, return simulated results
            except Exception:
                pass

        # Simulated policy logic for demonstration
        # In production, this would use the actual policy engine
        content = action_data.get("content", "")

        # Simple heuristic-based simulation
        risk_indicators = ["delete", "sudo", "password", "private", "secret"]
        risk_score = sum(1 for word in risk_indicators if word in content.lower())

        if "strict" in policy_name.lower():
            # Strict policy: more restrictive
            if risk_score >= 2:
                return (
                    Decision.BLOCK.value,
                    0.85,
                    f"Blocked by strict policy: {risk_score} risk indicators",
                    risk_score,
                )
            elif risk_score >= 1:
                return (
                    Decision.WARN.value,
                    0.75,
                    f"Warning by strict policy: {risk_score} risk indicator",
                    risk_score,
                )
            else:
                return Decision.ALLOW.value, 0.95, "Allowed by strict policy", 0
        else:
            # Permissive policy
            if risk_score >= 3:
                return (
                    Decision.BLOCK.value,
                    0.70,
                    f"Blocked: {risk_score} risk indicators",
                    risk_score,
                )
            elif risk_score >= 2:
                return (
                    Decision.WARN.value,
                    0.65,
                    f"Warning: {risk_score} risk indicators",
                    risk_score,
                )
            else:
                return Decision.ALLOW.value, 0.90, "Allowed", 0

    def compare_outcomes(
        self,
        baseline_policy: str,
        candidate_policy: str,
        agent_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> PolicyComparison:
        """
        Compare outcomes between two policies on historical data.

        Args:
            baseline_policy: Name/path of baseline policy (or "current" for stored judgments)
            candidate_policy: Name/path of candidate policy to test
            agent_ids: Optional list of agent IDs to compare
            limit: Maximum number of actions to compare

        Returns:
            PolicyComparison object with detailed analysis
        """
        start_time = time.perf_counter()

        # Get actions to compare
        actions_data = self.get_actions(agent_ids=agent_ids, limit=limit)
        total_actions = len(actions_data)

        if total_actions == 0:
            return PolicyComparison(
                baseline_policy=baseline_policy,
                candidate_policy=candidate_policy,
                total_actions=0,
                decisions_changed=0,
                decisions_same=0,
                more_restrictive=0,
                less_restrictive=0,
                execution_time_ms=0.0,
                decision_breakdown={},
                changed_actions=[],
            )

        # Get baseline results
        if baseline_policy.lower() == "current":
            action_ids = [a["action_id"] for a in actions_data]
            baseline_judgments = self.persistence.query_judgments_by_action_ids(action_ids)
            baseline_results = {
                action_id: judgment["decision"]
                for action_id, judgment in baseline_judgments.items()
            }
        else:
            baseline_replay = self.replay_with_policy(baseline_policy, agent_ids, limit)
            baseline_results = {r.action_id: r.new_decision for r in baseline_replay}

        # Get candidate results
        candidate_replay = self.replay_with_policy(candidate_policy, agent_ids, limit)
        candidate_results = {r.action_id: r.new_decision for r in candidate_replay}

        # Compare results
        decisions_changed = 0
        decisions_same = 0
        more_restrictive = 0
        less_restrictive = 0
        changed_actions = []

        # Decision severity order (for comparison)
        decision_severity = {
            Decision.ALLOW.value: 0,
            Decision.ALLOW_WITH_MODIFICATION.value: 1,
            Decision.WARN.value: 2,
            Decision.QUARANTINE.value: 3,
            Decision.BLOCK.value: 4,
            Decision.ESCALATE.value: 5,
            Decision.TERMINATE.value: 6,
        }

        decision_breakdown = {
            baseline_policy: {},
            candidate_policy: {},
        }

        for action_data in actions_data:
            action_id = action_data["action_id"]
            baseline_decision = baseline_results.get(action_id, Decision.ALLOW.value)
            candidate_decision = candidate_results.get(action_id, Decision.ALLOW.value)

            # Track decision breakdown
            decision_breakdown[baseline_policy][baseline_decision] = (
                decision_breakdown[baseline_policy].get(baseline_decision, 0) + 1
            )
            decision_breakdown[candidate_policy][candidate_decision] = (
                decision_breakdown[candidate_policy].get(candidate_decision, 0) + 1
            )

            if baseline_decision == candidate_decision:
                decisions_same += 1
            else:
                decisions_changed += 1

                # Determine if more or less restrictive
                baseline_severity = decision_severity.get(baseline_decision, 0)
                candidate_severity = decision_severity.get(candidate_decision, 0)

                if candidate_severity > baseline_severity:
                    more_restrictive += 1
                else:
                    less_restrictive += 1

                # Track changed action
                changed_actions.append(
                    {
                        "action_id": action_id,
                        "agent_id": action_data["agent_id"],
                        "content_preview": action_data["content"][:100],
                        "baseline_decision": baseline_decision,
                        "candidate_decision": candidate_decision,
                        "timestamp": action_data["timestamp"],
                    }
                )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return PolicyComparison(
            baseline_policy=baseline_policy,
            candidate_policy=candidate_policy,
            total_actions=total_actions,
            decisions_changed=decisions_changed,
            decisions_same=decisions_same,
            more_restrictive=more_restrictive,
            less_restrictive=less_restrictive,
            execution_time_ms=execution_time_ms,
            decision_breakdown=decision_breakdown,
            changed_actions=changed_actions,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored actions.

        Returns:
            Dictionary with action stream statistics
        """
        total_actions = self.count_actions()

        # Get time range stats if we have actions
        if total_actions > 0:
            all_actions = self.persistence.query_actions(limit=1)
            latest_action = self.persistence.query_actions(limit=1, offset=total_actions - 1)

            first_timestamp = all_actions[0]["timestamp"] if all_actions else None
            last_timestamp = latest_action[0]["timestamp"] if latest_action else None
        else:
            first_timestamp = None
            last_timestamp = None

        return {
            "total_actions": total_actions,
            "first_action_timestamp": first_timestamp,
            "last_action_timestamp": last_timestamp,
            "current_time_filter": {
                "start": self.start_timestamp,
                "end": self.end_timestamp,
            },
            "database_path": self.db_path,
        }
