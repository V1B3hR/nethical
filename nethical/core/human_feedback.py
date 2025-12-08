"""Phase 8: Human-in-the-Loop Operations

This module implements:
- Escalation queue with labeling interface
- Structured feedback tags (false_positive, missed_violation, policy_gap)
- Human review workflow for uncertain or critical decisions
- Structured feedback collection for model and rule improvement
- Median triage SLA tracking and optimization

Design:
- Builds on existing escalation_queue in SafetyJudge (governance.py)
- Provides structured feedback collection and storage
- Tracks response times and SLA metrics
- Enables continuous improvement feedback loop
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import statistics


class FeedbackTag(Enum):
    """Structured feedback tags for human review."""

    FALSE_POSITIVE = "false_positive"
    MISSED_VIOLATION = "missed_violation"
    POLICY_GAP = "policy_gap"
    CORRECT_DECISION = "correct_decision"
    NEEDS_CLARIFICATION = "needs_clarification"
    EDGE_CASE = "edge_case"


class ReviewStatus(Enum):
    """Status of escalated case review."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    DEFERRED = "deferred"


class ReviewPriority(Enum):
    """Priority levels for escalation review."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class HumanFeedback:
    """Human feedback on an escalated decision."""

    feedback_id: str
    judgment_id: str
    reviewer_id: str
    feedback_tags: List[FeedbackTag]
    rationale: str
    corrected_decision: Optional[str] = None
    confidence: float = 1.0
    reviewed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "judgment_id": self.judgment_id,
            "reviewer_id": self.reviewer_id,
            "feedback_tags": [tag.value for tag in self.feedback_tags],
            "rationale": self.rationale,
            "corrected_decision": self.corrected_decision,
            "confidence": self.confidence,
            "reviewed_at": self.reviewed_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanFeedback":
        """Create from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            judgment_id=data["judgment_id"],
            reviewer_id=data["reviewer_id"],
            feedback_tags=[FeedbackTag(tag) for tag in data["feedback_tags"]],
            rationale=data["rationale"],
            corrected_decision=data.get("corrected_decision"),
            confidence=data.get("confidence", 1.0),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EscalationCase:
    """Case in the escalation queue awaiting review."""

    case_id: str
    judgment_id: str
    action_id: str
    agent_id: str
    priority: ReviewPriority
    status: ReviewStatus
    escalated_at: datetime
    decision: str
    confidence: float
    violations: List[Dict[str, Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    started_review_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    feedback: Optional[HumanFeedback] = None

    def triage_time_seconds(self) -> Optional[float]:
        """Calculate time from escalation to review start."""
        if self.started_review_at:
            return (self.started_review_at - self.escalated_at).total_seconds()
        return None

    def resolution_time_seconds(self) -> Optional[float]:
        """Calculate time from escalation to completion."""
        if self.completed_at:
            return (self.completed_at - self.escalated_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "case_id": self.case_id,
            "judgment_id": self.judgment_id,
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "escalated_at": self.escalated_at.isoformat(),
            "decision": self.decision,
            "confidence": self.confidence,
            "violations": self.violations,
            "context": self.context,
            "assigned_to": self.assigned_to,
            "started_review_at": (
                self.started_review_at.isoformat() if self.started_review_at else None
            ),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "feedback": self.feedback.to_dict() if self.feedback else None,
        }


@dataclass
class SLAMetrics:
    """SLA tracking metrics."""

    median_triage_time_seconds: float
    p95_triage_time_seconds: float
    median_resolution_time_seconds: float
    p95_resolution_time_seconds: float
    total_cases: int
    pending_cases: int
    completed_cases: int
    sla_breaches: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "median_triage_time_seconds": self.median_triage_time_seconds,
            "p95_triage_time_seconds": self.p95_triage_time_seconds,
            "median_resolution_time_seconds": self.median_resolution_time_seconds,
            "p95_resolution_time_seconds": self.p95_resolution_time_seconds,
            "total_cases": self.total_cases,
            "pending_cases": self.pending_cases,
            "completed_cases": self.completed_cases,
            "sla_breaches": self.sla_breaches,
        }


class EscalationQueue:
    """Manages escalation queue with labeling interface and SLA tracking.

    Features:
    - Priority-based queue management
    - SLA tracking for triage and resolution times
    - Human feedback collection
    - Persistence to SQLite database
    """

    def __init__(
        self,
        storage_path: str = "./data/escalations.db",
        triage_sla_seconds: float = 3600,  # 1 hour default
        resolution_sla_seconds: float = 86400,  # 24 hours default
    ):
        """Initialize escalation queue.

        Args:
            storage_path: Path to SQLite database for persistence
            triage_sla_seconds: SLA target for starting review (seconds)
            resolution_sla_seconds: SLA target for completing review (seconds)
        """
        self.storage_path = Path(storage_path)
        self.triage_sla_seconds = triage_sla_seconds
        self.resolution_sla_seconds = resolution_sla_seconds

        # In-memory queue for fast access
        self.pending_cases: deque = deque()
        self.cases_by_id: Dict[str, EscalationCase] = {}

        # Metrics tracking
        self.triage_times: List[float] = []
        self.resolution_times: List[float] = []
        self.sla_breaches: int = 0

        # Initialize storage
        self._init_storage()
        self._load_pending_cases()

    def _init_storage(self) -> None:
        """Initialize SQLite storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Cases table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS escalation_cases (
                case_id TEXT PRIMARY KEY,
                judgment_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL,
                escalated_at TEXT NOT NULL,
                decision TEXT NOT NULL,
                confidence REAL NOT NULL,
                violations TEXT NOT NULL,
                context TEXT,
                assigned_to TEXT,
                started_review_at TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS human_feedback (
                feedback_id TEXT PRIMARY KEY,
                judgment_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                feedback_tags TEXT NOT NULL,
                rationale TEXT NOT NULL,
                corrected_decision TEXT,
                confidence REAL NOT NULL,
                reviewed_at TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (judgment_id) REFERENCES escalation_cases(judgment_id)
            )
        """
        )

        # Indices for performance
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_status ON escalation_cases(status)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_priority ON escalation_cases(priority)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_escalated_at ON escalation_cases(escalated_at)
        """
        )

        conn.commit()
        conn.close()

    def _load_pending_cases(self) -> None:
        """Load pending cases from storage."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT case_id, judgment_id, action_id, agent_id, priority, status,
                   escalated_at, decision, confidence, violations, context,
                   assigned_to, started_review_at, completed_at
            FROM escalation_cases
            WHERE status IN ('pending', 'in_review')
            ORDER BY priority DESC, escalated_at ASC
        """
        )

        for row in cursor.fetchall():
            case = EscalationCase(
                case_id=row[0],
                judgment_id=row[1],
                action_id=row[2],
                agent_id=row[3],
                priority=ReviewPriority(row[4]),
                status=ReviewStatus(row[5]),
                escalated_at=datetime.fromisoformat(row[6]),
                decision=row[7],
                confidence=row[8],
                violations=json.loads(row[9]),
                context=json.loads(row[10]) if row[10] else {},
                assigned_to=row[11],
                started_review_at=datetime.fromisoformat(row[12]) if row[12] else None,
                completed_at=datetime.fromisoformat(row[13]) if row[13] else None,
            )

            self.cases_by_id[case.case_id] = case
            if case.status == ReviewStatus.PENDING:
                self.pending_cases.append(case)

        conn.close()

    def add_case(
        self,
        judgment_id: str,
        action_id: str,
        agent_id: str,
        decision: str,
        confidence: float,
        violations: List[Dict[str, Any]],
        priority: ReviewPriority = ReviewPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ) -> EscalationCase:
        """Add a case to the escalation queue.

        Args:
            judgment_id: ID of the judgment being escalated
            action_id: ID of the action
            agent_id: ID of the agent
            decision: Decision that was made
            confidence: Confidence in the decision
            violations: List of violations detected
            priority: Priority level for review
            context: Additional context

        Returns:
            Created escalation case
        """
        import uuid

        case = EscalationCase(
            case_id=f"esc_{uuid.uuid4().hex[:12]}",
            judgment_id=judgment_id,
            action_id=action_id,
            agent_id=agent_id,
            priority=priority,
            status=ReviewStatus.PENDING,
            escalated_at=datetime.now(),
            decision=decision,
            confidence=confidence,
            violations=violations,
            context=context or {},
        )

        # Store in database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO escalation_cases
            (case_id, judgment_id, action_id, agent_id, priority, status,
             escalated_at, decision, confidence, violations, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                case.case_id,
                case.judgment_id,
                case.action_id,
                case.agent_id,
                case.priority.value,
                case.status.value,
                case.escalated_at.isoformat(),
                case.decision,
                case.confidence,
                json.dumps(case.violations),
                json.dumps(case.context),
            ),
        )

        conn.commit()
        conn.close()

        # Add to in-memory structures
        self.cases_by_id[case.case_id] = case
        self.pending_cases.append(case)

        return case

    def get_next_case(self, reviewer_id: str) -> Optional[EscalationCase]:
        """Get next case from queue for review.

        Prioritizes by:
        1. Priority level (highest first)
        2. Escalation time (oldest first)

        Args:
            reviewer_id: ID of the reviewer

        Returns:
            Next case to review, or None if queue is empty
        """
        if not self.pending_cases:
            return None

        # Get highest priority case
        case = self.pending_cases.popleft()
        case.status = ReviewStatus.IN_REVIEW
        case.assigned_to = reviewer_id
        case.started_review_at = datetime.now()

        # Track triage time
        triage_time = case.triage_time_seconds()
        if triage_time:
            self.triage_times.append(triage_time)
            if triage_time > self.triage_sla_seconds:
                self.sla_breaches += 1

        # Update database
        self._update_case(case)

        return case

    def submit_feedback(
        self,
        case_id: str,
        reviewer_id: str,
        feedback_tags: List[FeedbackTag],
        rationale: str,
        corrected_decision: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanFeedback:
        """Submit feedback for a case.

        Args:
            case_id: ID of the case
            reviewer_id: ID of the reviewer
            feedback_tags: Structured feedback tags
            rationale: Human explanation
            corrected_decision: Corrected decision if different
            confidence: Confidence in feedback
            metadata: Additional metadata

        Returns:
            Created feedback object
        """
        import uuid

        case = self.cases_by_id.get(case_id)
        if not case:
            raise ValueError(f"Case {case_id} not found")

        feedback = HumanFeedback(
            feedback_id=f"fb_{uuid.uuid4().hex[:12]}",
            judgment_id=case.judgment_id,
            reviewer_id=reviewer_id,
            feedback_tags=feedback_tags,
            rationale=rationale,
            corrected_decision=corrected_decision,
            confidence=confidence,
            reviewed_at=datetime.now(),
            metadata=metadata or {},
        )

        # Update case
        case.feedback = feedback
        case.status = ReviewStatus.COMPLETED
        case.completed_at = datetime.now()

        # Track resolution time
        resolution_time = case.resolution_time_seconds()
        if resolution_time:
            self.resolution_times.append(resolution_time)
            if resolution_time > self.resolution_sla_seconds:
                self.sla_breaches += 1

        # Store feedback
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO human_feedback
            (feedback_id, judgment_id, reviewer_id, feedback_tags, rationale,
             corrected_decision, confidence, reviewed_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                feedback.feedback_id,
                feedback.judgment_id,
                feedback.reviewer_id,
                json.dumps([tag.value for tag in feedback.feedback_tags]),
                feedback.rationale,
                feedback.corrected_decision,
                feedback.confidence,
                feedback.reviewed_at.isoformat(),
                json.dumps(feedback.metadata),
            ),
        )

        conn.commit()
        conn.close()

        # Update case in database
        self._update_case(case)

        return feedback

    def _update_case(self, case: EscalationCase) -> None:
        """Update case in database."""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE escalation_cases
            SET status = ?,
                assigned_to = ?,
                started_review_at = ?,
                completed_at = ?
            WHERE case_id = ?
        """,
            (
                case.status.value,
                case.assigned_to,
                case.started_review_at.isoformat() if case.started_review_at else None,
                case.completed_at.isoformat() if case.completed_at else None,
                case.case_id,
            ),
        )

        conn.commit()
        conn.close()

    def get_sla_metrics(self) -> SLAMetrics:
        """Calculate SLA metrics.

        Returns:
            SLA metrics object
        """

        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(sorted_data):
                return sorted_data[-1]
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        median_triage = statistics.median(self.triage_times) if self.triage_times else 0.0
        p95_triage = percentile(self.triage_times, 0.95) if self.triage_times else 0.0
        median_resolution = (
            statistics.median(self.resolution_times) if self.resolution_times else 0.0
        )
        p95_resolution = percentile(self.resolution_times, 0.95) if self.resolution_times else 0.0

        # Count cases by status
        pending = sum(1 for c in self.cases_by_id.values() if c.status == ReviewStatus.PENDING)
        completed = sum(1 for c in self.cases_by_id.values() if c.status == ReviewStatus.COMPLETED)

        return SLAMetrics(
            median_triage_time_seconds=median_triage,
            p95_triage_time_seconds=p95_triage,
            median_resolution_time_seconds=median_resolution,
            p95_resolution_time_seconds=p95_resolution,
            total_cases=len(self.cases_by_id),
            pending_cases=pending,
            completed_cases=completed,
            sla_breaches=self.sla_breaches,
        )

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of human feedback for continuous improvement.

        Returns:
            Summary dictionary with:
            - Tag counts
            - Correction rate
            - False positive rate
            - Missed violation rate
            - Policy gap rate
        """
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute("SELECT feedback_tags, corrected_decision FROM human_feedback")

        tag_counts = defaultdict(int)
        total_feedback = 0
        corrections = 0

        for row in cursor.fetchall():
            tags = json.loads(row[0])
            total_feedback += 1

            for tag in tags:
                tag_counts[tag] += 1

            if row[1]:  # corrected_decision
                corrections += 1

        conn.close()

        return {
            "total_feedback": total_feedback,
            "tag_counts": dict(tag_counts),
            "correction_rate": corrections / total_feedback if total_feedback > 0 else 0.0,
            "false_positive_rate": (
                tag_counts.get(FeedbackTag.FALSE_POSITIVE.value, 0) / total_feedback
                if total_feedback > 0
                else 0.0
            ),
            "missed_violation_rate": (
                tag_counts.get(FeedbackTag.MISSED_VIOLATION.value, 0) / total_feedback
                if total_feedback > 0
                else 0.0
            ),
            "policy_gap_rate": (
                tag_counts.get(FeedbackTag.POLICY_GAP.value, 0) / total_feedback
                if total_feedback > 0
                else 0.0
            ),
        }

    def list_pending_cases(self, limit: int = 10) -> List[EscalationCase]:
        """List pending cases.

        Args:
            limit: Maximum number of cases to return

        Returns:
            List of pending cases
        """
        return list(self.pending_cases)[:limit]

    def get_case(self, case_id: str) -> Optional[EscalationCase]:
        """Get case by ID.

        Args:
            case_id: Case ID

        Returns:
            Case if found, None otherwise
        """
        return self.cases_by_id.get(case_id)

    def update_case_status(
        self, case_id: str, status: ReviewStatus, reviewer_id: Optional[str] = None
    ) -> bool:
        """Update case status.

        Args:
            case_id: Case ID
            status: New status
            reviewer_id: Optional reviewer ID

        Returns:
            True if update successful, False otherwise
        """
        case = self.get_case(case_id)
        if not case:
            return False

        case.status = status
        if reviewer_id and not case.assigned_to:
            case.assigned_to = reviewer_id

        if status == ReviewStatus.IN_REVIEW and not case.started_review_at:
            case.started_review_at = datetime.now(timezone.utc)
        elif status in [ReviewStatus.COMPLETED, ReviewStatus.DEFERRED] and not case.completed_at:
            case.completed_at = datetime.now(timezone.utc)

        self._update_case(case)
        return True


def _demo():
    """Demonstrate the human feedback system functionality."""
    import tempfile
    import shutil
    from pathlib import Path

    print("=" * 70)
    print("Human Feedback System Demo")
    print("=" * 70)

    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    try:
        storage_path = str(Path(temp_dir) / "escalations.db")

        # Initialize escalation queue
        print("\n1. Initializing Escalation Queue...")
        queue = EscalationQueue(
            storage_path=storage_path,
            triage_sla_seconds=3600,  # 1 hour
            resolution_sla_seconds=86400,  # 24 hours
        )
        print(f"   ✓ Queue initialized with storage at: {storage_path}")
        print(f"   - Triage SLA: {queue.triage_sla_seconds}s (1 hour)")
        print(f"   - Resolution SLA: {queue.resolution_sla_seconds}s (24 hours)")

        # Add test cases
        print("\n2. Adding Cases to Escalation Queue...")

        test_cases = [
            {
                "judgment_id": "judg_001",
                "action_id": "act_001",
                "agent_id": "agent_alpha",
                "decision": "block",
                "confidence": 0.65,
                "violations": [
                    {"type": "safety", "severity": 4, "description": "Potential unsafe content"}
                ],
                "priority": ReviewPriority.HIGH,
                "context": {"source": "user_action", "timestamp": "2024-01-01T10:00:00Z"},
            },
            {
                "judgment_id": "judg_002",
                "action_id": "act_002",
                "agent_id": "agent_beta",
                "decision": "allow",
                "confidence": 0.55,
                "violations": [
                    {"type": "privacy", "severity": 3, "description": "Privacy concern"}
                ],
                "priority": ReviewPriority.MEDIUM,
                "context": {"source": "automated_scan", "timestamp": "2024-01-01T10:15:00Z"},
            },
            {
                "judgment_id": "judg_003",
                "action_id": "act_003",
                "agent_id": "agent_gamma",
                "decision": "terminate",
                "confidence": 0.45,
                "violations": [
                    {
                        "type": "security",
                        "severity": 5,
                        "description": "Critical security violation",
                    }
                ],
                "priority": ReviewPriority.EMERGENCY,
                "context": {"source": "security_monitor", "timestamp": "2024-01-01T10:30:00Z"},
            },
        ]

        for i, case_data in enumerate(test_cases, 1):
            case = queue.add_case(**case_data)
            print(f"\n   Case {i} added:")
            print(f"   - Case ID: {case.case_id}")
            print(f"   - Decision: {case.decision} (confidence: {case.confidence})")
            print(f"   - Priority: {case.priority.name}")
            print(f"   - Status: {case.status.value}")
            print(f"   - Violations: {len(case.violations)}")

        print(f"\n   ✓ Total cases in queue: {len(queue.pending_cases)}")

        # Review workflow
        print("\n3. Human Review Workflow...")
        reviewer_id = "reviewer_alice"
        reviewed_cases = []

        while len(reviewed_cases) < 3:
            case = queue.get_next_case(reviewer_id=reviewer_id)
            if not case:
                break

            reviewed_cases.append(case)
            print(f"\n   Reviewing Case #{len(reviewed_cases)}:")
            print(f"   - Case ID: {case.case_id}")
            print(f"   - Agent: {case.agent_id}")
            print(f"   - Decision: {case.decision} (confidence: {case.confidence})")
            print(f"   - Priority: {case.priority.name}")
            print(f"   - Status: {case.status.value}")
            print(f"   - Assigned to: {case.assigned_to}")

            # Submit different types of feedback
            if len(reviewed_cases) == 1:
                # False positive case
                feedback = queue.submit_feedback(
                    case_id=case.case_id,
                    reviewer_id=reviewer_id,
                    feedback_tags=[FeedbackTag.FALSE_POSITIVE],
                    rationale="Content was actually safe, detector was too aggressive",
                    corrected_decision="allow",
                    confidence=0.9,
                    metadata={"review_notes": "Edge case that needs policy clarification"},
                )
                print(f"   - Feedback: FALSE_POSITIVE")
                print(f"   - Rationale: {feedback.rationale}")
                print(f"   - Corrected: {feedback.corrected_decision}")

            elif len(reviewed_cases) == 2:
                # Correct decision
                feedback = queue.submit_feedback(
                    case_id=case.case_id,
                    reviewer_id=reviewer_id,
                    feedback_tags=[FeedbackTag.CORRECT_DECISION],
                    rationale="Decision was appropriate given the context",
                    confidence=0.95,
                )
                print(f"   - Feedback: CORRECT_DECISION")
                print(f"   - Rationale: {feedback.rationale}")

            else:
                # Policy gap
                feedback = queue.submit_feedback(
                    case_id=case.case_id,
                    reviewer_id=reviewer_id,
                    feedback_tags=[FeedbackTag.POLICY_GAP, FeedbackTag.EDGE_CASE],
                    rationale="Policy doesn't cover this scenario - needs update",
                    corrected_decision="escalate_further",
                    confidence=0.85,
                    metadata={"suggested_policy": "Add explicit handling for security violations"},
                )
                print(f"   - Feedback: POLICY_GAP, EDGE_CASE")
                print(f"   - Rationale: {feedback.rationale}")
                print(f"   - Corrected: {feedback.corrected_decision}")

        # SLA Metrics
        print("\n4. SLA Metrics...")
        metrics = queue.get_sla_metrics()
        print(f"   Total Cases: {metrics.total_cases}")
        print(f"   Pending Cases: {metrics.pending_cases}")
        print(f"   Completed Cases: {metrics.completed_cases}")
        print(f"   Median Triage Time: {metrics.median_triage_time_seconds:.2f}s")
        print(f"   Median Resolution Time: {metrics.median_resolution_time_seconds:.2f}s")
        print(f"   P95 Triage Time: {metrics.p95_triage_time_seconds:.2f}s")
        print(f"   P95 Resolution Time: {metrics.p95_resolution_time_seconds:.2f}s")
        print(f"   SLA Breaches: {metrics.sla_breaches}")

        # Feedback Summary
        print("\n5. Feedback Summary (for Continuous Improvement)...")
        summary = queue.get_feedback_summary()
        print(f"   Total Feedback: {summary['total_feedback']}")
        print(f"   Correction Rate: {summary['correction_rate']:.1%}")
        print(f"   False Positive Rate: {summary['false_positive_rate']:.1%}")
        print(f"   Missed Violation Rate: {summary['missed_violation_rate']:.1%}")
        print(f"   Policy Gap Rate: {summary['policy_gap_rate']:.1%}")
        print(f"   Tag Counts: {summary['tag_counts']}")

        # List pending cases
        print("\n6. Pending Cases...")
        pending = queue.list_pending_cases(limit=5)
        print(f"   Remaining pending cases: {len(pending)}")
        for case in pending:
            print(f"   - {case.case_id}: {case.priority.name} priority, {case.decision}")

        print("\n" + "=" * 70)
        print("Demo Complete!")
        print("=" * 70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Escalation queue initialization with SLA tracking")
        print("  ✓ Adding cases with different priorities")
        print("  ✓ Priority-based case retrieval")
        print("  ✓ Human review workflow with feedback submission")
        print("  ✓ Structured feedback tags (FALSE_POSITIVE, CORRECT_DECISION, POLICY_GAP)")
        print("  ✓ SLA metrics calculation")
        print("  ✓ Feedback summary for continuous improvement")
        print()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    _demo()
