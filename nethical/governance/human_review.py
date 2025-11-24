"""
Human Review Queue and SLA Management

This module implements the human review queue system with SLA tracking,
feedback taxonomy coverage, and reviewer drift metrics.

Production Readiness Checklist - Section 9: Human Review
- Review queue SLA dashboard live
- Feedback taxonomy coverage report
- Reviewer drift metrics < 5%
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """Priority levels for review items"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(Enum):
    """Status of a review item"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class FeedbackCategory(Enum):
    """Taxonomy of feedback categories"""
    SECURITY_ISSUE = "security_issue"
    ETHICS_VIOLATION = "ethics_violation"
    QUALITY_CONCERN = "quality_concern"
    PERFORMANCE_ISSUE = "performance_issue"
    DOCUMENTATION_GAP = "documentation_gap"
    COMPATIBILITY_ISSUE = "compatibility_issue"
    USER_EXPERIENCE = "user_experience"
    FALSE_POSITIVE = "false_positive"
    OTHER = "other"


@dataclass
class ReviewItem:
    """A single item in the review queue"""
    item_id: str
    item_type: str  # "plugin", "action", "decision", etc.
    priority: ReviewPriority
    status: ReviewStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    sla_deadline: Optional[datetime] = None
    feedback_category: Optional[FeedbackCategory] = None
    feedback_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_overdue(self) -> bool:
        """Check if item is past SLA deadline"""
        if not self.sla_deadline:
            return False
        return datetime.now(timezone.utc) > self.sla_deadline
    
    def time_in_queue(self) -> timedelta:
        """Calculate time item has been in queue"""
        end_time = self.completed_at or datetime.now(timezone.utc)
        return end_time - self.created_at
    
    def time_to_complete(self) -> Optional[timedelta]:
        """Calculate time from creation to completion"""
        if not self.completed_at:
            return None
        return self.completed_at - self.created_at


@dataclass
class ReviewerStats:
    """Statistics for a reviewer"""
    reviewer_id: str
    total_reviews: int = 0
    completed_reviews: int = 0
    avg_review_time_seconds: float = 0.0
    feedback_distribution: Dict[str, int] = field(default_factory=dict)
    accuracy_score: float = 1.0  # 0-1 scale
    drift_score: float = 0.0  # 0-1 scale, 0 is no drift


@dataclass
class SLAMetrics:
    """SLA performance metrics"""
    total_items: int
    pending_items: int
    completed_items: int
    overdue_items: int
    within_sla: int
    sla_compliance_rate: float
    avg_review_time_seconds: float
    p50_review_time_seconds: float
    p95_review_time_seconds: float
    timestamp: datetime


class HumanReviewQueue:
    """
    Manages human review queue with SLA tracking and metrics.
    
    This system provides:
    - Review queue management with priority and SLA tracking
    - SLA dashboard metrics
    - Feedback taxonomy coverage reporting
    - Reviewer drift detection
    
    Example:
        >>> queue = HumanReviewQueue()
        >>> item = queue.add_item("plugin-123", "plugin", ReviewPriority.HIGH)
        >>> queue.assign_item(item.item_id, "reviewer-1")
        >>> queue.complete_item(item.item_id, FeedbackCategory.SECURITY_ISSUE, "Found XSS")
        >>> metrics = queue.get_sla_metrics()
    """
    
    def __init__(
        self,
        sla_hours: Dict[ReviewPriority, int] = None,
        drift_threshold: float = 0.05  # 5%
    ):
        """
        Initialize the review queue.
        
        Args:
            sla_hours: SLA hours for each priority level
            drift_threshold: Maximum acceptable reviewer drift (0.05 = 5%)
        """
        self.sla_hours = sla_hours or {
            ReviewPriority.CRITICAL: 4,
            ReviewPriority.HIGH: 24,
            ReviewPriority.MEDIUM: 72,
            ReviewPriority.LOW: 168
        }
        self.drift_threshold = drift_threshold
        
        # In-memory storage
        self._items: Dict[str, ReviewItem] = {}
        self._reviewer_items: Dict[str, List[str]] = defaultdict(list)
        self._completed_reviews: List[ReviewItem] = []
        
        logger.info(
            f"Human review queue initialized with SLA hours: {self.sla_hours}, "
            f"drift threshold: {drift_threshold * 100}%"
        )
    
    def add_item(
        self,
        item_id: str,
        item_type: str,
        priority: ReviewPriority,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReviewItem:
        """
        Add a new item to the review queue.
        
        Args:
            item_id: Unique item identifier
            item_type: Type of item (plugin, action, decision, etc.)
            priority: Review priority level
            metadata: Additional metadata
        
        Returns:
            Created ReviewItem
        """
        now = datetime.now(timezone.utc)
        sla_deadline = now + timedelta(hours=self.sla_hours[priority])
        
        item = ReviewItem(
            item_id=item_id,
            item_type=item_type,
            priority=priority,
            status=ReviewStatus.PENDING,
            created_at=now,
            sla_deadline=sla_deadline,
            metadata=metadata or {}
        )
        
        self._items[item_id] = item
        logger.info(
            f"Added review item {item_id} ({item_type}) with priority {priority.value}, "
            f"SLA deadline: {sla_deadline.isoformat()}"
        )
        
        return item
    
    def assign_item(self, item_id: str, reviewer_id: str) -> bool:
        """
        Assign a review item to a reviewer.
        
        Args:
            item_id: Item identifier
            reviewer_id: Reviewer identifier
        
        Returns:
            True if successfully assigned
        """
        if item_id not in self._items:
            logger.error(f"Item {item_id} not found")
            return False
        
        item = self._items[item_id]
        if item.status != ReviewStatus.PENDING:
            logger.warning(f"Item {item_id} is not pending (status: {item.status.value})")
            return False
        
        item.assigned_to = reviewer_id
        item.assigned_at = datetime.now(timezone.utc)
        item.status = ReviewStatus.IN_PROGRESS
        
        self._reviewer_items[reviewer_id].append(item_id)
        
        logger.info(f"Assigned item {item_id} to reviewer {reviewer_id}")
        return True
    
    def complete_item(
        self,
        item_id: str,
        feedback_category: FeedbackCategory,
        feedback_text: str
    ) -> bool:
        """
        Mark a review item as completed.
        
        Args:
            item_id: Item identifier
            feedback_category: Category of feedback
            feedback_text: Feedback text
        
        Returns:
            True if successfully completed
        """
        if item_id not in self._items:
            logger.error(f"Item {item_id} not found")
            return False
        
        item = self._items[item_id]
        if item.status == ReviewStatus.COMPLETED:
            logger.warning(f"Item {item_id} already completed")
            return False
        
        item.status = ReviewStatus.COMPLETED
        item.completed_at = datetime.now(timezone.utc)
        item.feedback_category = feedback_category
        item.feedback_text = feedback_text
        
        self._completed_reviews.append(item)
        
        time_taken = item.time_to_complete()
        logger.info(
            f"Completed review item {item_id}, category: {feedback_category.value}, "
            f"time taken: {time_taken.total_seconds() if time_taken else 0}s"
        )
        
        return True
    
    def get_sla_metrics(self) -> SLAMetrics:
        """
        Get current SLA performance metrics.
        
        Returns:
            SLAMetrics with dashboard data
        """
        total_items = len(self._items)
        pending_items = sum(
            1 for item in self._items.values()
            if item.status == ReviewStatus.PENDING
        )
        in_progress = sum(
            1 for item in self._items.values()
            if item.status == ReviewStatus.IN_PROGRESS
        )
        completed_items = len(self._completed_reviews)
        
        overdue_items = sum(
            1 for item in self._items.values()
            if item.is_overdue() and item.status != ReviewStatus.COMPLETED
        )
        
        within_sla = sum(
            1 for item in self._completed_reviews
            if item.completed_at and item.sla_deadline
            and item.completed_at <= item.sla_deadline
        )
        
        sla_compliance_rate = (
            (within_sla / completed_items * 100.0)
            if completed_items > 0
            else 100.0
        )
        
        # Calculate review time statistics
        review_times = [
            item.time_to_complete().total_seconds()
            for item in self._completed_reviews
            if item.time_to_complete()
        ]
        
        avg_review_time = sum(review_times) / len(review_times) if review_times else 0.0
        
        # Calculate percentiles
        sorted_times = sorted(review_times)
        p50_time = sorted_times[len(sorted_times) // 2] if sorted_times else 0.0
        p95_idx = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_idx] if sorted_times else 0.0
        
        return SLAMetrics(
            total_items=total_items,
            pending_items=pending_items,
            completed_items=completed_items,
            overdue_items=overdue_items,
            within_sla=within_sla,
            sla_compliance_rate=sla_compliance_rate,
            avg_review_time_seconds=avg_review_time,
            p50_review_time_seconds=p50_time,
            p95_review_time_seconds=p95_time,
            timestamp=datetime.now(timezone.utc)
        )
    
    def get_feedback_taxonomy_coverage(self) -> Dict[str, Any]:
        """
        Get feedback taxonomy coverage report.
        
        Returns:
            Dictionary with coverage statistics
        """
        if not self._completed_reviews:
            return {
                "total_reviews": 0,
                "coverage_by_category": {},
                "coverage_percentage": 0.0,
                "uncovered_categories": list(FeedbackCategory.__members__.keys())
            }
        
        # Count feedback by category
        category_counts = Counter(
            item.feedback_category.value
            for item in self._completed_reviews
            if item.feedback_category
        )
        
        # Calculate coverage
        total_categories = len(FeedbackCategory)
        covered_categories = len(category_counts)
        coverage_percentage = (covered_categories / total_categories) * 100.0
        
        # Identify uncovered categories
        covered = set(category_counts.keys())
        all_categories = set(cat.value for cat in FeedbackCategory)
        uncovered = all_categories - covered
        
        return {
            "total_reviews": len(self._completed_reviews),
            "total_categories": total_categories,
            "covered_categories": covered_categories,
            "coverage_percentage": coverage_percentage,
            "coverage_by_category": dict(category_counts),
            "uncovered_categories": list(uncovered)
        }
    
    def get_reviewer_stats(self, reviewer_id: str) -> ReviewerStats:
        """
        Get statistics for a specific reviewer.
        
        Args:
            reviewer_id: Reviewer identifier
        
        Returns:
            ReviewerStats for the reviewer
        """
        reviewer_item_ids = self._reviewer_items.get(reviewer_id, [])
        reviewer_items = [
            self._items[iid] for iid in reviewer_item_ids
            if iid in self._items
        ]
        
        completed = [
            item for item in reviewer_items
            if item.status == ReviewStatus.COMPLETED
        ]
        
        # Calculate average review time
        review_times = [
            item.time_to_complete().total_seconds()
            for item in completed
            if item.time_to_complete()
        ]
        avg_time = sum(review_times) / len(review_times) if review_times else 0.0
        
        # Feedback distribution
        feedback_dist = Counter(
            item.feedback_category.value
            for item in completed
            if item.feedback_category
        )
        
        # Calculate drift score (simplified version)
        drift_score = self._calculate_reviewer_drift(reviewer_id, completed)
        
        return ReviewerStats(
            reviewer_id=reviewer_id,
            total_reviews=len(reviewer_items),
            completed_reviews=len(completed),
            avg_review_time_seconds=avg_time,
            feedback_distribution=dict(feedback_dist),
            drift_score=drift_score
        )
    
    def _calculate_reviewer_drift(
        self,
        reviewer_id: str,
        completed_items: List[ReviewItem]
    ) -> float:
        """
        Calculate reviewer drift score.
        
        Drift is measured as the deviation from the average feedback distribution
        across all reviewers.
        
        Returns:
            Drift score (0-1), where 0 is no drift and 1 is maximum drift
        """
        if not completed_items:
            return 0.0
        
        # Get reviewer's feedback distribution
        reviewer_dist = Counter(
            item.feedback_category.value
            for item in completed_items
            if item.feedback_category
        )
        
        # Get overall distribution
        overall_dist = Counter(
            item.feedback_category.value
            for item in self._completed_reviews
            if item.feedback_category
        )
        
        if not overall_dist:
            return 0.0
        
        # Calculate normalized distributions
        reviewer_total = sum(reviewer_dist.values())
        overall_total = sum(overall_dist.values())
        
        if reviewer_total == 0:
            return 0.0
        
        # Calculate drift as sum of absolute differences in proportions
        drift = 0.0
        all_categories = set(reviewer_dist.keys()) | set(overall_dist.keys())
        
        for category in all_categories:
            reviewer_prop = reviewer_dist.get(category, 0) / reviewer_total
            overall_prop = overall_dist.get(category, 0) / overall_total
            drift += abs(reviewer_prop - overall_prop)
        
        # Normalize drift (max possible drift is 2.0)
        return min(drift / 2.0, 1.0)
    
    def get_drift_report(self) -> Dict[str, Any]:
        """
        Get drift report for all reviewers.
        
        Returns:
            Dictionary with drift metrics for all reviewers
        """
        reviewers = list(self._reviewer_items.keys())
        
        drift_stats = {}
        high_drift_reviewers = []
        
        for reviewer_id in reviewers:
            stats = self.get_reviewer_stats(reviewer_id)
            drift_stats[reviewer_id] = {
                "drift_score": stats.drift_score,
                "drift_percentage": stats.drift_score * 100.0,
                "completed_reviews": stats.completed_reviews,
                "exceeds_threshold": stats.drift_score > self.drift_threshold
            }
            
            if stats.drift_score > self.drift_threshold:
                high_drift_reviewers.append(reviewer_id)
        
        max_drift = max(
            (s["drift_score"] for s in drift_stats.values()),
            default=0.0
        )
        avg_drift = (
            sum(s["drift_score"] for s in drift_stats.values()) / len(drift_stats)
            if drift_stats else 0.0
        )
        
        return {
            "total_reviewers": len(reviewers),
            "max_drift_score": max_drift,
            "max_drift_percentage": max_drift * 100.0,
            "avg_drift_score": avg_drift,
            "avg_drift_percentage": avg_drift * 100.0,
            "drift_threshold": self.drift_threshold,
            "drift_threshold_percentage": self.drift_threshold * 100.0,
            "high_drift_reviewers": high_drift_reviewers,
            "high_drift_count": len(high_drift_reviewers),
            "reviewer_drift_stats": drift_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data including SLA metrics, taxonomy coverage,
        and drift metrics.
        
        Returns:
            Complete dashboard data dictionary
        """
        sla_metrics = self.get_sla_metrics()
        taxonomy_coverage = self.get_feedback_taxonomy_coverage()
        drift_report = self.get_drift_report()
        
        return {
            "sla_metrics": {
                "total_items": sla_metrics.total_items,
                "pending_items": sla_metrics.pending_items,
                "completed_items": sla_metrics.completed_items,
                "overdue_items": sla_metrics.overdue_items,
                "within_sla": sla_metrics.within_sla,
                "sla_compliance_rate": sla_metrics.sla_compliance_rate,
                "avg_review_time_seconds": sla_metrics.avg_review_time_seconds,
                "p50_review_time_seconds": sla_metrics.p50_review_time_seconds,
                "p95_review_time_seconds": sla_metrics.p95_review_time_seconds
            },
            "taxonomy_coverage": taxonomy_coverage,
            "drift_metrics": drift_report,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
