"""Human-in-the-Loop (HITL) API endpoints for Phase 2.2.

This module provides REST API endpoints for:
- Escalation queue management
- Case review workflow
- Reviewer feedback collection
- Training module access
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path
from ..core.human_feedback import EscalationQueue, FeedbackTag, ReviewStatus
from .taxonomy_api import APIResponse


class HITLReviewAPI:
    """REST API for Human-in-the-Loop review system."""

    def __init__(self, storage_dir: str = "./hitl_data"):
        """Initialize HITL Review API.

        Args:
            storage_dir: Directory for HITL data storage
        """
        storage_path = Path(storage_dir) / "escalations.db"
        self.escalation_queue = EscalationQueue(storage_path=str(storage_path))

    def get_escalation_queue_endpoint(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """API endpoint: Get escalation queue.

        Args:
            status: Filter by status (pending, in_review, completed, deferred)
            priority: Filter by priority (low, medium, high, critical, emergency)
            limit: Maximum number of items

        Returns:
            Escalation queue
        """
        try:
            # Convert string status to enum if provided
            status_enum = None
            if status:
                try:
                    status_enum = ReviewStatus[status.upper()]
                except KeyError:
                    return APIResponse(
                        success=False,
                        error=f"Invalid status: {status}",
                        message="Failed to retrieve escalation queue",
                    ).to_dict()

            # Get queue items
            queue_items = [
                case.to_dict()
                for case in list(self.escalation_queue.pending_cases)[:limit]
            ]

            # Filter by status if provided
            if status_enum:
                queue_items = [
                    item
                    for item in queue_items
                    if item.get("status") == status_enum.value
                ]

            # Filter by priority if provided
            if priority:
                queue_items = [
                    item
                    for item in queue_items
                    if item.get("priority", "").lower() == priority.lower()
                ]

            return APIResponse(
                success=True,
                data={
                    "queue_items": queue_items,
                    "count": len(queue_items),
                    "filters": {"status": status, "priority": priority},
                },
                message="Escalation queue retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                message="Failed to retrieve escalation queue",
            ).to_dict()

    def get_case_details_endpoint(self, judgment_id: str) -> Dict[str, Any]:
        """API endpoint: Get detailed case information.

        Args:
            judgment_id: Judgment ID

        Returns:
            Case details
        """
        try:
            # Get escalation details
            case = self.escalation_queue.cases_by_id.get(judgment_id)

            if not case:
                return APIResponse(
                    success=False,
                    error="Case not found",
                    message="Failed to retrieve case details",
                ).to_dict()

            # Get any existing feedback from case
            feedback_history = [case.feedback.to_dict()] if case.feedback else []

            return APIResponse(
                success=True,
                data={
                    "case": case.to_dict(),
                    "feedback_history": feedback_history,
                    "has_feedback": len(feedback_history) > 0,
                },
                message="Case details retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve case details"
            ).to_dict()

    def submit_review_endpoint(
        self,
        judgment_id: str,
        reviewer_id: str,
        feedback_tags: List[str],
        rationale: str,
        corrected_decision: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """API endpoint: Submit case review.

        Args:
            judgment_id: Judgment ID
            reviewer_id: Reviewer ID
            feedback_tags: List of feedback tags
            rationale: Rationale for review
            corrected_decision: Corrected decision if applicable
            confidence: Confidence in review

        Returns:
            Submission response
        """
        try:
            # Convert string tags to enums
            tag_enums = []
            for tag_str in feedback_tags:
                try:
                    tag_enums.append(FeedbackTag[tag_str.upper()])
                except KeyError:
                    return APIResponse(
                        success=False,
                        error=f"Invalid feedback tag: {tag_str}",
                        message="Failed to submit review",
                    ).to_dict()

            # Submit feedback
            case = self.escalation_queue.cases_by_id.get(judgment_id)
            if not case:
                return APIResponse(
                    success=False,
                    error="Case not found",
                    message="Failed to submit review",
                ).to_dict()

            feedback = self.escalation_queue.submit_feedback(
                case_id=case.case_id,
                reviewer_id=reviewer_id,
                feedback_tags=tag_enums,
                rationale=rationale,
                corrected_decision=corrected_decision,
                confidence=confidence,
            )

            return APIResponse(
                success=True,
                data={
                    "feedback_id": feedback.feedback_id,
                    "judgment_id": judgment_id,
                    "submitted_at": feedback.reviewed_at.isoformat(),
                },
                message="Review submitted successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to submit review"
            ).to_dict()

    def get_reviewer_stats_endpoint(self, reviewer_id: str) -> Dict[str, Any]:
        """API endpoint: Get reviewer statistics.

        Args:
            reviewer_id: Reviewer ID

        Returns:
            Reviewer statistics
        """
        try:
            # Count cases assigned to reviewer
            assigned_cases = [
                case
                for case in self.escalation_queue.cases_by_id.values()
                if case.assigned_to == reviewer_id
            ]
            completed_cases = [
                case for case in assigned_cases if case.status == ReviewStatus.COMPLETED
            ]

            stats = {
                "reviewer_id": reviewer_id,
                "total_assigned": len(assigned_cases),
                "completed": len(completed_cases),
                "pending": len(
                    [c for c in assigned_cases if c.status == ReviewStatus.PENDING]
                ),
                "in_review": len(
                    [c for c in assigned_cases if c.status == ReviewStatus.IN_REVIEW]
                ),
            }

            return APIResponse(
                success=True,
                data=stats,
                message="Reviewer statistics retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                message="Failed to retrieve reviewer statistics",
            ).to_dict()

    def get_feedback_summary_endpoint(self, days: int = 30) -> Dict[str, Any]:
        """API endpoint: Get feedback summary.

        Args:
            days: Number of days to include

        Returns:
            Feedback summary
        """
        try:
            summary = self.escalation_queue.get_feedback_summary()

            return APIResponse(
                success=True,
                data=summary,
                message="Feedback summary retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                message="Failed to retrieve feedback summary",
            ).to_dict()

    def get_sla_metrics_endpoint(self) -> Dict[str, Any]:
        """API endpoint: Get SLA metrics.

        Returns:
            SLA metrics
        """
        try:
            metrics = self.escalation_queue.get_sla_metrics()

            return APIResponse(
                success=True,
                data=metrics.to_dict(),
                message="SLA metrics retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve SLA metrics"
            ).to_dict()

    def update_case_status_endpoint(
        self, judgment_id: str, status: str, reviewer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """API endpoint: Update case status.

        Note: This is a stub endpoint. Full implementation requires
        extending EscalationQueue with case status update methods.

        Args:
            judgment_id: Judgment ID
            status: New status (pending, in_review, completed, deferred)
            reviewer_id: Reviewer ID if claiming case

        Returns:
            Update response
        """
        try:
            # Convert status string to enum
            try:
                status_enum = ReviewStatus[status.upper()]
            except KeyError:
                return APIResponse(
                    success=False,
                    error=f"Invalid status: {status}",
                    message="Failed to update case status",
                ).to_dict()

            # Update case status in EscalationQueue
            success = self.escalation_queue.update_case_status(
                case_id=judgment_id, status=status_enum, reviewer_id=reviewer_id
            )

            if not success:
                return APIResponse(
                    success=False,
                    error=f"Case not found: {judgment_id}",
                    message="Failed to update case status",
                ).to_dict()

            return APIResponse(
                success=True,
                data={
                    "judgment_id": judgment_id,
                    "new_status": status_enum.value,
                    "reviewer_id": reviewer_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                message="Case status update validated (stub endpoint - full implementation pending)",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to update case status"
            ).to_dict()

    def get_training_cases_endpoint(
        self, category: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """API endpoint: Get training cases for reviewers.

        Args:
            category: Training category
            limit: Number of cases

        Returns:
            Training cases
        """
        try:
            # Get sample cases with high-quality feedback
            # This would pull from historical cases with consensus feedback
            training_cases = []

            # For now, return empty set - would need to implement in HumanFeedbackLoop

            return APIResponse(
                success=True,
                data={
                    "training_cases": training_cases,
                    "count": len(training_cases),
                    "category": category,
                },
                message="Training cases retrieved successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to retrieve training cases"
            ).to_dict()

    def batch_assign_cases_endpoint(
        self, reviewer_id: str, count: int = 10, priority: Optional[str] = None
    ) -> Dict[str, Any]:
        """API endpoint: Batch assign cases to reviewer.

        Args:
            reviewer_id: Reviewer ID
            count: Number of cases to assign
            priority: Priority filter

        Returns:
            Assigned cases
        """
        try:
            # Get pending cases
            queue_items = [
                case.to_dict()
                for case in list(self.escalation_queue.pending_cases)[: count * 2]
            ]

            # Filter by priority if provided
            if priority:
                queue_items = [
                    item
                    for item in queue_items
                    if item.get("priority", "").lower() == priority.lower()
                ]

            # Take requested count
            assigned_cases = queue_items[:count]

            # Update their status to in_review (would need implementation)

            return APIResponse(
                success=True,
                data={
                    "reviewer_id": reviewer_id,
                    "assigned_cases": assigned_cases,
                    "count": len(assigned_cases),
                },
                message=f"{len(assigned_cases)} cases assigned successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to assign cases"
            ).to_dict()
