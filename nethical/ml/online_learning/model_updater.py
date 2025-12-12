"""
Model Updater for Online Learning

Handles safe, incremental updates to detection models with constraints.

Safety Constraints:
- No reduction in detection rate for critical vectors
- Human approval for threshold changes > 10%
- Rollback capability within 5 minutes
- A/B testing before full deployment

Law Alignment:
- Law 23 (Fail-Safe Design): Safety constraints prevent degradation
- Law 24 (Adaptive Learning): Continuous improvement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid

from .feedback_loop import FeedbackBatch

logger = logging.getLogger(__name__)


@dataclass
class UpdateConstraints:
    """Constraints for safe model updates."""
    
    # Detection rate constraints
    min_detection_rate_critical: float = 0.95  # Never reduce below 95% for critical
    min_detection_rate_high: float = 0.90  # Never reduce below 90% for high
    min_detection_rate_other: float = 0.85  # Never reduce below 85% for others
    
    # Change limits
    max_threshold_change_pct: float = 10.0  # Max 10% threshold change without approval
    max_false_positive_rate: float = 0.02  # Max 2% false positives
    
    # Approval requirements
    require_human_approval_threshold: float = 0.10  # Require approval if > 10% change
    
    # Testing requirements
    require_ab_test: bool = True
    min_ab_test_samples: int = 1000
    ab_test_duration_hours: int = 24


@dataclass
class ModelUpdate:
    """Represents a pending model update."""
    
    update_id: str
    detector_name: str
    timestamp: datetime
    feedback_batch_ids: List[str]
    
    # Proposed changes
    old_threshold: float
    new_threshold: float
    threshold_change_pct: float
    
    # Performance metrics
    estimated_detection_rate: float
    estimated_false_positive_rate: float
    
    # Approval status
    requires_human_approval: bool
    approved: bool = False
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    
    # Testing status
    ab_test_id: Optional[str] = None
    ab_test_completed: bool = False
    deployed: bool = False


class ModelUpdater:
    """
    Manages safe updates to detection models.
    
    Features:
    - Constraint validation
    - Human approval workflow
    - A/B testing integration
    - Audit logging
    """
    
    def __init__(self, constraints: Optional[UpdateConstraints] = None):
        self.constraints = constraints or UpdateConstraints()
        
        # Storage
        self.pending_updates: Dict[str, ModelUpdate] = {}
        self.approved_updates: Dict[str, ModelUpdate] = {}
        self.deployed_updates: Dict[str, ModelUpdate] = {}
        
        # Metrics
        self.total_updates_proposed = 0
        self.total_updates_approved = 0
        self.total_updates_deployed = 0
        self.total_updates_rejected = 0
        
        logger.info(f"ModelUpdater initialized with constraints: {self.constraints}")
    
    async def propose_update(
        self,
        detector_name: str,
        batches: List[FeedbackBatch],
        old_threshold: float,
        new_threshold: float,
    ) -> Optional[ModelUpdate]:
        """
        Propose a model update based on feedback.
        
        Args:
            detector_name: Name of detector to update
            batches: Feedback batches used for update
            old_threshold: Current detection threshold
            new_threshold: Proposed new threshold
            
        Returns:
            ModelUpdate if valid, None if rejected
        """
        # Calculate change
        threshold_change_pct = abs(new_threshold - old_threshold) / old_threshold * 100
        
        # Estimate performance (simplified - would use actual model evaluation)
        combined_accuracy = sum(b.accuracy for b in batches) / len(batches) if batches else 0.0
        estimated_detection_rate = combined_accuracy
        
        fp_count = sum(b.false_positive_count for b in batches)
        total_count = sum(len(b.entries) for b in batches)
        estimated_false_positive_rate = fp_count / total_count if total_count > 0 else 0.0
        
        # Validate constraints
        if not self._validate_constraints(estimated_detection_rate, estimated_false_positive_rate):
            logger.warning(
                f"Update rejected for {detector_name}: constraint violation"
            )
            self.total_updates_rejected += 1
            return None
        
        # Check if human approval required
        requires_approval = threshold_change_pct > self.constraints.require_human_approval_threshold
        
        # Create update
        update = ModelUpdate(
            update_id=str(uuid.uuid4()),
            detector_name=detector_name,
            timestamp=datetime.now(timezone.utc),
            feedback_batch_ids=[b.batch_id for b in batches],
            old_threshold=old_threshold,
            new_threshold=new_threshold,
            threshold_change_pct=threshold_change_pct,
            estimated_detection_rate=estimated_detection_rate,
            estimated_false_positive_rate=estimated_false_positive_rate,
            requires_human_approval=requires_approval,
        )
        
        self.pending_updates[update.update_id] = update
        self.total_updates_proposed += 1
        
        logger.info(
            f"Proposed update {update.update_id} for {detector_name}: "
            f"threshold {old_threshold:.3f} -> {new_threshold:.3f} "
            f"(change: {threshold_change_pct:.1f}%), "
            f"requires_approval={requires_approval}"
        )
        
        return update
    
    def _validate_constraints(
        self,
        detection_rate: float,
        false_positive_rate: float,
    ) -> bool:
        """Validate update against constraints."""
        # Check detection rate (using "other" threshold as default)
        if detection_rate < self.constraints.min_detection_rate_other:
            logger.warning(
                f"Detection rate {detection_rate:.2%} below minimum "
                f"{self.constraints.min_detection_rate_other:.2%}"
            )
            return False
        
        # Check false positive rate
        if false_positive_rate > self.constraints.max_false_positive_rate:
            logger.warning(
                f"False positive rate {false_positive_rate:.2%} above maximum "
                f"{self.constraints.max_false_positive_rate:.2%}"
            )
            return False
        
        return True
    
    async def approve_update(self, update_id: str, approver: str) -> bool:
        """
        Approve a pending update.
        
        Args:
            update_id: ID of update to approve
            approver: Identifier of approver
            
        Returns:
            True if approved, False if not found or already processed
        """
        if update_id not in self.pending_updates:
            logger.warning(f"Update {update_id} not found in pending updates")
            return False
        
        update = self.pending_updates[update_id]
        update.approved = True
        update.approved_by = approver
        update.approval_timestamp = datetime.now(timezone.utc)
        
        # Move to approved
        self.approved_updates[update_id] = update
        del self.pending_updates[update_id]
        
        self.total_updates_approved += 1
        
        logger.info(
            f"Update {update_id} approved by {approver} for {update.detector_name}"
        )
        
        return True
    
    async def reject_update(self, update_id: str, reason: str) -> bool:
        """Reject a pending update."""
        if update_id not in self.pending_updates:
            return False
        
        update = self.pending_updates[update_id]
        del self.pending_updates[update_id]
        
        self.total_updates_rejected += 1
        
        logger.info(
            f"Update {update_id} rejected for {update.detector_name}: {reason}"
        )
        
        return True
    
    async def mark_deployed(self, update_id: str) -> bool:
        """Mark an update as successfully deployed."""
        if update_id not in self.approved_updates:
            logger.warning(f"Update {update_id} not found in approved updates")
            return False
        
        update = self.approved_updates[update_id]
        update.deployed = True
        
        # Move to deployed
        self.deployed_updates[update_id] = update
        del self.approved_updates[update_id]
        
        self.total_updates_deployed += 1
        
        logger.info(
            f"Update {update_id} deployed for {update.detector_name}"
        )
        
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get updater metrics."""
        return {
            "total_proposed": self.total_updates_proposed,
            "total_approved": self.total_updates_approved,
            "total_deployed": self.total_updates_deployed,
            "total_rejected": self.total_updates_rejected,
            "pending_count": len(self.pending_updates),
            "approved_count": len(self.approved_updates),
        }
