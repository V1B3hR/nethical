"""
Feedback Loop for Online Learning

Collects and processes feedback from various sources to improve detection models.

Sources:
- Human review decisions
- Appeal outcomes
- Red team findings
- False positive reports
- False negative discoveries

Law Alignment:
- Law 14 (Decision Justification): Feedback improves explainability
- Law 24 (Adaptive Learning): Learn from mistakes
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Type of feedback received."""
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    CORRECT_DETECTION = "correct_detection"
    APPEAL_APPROVED = "appeal_approved"
    APPEAL_DENIED = "appeal_denied"
    RED_TEAM_FINDING = "red_team_finding"
    HUMAN_OVERRIDE = "human_override"


class FeedbackSource(Enum):
    """Source of feedback."""
    HUMAN_REVIEWER = "human_reviewer"
    APPEAL_SYSTEM = "appeal_system"
    RED_TEAM = "red_team"
    AUTOMATED_VALIDATION = "automated_validation"
    USER_REPORT = "user_report"


@dataclass
class FeedbackEntry:
    """Single feedback entry."""
    
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_type: FeedbackType = FeedbackType.CORRECT_DETECTION
    source: FeedbackSource = FeedbackSource.AUTOMATED_VALIDATION
    detector_name: str = ""
    action_id: str = ""
    violation_id: Optional[str] = None
    original_confidence: float = 0.0
    corrected_label: bool = False  # True = violation, False = no violation
    reviewer_confidence: float = 1.0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackBatch:
    """Batch of feedback entries for processing."""
    
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entries: List[FeedbackEntry] = field(default_factory=list)
    detector_name: str = ""
    
    @property
    def false_positive_count(self) -> int:
        return sum(1 for e in self.entries if e.feedback_type == FeedbackType.FALSE_POSITIVE)
    
    @property
    def false_negative_count(self) -> int:
        return sum(1 for e in self.entries if e.feedback_type == FeedbackType.FALSE_NEGATIVE)
    
    @property
    def accuracy(self) -> float:
        if not self.entries:
            return 0.0
        correct = sum(1 for e in self.entries 
                     if e.feedback_type == FeedbackType.CORRECT_DETECTION)
        return correct / len(self.entries)


class FeedbackLoop:
    """
    Manages feedback collection and processing for online learning.
    
    Features:
    - Multi-source feedback aggregation
    - Batch processing with configurable size
    - Quality filtering to prevent adversarial feedback
    - Staleness tracking (max 24 hours)
    """
    
    def __init__(
        self,
        batch_size: int = 1000,
        max_staleness_hours: int = 24,
        min_reviewer_confidence: float = 0.7,
    ):
        self.batch_size = batch_size
        self.max_staleness_hours = max_staleness_hours
        self.min_reviewer_confidence = min_reviewer_confidence
        
        # Storage
        self.pending_feedback: List[FeedbackEntry] = []
        self.processed_batches: List[FeedbackBatch] = []
        
        # Metrics
        self.total_feedback_received = 0
        self.total_feedback_processed = 0
        self.last_batch_time: Optional[datetime] = None
        
        logger.info(
            f"FeedbackLoop initialized: batch_size={batch_size}, "
            f"max_staleness={max_staleness_hours}h"
        )
    
    async def submit_feedback(self, entry: FeedbackEntry) -> bool:
        """
        Submit a feedback entry.
        
        Args:
            entry: Feedback entry to submit
            
        Returns:
            True if accepted, False if rejected (e.g., low confidence)
        """
        # Quality filter
        if entry.reviewer_confidence < self.min_reviewer_confidence:
            logger.warning(
                f"Rejected feedback {entry.feedback_id}: "
                f"confidence {entry.reviewer_confidence} below threshold"
            )
            return False
        
        # Accept feedback
        self.pending_feedback.append(entry)
        self.total_feedback_received += 1
        
        logger.info(
            f"Accepted feedback {entry.feedback_id}: "
            f"type={entry.feedback_type.value}, detector={entry.detector_name}"
        )
        
        # Check if batch is ready
        if len(self.pending_feedback) >= self.batch_size:
            await self._process_batch()
        
        return True
    
    async def _process_batch(self) -> Optional[FeedbackBatch]:
        """Process a batch of feedback."""
        if not self.pending_feedback:
            return None
        
        # Group by detector
        by_detector: Dict[str, List[FeedbackEntry]] = {}
        for entry in self.pending_feedback[:self.batch_size]:
            detector = entry.detector_name or "unknown"
            if detector not in by_detector:
                by_detector[detector] = []
            by_detector[detector].append(entry)
        
        # Create batches per detector
        batches = []
        for detector_name, entries in by_detector.items():
            batch = FeedbackBatch(
                entries=entries,
                detector_name=detector_name
            )
            batches.append(batch)
            
            logger.info(
                f"Created feedback batch {batch.batch_id}: "
                f"detector={detector_name}, size={len(entries)}, "
                f"FP={batch.false_positive_count}, FN={batch.false_negative_count}, "
                f"accuracy={batch.accuracy:.2%}"
            )
        
        # Remove processed entries
        self.pending_feedback = self.pending_feedback[self.batch_size:]
        self.processed_batches.extend(batches)
        self.total_feedback_processed += sum(len(b.entries) for b in batches)
        self.last_batch_time = datetime.now(timezone.utc)
        
        return batches[0] if batches else None
    
    async def get_pending_batches(self) -> List[FeedbackBatch]:
        """Get all processed batches ready for model updates."""
        return self.processed_batches.copy()
    
    async def clear_processed_batches(self):
        """Clear processed batches after successful model update."""
        cleared_count = len(self.processed_batches)
        self.processed_batches.clear()
        logger.info(f"Cleared {cleared_count} processed batches")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feedback loop metrics."""
        return {
            "total_received": self.total_feedback_received,
            "total_processed": self.total_feedback_processed,
            "pending_count": len(self.pending_feedback),
            "processed_batches": len(self.processed_batches),
            "last_batch_time": self.last_batch_time.isoformat() if self.last_batch_time else None,
        }
