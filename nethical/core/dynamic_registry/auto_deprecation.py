"""
Auto-Deprecation System for Attack Vectors

This module automatically identifies and deprecates unused attack vectors
that are no longer relevant or detecting threats.

Process:
1. Flag for review (zero detections for 90 days + no known variants)
2. Human confirmation
3. Move to archive (not delete)

Alignment: Law 24 (Adaptive Learning), Law 15 (Audit Compliance)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class DeprecationReason(str, Enum):
    """Reasons for deprecation."""
    
    ZERO_DETECTIONS = "zero_detections"
    NO_VARIANTS = "no_variants"
    SUPERSEDED = "superseded"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    MANUAL_DEPRECATION = "manual_deprecation"


class ArchiveStatus(str, Enum):
    """Status of archived vectors."""
    
    FLAGGED = "flagged"
    PENDING_REVIEW = "pending_review"
    APPROVED_DEPRECATION = "approved_deprecation"
    ARCHIVED = "archived"
    RESTORED = "restored"


@dataclass
class VectorUsageStats:
    """Usage statistics for an attack vector."""
    
    vector_id: str
    total_detections: int
    last_detection: Optional[datetime]
    false_positive_count: int
    known_variants: int
    avg_confidence: float
    days_since_detection: int = 0


@dataclass
class DeprecationCandidate:
    """Candidate attack vector for deprecation."""
    
    vector_id: str
    reason: DeprecationReason
    usage_stats: VectorUsageStats
    flagged_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ArchiveStatus = ArchiveStatus.FLAGGED
    reviewer_notes: str = ""


class AutoDeprecation:
    """
    Automatically deprecate unused attack vectors.
    
    This component identifies attack vectors that are no longer
    detecting threats and manages their lifecycle through deprecation.
    
    Features:
    - Automatic flagging based on usage
    - Human review workflow
    - Archive management
    - Restoration capability
    """
    
    def __init__(
        self,
        zero_detection_days: int = 90,
        require_human_confirmation: bool = True
    ):
        """
        Initialize auto-deprecation system.
        
        Args:
            zero_detection_days: Days without detection before flagging
            require_human_confirmation: Require human approval
        """
        self.zero_detection_days = zero_detection_days
        self.require_human_confirmation = require_human_confirmation
        self.vector_stats: Dict[str, VectorUsageStats] = {}
        self.deprecation_candidates: Dict[str, DeprecationCandidate] = {}
        self.archived_vectors: Dict[str, DeprecationCandidate] = {}
        
        logger.info(
            f"AutoDeprecation initialized (threshold: {zero_detection_days} days)"
        )
    
    async def analyze_vector_usage(
        self,
        vector_id: str,
        detection_history: List[Dict[str, Any]]
    ) -> VectorUsageStats:
        """
        Analyze usage statistics for a vector.
        
        Args:
            vector_id: Vector to analyze
            detection_history: Historical detection records
            
        Returns:
            Usage statistics
        """
        # Calculate statistics
        total_detections = len(detection_history)
        
        # Find last detection
        last_detection = None
        if detection_history:
            sorted_history = sorted(
                detection_history,
                key=lambda x: x.get("timestamp", datetime.min),
                reverse=True
            )
            last_detection = sorted_history[0].get("timestamp")
        
        # Calculate days since last detection
        days_since_detection = 0
        if last_detection:
            delta = datetime.now(timezone.utc) - last_detection
            days_since_detection = delta.days
        else:
            days_since_detection = 999  # No detections ever
        
        # Count false positives
        false_positive_count = sum(
            1 for d in detection_history if d.get("false_positive", False)
        )
        
        # Calculate average confidence
        confidences = [d.get("confidence", 0.5) for d in detection_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Estimate known variants (simplified)
        known_variants = self._estimate_variants(detection_history)
        
        stats = VectorUsageStats(
            vector_id=vector_id,
            total_detections=total_detections,
            last_detection=last_detection,
            false_positive_count=false_positive_count,
            known_variants=known_variants,
            avg_confidence=avg_confidence,
            days_since_detection=days_since_detection
        )
        
        self.vector_stats[vector_id] = stats
        
        return stats
    
    def _estimate_variants(self, detection_history: List[Dict[str, Any]]) -> int:
        """Estimate number of attack variants from detection history."""
        # Group by signature/pattern similarity
        unique_signatures = set()
        
        for detection in detection_history:
            signature = detection.get("signature", "")
            # Simple hash-based deduplication
            unique_signatures.add(signature[:50])  # Truncate for grouping
        
        return len(unique_signatures)
    
    async def identify_deprecation_candidates(
        self,
        all_vectors: Dict[str, Any]
    ) -> List[DeprecationCandidate]:
        """
        Identify vectors that are candidates for deprecation.
        
        Args:
            all_vectors: Dictionary of all active vectors
            
        Returns:
            List of deprecation candidates
        """
        candidates = []
        
        for vector_id, vector_info in all_vectors.items():
            # Get or create usage stats
            if vector_id not in self.vector_stats:
                # Need detection history to analyze
                detection_history = vector_info.get("detection_history", [])
                await self.analyze_vector_usage(vector_id, detection_history)
            
            stats = self.vector_stats.get(vector_id)
            if not stats:
                continue
            
            # Check deprecation criteria
            reasons = self._check_deprecation_criteria(stats)
            
            if reasons:
                # Create deprecation candidate
                candidate = DeprecationCandidate(
                    vector_id=vector_id,
                    reason=reasons[0],  # Primary reason
                    usage_stats=stats
                )
                
                candidates.append(candidate)
                self.deprecation_candidates[vector_id] = candidate
                
                logger.info(
                    f"Flagged {vector_id} for deprecation: {reasons[0].value}"
                )
        
        return candidates
    
    def _check_deprecation_criteria(
        self,
        stats: VectorUsageStats
    ) -> List[DeprecationReason]:
        """Check if vector meets deprecation criteria."""
        reasons = []
        
        # Zero detections for threshold period
        if stats.days_since_detection >= self.zero_detection_days:
            reasons.append(DeprecationReason.ZERO_DETECTIONS)
        
        # No known variants
        if stats.known_variants == 0:
            reasons.append(DeprecationReason.NO_VARIANTS)
        
        # High false positive rate (>50%)
        if stats.total_detections > 0:
            fp_rate = stats.false_positive_count / stats.total_detections
            if fp_rate > 0.5:
                reasons.append(DeprecationReason.FALSE_POSITIVE_RATE)
        
        return reasons
    
    async def flag_for_review(
        self,
        vector_id: str,
        reason: DeprecationReason
    ) -> bool:
        """
        Flag a vector for human review.
        
        Args:
            vector_id: Vector to flag
            reason: Reason for flagging
            
        Returns:
            True if flagged successfully
        """
        if vector_id not in self.vector_stats:
            logger.warning(f"Vector {vector_id} not found in stats")
            return False
        
        # Create or update candidate
        if vector_id not in self.deprecation_candidates:
            candidate = DeprecationCandidate(
                vector_id=vector_id,
                reason=reason,
                usage_stats=self.vector_stats[vector_id]
            )
            self.deprecation_candidates[vector_id] = candidate
        
        candidate = self.deprecation_candidates[vector_id]
        candidate.status = ArchiveStatus.PENDING_REVIEW
        
        logger.info(f"Flagged {vector_id} for review: {reason.value}")
        
        return True
    
    async def approve_deprecation(
        self,
        vector_id: str,
        reviewer_notes: str = ""
    ) -> bool:
        """
        Approve deprecation of a vector.
        
        Args:
            vector_id: Vector to deprecate
            reviewer_notes: Notes from reviewer
            
        Returns:
            True if approved and archived
        """
        if vector_id not in self.deprecation_candidates:
            logger.warning(f"Vector {vector_id} not flagged for deprecation")
            return False
        
        candidate = self.deprecation_candidates[vector_id]
        candidate.status = ArchiveStatus.APPROVED_DEPRECATION
        candidate.reviewer_notes = reviewer_notes
        
        # Move to archive
        await self._archive_vector(candidate)
        
        logger.info(f"Approved deprecation of {vector_id}")
        
        return True
    
    async def _archive_vector(
        self,
        candidate: DeprecationCandidate
    ) -> None:
        """
        Move vector to archive.
        
        Note: Vectors are archived, not deleted, for future reference.
        """
        candidate.status = ArchiveStatus.ARCHIVED
        self.archived_vectors[candidate.vector_id] = candidate
        
        # Remove from active candidates
        if candidate.vector_id in self.deprecation_candidates:
            del self.deprecation_candidates[candidate.vector_id]
        
        logger.info(
            f"Archived vector {candidate.vector_id}: {candidate.reason.value}"
        )
    
    async def restore_vector(
        self,
        vector_id: str,
        restoration_reason: str
    ) -> bool:
        """
        Restore an archived vector to active status.
        
        Args:
            vector_id: Vector to restore
            restoration_reason: Reason for restoration
            
        Returns:
            True if restored successfully
        """
        if vector_id not in self.archived_vectors:
            logger.warning(f"Vector {vector_id} not in archive")
            return False
        
        candidate = self.archived_vectors[vector_id]
        candidate.status = ArchiveStatus.RESTORED
        candidate.reviewer_notes += f"\n[RESTORED] {restoration_reason}"
        
        # Remove from archive
        del self.archived_vectors[vector_id]
        
        # Reset usage stats
        if vector_id in self.vector_stats:
            stats = self.vector_stats[vector_id]
            stats.days_since_detection = 0
        
        logger.info(f"Restored vector {vector_id}: {restoration_reason}")
        
        return True
    
    def get_pending_reviews(self) -> List[DeprecationCandidate]:
        """Get list of vectors pending human review."""
        return [
            candidate for candidate in self.deprecation_candidates.values()
            if candidate.status == ArchiveStatus.PENDING_REVIEW
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deprecation statistics."""
        return {
            "total_vectors_tracked": len(self.vector_stats),
            "deprecation_candidates": len(self.deprecation_candidates),
            "pending_review": len(self.get_pending_reviews()),
            "archived_vectors": len(self.archived_vectors),
            "by_reason": self._count_by_reason(),
            "avg_days_since_detection": self._calculate_avg_days(),
        }
    
    def _count_by_reason(self) -> Dict[str, int]:
        """Count candidates by deprecation reason."""
        counts = {}
        for candidate in self.deprecation_candidates.values():
            reason = candidate.reason.value
            counts[reason] = counts.get(reason, 0) + 1
        return counts
    
    def _calculate_avg_days(self) -> float:
        """Calculate average days since last detection."""
        if not self.vector_stats:
            return 0.0
        
        total_days = sum(
            stats.days_since_detection for stats in self.vector_stats.values()
        )
        return total_days / len(self.vector_stats)
    
    def generate_deprecation_report(self) -> Dict[str, Any]:
        """Generate comprehensive deprecation report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": self.get_statistics(),
            "pending_review": [
                {
                    "vector_id": c.vector_id,
                    "reason": c.reason.value,
                    "days_since_detection": c.usage_stats.days_since_detection,
                    "total_detections": c.usage_stats.total_detections,
                    "flagged_at": c.flagged_at.isoformat(),
                }
                for c in self.get_pending_reviews()
            ],
            "archived": [
                {
                    "vector_id": c.vector_id,
                    "reason": c.reason.value,
                    "archived_at": c.flagged_at.isoformat(),
                    "reviewer_notes": c.reviewer_notes,
                }
                for c in self.archived_vectors.values()
            ],
        }
