"""Community Management for Plugin Contributions.

This module provides infrastructure for community contributions,
plugin reviews, and recognition, aligned with the broader Nethical system
principles (auditability, integrity, human-in-the-loop readiness, and
operational metrics).

Key Enhancements:
- Stronger typing, validation, and lifecycle state tracking
- Optional audit logging with tamper-evident hash chaining
- Lightweight persistence (JSONL) for submissions and reviews
- Helpful vote tracking with duplicate prevention
- Extended contributor and plugin statistics
- Reviewer assignment and review workflow helpers
- SLA and timing metrics for review lifecycle
- Templates with validation helpers
- System status snapshot for observability
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Iterable
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Current time in UTC."""
    return datetime.now(timezone.utc)


def _gen_id(prefix: str) -> str:
    """Generate a stable unique identifier with a prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_json(data: Any) -> str:
    """Compute sha256 over a canonical JSON representation."""
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Audit Logging (tamper-evident hash chaining)
# ---------------------------------------------------------------------------

class AuditLogger:
    """Append-only audit log with hash chaining (tamper-evident).

    Each event is stored as JSONL with fields:
      - ts: ISO timestamp
      - type: event type
      - data: event payload
      - prev_chain_hash: previous chain hash (or empty for first record)
      - chain_hash: sha256(prev_chain_hash + sha256_json(event payload blob))

    This is not a full Merkle tree but provides cryptographic anchoring
    via chained hashing. External anchoring (e.g., Merkle root pinning)
    can be layered on top if needed by higher-level governance.
    """

    def __init__(self, log_dir: str, filename: str = "community_audit.jsonl") -> None:
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, filename)
        _ensure_dir(self.log_dir)
        self._lock = threading.RLock()
        self._chain_hash_state_path = os.path.join(log_dir, ".chainhash")
        self._chain_hash = self._load_chain_state()

    def _load_chain_state(self) -> str:
        if os.path.exists(self._chain_hash_state_path):
            try:
                with open(self._chain_hash_state_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def _persist_chain_state(self, chain_hash: str) -> None:
        try:
            with open(self._chain_hash_state_path, "w", encoding="utf-8") as f:
                f.write(chain_hash)
        except Exception:
            # Non-fatal: system continues without persisting state
            pass

    def log_event(self, event_type: str, data: Dict[str, Any]) -> str:
        with self._lock:
            event = {
                "ts": utcnow().isoformat(),
                "type": event_type,
                "data": data,
            }
            payload_hash = _sha256_json(event)
            prev = self._chain_hash or ""
            chain_hash = sha256((prev + payload_hash).encode("utf-8")).hexdigest()

            record = {
                "ts": event["ts"],
                "type": event["type"],
                "data": data,  # unmodified payload for readability
                "prev_chain_hash": prev,
                "payload_hash": payload_hash,
                "chain_hash": chain_hash,
            }

            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

            self._chain_hash = chain_hash
            self._persist_chain_state(chain_hash)
            return chain_hash


# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------

class ReviewStatus(Enum):
    """Plugin review status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


@dataclass
class SubmissionEvent:
    """State transition / note for a submission."""
    ts: datetime
    status: ReviewStatus
    note: str
    actor: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "status": self.status.value,
            "note": self.note,
            "actor": self.actor,
        }


@dataclass
class PluginReview:
    """Plugin review from community member."""
    review_id: str
    plugin_id: str
    reviewer: str
    rating: float  # 1-5 stars
    comment: str
    review_date: datetime
    helpful_votes: int = 0
    helpful_voters: Set[str] = field(default_factory=set)
    moderated: bool = False
    moderation_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not (1.0 <= self.rating <= 5.0):
            raise ValueError("rating must be within [1.0, 5.0]")
        self.comment = self.comment.strip()

    def is_positive(self) -> bool:
        """Check if review is positive (>= 3 stars)."""
        return self.rating >= 3.0

    def add_helpful_vote(self, voter: str) -> bool:
        """Add a helpful vote; returns True if counted (no duplicates)."""
        if voter in self.helpful_voters:
            return False
        self.helpful_voters.add(voter)
        self.helpful_votes += 1
        return True

    def moderate(self, reason: str) -> None:
        """Mark this review as moderated (hidden/flagged)."""
        self.moderated = True
        self.moderation_reason = reason.strip()


@dataclass
class PluginSubmission:
    """Plugin submission for marketplace."""
    submission_id: str
    plugin_id: str
    author: str
    submission_date: datetime
    status: ReviewStatus
    reviewer_notes: List[str] = field(default_factory=list)
    assigned_reviewers: Set[str] = field(default_factory=set)
    status_history: List[SubmissionEvent] = field(default_factory=list)
    plugin_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_note(self, note: str, actor: str = "system") -> None:
        """Add reviewer note."""
        ts = utcnow()
        self.reviewer_notes.append(f"[{ts.isoformat()}] {note}")
        # Mirror notes to status history as a non-transition entry
        self.status_history.append(
            SubmissionEvent(ts=ts, status=self.status, note=note, actor=actor)
        )

    def assign_reviewer(self, reviewer: str) -> bool:
        """Assign a reviewer to this submission."""
        reviewer = reviewer.strip()
        if not reviewer:
            return False
        if reviewer in self.assigned_reviewers:
            return False
        self.assigned_reviewers.add(reviewer)
        self.add_note(f"Reviewer assigned: {reviewer}", actor="system")
        return True

    def transition(self, new_status: ReviewStatus, note: str, actor: str = "system") -> None:
        """Transition submission status with auditing."""
        ts = utcnow()
        self.status = new_status
        self.status_history.append(SubmissionEvent(ts=ts, status=new_status, note=note, actor=actor))
        # Also append human-readable note
        self.reviewer_notes.append(f"[{ts.isoformat()}] ({actor}) {new_status.value.upper()}: {note}")


@dataclass
class ContributionTemplate:
    """Template for plugin contributions."""
    template_id: str
    name: str
    description: str
    required_files: List[str]
    guidelines: List[str]
    examples: Dict[str, str] = field(default_factory=dict)

    def validate_submission_files(self, files_present: Iterable[str]) -> Dict[str, Any]:
        """Validate required files against a provided iterable of filenames."""
        present = set(files_present)
        missing = [f for f in self.required_files if f not in present]
        return {
            "is_valid": len(missing) == 0,
            "missing": missing,
            "required": list(self.required_files),
        }


# ---------------------------------------------------------------------------
# Persistence (lightweight JSONL)
# ---------------------------------------------------------------------------

class _JsonlStore:
    """Simple append-only JSONL store for submissions and reviews."""

    def __init__(self, storage_dir: str) -> None:
        self.dir = storage_dir
        _ensure_dir(self.dir)
        self.submissions_path = os.path.join(self.dir, "submissions.jsonl")
        self.reviews_path = os.path.join(self.dir, "reviews.jsonl")
        _ensure_dir(os.path.dirname(self.submissions_path))
        _ensure_dir(os.path.dirname(self.reviews_path))
        self._lock = threading.RLock()

    def append_submission(self, submission: PluginSubmission) -> None:
        with self._lock, open(self.submissions_path, "a", encoding="utf-8") as f:
            payload = asdict(submission)
            payload["submission_date"] = submission.submission_date.isoformat()
            payload["status"] = submission.status.value
            payload["status_history"] = [e.to_dict() for e in submission.status_history]
            # Sets to sorted lists for determinism
            payload["assigned_reviewers"] = sorted(list(submission.assigned_reviewers))
            f.write(json.dumps(payload, sort_keys=True) + "\n")

    def append_review(self, review: PluginReview) -> None:
        with self._lock, open(self.reviews_path, "a", encoding="utf-8") as f:
            payload = asdict(review)
            payload["review_date"] = review.review_date.isoformat()
            payload["helpful_voters"] = sorted(list(review.helpful_voters))
            f.write(json.dumps(payload, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Community Manager
# ---------------------------------------------------------------------------

class CommunityManager:
    """Manage community contributions and reviews.

    This class handles plugin submissions, reviews, and community recognition,
    with audit logging and optional lightweight persistence.

    Example:
        >>> community = CommunityManager()
        >>> submission = community.submit_plugin("my-plugin", "author")
        >>> community.add_review("my-plugin", "reviewer", 4.5, "Great plugin!")
        >>> stats = community.get_contributor_stats("author")
    """

    def __init__(
        self,
        storage_dir: str = "./nethical_community",
        enable_audit: bool = True,
        persist_jsonl: bool = True,
    ):
        """Initialize community manager.

        Args:
            storage_dir: Directory for community data (audit logs, stores)
            enable_audit: Enable tamper-evident audit logging for events
            persist_jsonl: Persist submissions/reviews to JSONL append-only files
        """
        self.storage_dir = storage_dir
        _ensure_dir(self.storage_dir)

        # In-memory indices
        self._submissions: Dict[str, PluginSubmission] = {}
        self._reviews: Dict[str, List[PluginReview]] = defaultdict(list)
        self._plugin_to_submissions: Dict[str, Set[str]] = defaultdict(set)

        # Stats cache (basic, computed on demand)
        self._contributor_stats: Dict[str, Dict[str, Any]] = {}

        # Infra
        self._lock = threading.RLock()
        self._store = _JsonlStore(os.path.join(self.storage_dir, "store")) if persist_jsonl else None
        self._audit = AuditLogger(os.path.join(self.storage_dir, "audit")) if enable_audit else None

    # --------------------------
    # Submissions
    # --------------------------

    def submit_plugin(
        self,
        plugin_id: str,
        author: str,
        *,
        plugin_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actor: str = "author",
    ) -> PluginSubmission:
        """Submit a plugin for review.

        Args:
            plugin_id: Plugin identifier
            author: Plugin author
            plugin_version: Optional semantic version or tag
            metadata: Arbitrary metadata for the submission (e.g., repo URL)
            actor: Who initiated the submission (for audit trails)

        Returns:
            Submission record
        """
        plugin_id = plugin_id.strip()
        author = author.strip()
        if not plugin_id or not author:
            raise ValueError("plugin_id and author are required")

        submission_id = _gen_id("sub")
        submission = PluginSubmission(
            submission_id=submission_id,
            plugin_id=plugin_id,
            author=author,
            submission_date=utcnow(),
            status=ReviewStatus.PENDING,
            plugin_version=plugin_version,
            metadata=metadata or {},
        )
        submission.transition(ReviewStatus.PENDING, "Submission created", actor=actor)

        with self._lock:
            self._submissions[submission_id] = submission
            self._plugin_to_submissions[plugin_id].add(submission_id)

            if self._store:
                self._store.append_submission(submission)

            if self._audit:
                self._audit.log_event(
                    "submission_created",
                    {
                        "submission_id": submission_id,
                        "plugin_id": plugin_id,
                        "author": author,
                        "plugin_version": plugin_version,
                        "metadata": submission.metadata,
                    },
                )

        return submission

    def get_submission(self, submission_id: str) -> Optional[PluginSubmission]:
        with self._lock:
            return self._submissions.get(submission_id)

    def list_submissions(
        self,
        *,
        author: Optional[str] = None,
        plugin_id: Optional[str] = None,
        status: Optional[ReviewStatus] = None,
    ) -> List[PluginSubmission]:
        """List submissions with optional filters."""
        with self._lock:
            subs = list(self._submissions.values())

        if author:
            subs = [s for s in subs if s.author == author]
        if plugin_id:
            subs = [s for s in subs if s.plugin_id == plugin_id]
        if status:
            subs = [s for s in subs if s.status == status]
        subs.sort(key=lambda s: s.submission_date, reverse=True)
        return subs

    def assign_reviewer(self, submission_id: str, reviewer: str, *, actor: str = "system") -> bool:
        """Assign a reviewer to a submission."""
        with self._lock:
            sub = self._submissions.get(submission_id)
            if not sub:
                return False
            changed = sub.assign_reviewer(reviewer)
            if changed and self._audit:
                self._audit.log_event(
                    "reviewer_assigned",
                    {"submission_id": submission_id, "reviewer": reviewer, "actor": actor},
                )
            if changed and self._store:
                self._store.append_submission(sub)
            return changed

    def approve_submission(self, submission_id: str, *, actor: str = "reviewer") -> bool:
        """Approve a plugin submission.

        Args:
            submission_id: Submission identifier

        Returns:
            True if successful
        """
        with self._lock:
            submission = self._submissions.get(submission_id)
            if submission:
                submission.transition(ReviewStatus.APPROVED, "Submission approved", actor=actor)

                if self._store:
                    self._store.append_submission(submission)
                if self._audit:
                    self._audit.log_event(
                        "submission_approved", {"submission_id": submission_id, "actor": actor}
                    )
                return True
            return False

    def reject_submission(self, submission_id: str, reason: str, *, actor: str = "reviewer") -> bool:
        """Reject a plugin submission.

        Args:
            submission_id: Submission identifier
            reason: Rejection reason

        Returns:
            True if successful
        """
        reason = reason.strip() or "No reason provided"
        with self._lock:
            submission = self._submissions.get(submission_id)
            if submission:
                submission.transition(ReviewStatus.REJECTED, f"Submission rejected: {reason}", actor=actor)

                if self._store:
                    self._store.append_submission(submission)
                if self._audit:
                    self._audit.log_event(
                        "submission_rejected",
                        {"submission_id": submission_id, "reason": reason, "actor": actor},
                    )
                return True
            return False

    def request_changes(self, submission_id: str, note: str, *, actor: str = "reviewer") -> bool:
        """Move submission to NEEDS_CHANGES with a note."""
        note = note.strip() or "Changes requested"
        with self._lock:
            submission = self._submissions.get(submission_id)
            if submission:
                submission.transition(ReviewStatus.NEEDS_CHANGES, note, actor=actor)

                if self._store:
                    self._store.append_submission(submission)
                if self._audit:
                    self._audit.log_event(
                        "submission_needs_changes",
                        {"submission_id": submission_id, "note": note, "actor": actor},
                    )
                return True
            return False

    # --------------------------
    # Reviews
    # --------------------------

    def add_review(
        self,
        plugin_id: str,
        reviewer: str,
        rating: float,
        comment: str,
        *,
        moderated: bool = False,
        moderation_reason: Optional[str] = None,
    ) -> PluginReview:
        """Add a review for a plugin.

        Args:
            plugin_id: Plugin identifier
            reviewer: Reviewer username
            rating: Rating (1-5)
            comment: Review comment
            moderated: Immediately mark as moderated/hidden
            moderation_reason: Reason, if moderated

        Returns:
            Review record
        """
        plugin_id = plugin_id.strip()
        reviewer = reviewer.strip()
        if not plugin_id or not reviewer:
            raise ValueError("plugin_id and reviewer are required")

        review_id = _gen_id("rev")
        review = PluginReview(
            review_id=review_id,
            plugin_id=plugin_id,
            reviewer=reviewer,
            rating=rating,
            comment=comment,
            review_date=utcnow(),
            moderated=moderated,
            moderation_reason=moderation_reason,
        )

        with self._lock:
            self._reviews[plugin_id].append(review)

            if self._store:
                self._store.append_review(review)

            if self._audit:
                self._audit.log_event(
                    "review_added",
                    {
                        "review_id": review_id,
                        "plugin_id": plugin_id,
                        "reviewer": reviewer,
                        "rating": rating,
                        "moderated": moderated,
                        "moderation_reason": moderation_reason,
                    },
                )

        return review

    def add_helpful_vote(self, review_id: str, plugin_id: str, voter: str) -> bool:
        """Mark a review as helpful; no duplicate votes permitted per voter."""
        voter = voter.strip()
        if not voter:
            return False

        with self._lock:
            reviews = self._reviews.get(plugin_id, [])
            for r in reviews:
                if r.review_id == review_id:
                    counted = r.add_helpful_vote(voter)
                    if counted and self._audit:
                        self._audit.log_event(
                            "review_helpful_voted",
                            {"review_id": review_id, "plugin_id": plugin_id, "voter": voter},
                        )
                    if counted and self._store:
                        self._store.append_review(r)
                    return counted
        return False

    def moderate_review(self, review_id: str, plugin_id: str, reason: str, *, actor: str = "moderator") -> bool:
        """Flag a review as moderated/hidden."""
        reason = reason.strip() or "Policy violation"
        with self._lock:
            reviews = self._reviews.get(plugin_id, [])
            for r in reviews:
                if r.review_id == review_id:
                    r.moderate(reason)
                    if self._audit:
                        self._audit.log_event(
                            "review_moderated",
                            {
                                "review_id": review_id,
                                "plugin_id": plugin_id,
                                "reason": reason,
                                "actor": actor,
                            },
                        )
                    if self._store:
                        self._store.append_review(r)
                    return True
        return False

    def get_reviews(self, plugin_id: str, include_moderated: bool = False) -> List[PluginReview]:
        """Get all reviews for a plugin.

        Args:
            plugin_id: Plugin identifier
            include_moderated: Include moderated/hidden reviews

        Returns:
            List of reviews
        """
        with self._lock:
            reviews = list(self._reviews.get(plugin_id, []))
        if not include_moderated:
            reviews = [r for r in reviews if not r.moderated]
        reviews.sort(key=lambda r: r.review_date, reverse=True)
        return reviews

    def get_average_rating(self, plugin_id: str) -> float:
        """Get average rating for a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Average rating
        """
        reviews = self.get_reviews(plugin_id)
        if not reviews:
            return 0.0
        return sum(r.rating for r in reviews) / len(reviews)

    def get_rating_histogram(self, plugin_id: str) -> Dict[int, int]:
        """Return counts of ratings 1..5 for a plugin."""
        hist = Counter(int(round(r.rating)) for r in self.get_reviews(plugin_id))
        return {k: hist.get(k, 0) for k in range(1, 6)}

    # --------------------------
    # Stats and Scores
    # --------------------------

    def get_contributor_stats(self, author: str) -> Dict[str, Any]:
        """Get statistics for a contributor.

        Args:
            author: Contributor username

        Returns:
            Contributor statistics
        """
        with self._lock:
            submissions = [s for s in self._submissions.values() if s.author == author]
        total = len(submissions)
        approved = sum(1 for s in submissions if s.status == ReviewStatus.APPROVED)
        pending = sum(1 for s in submissions if s.status == ReviewStatus.PENDING)
        rejected = sum(1 for s in submissions if s.status == ReviewStatus.REJECTED)
        needs_changes = sum(1 for s in submissions if s.status == ReviewStatus.NEEDS_CHANGES)

        # Time-to-approval metrics
        approval_durations: List[float] = []
        for s in submissions:
            created_ts = next((e.ts for e in s.status_history if e.status == ReviewStatus.PENDING), None)
            approved_ts = next((e.ts for e in s.status_history if e.status == ReviewStatus.APPROVED), None)
            if created_ts and approved_ts:
                approval_durations.append((approved_ts - created_ts).total_seconds())

        avg_tta = sum(approval_durations) / len(approval_durations) if approval_durations else None

        # Helpful votes across author's plugins
        helpful_votes_total = 0
        total_reviews = 0
        with self._lock:
            for s in submissions:
                for r in self._reviews.get(s.plugin_id, []):
                    helpful_votes_total += r.helpful_votes
                    total_reviews += 1

        acceptance_rate = (approved / total) if total else 0.0
        helpful_votes_per_review = (helpful_votes_total / total_reviews) if total_reviews else 0.0

        return {
            "author": author,
            "total_submissions": total,
            "approved": approved,
            "pending": pending,
            "rejected": rejected,
            "needs_changes": needs_changes,
            "acceptance_rate": round(acceptance_rate, 4),
            "avg_time_to_approval_seconds": avg_tta,
            "helpful_votes_total": helpful_votes_total,
            "helpful_votes_per_review": round(helpful_votes_per_review, 4),
        }

    def get_plugin_stats(self, plugin_id: str) -> Dict[str, Any]:
        """Aggregate stats for a plugin."""
        reviews = self.get_reviews(plugin_id)
        avg = self.get_average_rating(plugin_id)
        hist = self.get_rating_histogram(plugin_id)
        return {
            "plugin_id": plugin_id,
            "average_rating": round(avg, 4),
            "total_reviews": len(reviews),
            "rating_histogram": hist,
            "positive_share": round((sum(1 for r in reviews if r.is_positive()) / len(reviews)) if reviews else 0.0, 4),
            "helpful_votes_total": sum(r.helpful_votes for r in reviews),
        }

    def compute_trust_score(
        self,
        *,
        average_rating: float,
        review_count: int,
        approval_ratio: float,
        helpful_votes: int,
    ) -> float:
        """Compute a plugin/community trust score (0..1) using a simple blend.

        - Normalize average rating to 0..1 scale
        - Weight by a saturation function of review_count
        - Include approval_ratio and helpful_votes normalization
        """
        # Rating normalized
        rating_norm = (average_rating - 1.0) / 4.0 if review_count > 0 else 0.0

        # Saturation of volume (bounded at ~200 reviews)
        volume_weight = min(review_count / 200.0, 1.0)

        # Helpful votes normalized against a soft cap
        helpful_norm = min(helpful_votes / 100.0, 1.0)

        # Blend with weights (tunable)
        score = (
            0.5 * rating_norm
            + 0.2 * volume_weight
            + 0.2 * approval_ratio
            + 0.1 * helpful_norm
        )
        return max(0.0, min(1.0, score))

    # --------------------------
    # Templates
    # --------------------------

    def get_contribution_template(self, template_name: str = "default") -> ContributionTemplate:
        """Get contribution template.

        Args:
            template_name: Template identifier

        Returns:
            Contribution template
        """
        return ContributionTemplate(
            template_id=template_name,
            name="Default Plugin Template",
            description="Standard template for plugin contributions",
            required_files=[
                "plugin.py",
                "README.md",
                "requirements.txt",
                "tests.py",
            ],
            guidelines=[
                "Follow PEP 8 style guidelines",
                "Include comprehensive documentation",
                "Provide unit tests with >80% coverage",
                "Use type hints throughout",
                "Include example usage",
            ],
            examples={
                "plugin.py": "# Example plugin implementation",
                "README.md": "# Plugin Name\n\nDescription...",
            },
        )

    # --------------------------
    # System status and introspection
    # --------------------------

    def get_system_status(self) -> Dict[str, Any]:
        """Return a snapshot of community system status for observability."""
        with self._lock:
            total_submissions = len(self._submissions)
            by_status = Counter(s.status.value for s in self._submissions.values())
            total_reviews = sum(len(v) for v in self._reviews.values())
            total_plugins = len({s.plugin_id for s in self._submissions.values()})
            assigned = sum(len(s.assigned_reviewers) for s in self._submissions.values())

        return {
            "storage_dir": self.storage_dir,
            "audit_enabled": self._audit is not None,
            "persistence_enabled": self._store is not None,
            "counts": {
                "submissions": total_submissions,
                "submissions_by_status": dict(by_status),
                "reviews": total_reviews,
                "plugins": total_plugins,
                "assigned_reviewers": assigned,
            },
            "timestamp": utcnow().isoformat(),
        }
