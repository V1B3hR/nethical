"""Community Management for Plugin Contributions.

This module provides infrastructure for community contributions,
plugin reviews, and recognition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime


class ReviewStatus(Enum):
    """Plugin review status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_CHANGES = "needs_changes"


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
    
    def is_positive(self) -> bool:
        """Check if review is positive (>= 3 stars)."""
        return self.rating >= 3.0


@dataclass
class PluginSubmission:
    """Plugin submission for marketplace."""
    submission_id: str
    plugin_id: str
    author: str
    submission_date: datetime
    status: ReviewStatus
    reviewer_notes: List[str] = field(default_factory=list)
    
    def add_note(self, note: str):
        """Add reviewer note."""
        self.reviewer_notes.append(f"[{datetime.now().isoformat()}] {note}")


@dataclass
class ContributionTemplate:
    """Template for plugin contributions."""
    template_id: str
    name: str
    description: str
    required_files: List[str]
    guidelines: List[str]
    examples: Dict[str, str] = field(default_factory=dict)


class CommunityManager:
    """Manage community contributions and reviews.
    
    This class handles plugin submissions, reviews, and community recognition.
    
    Example:
        >>> community = CommunityManager()
        >>> submission = community.submit_plugin("my-plugin", "author")
        >>> community.add_review("my-plugin", "reviewer", 4.5, "Great plugin!")
        >>> stats = community.get_contributor_stats("author")
    """
    
    def __init__(self, storage_dir: str = "./nethical_community"):
        """Initialize community manager.
        
        Args:
            storage_dir: Directory for community data
        """
        self.storage_dir = storage_dir
        self._submissions: Dict[str, PluginSubmission] = {}
        self._reviews: Dict[str, List[PluginReview]] = {}
        self._contributor_stats: Dict[str, Dict] = {}
    
    def submit_plugin(
        self,
        plugin_id: str,
        author: str
    ) -> PluginSubmission:
        """Submit a plugin for review.
        
        Args:
            plugin_id: Plugin identifier
            author: Plugin author
            
        Returns:
            Submission record
        """
        submission_id = f"sub_{plugin_id}_{int(datetime.now().timestamp())}"
        submission = PluginSubmission(
            submission_id=submission_id,
            plugin_id=plugin_id,
            author=author,
            submission_date=datetime.now(),
            status=ReviewStatus.PENDING
        )
        
        self._submissions[submission_id] = submission
        return submission
    
    def add_review(
        self,
        plugin_id: str,
        reviewer: str,
        rating: float,
        comment: str
    ) -> PluginReview:
        """Add a review for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            reviewer: Reviewer username
            rating: Rating (1-5)
            comment: Review comment
            
        Returns:
            Review record
        """
        review_id = f"rev_{plugin_id}_{int(datetime.now().timestamp())}"
        review = PluginReview(
            review_id=review_id,
            plugin_id=plugin_id,
            reviewer=reviewer,
            rating=rating,
            comment=comment,
            review_date=datetime.now()
        )
        
        if plugin_id not in self._reviews:
            self._reviews[plugin_id] = []
        self._reviews[plugin_id].append(review)
        
        return review
    
    def get_reviews(self, plugin_id: str) -> List[PluginReview]:
        """Get all reviews for a plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            List of reviews
        """
        return self._reviews.get(plugin_id, [])
    
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
    
    def approve_submission(self, submission_id: str) -> bool:
        """Approve a plugin submission.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            True if successful
        """
        submission = self._submissions.get(submission_id)
        if submission:
            submission.status = ReviewStatus.APPROVED
            submission.add_note("Submission approved")
            return True
        return False
    
    def reject_submission(self, submission_id: str, reason: str) -> bool:
        """Reject a plugin submission.
        
        Args:
            submission_id: Submission identifier
            reason: Rejection reason
            
        Returns:
            True if successful
        """
        submission = self._submissions.get(submission_id)
        if submission:
            submission.status = ReviewStatus.REJECTED
            submission.add_note(f"Submission rejected: {reason}")
            return True
        return False
    
    def get_contributor_stats(self, author: str) -> Dict:
        """Get statistics for a contributor.
        
        Args:
            author: Contributor username
            
        Returns:
            Contributor statistics
        """
        submissions = [s for s in self._submissions.values() if s.author == author]
        
        return {
            'author': author,
            'total_submissions': len(submissions),
            'approved': sum(1 for s in submissions if s.status == ReviewStatus.APPROVED),
            'pending': sum(1 for s in submissions if s.status == ReviewStatus.PENDING),
            'rejected': sum(1 for s in submissions if s.status == ReviewStatus.REJECTED),
        }
    
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
                "tests.py"
            ],
            guidelines=[
                "Follow PEP 8 style guidelines",
                "Include comprehensive documentation",
                "Provide unit tests with >80% coverage",
                "Use type hints throughout",
                "Include example usage"
            ],
            examples={
                "plugin.py": "# Example plugin implementation",
                "README.md": "# Plugin Name\n\nDescription..."
            }
        )
