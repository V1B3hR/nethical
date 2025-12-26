"""
Feedback and Fine-tuning Infrastructure for Universal Vector Language.

This module provides hooks for supervised fine-tuning using user-labeled
action-law pairs, feedback logging, and continuous performance improvement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback for fine-tuning."""
    CORRECT_CLASSIFICATION = "correct_classification"
    INCORRECT_CLASSIFICATION = "incorrect_classification"
    MISSING_PRIMITIVE = "missing_primitive"
    FALSE_POSITIVE_PRIMITIVE = "false_positive_primitive"
    LAW_MISMATCH = "law_mismatch"
    RISK_SCORE_TOO_HIGH = "risk_score_too_high"
    RISK_SCORE_TOO_LOW = "risk_score_too_low"
    USER_OVERRIDE = "user_override"


class FeedbackSource(str, Enum):
    """Source of feedback."""
    USER = "user"
    AUTOMATED_TEST = "automated_test"
    HUMAN_REVIEWER = "human_reviewer"
    BENCHMARK = "benchmark"
    PRODUCTION = "production"


@dataclass
class ActionLawPair:
    """Training pair of action and its correct law classifications."""
    
    action_text: str
    action_type: str
    context: Dict[str, Any]
    
    # Ground truth labels
    correct_laws: List[int]
    correct_primitives: List[str]
    correct_risk_score: float
    correct_decision: str
    
    # Metadata
    action_hash: str
    timestamp: datetime
    source: FeedbackSource
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.action_hash:
            self.action_hash = hashlib.sha256(
                self.action_text.encode('utf-8')
            ).hexdigest()[:16]


@dataclass
class FeedbackEntry:
    """Single feedback entry for model improvement."""
    
    feedback_id: str
    feedback_type: FeedbackType
    source: FeedbackSource
    
    # Original evaluation
    action_text: str
    action_type: str
    context: Dict[str, Any]
    predicted_laws: List[int]
    predicted_primitives: List[str]
    predicted_risk_score: float
    predicted_decision: str
    
    # Corrected/expected values
    expected_laws: Optional[List[int]] = None
    expected_primitives: Optional[List[str]] = None
    expected_risk_score: Optional[float] = None
    expected_decision: Optional[str] = None
    
    # Feedback details
    comment: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # Timestamps
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
        from uuid import uuid4
        if not self.feedback_id:
            self.feedback_id = f"fb_{uuid4().hex[:12]}"


class FeedbackLogger:
    """Logger for collecting feedback data for fine-tuning."""
    
    def __init__(
        self,
        log_path: Optional[str] = None,
        auto_export: bool = True,
        export_format: str = "jsonl",
    ):
        """Initialize feedback logger.
        
        Args:
            log_path: Path to feedback log file
            auto_export: Whether to auto-export feedback to file
            export_format: Format for export (jsonl, json, csv)
        """
        self.log_path = Path(log_path) if log_path else Path("./feedback_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.auto_export = auto_export
        self.export_format = export_format
        
        # In-memory storage
        self.feedback_entries: List[FeedbackEntry] = []
        self.action_law_pairs: List[ActionLawPair] = []
        
        # Statistics
        self.feedback_counts: Dict[FeedbackType, int] = {}
        self.accuracy_metrics: Dict[str, float] = {}
        
        logger.info(f"FeedbackLogger initialized at {self.log_path}")
    
    def log_feedback(
        self,
        feedback_type: FeedbackType,
        source: FeedbackSource,
        action_text: str,
        action_type: str,
        context: Dict[str, Any],
        predicted_laws: List[int],
        predicted_primitives: List[str],
        predicted_risk_score: float,
        predicted_decision: str,
        expected_laws: Optional[List[int]] = None,
        expected_primitives: Optional[List[str]] = None,
        expected_risk_score: Optional[float] = None,
        expected_decision: Optional[str] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackEntry:
        """Log a feedback entry.
        
        Args:
            feedback_type: Type of feedback
            source: Source of feedback
            action_text: The action that was evaluated
            action_type: Type of action
            context: Action context
            predicted_laws: Laws that were predicted
            predicted_primitives: Primitives that were detected
            predicted_risk_score: Predicted risk score
            predicted_decision: Predicted decision
            expected_laws: Correct laws (if available)
            expected_primitives: Correct primitives (if available)
            expected_risk_score: Correct risk score (if available)
            expected_decision: Correct decision (if available)
            comment: Optional comment
            metadata: Optional metadata
            
        Returns:
            Created FeedbackEntry
        """
        from uuid import uuid4
        
        entry = FeedbackEntry(
            feedback_id=f"fb_{uuid4().hex[:12]}",
            feedback_type=feedback_type,
            source=source,
            action_text=action_text,
            action_type=action_type,
            context=context,
            predicted_laws=predicted_laws,
            predicted_primitives=predicted_primitives,
            predicted_risk_score=predicted_risk_score,
            predicted_decision=predicted_decision,
            expected_laws=expected_laws,
            expected_primitives=expected_primitives,
            expected_risk_score=expected_risk_score,
            expected_decision=expected_decision,
            comment=comment,
            metadata=metadata or {},
        )
        
        self.feedback_entries.append(entry)
        
        # Update counts
        self.feedback_counts[feedback_type] = self.feedback_counts.get(feedback_type, 0) + 1
        
        # Auto-export if enabled
        if self.auto_export:
            self._export_entry(entry)
        
        # Create training pair if we have ground truth
        if expected_laws and expected_primitives:
            pair = ActionLawPair(
                action_text=action_text,
                action_type=action_type,
                context=context,
                correct_laws=expected_laws,
                correct_primitives=expected_primitives,
                correct_risk_score=expected_risk_score or predicted_risk_score,
                correct_decision=expected_decision or predicted_decision,
                action_hash="",
                timestamp=datetime.now(timezone.utc),
                source=source,
            )
            self.action_law_pairs.append(pair)
        
        logger.info(
            f"Logged feedback: {feedback_type.value} from {source.value} "
            f"for action: {action_text[:50]}..."
        )
        
        return entry
    
    def _export_entry(self, entry: FeedbackEntry):
        """Export a single feedback entry to file."""
        timestamp_str = entry.timestamp.strftime("%Y%m%d")
        filename = self.log_path / f"feedback_{timestamp_str}.{self.export_format}"
        
        if self.export_format == "jsonl":
            with open(filename, 'a') as f:
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                f.write(json.dumps(entry_dict) + '\n')
        elif self.export_format == "json":
            # Append to JSON array
            if filename.exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            entry_dict = asdict(entry)
            entry_dict['timestamp'] = entry.timestamp.isoformat()
            data.append(entry_dict)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
    
    def export_training_data(
        self,
        output_path: Optional[str] = None,
        format: str = "jsonl",
        min_confidence: float = 0.7,
    ) -> str:
        """Export training data for fine-tuning.
        
        Args:
            output_path: Path to output file
            format: Output format (jsonl, json, csv)
            min_confidence: Minimum confidence for training pairs
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self.log_path / f"training_data_{timestamp}.{format}"
        else:
            output_path = Path(output_path)
        
        # Filter training pairs by confidence
        high_confidence_pairs = [
            pair for pair in self.action_law_pairs
            if pair.confidence >= min_confidence
        ]
        
        if format == "jsonl":
            with open(output_path, 'w') as f:
                for pair in high_confidence_pairs:
                    pair_dict = asdict(pair)
                    pair_dict['timestamp'] = pair.timestamp.isoformat()
                    f.write(json.dumps(pair_dict) + '\n')
        elif format == "json":
            data = []
            for pair in high_confidence_pairs:
                pair_dict = asdict(pair)
                pair_dict['timestamp'] = pair.timestamp.isoformat()
                data.append(pair_dict)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if high_confidence_pairs:
                    fieldnames = asdict(high_confidence_pairs[0]).keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for pair in high_confidence_pairs:
                        pair_dict = asdict(pair)
                        pair_dict['timestamp'] = pair.timestamp.isoformat()
                        pair_dict['correct_laws'] = json.dumps(pair_dict['correct_laws'])
                        pair_dict['correct_primitives'] = json.dumps(pair_dict['correct_primitives'])
                        pair_dict['context'] = json.dumps(pair_dict['context'])
                        writer.writerow(pair_dict)
        
        logger.info(
            f"Exported {len(high_confidence_pairs)} training pairs to {output_path}"
        )
        return str(output_path)
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Calculate accuracy metrics from feedback.
        
        Returns:
            Dictionary of accuracy metrics
        """
        if not self.feedback_entries:
            return {}
        
        total = len(self.feedback_entries)
        correct = sum(
            1 for entry in self.feedback_entries
            if entry.feedback_type == FeedbackType.CORRECT_CLASSIFICATION
        )
        
        # Calculate per-component accuracy
        law_matches = 0
        primitive_matches = 0
        risk_matches = 0
        decision_matches = 0
        
        for entry in self.feedback_entries:
            if entry.expected_laws and entry.predicted_laws == entry.expected_laws:
                law_matches += 1
            if entry.expected_primitives and entry.predicted_primitives == entry.expected_primitives:
                primitive_matches += 1
            if entry.expected_risk_score and abs(entry.predicted_risk_score - entry.expected_risk_score) < 0.1:
                risk_matches += 1
            if entry.expected_decision and entry.predicted_decision == entry.expected_decision:
                decision_matches += 1
        
        metrics = {
            "overall_accuracy": correct / total if total > 0 else 0.0,
            "law_accuracy": law_matches / total if total > 0 else 0.0,
            "primitive_accuracy": primitive_matches / total if total > 0 else 0.0,
            "risk_accuracy": risk_matches / total if total > 0 else 0.0,
            "decision_accuracy": decision_matches / total if total > 0 else 0.0,
            "total_feedback_entries": total,
            "correct_classifications": correct,
        }
        
        self.accuracy_metrics = metrics
        return metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback logger statistics."""
        return {
            "total_feedback_entries": len(self.feedback_entries),
            "total_training_pairs": len(self.action_law_pairs),
            "feedback_by_type": dict(self.feedback_counts),
            "accuracy_metrics": self.get_accuracy_metrics(),
            "log_path": str(self.log_path),
            "auto_export": self.auto_export,
        }
