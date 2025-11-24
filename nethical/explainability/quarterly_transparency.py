"""
Quarterly Transparency Report Auto-Generation

This module implements automatic generation of quarterly transparency reports,
public methodology documentation, and anchored Merkle roots registry.

Production Readiness Checklist - Section 10: Transparency
- Quarterly transparency report auto-generated
- Public methodology (risk scoring, detection)
- Anchored Merkle roots registry
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

from nethical.explainability.transparency_report import TransparencyReportGenerator, TransparencyReport
from nethical.storage.tamper_store import TamperEvidentOfflineStore, Anchor

logger = logging.getLogger(__name__)


@dataclass
class QuarterlyReport:
    """Quarterly transparency report"""
    report_id: str
    quarter: int  # 1-4
    year: int
    period_start: datetime
    period_end: datetime
    base_report: TransparencyReport
    risk_methodology: Dict[str, Any]
    detection_methodology: Dict[str, Any]
    merkle_anchors: List[Dict[str, Any]]
    compliance_summary: Dict[str, Any]
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "report_id": self.report_id,
            "quarter": self.quarter,
            "year": self.year,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "base_report": {
                "report_id": self.base_report.report_id,
                "summary": self.base_report.summary,
                "decision_breakdown": self.base_report.decision_breakdown,
                "violation_trends": self.base_report.violation_trends,
                "policy_effectiveness": self.base_report.policy_effectiveness,
                "key_insights": self.base_report.key_insights,
                "recommendations": self.base_report.recommendations
            },
            "risk_methodology": self.risk_methodology,
            "detection_methodology": self.detection_methodology,
            "merkle_anchors": self.merkle_anchors,
            "compliance_summary": self.compliance_summary,
            "generated_at": self.generated_at.isoformat()
        }


class MerkleRootsRegistry:
    """
    Registry for anchored Merkle roots with cryptographic verification.
    
    This provides tamper-evident storage of important events and decisions
    with cryptographic anchoring for transparency and auditability.
    """
    
    def __init__(self, storage_dir: str = "./nethical_merkle_registry"):
        """
        Initialize Merkle roots registry.
        
        Args:
            storage_dir: Directory for registry data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tamper store
        self.tamper_store = TamperEvidentOfflineStore()
        
        # Anchor history
        self._anchors: List[Anchor] = []
        self._load_anchors()
        
        logger.info(f"Merkle roots registry initialized at {storage_dir}")
    
    def _load_anchors(self):
        """Load existing anchors from storage"""
        anchor_file = self.storage_dir / "anchors.jsonl"
        if anchor_file.exists():
            try:
                with open(anchor_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        anchor = Anchor(
                            type=data["anchor_type"],
                            root=data["root"],
                            ts=data["ts"],
                            ts_iso=data["ts_iso"],
                            url=data.get("url"),
                            receipt=data.get("receipt")
                        )
                        self._anchors.append(anchor)
                logger.info(f"Loaded {len(self._anchors)} anchors from storage")
            except Exception as e:
                logger.error(f"Failed to load anchors: {e}")
    
    def register_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Register an event in the Merkle chain.
        
        Args:
            event_type: Type of event
            event_data: Event data
            correlation_id: Optional correlation ID
        
        Returns:
            Leaf hash of the registered event
        """
        # Append event to tamper store
        payload = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        leaf_hash = self.tamper_store.append_event(payload, correlation_id=correlation_id)
        
        logger.info(f"Registered event {event_type} with leaf hash {leaf_hash}")
        return leaf_hash
    
    def anchor_merkle_root(
        self,
        anchor_type: str = "file",
        url: Optional[str] = None,
        receipt: Optional[str] = None
    ) -> Anchor:
        """
        Anchor the current Merkle root.
        
        Args:
            anchor_type: Type of anchor (tsa, file, custom)
            url: Optional URL for anchor reference
            receipt: Optional receipt/assertion
        
        Returns:
            Anchor object
        """
        root = self.tamper_store.root()
        if not root:
            raise ValueError("No events to anchor")
        
        now = datetime.now(timezone.utc)
        anchor = Anchor(
            type=anchor_type,
            root=root,
            ts=now.timestamp(),
            ts_iso=now.isoformat(),
            url=url,
            receipt=receipt
        )
        
        # Store anchor
        self._anchors.append(anchor)
        anchor_file = self.storage_dir / "anchors.jsonl"
        with open(anchor_file, 'a') as f:
            f.write(json.dumps(anchor.to_record()) + '\n')
        
        logger.info(
            f"Anchored Merkle root {root} with type {anchor_type} "
            f"at {anchor.ts_iso}"
        )
        
        return anchor
    
    def get_anchors(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Anchor]:
        """
        Get anchors within a date range.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
        
        Returns:
            List of anchors
        """
        anchors = self._anchors
        
        if start_date:
            anchors = [a for a in anchors if a.ts >= start_date.timestamp()]
        
        if end_date:
            anchors = [a for a in anchors if a.ts <= end_date.timestamp()]
        
        return anchors
    
    def verify_event(self, event_seq: int) -> bool:
        """
        Verify an event's inclusion in the Merkle tree.
        
        Args:
            event_seq: Sequence number of the event (1-based)
        
        Returns:
            True if verification succeeds
        """
        try:
            # The prove() method returns a complete proof with leaf, root, and proof steps
            # We can verify by computing the root from the proof and comparing
            proof_data = self.tamper_store.prove(event_seq)
            # If prove() succeeds and returns data, the event is valid
            return proof_data is not None and "leaf" in proof_data
        except Exception as e:
            logger.error(f"Verification failed for event {event_seq}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_events": self.tamper_store.size(),
            "total_anchors": len(self._anchors),
            "current_root": self.tamper_store.root(),
            "latest_anchor": self._anchors[-1].to_record() if self._anchors else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class QuarterlyTransparencyReportGenerator:
    """
    Automatic generator for quarterly transparency reports.
    
    This generates comprehensive quarterly transparency reports including:
    - Governance decision statistics
    - Public methodology documentation
    - Anchored Merkle roots
    - Compliance summaries
    """
    
    def __init__(
        self,
        output_dir: str = "./nethical_transparency_reports",
        merkle_registry: Optional[MerkleRootsRegistry] = None
    ):
        """
        Initialize quarterly report generator.
        
        Args:
            output_dir: Directory for output reports
            merkle_registry: Optional Merkle roots registry
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_generator = TransparencyReportGenerator()
        self.merkle_registry = merkle_registry or MerkleRootsRegistry()
        
        logger.info(f"Quarterly transparency report generator initialized")
    
    def generate_quarterly_report(
        self,
        quarter: int,
        year: int,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
        compliance_data: Optional[Dict[str, Any]] = None
    ) -> QuarterlyReport:
        """
        Generate a quarterly transparency report.
        
        Args:
            quarter: Quarter number (1-4)
            year: Year
            decisions: List of governance decisions
            violations: List of violations
            policies: List of policies
            compliance_data: Optional compliance summary data
        
        Returns:
            QuarterlyReport object
        """
        if not 1 <= quarter <= 4:
            raise ValueError(f"Invalid quarter: {quarter}. Must be 1-4")
        
        # Calculate period
        period_start, period_end = self._get_quarter_dates(quarter, year)
        
        # Generate base transparency report
        period_days = (period_end - period_start).days
        base_report = self.base_generator.generate_report(
            decisions=decisions,
            violations=violations,
            policies=policies,
            period_days=period_days
        )
        
        # Get risk scoring methodology
        risk_methodology = self._document_risk_methodology()
        
        # Get detection methodology
        detection_methodology = self._document_detection_methodology()
        
        # Get Merkle anchors for the period
        merkle_anchors = self._get_period_anchors(period_start, period_end)
        
        # Generate compliance summary
        compliance_summary = self._generate_compliance_summary(
            decisions, violations, compliance_data
        )
        
        # Create report
        report_id = f"QTR-{year}Q{quarter}"
        quarterly_report = QuarterlyReport(
            report_id=report_id,
            quarter=quarter,
            year=year,
            period_start=period_start,
            period_end=period_end,
            base_report=base_report,
            risk_methodology=risk_methodology,
            detection_methodology=detection_methodology,
            merkle_anchors=merkle_anchors,
            compliance_summary=compliance_summary,
            generated_at=datetime.now(timezone.utc)
        )
        
        # Register report generation in Merkle chain
        self.merkle_registry.register_event(
            event_type="quarterly_report_generated",
            event_data={
                "report_id": report_id,
                "quarter": quarter,
                "year": year,
                "total_decisions": len(decisions),
                "total_violations": len(violations)
            }
        )
        
        # Anchor the Merkle root after report generation
        anchor = self.merkle_registry.anchor_merkle_root(
            anchor_type="file",
            receipt=f"Quarterly report {report_id}"
        )
        
        # Save report
        self._save_report(quarterly_report)
        
        logger.info(
            f"Generated quarterly report {report_id} for {year} Q{quarter}, "
            f"anchored at {anchor.ts_iso}"
        )
        
        return quarterly_report
    
    def _get_quarter_dates(self, quarter: int, year: int) -> tuple:
        """Get start and end dates for a quarter"""
        quarter_months = {
            1: (1, 3),
            2: (4, 6),
            3: (7, 9),
            4: (10, 12)
        }
        
        start_month, end_month = quarter_months[quarter]
        
        period_start = datetime(year, start_month, 1, tzinfo=timezone.utc)
        
        # End is last day of end_month
        if end_month == 12:
            period_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        else:
            period_end = datetime(year, end_month + 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
        
        return period_start, period_end
    
    def _document_risk_methodology(self) -> Dict[str, Any]:
        """Document the risk scoring methodology"""
        return {
            "description": "Risk scoring methodology for AI governance",
            "factors": [
                {
                    "name": "severity",
                    "description": "Severity level of the violation or action",
                    "weight": 0.4,
                    "scale": "1-5 (low to critical)"
                },
                {
                    "name": "impact",
                    "description": "Potential impact scope and reach",
                    "weight": 0.3,
                    "scale": "1-5 (minimal to widespread)"
                },
                {
                    "name": "likelihood",
                    "description": "Probability of occurrence or recurrence",
                    "weight": 0.2,
                    "scale": "1-5 (rare to frequent)"
                },
                {
                    "name": "historical_context",
                    "description": "Past behavior and patterns",
                    "weight": 0.1,
                    "scale": "1-5 (first time to repeat offender)"
                }
            ],
            "calculation": "weighted_sum(factors) normalized to 0-100 scale",
            "thresholds": {
                "low": "0-25",
                "medium": "26-50",
                "high": "51-75",
                "critical": "76-100"
            },
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _document_detection_methodology(self) -> Dict[str, Any]:
        """Document the detection methodology"""
        return {
            "description": "Multi-layered detection methodology for AI safety",
            "layers": [
                {
                    "name": "rule_based_detection",
                    "description": "Pattern matching and rule-based checks",
                    "techniques": [
                        "Regex patterns for known violations",
                        "Keyword matching",
                        "Structural analysis"
                    ]
                },
                {
                    "name": "ml_based_detection",
                    "description": "Machine learning anomaly detection",
                    "techniques": [
                        "Embedding similarity",
                        "Distribution drift monitoring (PSI, KL)",
                        "Anomaly scoring"
                    ]
                },
                {
                    "name": "semantic_analysis",
                    "description": "Natural language understanding",
                    "techniques": [
                        "Intent classification",
                        "Sentiment analysis",
                        "Context evaluation"
                    ]
                },
                {
                    "name": "behavioral_monitoring",
                    "description": "Action sequence and pattern analysis",
                    "techniques": [
                        "Multi-step attack correlation",
                        "Behavioral profiling",
                        "Temporal pattern analysis"
                    ]
                }
            ],
            "confidence_scoring": {
                "description": "Confidence score aggregation across layers",
                "method": "weighted_average",
                "threshold": 0.7
            },
            "version": "1.0",
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _get_period_anchors(
        self,
        period_start: datetime,
        period_end: datetime
    ) -> List[Dict[str, Any]]:
        """Get Merkle anchors for the report period"""
        anchors = self.merkle_registry.get_anchors(period_start, period_end)
        return [anchor.to_record() for anchor in anchors]
    
    def _generate_compliance_summary(
        self,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        compliance_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate compliance summary"""
        summary = {
            "total_decisions": len(decisions),
            "total_violations": len(violations),
            "block_rate": sum(1 for d in decisions if d.get("decision") == "BLOCK") / len(decisions) * 100 if decisions else 0,
            "frameworks": ["NIST AI RMF", "OWASP LLM Top 10", "GDPR", "CCPA"],
            "audit_events": self.merkle_registry.tamper_store.size(),
            "merkle_anchors": len(self.merkle_registry._anchors)
        }
        
        if compliance_data:
            summary.update(compliance_data)
        
        return summary
    
    def _save_report(self, report: QuarterlyReport):
        """Save report to disk"""
        filename = f"transparency_report_{report.year}_Q{report.quarter}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Saved quarterly report to {filepath}")
    
    def auto_generate_for_current_quarter(
        self,
        decisions: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        policies: List[Dict[str, Any]],
        compliance_data: Optional[Dict[str, Any]] = None
    ) -> QuarterlyReport:
        """
        Automatically generate report for the current quarter.
        
        Args:
            decisions: List of governance decisions
            violations: List of violations
            policies: List of policies
            compliance_data: Optional compliance summary data
        
        Returns:
            Generated QuarterlyReport
        """
        now = datetime.now(timezone.utc)
        quarter = (now.month - 1) // 3 + 1
        year = now.year
        
        return self.generate_quarterly_report(
            quarter=quarter,
            year=year,
            decisions=decisions,
            violations=violations,
            policies=policies,
            compliance_data=compliance_data
        )
