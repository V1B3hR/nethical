"""
Fairness Metrics Collector

Collects and computes fairness metrics including Statistical Parity,
Disparate Impact, and Equal Opportunity for protected attributes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


@dataclass
class DecisionRecord:
    """Record of a decision for fairness analysis"""
    decision: str  # "allow" or "deny"
    protected_group: str
    timestamp: datetime
    context: Dict[str, Any]


class FairnessMetricsCollector:
    """
    Fairness Metrics Collector
    
    Computes fairness metrics across protected attributes:
    - Statistical Parity: P(allow|protected) - P(allow|unprotected)
    - Disparate Impact: P(allow|protected) / P(allow|unprotected)
    - Equal Opportunity: TPR(protected) - TPR(unprotected)
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        window_hours: int = 24,
    ):
        """
        Initialize fairness metrics collector.
        
        Args:
            protected_attributes: List of protected attributes to monitor
            window_hours: Time window for metrics computation
        """
        self.protected_attributes = protected_attributes
        self.window_hours = window_hours
        self._decisions: List[DecisionRecord] = []
        self._max_decisions = 100000
    
    def record_decision(
        self,
        decision: str,
        protected_group: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a decision for fairness analysis.
        
        Args:
            decision: Decision outcome ("allow" or "deny")
            protected_group: Protected group identifier (or None)
            context: Additional context
        """
        record = DecisionRecord(
            decision=decision,
            protected_group=protected_group or "unprotected",
            timestamp=datetime.utcnow(),
            context=context or {},
        )
        
        self._decisions.append(record)
        
        # Trim old decisions
        if len(self._decisions) > self._max_decisions:
            self._decisions.pop(0)
    
    def get_statistical_parity(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Statistical Parity metric.
        
        Formula: P(decision=allow|protected) - P(decision=allow|unprotected)
        Threshold: |difference| <= 0.10 (healthy)
        
        Args:
            attribute: Specific attribute to analyze (None = all)
        
        Returns:
            Statistical parity metrics
        """
        recent = self._get_recent_decisions()
        
        if not recent:
            return {
                "difference": 0.0,
                "protected_rate": 0.0,
                "unprotected_rate": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }
        
        # Group by protected status
        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]
        
        # Calculate approval rates
        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")
        
        protected_rate = protected_allows / len(protected) if protected else 0.0
        unprotected_rate = unprotected_allows / len(unprotected) if unprotected else 0.0
        
        difference = protected_rate - unprotected_rate
        
        # Determine status
        abs_diff = abs(difference)
        if abs_diff <= 0.10:
            status = "healthy"
        elif abs_diff <= 0.20:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "difference": difference,
            "protected_rate": protected_rate,
            "unprotected_rate": unprotected_rate,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold": 0.10,
        }
    
    def get_disparate_impact(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Disparate Impact Ratio.
        
        Formula: P(decision=allow|protected) / P(decision=allow|unprotected)
        Threshold: 0.80 <= ratio <= 1.25 (healthy)
        
        Args:
            attribute: Specific attribute to analyze
        
        Returns:
            Disparate impact metrics
        """
        recent = self._get_recent_decisions()
        
        if not recent:
            return {
                "ratio": 1.0,
                "protected_rate": 0.0,
                "unprotected_rate": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }
        
        # Group by protected status
        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]
        
        # Calculate approval rates
        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")
        
        protected_rate = protected_allows / len(protected) if protected else 0.0
        unprotected_rate = unprotected_allows / len(unprotected) if unprotected else 1.0
        
        # Calculate ratio (avoid division by zero)
        ratio = protected_rate / unprotected_rate if unprotected_rate > 0 else 0.0
        
        # Determine status
        if 0.80 <= ratio <= 1.25:
            status = "healthy"
        elif 0.70 <= ratio <= 1.40:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "ratio": ratio,
            "protected_rate": protected_rate,
            "unprotected_rate": unprotected_rate,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold_min": 0.80,
            "threshold_max": 1.25,
        }
    
    def get_equal_opportunity(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Equal Opportunity metric.
        
        Formula: TPR(protected) - TPR(unprotected)
        Note: Requires ground truth labels for true positive rate
        
        Args:
            attribute: Specific attribute to analyze
        
        Returns:
            Equal opportunity metrics
        """
        recent = self._get_recent_decisions()
        
        if not recent:
            return {
                "difference": 0.0,
                "protected_tpr": 0.0,
                "unprotected_tpr": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }
        
        # For now, use approval rate as proxy for TPR
        # In production, would use actual ground truth labels
        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]
        
        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")
        
        protected_tpr = protected_allows / len(protected) if protected else 0.0
        unprotected_tpr = unprotected_allows / len(unprotected) if unprotected else 0.0
        
        difference = protected_tpr - unprotected_tpr
        
        # Determine status
        abs_diff = abs(difference)
        if abs_diff <= 0.10:
            status = "healthy"
        elif abs_diff <= 0.20:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "difference": difference,
            "protected_tpr": protected_tpr,
            "unprotected_tpr": unprotected_tpr,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold": 0.10,
            "note": "Using approval rate as TPR proxy",
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all fairness metrics"""
        sp = self.get_statistical_parity()
        di = self.get_disparate_impact()
        eo = self.get_equal_opportunity()
        
        # Determine overall status
        statuses = [sp["status"], di["status"], eo["status"]]
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "overall_status": overall_status,
            "statistical_parity": sp,
            "disparate_impact": di,
            "equal_opportunity": eo,
            "timestamp": datetime.utcnow().isoformat(),
            "protected_attributes": self.protected_attributes,
        }
    
    def _get_recent_decisions(self) -> List[DecisionRecord]:
        """Get decisions within time window"""
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        return [d for d in self._decisions if d.timestamp > cutoff]
    
    def get_by_attribute(self, attribute: str) -> Dict[str, Any]:
        """Get fairness metrics for specific protected attribute"""
        # Placeholder - would filter by specific attribute
        return {
            "attribute": attribute,
            "statistical_parity": self.get_statistical_parity(attribute),
            "disparate_impact": self.get_disparate_impact(attribute),
            "equal_opportunity": self.get_equal_opportunity(attribute),
        }
