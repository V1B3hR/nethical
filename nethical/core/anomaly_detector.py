"""Phase 7: Anomaly & Drift Detection Implementation.

This module implements:
- Sequence anomaly scoring (n-gram based)
- Distribution shift detection (PSI / KL divergence)
- Alert pipeline for drift events
- Behavioral anomaly detection for unusual agent patterns
- Statistical drift monitoring to detect changes from baseline
- Automated alerts when drift exceeds thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math
import json


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    SEQUENCE = "sequence"  # Unusual action sequences
    FREQUENCY = "frequency"  # Unusual frequency patterns
    BEHAVIORAL = "behavioral"  # Unusual agent behavior
    DISTRIBUTIONAL = "distributional"  # Distribution shift


class DriftSeverity(str, Enum):
    """Severity levels for drift alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly or drift."""
    alert_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: DriftSeverity
    
    # Detection details
    agent_id: Optional[str] = None
    cohort: Optional[str] = None
    anomaly_score: float = 0.0
    threshold: float = 0.0
    
    # Evidence
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Actions taken
    auto_escalated: bool = False
    quarantine_recommended: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'agent_id': self.agent_id,
            'cohort': self.cohort,
            'anomaly_score': self.anomaly_score,
            'threshold': self.threshold,
            'description': self.description,
            'evidence': self.evidence,
            'auto_escalated': self.auto_escalated,
            'quarantine_recommended': self.quarantine_recommended
        }


@dataclass
class DriftMetrics:
    """Metrics for drift detection."""
    
    # Distribution metrics
    psi_score: float = 0.0  # Population Stability Index
    kl_divergence: float = 0.0  # Kullback-Leibler divergence
    
    # Drift flags
    psi_drift_detected: bool = False
    kl_drift_detected: bool = False
    
    # Thresholds
    psi_threshold: float = 0.2
    kl_threshold: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'psi_score': self.psi_score,
            'kl_divergence': self.kl_divergence,
            'psi_drift_detected': self.psi_drift_detected,
            'kl_drift_detected': self.kl_drift_detected,
            'psi_threshold': self.psi_threshold,
            'kl_threshold': self.kl_threshold,
            'drift_detected': self.psi_drift_detected or self.kl_drift_detected
        }


class SequenceAnomalyDetector:
    """N-gram based sequence anomaly detection."""
    
    def __init__(
        self,
        n: int = 3,
        min_frequency: int = 2,
        anomaly_threshold: float = 0.8
    ):
        """Initialize sequence anomaly detector.
        
        Args:
            n: N-gram size (default: 3)
            min_frequency: Minimum frequency to be considered normal (default: 2)
            anomaly_threshold: Threshold for anomaly score (default: 0.8)
        """
        self.n = n
        self.min_frequency = min_frequency
        self.anomaly_threshold = anomaly_threshold
        
        # N-gram frequency tracking
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.total_ngrams = 0
        
        # Sequence history per agent
        self.agent_sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def record_action(self, agent_id: str, action_type: str) -> None:
        """Record an action in agent's sequence.
        
        Args:
            agent_id: Agent identifier
            action_type: Type of action taken
        """
        self.agent_sequences[agent_id].append(action_type)
        
        # Update n-gram counts if we have enough history
        sequence = list(self.agent_sequences[agent_id])
        if len(sequence) >= self.n:
            ngram = tuple(sequence[-self.n:])
            self.ngram_counts[ngram] += 1
            self.total_ngrams += 1
    
    def detect_anomaly(self, agent_id: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect sequence anomaly for agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Tuple of (is_anomalous, anomaly_score, evidence)
        """
        sequence = list(self.agent_sequences[agent_id])
        
        if len(sequence) < self.n:
            return False, 0.0, {'reason': 'insufficient_history'}
        
        # Get recent n-gram
        recent_ngram = tuple(sequence[-self.n:])
        
        # Calculate anomaly score based on rarity
        ngram_count = self.ngram_counts.get(recent_ngram, 0)
        
        if self.total_ngrams == 0:
            return False, 0.0, {'reason': 'no_baseline'}
        
        # Frequency-based anomaly score
        frequency = ngram_count / self.total_ngrams
        
        # Anomaly score: high when frequency is low
        anomaly_score = 1.0 - min(frequency * 100, 1.0)
        
        # Check if unseen n-gram
        is_unseen = ngram_count < self.min_frequency
        
        evidence = {
            'ngram': list(recent_ngram),
            'frequency': frequency,
            'seen_count': ngram_count,
            'is_unseen': is_unseen,
            'total_ngrams': self.total_ngrams
        }
        
        is_anomalous = anomaly_score >= self.anomaly_threshold
        
        return is_anomalous, anomaly_score, evidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'n': self.n,
            'total_ngrams': self.total_ngrams,
            'unique_ngrams': len(self.ngram_counts),
            'tracked_agents': len(self.agent_sequences),
            'anomaly_threshold': self.anomaly_threshold
        }


class DistributionDriftDetector:
    """Statistical drift detection using PSI and KL divergence."""
    
    def __init__(
        self,
        num_bins: int = 10,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1
    ):
        """Initialize distribution drift detector.
        
        Args:
            num_bins: Number of bins for discretization (default: 10)
            psi_threshold: PSI threshold for drift alert (default: 0.2)
            kl_threshold: KL divergence threshold (default: 0.1)
        """
        self.num_bins = num_bins
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        
        # Baseline distribution (risk scores)
        self.baseline_scores: List[float] = []
        self.baseline_distribution: Optional[Dict[int, float]] = None
        
        # Current distribution
        self.current_scores: List[float] = []
        
        # Drift history
        self.drift_history: List[Tuple[datetime, DriftMetrics]] = []
    
    def set_baseline(self, scores: List[float]) -> None:
        """Set baseline distribution from risk scores.
        
        Args:
            scores: List of risk scores (0-1)
        """
        self.baseline_scores = scores.copy()
        self.baseline_distribution = self._compute_distribution(scores)
    
    def add_score(self, score: float) -> None:
        """Add current score for drift tracking.
        
        Args:
            score: Risk score (0-1)
        """
        self.current_scores.append(score)
    
    def detect_drift(self) -> DriftMetrics:
        """Detect distribution drift between baseline and current.
        
        Returns:
            Drift metrics with PSI and KL divergence
        """
        if not self.baseline_distribution or len(self.current_scores) < 30:
            # Need baseline and sufficient current data
            return DriftMetrics()
        
        # Compute current distribution
        current_distribution = self._compute_distribution(self.current_scores)
        
        # Calculate PSI
        psi = self._calculate_psi(self.baseline_distribution, current_distribution)
        
        # Calculate KL divergence
        kl = self._calculate_kl_divergence(self.baseline_distribution, current_distribution)
        
        # Create metrics
        metrics = DriftMetrics(
            psi_score=psi,
            kl_divergence=kl,
            psi_drift_detected=psi > self.psi_threshold,
            kl_drift_detected=kl > self.kl_threshold,
            psi_threshold=self.psi_threshold,
            kl_threshold=self.kl_threshold
        )
        
        # Record in history
        self.drift_history.append((datetime.utcnow(), metrics))
        
        return metrics
    
    def _compute_distribution(self, scores: List[float]) -> Dict[int, float]:
        """Compute binned distribution from scores.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary mapping bin index to probability
        """
        if not scores:
            return {}
        
        # Create bins
        bin_counts = defaultdict(int)
        for score in scores:
            bin_idx = min(int(score * self.num_bins), self.num_bins - 1)
            bin_counts[bin_idx] += 1
        
        # Convert to probabilities
        total = len(scores)
        distribution = {
            bin_idx: count / total
            for bin_idx, count in bin_counts.items()
        }
        
        return distribution
    
    def _calculate_psi(
        self,
        baseline: Dict[int, float],
        current: Dict[int, float]
    ) -> float:
        """Calculate Population Stability Index (PSI).
        
        PSI = Σ (current% - baseline%) * ln(current% / baseline%)
        """
        psi = 0.0
        
        all_bins = set(baseline.keys()) | set(current.keys())
        
        for bin_idx in all_bins:
            baseline_pct = baseline.get(bin_idx, 0.001)  # Small value to avoid division by zero
            current_pct = current.get(bin_idx, 0.001)
            
            psi += (current_pct - baseline_pct) * math.log(current_pct / baseline_pct)
        
        return psi
    
    def _calculate_kl_divergence(
        self,
        baseline: Dict[int, float],
        current: Dict[int, float]
    ) -> float:
        """Calculate Kullback-Leibler divergence.
        
        KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        """
        kl = 0.0
        
        all_bins = set(baseline.keys()) | set(current.keys())
        
        for bin_idx in all_bins:
            p = current.get(bin_idx, 0.001)  # Current distribution
            q = baseline.get(bin_idx, 0.001)  # Baseline distribution
            
            kl += p * math.log(p / q)
        
        return kl
    
    def reset_current(self) -> None:
        """Reset current distribution tracking."""
        self.current_scores.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        recent_drift = None
        if self.drift_history:
            _, recent_drift = self.drift_history[-1]
        
        return {
            'num_bins': self.num_bins,
            'baseline_size': len(self.baseline_scores),
            'current_size': len(self.current_scores),
            'drift_checks_performed': len(self.drift_history),
            'recent_drift': recent_drift.to_dict() if recent_drift else None,
            'psi_threshold': self.psi_threshold,
            'kl_threshold': self.kl_threshold
        }


class AnomalyDriftMonitor:
    """Integrated anomaly and drift detection system."""
    
    def __init__(
        self,
        sequence_n: int = 3,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
        storage_path: Optional[str] = None
    ):
        """Initialize anomaly and drift monitor.
        
        Args:
            sequence_n: N-gram size for sequence detection
            psi_threshold: PSI threshold for drift
            kl_threshold: KL divergence threshold
            storage_path: Path for storing alerts
        """
        self.storage_path = storage_path
        
        # Detectors
        self.sequence_detector = SequenceAnomalyDetector(n=sequence_n)
        self.drift_detector = DistributionDriftDetector(
            psi_threshold=psi_threshold,
            kl_threshold=kl_threshold
        )
        
        # Alert tracking
        self.alerts: List[AnomalyAlert] = []
        
        # Alert thresholds by type
        self.alert_thresholds = {
            AnomalyType.SEQUENCE: 0.8,
            AnomalyType.FREQUENCY: 0.75,
            AnomalyType.BEHAVIORAL: 0.7,
            AnomalyType.DISTRIBUTIONAL: 0.5
        }
        
        # Behavioral tracking
        self.agent_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_last_seen: Dict[str, datetime] = {}
    
    def record_action(
        self,
        agent_id: str,
        action_type: str,
        risk_score: float,
        cohort: Optional[str] = None
    ) -> Optional[AnomalyAlert]:
        """Record an action and check for anomalies.
        
        Args:
            agent_id: Agent identifier
            action_type: Type of action
            risk_score: Risk score for the action
            cohort: Optional cohort identifier
            
        Returns:
            Alert if anomaly detected, None otherwise
        """
        # Update sequence detector
        self.sequence_detector.record_action(agent_id, action_type)
        
        # Update drift detector
        self.drift_detector.add_score(risk_score)
        
        # Update behavioral tracking
        self.agent_action_counts[agent_id][action_type] += 1
        self.agent_last_seen[agent_id] = datetime.utcnow()
        
        # Check for sequence anomaly
        is_anomalous, anomaly_score, evidence = self.sequence_detector.detect_anomaly(agent_id)
        
        if is_anomalous:
            alert = self._create_alert(
                anomaly_type=AnomalyType.SEQUENCE,
                agent_id=agent_id,
                cohort=cohort,
                anomaly_score=anomaly_score,
                evidence=evidence
            )
            return alert
        
        return None
    
    def check_drift(self, cohort: Optional[str] = None) -> Optional[AnomalyAlert]:
        """Check for distribution drift.
        
        Args:
            cohort: Optional cohort identifier
            
        Returns:
            Alert if drift detected, None otherwise
        """
        drift_metrics = self.drift_detector.detect_drift()
        
        if drift_metrics.psi_drift_detected or drift_metrics.kl_drift_detected:
            # Determine severity
            if drift_metrics.psi_score > 0.5 or drift_metrics.kl_divergence > 0.3:
                severity = DriftSeverity.CRITICAL
            elif drift_metrics.psi_score > 0.3 or drift_metrics.kl_divergence > 0.15:
                severity = DriftSeverity.WARNING
            else:
                severity = DriftSeverity.INFO
            
            alert = AnomalyAlert(
                alert_id=f"drift_{int(datetime.utcnow().timestamp() * 1000)}",
                timestamp=datetime.utcnow(),
                anomaly_type=AnomalyType.DISTRIBUTIONAL,
                severity=severity,
                cohort=cohort,
                anomaly_score=max(drift_metrics.psi_score, drift_metrics.kl_divergence),
                threshold=self.alert_thresholds[AnomalyType.DISTRIBUTIONAL],
                description=f"Distribution drift detected: PSI={drift_metrics.psi_score:.3f}, KL={drift_metrics.kl_divergence:.3f}",
                evidence=drift_metrics.to_dict(),
                auto_escalated=severity == DriftSeverity.CRITICAL,
                quarantine_recommended=severity == DriftSeverity.CRITICAL
            )
            
            self.alerts.append(alert)
            
            if self.storage_path:
                self._persist_alert(alert)
            
            return alert
        
        return None
    
    def check_behavioral_anomaly(
        self,
        agent_id: str,
        cohort: Optional[str] = None
    ) -> Optional[AnomalyAlert]:
        """Check for behavioral anomalies in agent activity.
        
        Args:
            agent_id: Agent identifier
            cohort: Optional cohort identifier
            
        Returns:
            Alert if behavioral anomaly detected, None otherwise
        """
        if agent_id not in self.agent_action_counts:
            return None
        
        action_counts = self.agent_action_counts[agent_id]
        total_actions = sum(action_counts.values())
        
        # Check for unusual action distribution
        # High concentration in one action type is suspicious
        if total_actions > 0:
            max_count = max(action_counts.values())
            concentration = max_count / total_actions
            
            # Anomaly if > 80% of actions are same type
            if concentration > 0.8 and total_actions >= 10:
                anomaly_score = concentration
                
                alert = self._create_alert(
                    anomaly_type=AnomalyType.BEHAVIORAL,
                    agent_id=agent_id,
                    cohort=cohort,
                    anomaly_score=anomaly_score,
                    evidence={
                        'total_actions': total_actions,
                        'action_concentration': concentration,
                        'dominant_action': max(action_counts, key=action_counts.get),
                        'action_distribution': dict(action_counts)
                    }
                )
                
                return alert
        
        return None
    
    def _create_alert(
        self,
        anomaly_type: AnomalyType,
        agent_id: str,
        cohort: Optional[str],
        anomaly_score: float,
        evidence: Dict[str, Any]
    ) -> AnomalyAlert:
        """Create and log an anomaly alert."""
        threshold = self.alert_thresholds[anomaly_type]
        
        # Determine severity
        if anomaly_score >= 0.9:
            severity = DriftSeverity.CRITICAL
        elif anomaly_score >= 0.75:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.INFO
        
        # Create description
        descriptions = {
            AnomalyType.SEQUENCE: f"Unusual action sequence detected (score: {anomaly_score:.3f})",
            AnomalyType.BEHAVIORAL: f"Behavioral anomaly detected (score: {anomaly_score:.3f})",
            AnomalyType.FREQUENCY: f"Frequency anomaly detected (score: {anomaly_score:.3f})"
        }
        
        alert = AnomalyAlert(
            alert_id=f"{anomaly_type.value}_{int(datetime.utcnow().timestamp() * 1000)}",
            timestamp=datetime.utcnow(),
            anomaly_type=anomaly_type,
            severity=severity,
            agent_id=agent_id,
            cohort=cohort,
            anomaly_score=anomaly_score,
            threshold=threshold,
            description=descriptions.get(anomaly_type, "Anomaly detected"),
            evidence=evidence,
            auto_escalated=severity == DriftSeverity.CRITICAL,
            quarantine_recommended=severity == DriftSeverity.CRITICAL
        )
        
        self.alerts.append(alert)
        
        if self.storage_path:
            self._persist_alert(alert)
        
        return alert
    
    def _persist_alert(self, alert: AnomalyAlert) -> None:
        """Persist alert to storage."""
        if not self.storage_path:
            return
        
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Append to alerts log
            log_file = os.path.join(self.storage_path, "anomaly_alerts.jsonl")
            with open(log_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception:
            pass  # Silent fail for logging
    
    def set_baseline_distribution(self, scores: List[float]) -> None:
        """Set baseline distribution for drift detection.
        
        Args:
            scores: List of baseline risk scores
        """
        self.drift_detector.set_baseline(scores)
    
    def get_alerts(
        self,
        severity: Optional[DriftSeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        limit: Optional[int] = None
    ) -> List[AnomalyAlert]:
        """Get alerts with optional filtering.
        
        Args:
            severity: Optional severity filter
            anomaly_type: Optional type filter
            limit: Optional limit on number of alerts
            
        Returns:
            List of alerts matching criteria
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if anomaly_type:
            alerts = [a for a in alerts if a.anomaly_type == anomaly_type]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.
        
        Returns:
            Statistics dictionary
        """
        # Alert counts by severity
        alert_counts = {
            'total': len(self.alerts),
            'by_severity': {
                DriftSeverity.INFO.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.INFO),
                DriftSeverity.WARNING.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.WARNING),
                DriftSeverity.CRITICAL.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.CRITICAL)
            },
            'by_type': {
                AnomalyType.SEQUENCE.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.SEQUENCE),
                AnomalyType.BEHAVIORAL.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.BEHAVIORAL),
                AnomalyType.DISTRIBUTIONAL.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.DISTRIBUTIONAL)
            }
        }
        
        return {
            'alerts': alert_counts,
            'sequence_detector': self.sequence_detector.get_statistics(),
            'drift_detector': self.drift_detector.get_statistics(),
            'tracked_agents': len(self.agent_action_counts)
        }
    
    def export_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export alerts for analysis.
        
        Args:
            limit: Optional limit on number of alerts
            
        Returns:
            List of alert dictionaries
        """
        alerts_to_export = self.alerts[-limit:] if limit else self.alerts
        return [a.to_dict() for a in alerts_to_export]
