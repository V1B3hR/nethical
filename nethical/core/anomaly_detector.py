"""Phase 7: Anomaly & Drift Detection Implementation.

This module implements:
- Sequence anomaly scoring (n-gram based, sliding-window + smoothed rarity)
- Burst frequency anomaly detection (rate spikes in time windows)
- Distribution shift detection (PSI / KL divergence / Jensen–Shannon distance)
- Alert pipeline for drift events with severity mapping and cooldowns
- Behavioral anomaly detection for unusual agent patterns (entropy/concentration)
- Statistical drift monitoring to detect changes from baseline (per cohort)
- Automated alerts when drift exceeds thresholds
- Persistent alert logging (JSONL)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Deque
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math
import json
import uuid


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    SEQUENCE = "sequence"        # Unusual action sequences
    FREQUENCY = "frequency"      # Unusual frequency patterns (bursts)
    BEHAVIORAL = "behavioral"    # Unusual agent behavior (e.g., low entropy)
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
            'timestamp': (self.timestamp.replace(microsecond=0).isoformat() + 'Z') if self.timestamp else None,
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'agent_id': self.agent_id,
            'cohort': self.cohort,
            'anomaly_score': float(self.anomaly_score),
            'threshold': float(self.threshold),
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
    kl_divergence: float = 0.0  # Kullback-Leibler divergence (current || baseline)
    js_distance: float = 0.0  # Jensen–Shannon distance (symmetric, bounded [0,1])

    # Drift flags
    psi_drift_detected: bool = False
    kl_drift_detected: bool = False
    js_drift_detected: bool = False

    # Thresholds
    psi_threshold: float = 0.2
    kl_threshold: float = 0.1
    js_threshold: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'psi_score': self.psi_score,
            'kl_divergence': self.kl_divergence,
            'js_distance': self.js_distance,
            'psi_drift_detected': self.psi_drift_detected,
            'kl_drift_detected': self.kl_drift_detected,
            'js_drift_detected': self.js_drift_detected,
            'psi_threshold': self.psi_threshold,
            'kl_threshold': self.kl_threshold,
            'js_threshold': self.js_threshold,
            'drift_detected': any([self.psi_drift_detected, self.kl_drift_detected, self.js_drift_detected])
        }


class SequenceAnomalyDetector:
    """N-gram based sequence anomaly detection with sliding window and smoothed rarity scoring."""

    def __init__(
        self,
        n: int = 3,
        min_frequency: int = 2,
        anomaly_threshold: float = 0.8,
        window_size: int = 10000,
        smoothing_alpha: float = 1.0
    ):
        """Initialize sequence anomaly detector.

        Args:
            n: N-gram size (default: 3)
            min_frequency: Minimum frequency to be considered normal (default: 2)
            anomaly_threshold: Threshold for anomaly score (0..1) (default: 0.8)
            window_size: Sliding window size of recent n-grams to track
            smoothing_alpha: Additive smoothing constant for probability estimation
        """
        self.n = n
        self.min_frequency = min_frequency
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size
        self.alpha = smoothing_alpha

        # N-gram frequency tracking within a sliding window
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.ngram_window: Deque[Tuple[str, ...]] = deque()  # we manually manage length to decrement counts
        self.total_ngrams: int = 0

        # Sequence history per agent
        self.agent_sequences: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=200))

    def record_action(self, agent_id: str, action_type: str) -> None:
        """Record an action in agent's sequence, update n-grams and sliding window."""
        self.agent_sequences[agent_id].append(action_type)

        # Update n-gram counts if we have enough history
        sequence = self.agent_sequences[agent_id]
        if len(sequence) >= self.n:
            ngram = tuple(list(sequence)[-self.n:])

            # Manage sliding window counts
            # Pop oldest if exceeding window size
            if self.total_ngrams >= self.window_size and self.ngram_window:
                old = self.ngram_window.popleft()
                self.ngram_counts[old] -= 1
                if self.ngram_counts[old] <= 0:
                    del self.ngram_counts[old]
                self.total_ngrams -= 1

            self.ngram_window.append(ngram)
            self.ngram_counts[ngram] += 1
            self.total_ngrams += 1

    def _probability(self, ngram: Tuple[str, ...]) -> float:
        """Compute smoothed probability of an n-gram within the sliding window."""
        vocab_size = max(1, len(self.ngram_counts))
        count = self.ngram_counts.get(ngram, 0)
        return (count + self.alpha) / (self.total_ngrams + self.alpha * vocab_size)

    def detect_anomaly(self, agent_id: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect sequence anomaly for agent.

        Returns:
            Tuple of (is_anomalous, anomaly_score [0..1], evidence)
        """
        sequence = self.agent_sequences[agent_id]

        if len(sequence) < self.n or self.total_ngrams == 0:
            return False, 0.0, {'reason': 'insufficient_baseline_or_history'}

        recent_ngram = tuple(list(sequence)[-self.n:])
        prob = self._probability(recent_ngram)

        # Normalize -log(prob) to [0,1] using the minimal possible probability under smoothing
        vocab_size = max(1, len(self.ngram_counts)) + 1  # allow for unseen
        min_prob = self.alpha / (self.total_ngrams + self.alpha * vocab_size)
        max_log = -math.log(min_prob)
        score = min(1.0, (-math.log(prob)) / max_log if max_log > 0 else 0.0)

        # Boost score if essentially unseen or very rare
        seen_count = self.ngram_counts.get(recent_ngram, 0)
        is_unseen = seen_count < self.min_frequency
        if is_unseen:
            score = min(1.0, score + 0.1)

        evidence = {
            'ngram': list(recent_ngram),
            'probability': prob,
            'seen_count': seen_count,
            'is_unseen': is_unseen,
            'total_ngrams_window': self.total_ngrams,
            'window_size': self.window_size,
            'vocab_size': len(self.ngram_counts)
        }

        is_anomalous = score >= self.anomaly_threshold
        return is_anomalous, score, evidence

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'n': self.n,
            'total_ngrams_in_window': self.total_ngrams,
            'unique_ngrams': len(self.ngram_counts),
            'tracked_agents': len(self.agent_sequences),
            'anomaly_threshold': self.anomaly_threshold,
            'smoothing_alpha': self.alpha,
            'window_size': self.window_size
        }


class DistributionDriftDetector:
    """Statistical drift detection using PSI, KL divergence, and Jensen–Shannon distance."""

    def __init__(
        self,
        num_bins: int = 10,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
        js_threshold: float = 0.1,
        min_current_samples: int = 30,
        epsilon: float = 1e-6
    ):
        """Initialize distribution drift detector.

        Args:
            num_bins: Number of bins for discretization (default: 10)
            psi_threshold: PSI threshold for drift alert (default: 0.2)
            kl_threshold: KL divergence threshold (default: 0.1)
            js_threshold: JSD threshold (default: 0.1)
            min_current_samples: Minimum current scores required to perform drift detection
            epsilon: Smoothing value to avoid zero probabilities
        """
        self.num_bins = max(2, num_bins)
        self.psi_threshold = psi_threshold
        self.kl_threshold = kl_threshold
        self.js_threshold = js_threshold
        self.min_current_samples = min_current_samples
        self.epsilon = epsilon

        # Baseline distribution (risk scores)
        self.baseline_scores: List[float] = []
        self.baseline_distribution: Optional[List[float]] = None  # length num_bins

        # Current distribution
        self.current_scores: List[float] = []

        # Drift history
        self.drift_history: List[Tuple[datetime, DriftMetrics]] = []

    @staticmethod
    def _clip01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def set_baseline(self, scores: List[float]) -> None:
        """Set baseline distribution from risk scores (values expected in [0,1])."""
        self.baseline_scores = [self._clip01(s) for s in scores]
        self.baseline_distribution = self._compute_distribution(self.baseline_scores)

    def add_score(self, score: float) -> None:
        """Add current score for drift tracking (value expected in [0,1])."""
        self.current_scores.append(self._clip01(score))

    def detect_drift(self) -> DriftMetrics:
        """Detect distribution drift between baseline and current."""
        if not self.baseline_distribution or len(self.current_scores) < self.min_current_samples:
            # Need baseline and sufficient current data
            return DriftMetrics(
                psi_threshold=self.psi_threshold,
                kl_threshold=self.kl_threshold,
                js_threshold=self.js_threshold
            )

        # Compute current distribution
        current_distribution = self._compute_distribution(self.current_scores)

        # Calculate metrics
        psi = self._calculate_psi(self.baseline_distribution, current_distribution)
        kl = self._calculate_kl_divergence(self.baseline_distribution, current_distribution)
        js = self._calculate_js_distance(self.baseline_distribution, current_distribution)

        # Create metrics
        metrics = DriftMetrics(
            psi_score=psi,
            kl_divergence=kl,
            js_distance=js,
            psi_drift_detected=psi > self.psi_threshold,
            kl_drift_detected=kl > self.kl_threshold,
            js_drift_detected=js > self.js_threshold,
            psi_threshold=self.psi_threshold,
            kl_threshold=self.kl_threshold,
            js_threshold=self.js_threshold
        )

        # Record in history
        self.drift_history.append((datetime.utcnow(), metrics))
        # Reset current buffer after detection to avoid over-accumulation
        self.reset_current()

        return metrics

    def _compute_distribution(self, scores: List[float]) -> List[float]:
        """Compute binned distribution with smoothing over fixed bins [0..num_bins-1]."""
        counts = [0] * self.num_bins
        if scores:
            for s in scores:
                # Map [0,1] to 0..num_bins-1
                bin_idx = min(int(s * self.num_bins), self.num_bins - 1)
                counts[bin_idx] += 1

        # Add epsilon smoothing to all bins and normalize
        smoothed = [c + self.epsilon for c in counts]
        total = sum(smoothed)
        return [c / total for c in smoothed]

    def _calculate_psi(self, baseline: List[float], current: List[float]) -> float:
        """Calculate PSI: Σ (cur - base) * ln(cur/base)."""
        psi = 0.0
        for b, c in zip(baseline, current):
            # Values are already smoothed and non-zero
            psi += (c - b) * math.log(c / b)
        return psi

    def _calculate_kl_divergence(self, baseline: List[float], current: List[float]) -> float:
        """Calculate KL(current || baseline): Σ current * log(current/baseline)."""
        kl = 0.0
        for b, c in zip(baseline, current):
            kl += c * math.log(c / b)
        return kl

    def _calculate_js_distance(self, baseline: List[float], current: List[float]) -> float:
        """Calculate Jensen–Shannon distance: sqrt(0.5*KL(P||M)+0.5*KL(Q||M))."""
        m = [(b + c) / 2.0 for b, c in zip(baseline, current)]
        # KL(P||M) + KL(Q||M)
        kl_pm = sum(c * math.log(c / m_i) for c, m_i in zip(current, m))
        kl_qm = sum(b * math.log(b / m_i) for b, m_i in zip(baseline, m))
        js_div = 0.5 * (kl_pm + kl_qm)
        # Convert divergence to distance (bounded [0,1] if log base 2; with natural log it's scaled)
        # To keep consistent scale in [0,1], we can map using sqrt and cap at 1
        distance = math.sqrt(max(0.0, js_div))
        return min(distance, 1.0)

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
            'kl_threshold': self.kl_threshold,
            'js_threshold': self.js_threshold
        }


class AnomalyDriftMonitor:
    """Integrated anomaly and drift detection system with per-cohort drift tracking and alert cooldowns."""

    def __init__(
        self,
        sequence_n: int = 3,
        psi_threshold: float = 0.2,
        kl_threshold: float = 0.1,
        js_threshold: float = 0.1,
        storage_path: Optional[str] = None,
        # Frequency burst detection config
        frequency_window_seconds: int = 60,
        frequency_burst_threshold: int = 30,
        # Alerting
        alert_cooldown_seconds: int = 300
    ):
        """Initialize anomaly and drift monitor.

        Args:
            sequence_n: N-gram size for sequence detection
            psi_threshold: PSI threshold for drift
            kl_threshold: KL divergence threshold
            js_threshold: JSD threshold
            storage_path: Path for storing alerts (JSONL)
            frequency_window_seconds: Sliding time window for frequency burst detection
            frequency_burst_threshold: Max actions per window before alert
            alert_cooldown_seconds: Cooldown to deduplicate alerts of same type/agent/cohort
        """
        self.storage_path = storage_path

        # Detectors
        self.sequence_detector = SequenceAnomalyDetector(n=sequence_n)
        # Default drift detector (for None cohort)
        self.drift_detector = DistributionDriftDetector(
            psi_threshold=psi_threshold,
            kl_threshold=kl_threshold,
            js_threshold=js_threshold
        )
        # Per-cohort drift detectors
        self._drift_detectors: Dict[Optional[str], DistributionDriftDetector] = {None: self.drift_detector}

        # Alert tracking
        self.alerts: List[AnomalyAlert] = []
        self._last_alert_at: Dict[Tuple[AnomalyType, Optional[str], Optional[str]], datetime] = {}
        self._alert_cooldown = timedelta(seconds=alert_cooldown_seconds)

        # Alert thresholds by type (score thresholds 0..1)
        self.alert_thresholds = {
            AnomalyType.SEQUENCE: 0.8,
            AnomalyType.FREQUENCY: 0.75,
            AnomalyType.BEHAVIORAL: 0.7,
            AnomalyType.DISTRIBUTIONAL: 0.5
        }

        # Behavioral tracking
        self.agent_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.agent_last_seen: Dict[str, datetime] = {}

        # Frequency burst tracking (timestamps of recent actions)
        self.frequency_window = timedelta(seconds=frequency_window_seconds)
        self.frequency_burst_threshold = max(1, frequency_burst_threshold)
        self.agent_event_times: Dict[str, Deque[datetime]] = defaultdict(deque)

    def _get_drift_detector(self, cohort: Optional[str]) -> DistributionDriftDetector:
        """Get or create a drift detector for a cohort."""
        if cohort not in self._drift_detectors:
            base = self._drift_detectors[None]
            self._drift_detectors[cohort] = DistributionDriftDetector(
                num_bins=base.num_bins,
                psi_threshold=base.psi_threshold,
                kl_threshold=base.kl_threshold,
                js_threshold=base.js_threshold,
                min_current_samples=base.min_current_samples,
                epsilon=base.epsilon
            )
        return self._drift_detectors[cohort]

    def record_action(
        self,
        agent_id: str,
        action_type: str,
        risk_score: float,
        cohort: Optional[str] = None
    ) -> Optional[AnomalyAlert]:
        """Record an action and check for anomalies.

        Returns:
            Alert if anomaly detected, None otherwise
        """
        now = datetime.utcnow()

        # Update sequence detector
        self.sequence_detector.record_action(agent_id, action_type)

        # Update cohort-specific drift detector
        self._get_drift_detector(cohort).add_score(risk_score)

        # Update behavioral tracking
        self.agent_action_counts[agent_id][action_type] += 1
        self.agent_last_seen[agent_id] = now

        # Update frequency burst tracking
        q = self.agent_event_times[agent_id]
        q.append(now)
        # Drop events outside window
        cutoff = now - self.frequency_window
        while q and q[0] < cutoff:
            q.popleft()

        # 1) Frequency burst anomaly
        if len(q) >= self.frequency_burst_threshold:
            # Score grows with magnitude of the burst relative to threshold
            burst_score = min(1.0, len(q) / float(self.frequency_burst_threshold))
            evidence = {
                'window_seconds': int(self.frequency_window.total_seconds()),
                'events_in_window': len(q),
                'threshold': self.frequency_burst_threshold,
                'recent_events_iso': [t.replace(microsecond=0).isoformat() + 'Z' for t in list(q)[-5:]],
                'risk_score_last': risk_score
            }
            alert = self._create_alert(
                anomaly_type=AnomalyType.FREQUENCY,
                agent_id=agent_id,
                cohort=cohort,
                anomaly_score=burst_score,
                evidence=evidence
            )
            if alert:
                return alert

        # 2) Sequence anomaly
        is_anomalous, anomaly_score, evidence = self.sequence_detector.detect_anomaly(agent_id)
        if is_anomalous:
            alert = self._create_alert(
                anomaly_type=AnomalyType.SEQUENCE,
                agent_id=agent_id,
                cohort=cohort,
                anomaly_score=anomaly_score,
                evidence=evidence
            )
            if alert:
                return alert

        # 3) Behavioral anomaly (distribution concentration/entropy)
        behavioral_alert = self.check_behavioral_anomaly(agent_id, cohort=cohort)
        if behavioral_alert:
            return behavioral_alert

        return None

    def check_drift(self, cohort: Optional[str] = None) -> Optional[AnomalyAlert]:
        """Check for distribution drift for a specific cohort (or default)."""
        detector = self._get_drift_detector(cohort)
        drift_metrics = detector.detect_drift()

        if drift_metrics.psi_drift_detected or drift_metrics.kl_drift_detected or drift_metrics.js_drift_detected:
            # Determine severity from combined signals
            score_for_severity = max(drift_metrics.psi_score, drift_metrics.kl_divergence, drift_metrics.js_distance)
            if drift_metrics.psi_score > 0.5 or drift_metrics.kl_divergence > 0.3 or drift_metrics.js_distance > 0.25:
                severity = DriftSeverity.CRITICAL
            elif drift_metrics.psi_score > 0.3 or drift_metrics.kl_divergence > 0.15 or drift_metrics.js_distance > 0.18:
                severity = DriftSeverity.WARNING
            else:
                severity = DriftSeverity.INFO

            alert = AnomalyAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                anomaly_type=AnomalyType.DISTRIBUTIONAL,
                severity=severity,
                cohort=cohort,
                anomaly_score=score_for_severity,
                threshold=self.alert_thresholds[AnomalyType.DISTRIBUTIONAL],
                description=f"Distribution drift detected: PSI={drift_metrics.psi_score:.3f}, KL={drift_metrics.kl_divergence:.3f}, JS={drift_metrics.js_distance:.3f}",
                evidence=drift_metrics.to_dict(),
                auto_escalated=severity == DriftSeverity.CRITICAL,
                quarantine_recommended=severity == DriftSeverity.CRITICAL
            )

            if not self._is_in_cooldown(AnomalyType.DISTRIBUTIONAL, None, cohort):
                self._record_alert(alert)
                return alert

        return None

    def check_behavioral_anomaly(
        self,
        agent_id: str,
        cohort: Optional[str] = None
    ) -> Optional[AnomalyAlert]:
        """Check for behavioral anomalies in agent activity."""
        if agent_id not in self.agent_action_counts:
            return None

        action_counts = self.agent_action_counts[agent_id]
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return None

        # Concentration (max share) and entropy
        max_count = max(action_counts.values())
        concentration = max_count / total_actions
        # Shannon entropy in nats normalized by log(k)
        k = len(action_counts)
        probs = [c / total_actions for c in action_counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        norm_entropy = entropy / math.log(k) if k > 1 else 0.0  # [0..1]

        # Anomaly if > 80% of actions are same type with sufficient volume or very low entropy
        if (concentration > 0.8 and total_actions >= 10) or (norm_entropy < 0.2 and total_actions >= 10):
            # Score blends both signals
            score = max(concentration, 1.0 - norm_entropy)
            evidence = {
                'total_actions': total_actions,
                'action_concentration': concentration,
                'dominant_action': max(action_counts, key=action_counts.get),
                'normalized_entropy': norm_entropy,
                'action_distribution': dict(action_counts)
            }

            alert = self._create_alert(
                anomaly_type=AnomalyType.BEHAVIORAL,
                agent_id=agent_id,
                cohort=cohort,
                anomaly_score=score,
                evidence=evidence
            )
            return alert

        return None

    def _is_in_cooldown(self, anomaly_type: AnomalyType, agent_id: Optional[str], cohort: Optional[str]) -> bool:
        key = (anomaly_type, agent_id, cohort)
        last = self._last_alert_at.get(key)
        if last and (datetime.utcnow() - last) < self._alert_cooldown:
            return True
        return False

    def _record_alert(self, alert: AnomalyAlert) -> None:
        """Record and persist an alert."""
        key = (alert.anomaly_type, alert.agent_id, alert.cohort)
        self._last_alert_at[key] = alert.timestamp
        self.alerts.append(alert)
        if self.storage_path:
            self._persist_alert(alert)

    def _create_alert(
        self,
        anomaly_type: AnomalyType,
        agent_id: Optional[str],
        cohort: Optional[str],
        anomaly_score: float,
        evidence: Dict[str, Any]
    ) -> Optional[AnomalyAlert]:
        """Create and log an anomaly alert with cooldown deduplication."""
        threshold = self.alert_thresholds[anomaly_type]

        # Deduplicate via cooldown
        if self._is_in_cooldown(anomaly_type, agent_id, cohort):
            return None

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
            AnomalyType.FREQUENCY: f"Frequency burst detected (score: {anomaly_score:.3f})"
        }

        alert = AnomalyAlert(
            alert_id=str(uuid.uuid4()),
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

        self._record_alert(alert)
        return alert

    def _persist_alert(self, alert: AnomalyAlert) -> None:
        """Persist alert to storage as JSONL."""
        if not self.storage_path:
            return

        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)

            # Append to alerts log
            log_file = os.path.join(self.storage_path, "anomaly_alerts.jsonl")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + '\n')
        except Exception:
            # Silent fail for logging
            pass

    def set_baseline_distribution(self, scores: List[float], cohort: Optional[str] = None) -> None:
        """Set baseline distribution for drift detection (optionally per cohort)."""
        self._get_drift_detector(cohort).set_baseline(scores)

    def get_alerts(
        self,
        severity: Optional[DriftSeverity] = None,
        anomaly_type: Optional[AnomalyType] = None,
        limit: Optional[int] = None
    ) -> List[AnomalyAlert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if anomaly_type:
            alerts = [a for a in alerts if a.anomaly_type == anomaly_type]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        # Alert counts by severity and type
        alert_counts = {
            'total': len(self.alerts),
            'by_severity': {
                DriftSeverity.INFO.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.INFO),
                DriftSeverity.WARNING.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.WARNING),
                DriftSeverity.CRITICAL.value: sum(1 for a in self.alerts if a.severity == DriftSeverity.CRITICAL)
            },
            'by_type': {
                AnomalyType.SEQUENCE.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.SEQUENCE),
                AnomalyType.FREQUENCY.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.FREQUENCY),
                AnomalyType.BEHAVIORAL.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.BEHAVIORAL),
                AnomalyType.DISTRIBUTIONAL.value: sum(1 for a in self.alerts if a.anomaly_type == AnomalyType.DISTRIBUTIONAL)
            }
        }

        # Per-cohort drift stats
        cohorts_stats = {}
        for cohort, det in self._drift_detectors.items():
            key = cohort if cohort is not None else "__default__"
            cohorts_stats[key] = det.get_statistics()

        return {
            'alerts': alert_counts,
            'sequence_detector': self.sequence_detector.get_statistics(),
            'drift_detectors': cohorts_stats,
            'tracked_agents': len(self.agent_action_counts),
            'frequency_burst_config': {
                'window_seconds': int(self.frequency_window.total_seconds()),
                'burst_threshold': self.frequency_burst_threshold
            },
            'alert_cooldown_seconds': int(self._alert_cooldown.total_seconds())
        }

    def export_alerts(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export alerts for analysis."""
        alerts_to_export = self.alerts[-limit:] if limit else self.alerts
        return [a.to_dict() for a in alerts_to_export]
