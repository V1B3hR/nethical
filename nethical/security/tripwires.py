"""Instant tripwire checks for the Adaptive Guardian.

Layer 1: Always-active checks that catch critical threats with minimal overhead.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Deque

from .guardian_modes import TripwireSensitivity, severity_matches_sensitivity


@dataclass
class TripwireAlert:
    """Alert triggered by a tripwire."""
    
    tripwire_type: str
    severity: str
    module: str
    description: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class Tripwires:
    """Instant tripwire checks with configurable sensitivity.
    
    Tripwires are always active but their sensitivity varies by mode:
    - Hard limits always trigger (e.g., response time > 5s)
    - Other checks trigger based on current sensitivity level
    """
    
    # Hard limits that always trigger regardless of mode
    HARD_RESPONSE_TIME_LIMIT_MS = 5000.0
    HARD_ERROR_RATE_THRESHOLD = 0.5  # 50% error rate
    
    # Soft limits that depend on mode
    SOFT_RESPONSE_TIME_SPIKE_MULTIPLIER = 10.0  # 10x baseline
    SOFT_ERROR_RATE_THRESHOLD = 0.1  # 10% error rate
    
    # Sliding window sizes
    WINDOW_SIZE = 100
    WINDOW_DURATION_S = 300  # 5 minutes
    
    def __init__(self):
        """Initialize tripwires."""
        self._lock = Lock()
        
        # Module baselines (calculated from historical data)
        self._baselines: Dict[str, float] = {}
        
        # Sliding windows for each module
        self._response_times: Dict[str, Deque[tuple[float, float]]] = {}
        self._error_counts: Dict[str, Deque[tuple[float, bool]]] = {}
        
        # Alert tracking
        self._recent_alerts: Deque[TripwireAlert] = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            "total_checks": 0,
            "total_alerts": 0,
            "alerts_by_type": {},
            "alerts_by_module": {},
        }
    
    def check(
        self,
        module: str,
        response_time_ms: float,
        decision: str,
        error: bool,
        sensitivity: TripwireSensitivity,
    ) -> Optional[TripwireAlert]:
        """Check for tripwire violations.
        
        Args:
            module: Module name
            response_time_ms: Response time in milliseconds
            decision: Decision made (ALLOW, BLOCK, etc.)
            error: Whether an error occurred
            sensitivity: Current tripwire sensitivity
            
        Returns:
            TripwireAlert if violation detected, None otherwise
        """
        with self._lock:
            self._stats["total_checks"] += 1
            now = time.time()
            
            # Initialize module tracking if needed
            if module not in self._response_times:
                self._response_times[module] = deque(maxlen=self.WINDOW_SIZE)
                self._error_counts[module] = deque(maxlen=self.WINDOW_SIZE)
                self._baselines[module] = response_time_ms
            
            # Add to sliding windows
            self._response_times[module].append((now, response_time_ms))
            self._error_counts[module].append((now, error))
            
            # Clean old data from windows
            self._clean_windows(module, now)
            
            # Check 1: Hard response time limit (always)
            if response_time_ms > self.HARD_RESPONSE_TIME_LIMIT_MS:
                return self._create_alert(
                    tripwire_type="hard_response_time",
                    severity="CRITICAL",
                    module=module,
                    description=f"Response time exceeded hard limit",
                    metric_value=response_time_ms,
                    threshold=self.HARD_RESPONSE_TIME_LIMIT_MS,
                )
            
            # Check 2: Response time spike (10x baseline)
            baseline = self._get_baseline(module)
            if response_time_ms > baseline * self.SOFT_RESPONSE_TIME_SPIKE_MULTIPLIER:
                alert = self._create_alert(
                    tripwire_type="response_time_spike",
                    severity="HIGH",
                    module=module,
                    description=f"Response time spike detected ({response_time_ms:.2f}ms vs {baseline:.2f}ms baseline)",
                    metric_value=response_time_ms,
                    threshold=baseline * self.SOFT_RESPONSE_TIME_SPIKE_MULTIPLIER,
                )
                if severity_matches_sensitivity("HIGH", sensitivity):
                    return alert
            
            # Check 3: Error rate in sliding window
            error_rate = self._calculate_error_rate(module)
            
            # Hard error rate limit (always)
            if error_rate > self.HARD_ERROR_RATE_THRESHOLD:
                return self._create_alert(
                    tripwire_type="hard_error_rate",
                    severity="CRITICAL",
                    module=module,
                    description=f"Error rate exceeded hard limit",
                    metric_value=error_rate,
                    threshold=self.HARD_ERROR_RATE_THRESHOLD,
                )
            
            # Soft error rate limit (mode-dependent)
            if error_rate > self.SOFT_ERROR_RATE_THRESHOLD:
                alert = self._create_alert(
                    tripwire_type="soft_error_rate",
                    severity="MEDIUM",
                    module=module,
                    description=f"Elevated error rate detected",
                    metric_value=error_rate,
                    threshold=self.SOFT_ERROR_RATE_THRESHOLD,
                )
                if severity_matches_sensitivity("MEDIUM", sensitivity):
                    return alert
            
            # Check 4: Critical decisions detection
            if decision in ["BLOCK", "QUARANTINE", "ESCALATE"] and error:
                alert = self._create_alert(
                    tripwire_type="critical_decision_with_error",
                    severity="HIGH",
                    module=module,
                    description=f"Critical decision ({decision}) with error",
                    metric_value=1.0,
                    threshold=1.0,
                )
                if severity_matches_sensitivity("HIGH", sensitivity):
                    return alert
            
            return None
    
    def _clean_windows(self, module: str, now: float) -> None:
        """Remove old entries from sliding windows."""
        cutoff = now - self.WINDOW_DURATION_S
        
        # Clean response times
        while (
            self._response_times[module]
            and self._response_times[module][0][0] < cutoff
        ):
            self._response_times[module].popleft()
        
        # Clean error counts
        while (
            self._error_counts[module]
            and self._error_counts[module][0][0] < cutoff
        ):
            self._error_counts[module].popleft()
    
    def _get_baseline(self, module: str) -> float:
        """Get response time baseline for module.
        
        Uses Welford's algorithm for running mean if enough data,
        otherwise returns initial baseline.
        """
        if len(self._response_times[module]) < 10:
            return self._baselines[module]
        
        # Calculate mean of recent response times
        times = [rt for _, rt in self._response_times[module]]
        mean = sum(times) / len(times)
        
        # Update baseline (moving average)
        self._baselines[module] = 0.9 * self._baselines[module] + 0.1 * mean
        
        return self._baselines[module]
    
    def _calculate_error_rate(self, module: str) -> float:
        """Calculate error rate in sliding window."""
        if not self._error_counts[module]:
            return 0.0
        
        errors = sum(1 for _, is_error in self._error_counts[module] if is_error)
        total = len(self._error_counts[module])
        
        return errors / total if total > 0 else 0.0
    
    def _create_alert(
        self,
        tripwire_type: str,
        severity: str,
        module: str,
        description: str,
        metric_value: float,
        threshold: float,
    ) -> TripwireAlert:
        """Create and record a tripwire alert."""
        alert = TripwireAlert(
            tripwire_type=tripwire_type,
            severity=severity,
            module=module,
            description=description,
            metric_value=metric_value,
            threshold=threshold,
        )
        
        self._recent_alerts.append(alert)
        
        # Update statistics
        self._stats["total_alerts"] += 1
        self._stats["alerts_by_type"][tripwire_type] = (
            self._stats["alerts_by_type"].get(tripwire_type, 0) + 1
        )
        self._stats["alerts_by_module"][module] = (
            self._stats["alerts_by_module"].get(module, 0) + 1
        )
        
        return alert
    
    def get_recent_alerts(self, limit: int = 100) -> list[TripwireAlert]:
        """Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        with self._lock:
            return list(self._recent_alerts)[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get tripwire statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                **self._stats,
                "modules_tracked": len(self._baselines),
                "current_baselines": dict(self._baselines),
            }
    
    def reset_module(self, module: str) -> None:
        """Reset tracking data for a module.
        
        Args:
            module: Module name
        """
        with self._lock:
            if module in self._response_times:
                del self._response_times[module]
            if module in self._error_counts:
                del self._error_counts[module]
            if module in self._baselines:
                del self._baselines[module]
