"""Adaptive Guardian - Intelligent Throttling Security System.

Main implementation of the intelligent, adaptive security monitoring system
that automatically adjusts its intensity based on current threat landscape.
"""

import asyncio
import time
import threading
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable
from functools import wraps

from .guardian_modes import (
    GuardianMode,
    TripwireSensitivity,
    get_mode_config,
    ModeConfig,
)
from .tripwires import Tripwires, TripwireAlert
from .track_analyzer import TrackAnalyzer, ThreatAnalysis
from .watchdog import Watchdog, WatchdogAlert

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    """Record of a metric recording."""
    
    module: str
    response_time_ms: float
    decision: str
    error: bool
    timestamp: float = field(default_factory=time.time)
    alert: Optional[TripwireAlert] = None


@dataclass
class GuardianStatistics:
    """Statistics tracked by the Guardian."""
    
    # Mode statistics
    mode_durations: Dict[GuardianMode, float] = field(default_factory=dict)
    mode_switches: Dict[str, int] = field(default_factory=dict)  # "FROM->TO": count
    current_mode_start: float = field(default_factory=time.time)
    
    # Metric statistics
    total_metrics_recorded: int = 0
    total_alerts_triggered: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    alerts_by_module: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Performance statistics
    avg_overhead_ms: float = 0.0
    max_overhead_ms: float = 0.0
    
    # Lockdown statistics
    manual_lockdowns: int = 0
    automatic_lockdowns: int = 0
    lockdown_duration_total_s: float = 0.0


class AdaptiveGuardian:
    """Adaptive Guardian - Intelligent security monitoring with automatic mode adaptation.
    
    Operates in 5 modes with increasing security intensity:
    SPRINT -> CRUISE -> ALERT -> DEFENSE -> LOCKDOWN
    
    Features:
    - Automatic mode adaptation based on threat landscape
    - Layered security: tripwires, metrics, pulse analysis, watchdog
    - Cross-module correlation detection
    - Manual lockdown/clear functions
    - Comprehensive statistics tracking
    """
    
    def __init__(
        self,
        pulse_callback: Optional[Callable[[ThreatAnalysis], None]] = None,
        alert_callback: Optional[Callable[[TripwireAlert], None]] = None,
        watchdog_callback: Optional[Callable[[WatchdogAlert], None]] = None,
    ):
        """Initialize Adaptive Guardian.
        
        Args:
            pulse_callback: Function to call on each pulse analysis
            alert_callback: Function to call when tripwire alerts
            watchdog_callback: Function to call when watchdog alerts
        """
        # Current mode
        self._current_mode = GuardianMode.CRUISE
        self._mode_lock = threading.Lock()
        self._manual_lockdown = False
        self._lockdown_reason: Optional[str] = None
        
        # Components
        self._tripwires = Tripwires()
        self._track_analyzer = TrackAnalyzer()
        self._watchdog = Watchdog(alert_callback=self._handle_watchdog_alert)
        
        # Callbacks
        self._pulse_callback = pulse_callback
        self._alert_callback = alert_callback
        self._watchdog_callback = watchdog_callback
        
        # Pulse thread
        self._pulse_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Statistics
        self._stats = GuardianStatistics()
        
        # Initialize mode durations
        for mode in GuardianMode:
            self._stats.mode_durations[mode] = 0.0
        
        # Start watchdog
        self._watchdog.start()
        
        logger.info(f"Adaptive Guardian initialized in {self._current_mode.value} mode")
    
    def start(self) -> None:
        """Start the Guardian's background pulse analysis."""
        if self._running:
            logger.warning("Guardian already running")
            return
        
        self._running = True
        self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
        self._pulse_thread.start()
        logger.info("Guardian pulse analysis started")
    
    def stop(self) -> None:
        """Stop the Guardian."""
        if not self._running:
            return
        
        self._running = False
        if self._pulse_thread:
            self._pulse_thread.join(timeout=5.0)
        
        self._watchdog.stop()
        logger.info("Guardian stopped")
    
    def record_metric(
        self,
        module: str,
        response_time_ms: float,
        decision: str = "ALLOW",
        error: bool = False,
    ) -> MetricRecord:
        """Record a metric and check for violations.
        
        This is the main entry point for monitoring. Should be called for
        every significant operation.
        
        Args:
            module: Name of the module (e.g., "SafetyJudge")
            response_time_ms: Response time in milliseconds
            decision: Decision made (ALLOW, BLOCK, etc.)
            error: Whether an error occurred
            
        Returns:
            MetricRecord with any alerts triggered
        """
        start = time.perf_counter()
        
        # Get current mode and sensitivity
        with self._mode_lock:
            current_mode = self._current_mode
        
        config = get_mode_config(current_mode)
        sensitivity = config.tripwire_sensitivity
        
        # Check tripwires (Layer 1 - instant)
        alert = self._tripwires.check(
            module=module,
            response_time_ms=response_time_ms,
            decision=decision,
            error=error,
            sensitivity=sensitivity,
        )
        
        # Record in track analyzer (Layer 3 - background)
        self._track_analyzer.record_metric(module, response_time_ms, error)
        
        # Handle alert if triggered
        if alert:
            self._handle_alert(alert)
        
        # Update statistics
        self._stats.total_metrics_recorded += 1
        
        # Track overhead
        overhead_ms = (time.perf_counter() - start) * 1000
        if overhead_ms > self._stats.max_overhead_ms:
            self._stats.max_overhead_ms = overhead_ms
        
        prev_avg = self._stats.avg_overhead_ms
        count = self._stats.total_metrics_recorded
        self._stats.avg_overhead_ms = (prev_avg * (count - 1) + overhead_ms) / count
        
        return MetricRecord(
            module=module,
            response_time_ms=response_time_ms,
            decision=decision,
            error=error,
            alert=alert,
        )
    
    def trigger_lockdown(self, reason: str = "manual") -> None:
        """Trigger manual lockdown.
        
        Args:
            reason: Reason for lockdown
        """
        with self._mode_lock:
            if self._current_mode == GuardianMode.LOCKDOWN and self._manual_lockdown:
                logger.warning("Already in manual lockdown")
                return
            
            prev_mode = self._current_mode
            self._switch_mode(GuardianMode.LOCKDOWN)
            self._manual_lockdown = True
            self._lockdown_reason = reason
            self._stats.manual_lockdowns += 1
            
            logger.warning(
                f"Manual lockdown triggered: {reason} (from {prev_mode.value})"
            )
    
    def clear_lockdown(self) -> None:
        """Clear manual lockdown and return to automatic mode selection."""
        with self._mode_lock:
            if not self._manual_lockdown:
                logger.warning("Not in manual lockdown")
                return
            
            self._manual_lockdown = False
            self._lockdown_reason = None
            
            # Analyze and switch to appropriate mode
            analysis = self._track_analyzer.analyze()
            self._switch_mode(analysis.recommended_mode)
            
            logger.info(
                f"Manual lockdown cleared, switched to {self._current_mode.value}"
            )
    
    def get_mode(self) -> GuardianMode:
        """Get current guardian mode.
        
        Returns:
            Current mode
        """
        with self._mode_lock:
            return self._current_mode
    
    def set_external_threat_level(self, level: float) -> None:
        """Set external threat level from threat intelligence.
        
        Args:
            level: Threat level (0.0-1.0)
        """
        self._track_analyzer.set_external_threat_level(level)
    
    def record_correlation(self, module1: str, module2: str) -> None:
        """Record a cross-module correlation event.
        
        Args:
            module1: First module name
            module2: Second module name
        """
        self._track_analyzer.record_correlation(module1, module2)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status.
        
        Returns:
            Dictionary with complete status information
        """
        with self._mode_lock:
            mode = self._current_mode
            config = get_mode_config(mode)
            mode_duration = time.time() - self._stats.current_mode_start
        
        # Get analysis
        analysis = self._track_analyzer.analyze()
        
        return {
            "current_mode": mode.value,
            "mode_emoji": config.emoji,
            "mode_description": config.description,
            "mode_duration_s": mode_duration,
            "manual_lockdown": self._manual_lockdown,
            "lockdown_reason": self._lockdown_reason,
            "threat_analysis": {
                "overall_score": analysis.overall_threat_score,
                "recommended_mode": analysis.recommended_mode.value,
                "contributing_factors": analysis.contributing_factors,
                "anomaly_modules": analysis.anomaly_modules,
                "alert_count": analysis.alert_count,
                "error_rate": analysis.error_rate,
                "response_time_trend": analysis.response_time_trend,
                "correlation_score": analysis.correlation_score,
            },
            "performance": {
                "avg_overhead_ms": self._stats.avg_overhead_ms,
                "max_overhead_ms": self._stats.max_overhead_ms,
                "target_overhead_ms": config.overhead_ms,
                "pulse_interval_s": config.pulse_interval_s,
            },
            "statistics": self._get_statistics_dict(),
            "watchdog": self._watchdog.get_status(),
        }
    
    def get_statistics(self) -> GuardianStatistics:
        """Get Guardian statistics.
        
        Returns:
            GuardianStatistics object
        """
        with self._mode_lock:
            # Update current mode duration
            mode_duration = time.time() - self._stats.current_mode_start
            self._stats.mode_durations[self._current_mode] += mode_duration
            self._stats.current_mode_start = time.time()
            
            return self._stats
    
    def _get_statistics_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        stats = self.get_statistics()
        
        return {
            "total_metrics_recorded": stats.total_metrics_recorded,
            "total_alerts_triggered": stats.total_alerts_triggered,
            "alerts_by_severity": dict(stats.alerts_by_severity),
            "alerts_by_module": dict(stats.alerts_by_module),
            "mode_durations": {
                mode.value: duration for mode, duration in stats.mode_durations.items()
            },
            "mode_switches": dict(stats.mode_switches),
            "manual_lockdowns": stats.manual_lockdowns,
            "automatic_lockdowns": stats.automatic_lockdowns,
            "lockdown_duration_total_s": stats.lockdown_duration_total_s,
        }
    
    def _pulse_loop(self) -> None:
        """Background pulse analysis loop."""
        logger.info("Guardian pulse loop started")
        
        while self._running:
            # Send watchdog heartbeat
            self._watchdog.heartbeat()
            
            # Get current mode config for pulse interval
            with self._mode_lock:
                mode = self._current_mode
                manual_lockdown = self._manual_lockdown
            
            config = get_mode_config(mode)
            pulse_interval = config.pulse_interval_s
            
            # Sleep for pulse interval
            time.sleep(pulse_interval)
            
            # Skip analysis if in manual lockdown
            if manual_lockdown:
                continue
            
            # Perform pulse analysis
            analysis = self._track_analyzer.analyze()
            
            # Call pulse callback if provided
            if self._pulse_callback:
                try:
                    self._pulse_callback(analysis)
                except Exception as e:
                    logger.error(f"Error in pulse callback: {e}")
            
            # Check if mode should change
            if analysis.recommended_mode != mode:
                with self._mode_lock:
                    # Double-check we're not in manual lockdown
                    if not self._manual_lockdown:
                        self._switch_mode(analysis.recommended_mode)
                        
                        # Track automatic lockdowns
                        if analysis.recommended_mode == GuardianMode.LOCKDOWN:
                            self._stats.automatic_lockdowns += 1
    
    def _switch_mode(self, new_mode: GuardianMode) -> None:
        """Switch to a new mode (must be called with lock held).
        
        Args:
            new_mode: New mode to switch to
        """
        old_mode = self._current_mode
        
        if old_mode == new_mode:
            return
        
        # Update duration statistics
        mode_duration = time.time() - self._stats.current_mode_start
        self._stats.mode_durations[old_mode] += mode_duration
        
        # Track lockdown duration
        if old_mode == GuardianMode.LOCKDOWN:
            self._stats.lockdown_duration_total_s += mode_duration
        
        # Switch mode
        self._current_mode = new_mode
        self._stats.current_mode_start = time.time()
        
        # Track switch
        switch_key = f"{old_mode.value}->{new_mode.value}"
        self._stats.mode_switches[switch_key] = (
            self._stats.mode_switches.get(switch_key, 0) + 1
        )
        
        old_config = get_mode_config(old_mode)
        new_config = get_mode_config(new_mode)
        
        logger.info(
            f"Guardian mode switch: {old_mode.value} {old_config.emoji} -> "
            f"{new_mode.value} {new_config.emoji}"
        )
    
    def _handle_alert(self, alert: TripwireAlert) -> None:
        """Handle a tripwire alert.
        
        Args:
            alert: The alert that was triggered
        """
        # Update statistics
        self._stats.total_alerts_triggered += 1
        self._stats.alerts_by_severity[alert.severity] += 1
        self._stats.alerts_by_module[alert.module] += 1
        
        # Record in track analyzer
        self._track_analyzer.record_alert(alert)
        
        # Call alert callback if provided
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(
            f"Tripwire alert: {alert.severity} - {alert.description} "
            f"(module: {alert.module})"
        )
    
    def _handle_watchdog_alert(self, alert: WatchdogAlert) -> None:
        """Handle a watchdog alert.
        
        Args:
            alert: Watchdog alert
        """
        if self._watchdog_callback:
            try:
                self._watchdog_callback(alert)
            except Exception as e:
                logger.error(f"Error in watchdog callback: {e}")


# Global singleton instance
_guardian_instance: Optional[AdaptiveGuardian] = None
_guardian_lock = threading.Lock()


def get_guardian() -> AdaptiveGuardian:
    """Get the global Guardian instance (singleton).
    
    Returns:
        Global AdaptiveGuardian instance
    """
    global _guardian_instance
    
    if _guardian_instance is None:
        with _guardian_lock:
            if _guardian_instance is None:
                _guardian_instance = AdaptiveGuardian()
                _guardian_instance.start()
    
    return _guardian_instance


def record_metric(
    module: str,
    response_time_ms: float,
    decision: str = "ALLOW",
    error: bool = False,
) -> MetricRecord:
    """Record a metric using the global Guardian.
    
    Args:
        module: Name of the module
        response_time_ms: Response time in milliseconds
        decision: Decision made
        error: Whether an error occurred
        
    Returns:
        MetricRecord with any alerts
    """
    guardian = get_guardian()
    return guardian.record_metric(module, response_time_ms, decision, error)


def trigger_lockdown(reason: str = "manual") -> None:
    """Trigger manual lockdown on the global Guardian.
    
    Args:
        reason: Reason for lockdown
    """
    guardian = get_guardian()
    guardian.trigger_lockdown(reason)


def clear_lockdown() -> None:
    """Clear manual lockdown on the global Guardian."""
    guardian = get_guardian()
    guardian.clear_lockdown()


def get_mode() -> GuardianMode:
    """Get current mode from the global Guardian.
    
    Returns:
        Current GuardianMode
    """
    guardian = get_guardian()
    return guardian.get_mode()


def get_status() -> Dict[str, Any]:
    """Get status from the global Guardian.
    
    Returns:
        Status dictionary
    """
    guardian = get_guardian()
    return guardian.get_status()


def monitored(module: str):
    """Decorator to monitor a function with the Guardian.
    
    Args:
        module: Module name for tracking
        
    Example:
        @monitored("SafetyJudge")
        async def evaluate(self, action):
            ...
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                error = False
                decision = "ALLOW"
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Try to extract decision from result
                    if hasattr(result, "decision"):
                        decision = str(result.decision)
                    
                    return result
                except Exception as e:
                    error = True
                    raise
                finally:
                    response_time_ms = (time.perf_counter() - start) * 1000
                    record_metric(module, response_time_ms, decision, error)
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                error = False
                decision = "ALLOW"
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Try to extract decision from result
                    if hasattr(result, "decision"):
                        decision = str(result.decision)
                    
                    return result
                except Exception as e:
                    error = True
                    raise
                finally:
                    response_time_ms = (time.perf_counter() - start) * 1000
                    record_metric(module, response_time_ms, decision, error)
            
            return sync_wrapper
    
    return decorator
