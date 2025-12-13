"""Independent watchdog process for the Adaptive Guardian.

"Who watches the watchmen?" - The Watchdog does.
"""

import time
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class WatchdogAlert:
    """Alert from the watchdog."""
    
    alert_type: str
    description: str
    timestamp: float
    guardian_last_heartbeat: float
    time_since_heartbeat: float


class Watchdog:
    """Independent watchdog that monitors the Guardian itself.
    
    Runs in a separate thread and alerts if Guardian becomes unresponsive.
    """
    
    HEARTBEAT_INTERVAL_S = 5  # Guardian should heartbeat every 5s
    HEARTBEAT_TIMEOUT_S = 15  # Alert if no heartbeat for 15s
    CHECK_INTERVAL_S = 3  # How often watchdog checks
    
    def __init__(self, alert_callback: Optional[Callable[[WatchdogAlert], None]] = None):
        """Initialize watchdog.
        
        Args:
            alert_callback: Function to call when watchdog detects an issue
        """
        self._alert_callback = alert_callback
        
        self._last_heartbeat = time.time()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "started_at": None,
            "total_checks": 0,
            "total_alerts": 0,
            "guardian_restarts_detected": 0,
            "longest_silence_s": 0.0,
        }
    
    def start(self) -> None:
        """Start the watchdog thread."""
        if self._running:
            logger.warning("Watchdog already running")
            return
        
        self._running = True
        self._stats["started_at"] = time.time()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        logger.info("Watchdog started")
    
    def stop(self) -> None:
        """Stop the watchdog thread."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Watchdog stopped")
    
    def heartbeat(self) -> None:
        """Record a heartbeat from the Guardian.
        
        Should be called regularly by the Guardian to indicate it's alive.
        """
        with self._lock:
            self._last_heartbeat = time.time()
    
    def _watchdog_loop(self) -> None:
        """Main watchdog loop."""
        logger.info("Watchdog loop started")
        
        prev_heartbeat = self._last_heartbeat
        
        while self._running:
            time.sleep(self.CHECK_INTERVAL_S)
            
            with self._lock:
                now = time.time()
                time_since_heartbeat = now - self._last_heartbeat
                self._stats["total_checks"] += 1
                
                # Update longest silence
                if time_since_heartbeat > self._stats["longest_silence_s"]:
                    self._stats["longest_silence_s"] = time_since_heartbeat
                
                # Check if Guardian is responsive
                if time_since_heartbeat > self.HEARTBEAT_TIMEOUT_S:
                    alert = WatchdogAlert(
                        alert_type="guardian_unresponsive",
                        description=f"Guardian has not sent heartbeat for {time_since_heartbeat:.1f}s",
                        timestamp=now,
                        guardian_last_heartbeat=self._last_heartbeat,
                        time_since_heartbeat=time_since_heartbeat,
                    )
                    
                    self._stats["total_alerts"] += 1
                    logger.error(f"Watchdog alert: {alert.description}")
                    
                    if self._alert_callback:
                        try:
                            self._alert_callback(alert)
                        except Exception as e:
                            logger.error(f"Error in watchdog alert callback: {e}")
                
                # Detect Guardian restart (heartbeat goes backwards in time)
                if self._last_heartbeat < prev_heartbeat:
                    self._stats["guardian_restarts_detected"] += 1
                    logger.warning("Guardian restart detected")
                
                prev_heartbeat = self._last_heartbeat
    
    def get_status(self) -> dict:
        """Get watchdog status.
        
        Returns:
            Dictionary with watchdog status
        """
        with self._lock:
            now = time.time()
            time_since_heartbeat = now - self._last_heartbeat
            
            return {
                "running": self._running,
                "last_heartbeat": self._last_heartbeat,
                "time_since_heartbeat_s": time_since_heartbeat,
                "is_guardian_responsive": time_since_heartbeat < self.HEARTBEAT_TIMEOUT_S,
                "stats": dict(self._stats),
            }
    
    def get_statistics(self) -> dict:
        """Get watchdog statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            uptime_s = (
                time.time() - self._stats["started_at"]
                if self._stats["started_at"]
                else 0.0
            )
            
            return {
                **self._stats,
                "uptime_s": uptime_s,
                "uptime_hours": uptime_s / 3600,
            }
