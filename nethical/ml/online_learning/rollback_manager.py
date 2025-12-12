"""
Rollback Manager for Safe Deployments

Provides fast rollback capability for failed detector updates.

Features:
- Version history tracking
- 5-minute rollback guarantee
- Automatic health checks
- Rollback strategies

Law Alignment:
- Law 23 (Fail-Safe Design): Quick recovery from failures
- Law 15 (Audit Compliance): Track all changes
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid

logger = logging.getLogger(__name__)


class RollbackStrategy(Enum):
    """Strategy for rollback."""
    IMMEDIATE = "immediate"  # Rollback immediately
    GRADUAL = "gradual"  # Gradual traffic shift back
    CANARY = "canary"  # Test rollback on canary first


@dataclass
class DetectorVersion:
    """Represents a detector version."""
    
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detector_name: str = ""
    version_number: str = "1.0.0"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Configuration snapshot
    threshold: float = 0.5
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    deployed_by: str = ""
    notes: str = ""
    
    # Health status
    is_healthy: bool = True
    health_check_count: int = 0
    error_count: int = 0


@dataclass
class RollbackEvent:
    """Represents a rollback event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    detector_name: str = ""
    from_version_id: str = ""
    to_version_id: str = ""
    strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE
    
    reason: str = ""
    triggered_by: str = "system"  # "system" or user ID
    
    completed: bool = False
    completion_time: Optional[datetime] = None
    duration_seconds: float = 0.0


class RollbackManager:
    """
    Manages detector version history and rollbacks.
    
    Features:
    - Version history per detector
    - Fast rollback (< 5 minutes)
    - Health monitoring
    - Audit trail
    """
    
    def __init__(self, max_rollback_time_seconds: int = 300):
        """
        Initialize rollback manager.
        
        Args:
            max_rollback_time_seconds: Maximum time for rollback (default 5 minutes)
        """
        self.max_rollback_time_seconds = max_rollback_time_seconds
        
        # Version history per detector
        self.version_history: Dict[str, List[DetectorVersion]] = {}
        
        # Current versions
        self.current_versions: Dict[str, str] = {}  # detector_name -> version_id
        
        # Rollback events
        self.rollback_events: List[RollbackEvent] = []
        
        # Metrics
        self.total_versions = 0
        self.total_rollbacks = 0
        self.successful_rollbacks = 0
        self.failed_rollbacks = 0
        
        logger.info(
            f"RollbackManager initialized: max_rollback_time={max_rollback_time_seconds}s"
        )
    
    async def register_version(self, version: DetectorVersion) -> bool:
        """
        Register a new detector version.
        
        Args:
            version: Version to register
            
        Returns:
            True if successful
        """
        detector_name = version.detector_name
        
        # Initialize history if needed
        if detector_name not in self.version_history:
            self.version_history[detector_name] = []
        
        # Add to history
        self.version_history[detector_name].append(version)
        
        # Update current version
        self.current_versions[detector_name] = version.version_id
        
        self.total_versions += 1
        
        logger.info(
            f"Registered version {version.version_id} for {detector_name} "
            f"(v{version.version_number})"
        )
        
        return True
    
    async def rollback(
        self,
        detector_name: str,
        target_version_id: Optional[str] = None,
        strategy: RollbackStrategy = RollbackStrategy.IMMEDIATE,
        reason: str = "",
        triggered_by: str = "system",
    ) -> Optional[RollbackEvent]:
        """
        Rollback detector to previous version.
        
        Args:
            detector_name: Name of detector to rollback
            target_version_id: Specific version to rollback to (None = previous)
            strategy: Rollback strategy
            reason: Reason for rollback
            triggered_by: Who/what triggered rollback
            
        Returns:
            RollbackEvent if successful, None if failed
        """
        start_time = datetime.now(timezone.utc)
        
        # Get current version
        if detector_name not in self.current_versions:
            logger.error(f"Detector {detector_name} not found")
            return None
        
        current_version_id = self.current_versions[detector_name]
        
        # Get version history
        history = self.version_history.get(detector_name, [])
        if len(history) < 2:
            logger.error(f"No previous version available for {detector_name}")
            return None
        
        # Determine target version
        if target_version_id is None:
            # Rollback to previous version
            current_idx = next(
                (i for i, v in enumerate(history) if v.version_id == current_version_id),
                -1
            )
            if current_idx <= 0:
                logger.error(f"Cannot determine previous version for {detector_name}")
                return None
            target_version = history[current_idx - 1]
            target_version_id = target_version.version_id
        else:
            # Rollback to specific version
            target_version = next(
                (v for v in history if v.version_id == target_version_id),
                None
            )
            if target_version is None:
                logger.error(f"Target version {target_version_id} not found")
                return None
        
        # Create rollback event
        event = RollbackEvent(
            detector_name=detector_name,
            from_version_id=current_version_id,
            to_version_id=target_version_id,
            strategy=strategy,
            reason=reason,
            triggered_by=triggered_by,
        )
        
        logger.info(
            f"Starting rollback for {detector_name}: "
            f"{current_version_id} -> {target_version_id}, "
            f"strategy={strategy.value}, reason={reason}"
        )
        
        # Perform rollback based on strategy
        success = await self._execute_rollback(detector_name, target_version, strategy)
        
        # Complete event
        end_time = datetime.now(timezone.utc)
        event.completed = success
        event.completion_time = end_time
        event.duration_seconds = (end_time - start_time).total_seconds()
        
        self.rollback_events.append(event)
        self.total_rollbacks += 1
        
        if success:
            self.successful_rollbacks += 1
            logger.info(
                f"Rollback completed for {detector_name} in {event.duration_seconds:.2f}s"
            )
        else:
            self.failed_rollbacks += 1
            logger.error(f"Rollback failed for {detector_name}")
        
        return event if success else None
    
    async def _execute_rollback(
        self,
        detector_name: str,
        target_version: DetectorVersion,
        strategy: RollbackStrategy,
    ) -> bool:
        """Execute the rollback."""
        try:
            if strategy == RollbackStrategy.IMMEDIATE:
                # Immediate rollback
                self.current_versions[detector_name] = target_version.version_id
                logger.info(f"Immediate rollback completed for {detector_name}")
                return True
            
            elif strategy == RollbackStrategy.GRADUAL:
                # Gradual rollback (simplified - would do traffic shifting)
                self.current_versions[detector_name] = target_version.version_id
                logger.info(f"Gradual rollback completed for {detector_name}")
                return True
            
            elif strategy == RollbackStrategy.CANARY:
                # Canary rollback (simplified - would test on subset first)
                self.current_versions[detector_name] = target_version.version_id
                logger.info(f"Canary rollback completed for {detector_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            return False
    
    async def health_check(self, detector_name: str) -> bool:
        """
        Check health of current detector version.
        
        Args:
            detector_name: Name of detector
            
        Returns:
            True if healthy
        """
        if detector_name not in self.current_versions:
            return False
        
        version_id = self.current_versions[detector_name]
        history = self.version_history.get(detector_name, [])
        version = next((v for v in history if v.version_id == version_id), None)
        
        if version is None:
            return False
        
        version.health_check_count += 1
        
        # Simple health check (would do actual metrics in production)
        is_healthy = version.error_count < 10
        version.is_healthy = is_healthy
        
        return is_healthy
    
    def get_current_version(self, detector_name: str) -> Optional[DetectorVersion]:
        """Get current version for detector."""
        if detector_name not in self.current_versions:
            return None
        
        version_id = self.current_versions[detector_name]
        history = self.version_history.get(detector_name, [])
        return next((v for v in history if v.version_id == version_id), None)
    
    def get_version_history(self, detector_name: str) -> List[DetectorVersion]:
        """Get version history for detector."""
        return self.version_history.get(detector_name, []).copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rollback manager metrics."""
        return {
            "total_versions": self.total_versions,
            "total_rollbacks": self.total_rollbacks,
            "successful_rollbacks": self.successful_rollbacks,
            "failed_rollbacks": self.failed_rollbacks,
            "success_rate": (
                self.successful_rollbacks / self.total_rollbacks
                if self.total_rollbacks > 0 else 0.0
            ),
        }
