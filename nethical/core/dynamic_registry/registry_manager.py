"""
Registry Manager for Dynamic Attack Registry

This module manages the complete lifecycle of the dynamic attack registry,
coordinating auto-registration and auto-deprecation processes.

Features:
- Lifecycle management
- Integration with existing attack registry
- Monitoring and reporting
- Configuration management

Alignment: Law 24 (Adaptive Learning), Law 15 (Audit Compliance)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .auto_registration import AutoRegistration, AttackPattern, RegistrationStage
from .auto_deprecation import AutoDeprecation, DeprecationCandidate, ArchiveStatus

logger = logging.getLogger(__name__)


@dataclass
class RegistryHealth:
    """Health metrics for the dynamic registry."""
    
    timestamp: datetime
    total_active_vectors: int
    pending_registration: int
    pending_deprecation: int
    archived_vectors: int
    registration_success_rate: float
    deprecation_rate: float
    overall_health: str  # "HEALTHY", "DEGRADED", "CRITICAL"


class RegistryManager:
    """
    Manages the dynamic attack registry lifecycle.
    
    This component coordinates automatic registration of new attack
    patterns and deprecation of unused vectors, maintaining a healthy
    and up-to-date registry.
    
    Features:
    - Coordinated lifecycle management
    - Health monitoring
    - Integration with existing registry
    - Automated maintenance
    """
    
    def __init__(
        self,
        attack_registry: Optional[Dict[str, Any]] = None,
        auto_registration: Optional[AutoRegistration] = None,
        auto_deprecation: Optional[AutoDeprecation] = None
    ):
        """
        Initialize the registry manager.
        
        Args:
            attack_registry: Existing attack registry (from core.attack_registry)
            auto_registration: Auto-registration component
            auto_deprecation: Auto-deprecation component
        """
        self.attack_registry = attack_registry or {}
        self.auto_registration = auto_registration or AutoRegistration()
        self.auto_deprecation = auto_deprecation or AutoDeprecation()
        
        self.health_history: List[RegistryHealth] = []
        self.maintenance_enabled = True
        
        logger.info("RegistryManager initialized")
    
    async def run_maintenance_cycle(self) -> RegistryHealth:
        """
        Run a complete maintenance cycle.
        
        This includes:
        - Identifying new patterns for registration
        - Processing pending registrations
        - Identifying vectors for deprecation
        - Processing pending deprecations
        
        Returns:
            Health metrics after maintenance
        """
        logger.info("Starting registry maintenance cycle")
        
        try:
            # Process registrations
            await self._process_registrations()
            
            # Process deprecations
            await self._process_deprecations()
            
            # Calculate health metrics
            health = await self._calculate_health()
            
            self.health_history.append(health)
            
            logger.info(
                f"Maintenance cycle complete. Health: {health.overall_health}"
            )
            
            return health
            
        except Exception as e:
            logger.error(f"Error in maintenance cycle: {e}")
            raise
    
    async def _process_registrations(self) -> None:
        """Process pending registrations."""
        # Get patterns that need processing
        pending_patterns = [
            p for p in self.auto_registration.discovered_patterns.values()
            if p.stage not in [RegistrationStage.DEPLOYED, RegistrationStage.REJECTED]
        ]
        
        logger.info(f"Processing {len(pending_patterns)} pending registrations")
        
        for pattern in pending_patterns:
            try:
                # Processing happens automatically in auto_registration
                # Here we just log and monitor
                if pattern.stage == RegistrationStage.AB_TESTING:
                    # Simulate A/B test completion check
                    if await self._is_ab_test_complete(pattern.pattern_id):
                        await self._finalize_deployment(pattern)
            except Exception as e:
                logger.error(f"Error processing pattern {pattern.pattern_id}: {e}")
    
    async def _is_ab_test_complete(self, pattern_id: str) -> bool:
        """Check if A/B test is complete for a pattern."""
        # In production, would check actual A/B test metrics
        # For now, simplified logic
        pattern = self.auto_registration.discovered_patterns.get(pattern_id)
        if not pattern:
            return False
        
        # Check if pattern has been in AB testing long enough
        # (simplified - production would check metrics)
        return True  # Assume complete for now
    
    async def _finalize_deployment(self, pattern: AttackPattern) -> None:
        """Finalize deployment of a pattern."""
        pattern.stage = RegistrationStage.DEPLOYED
        
        # Add to main attack registry
        self.attack_registry[pattern.pattern_id] = {
            "id": pattern.pattern_id,
            "category": pattern.category,
            "signature": pattern.signature,
            "description": pattern.description,
            "severity": pattern.severity,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "discovered_by": pattern.discovered_by,
        }
        
        logger.info(f"Finalized deployment of {pattern.pattern_id}")
    
    async def _process_deprecations(self) -> None:
        """Process deprecations."""
        # Analyze usage for all active vectors
        candidates = await self.auto_deprecation.identify_deprecation_candidates(
            self.attack_registry
        )
        
        logger.info(f"Identified {len(candidates)} deprecation candidates")
        
        # Auto-approve low-risk deprecations
        for candidate in candidates:
            # Auto-approve if very clear case (zero detections + no variants)
            if self._should_auto_deprecate(candidate):
                await self.auto_deprecation.approve_deprecation(
                    candidate.vector_id,
                    "Auto-approved: Clear deprecation criteria met"
                )
                
                # Remove from main registry
                if candidate.vector_id in self.attack_registry:
                    del self.attack_registry[candidate.vector_id]
                
                logger.info(f"Auto-deprecated {candidate.vector_id}")
    
    def _should_auto_deprecate(self, candidate: DeprecationCandidate) -> bool:
        """Check if candidate should be auto-deprecated."""
        stats = candidate.usage_stats
        
        # Very conservative auto-deprecation criteria
        if (stats.days_since_detection > 180 and  # 6+ months
            stats.total_detections == 0 and
            stats.known_variants == 0):
            return True
        
        return False
    
    async def _calculate_health(self) -> RegistryHealth:
        """Calculate registry health metrics."""
        # Count vectors by status
        total_active = len(self.attack_registry)
        
        pending_registration = sum(
            1 for p in self.auto_registration.discovered_patterns.values()
            if p.stage not in [RegistrationStage.DEPLOYED, RegistrationStage.REJECTED]
        )
        
        pending_deprecation = len(self.auto_deprecation.get_pending_reviews())
        archived = len(self.auto_deprecation.archived_vectors)
        
        # Calculate rates
        total_patterns = len(self.auto_registration.discovered_patterns)
        if total_patterns > 0:
            deployed = sum(
                1 for p in self.auto_registration.discovered_patterns.values()
                if p.stage == RegistrationStage.DEPLOYED
            )
            registration_success_rate = deployed / total_patterns
        else:
            registration_success_rate = 1.0
        
        # Deprecation rate
        total_analyzed = len(self.auto_deprecation.vector_stats)
        if total_analyzed > 0:
            deprecation_rate = archived / total_analyzed
        else:
            deprecation_rate = 0.0
        
        # Determine overall health
        overall_health = self._determine_health_status(
            pending_registration,
            pending_deprecation,
            registration_success_rate
        )
        
        return RegistryHealth(
            timestamp=datetime.now(timezone.utc),
            total_active_vectors=total_active,
            pending_registration=pending_registration,
            pending_deprecation=pending_deprecation,
            archived_vectors=archived,
            registration_success_rate=registration_success_rate,
            deprecation_rate=deprecation_rate,
            overall_health=overall_health
        )
    
    def _determine_health_status(
        self,
        pending_registration: int,
        pending_deprecation: int,
        success_rate: float
    ) -> str:
        """Determine overall health status."""
        # Critical conditions
        if success_rate < 0.5 or pending_registration > 50:
            return "CRITICAL"
        
        # Degraded conditions
        if success_rate < 0.8 or pending_registration > 20 or pending_deprecation > 10:
            return "DEGRADED"
        
        # Otherwise healthy
        return "HEALTHY"
    
    async def register_new_pattern(
        self,
        category: str,
        signature: str,
        description: str,
        severity: str = "MEDIUM"
    ) -> str:
        """
        Register a new attack pattern.
        
        Args:
            category: Attack category
            signature: Attack signature
            description: Description
            severity: Severity level
            
        Returns:
            Pattern ID
        """
        pattern_id = await self.auto_registration.register_attack_pattern(
            category=category,
            signature=signature,
            description=description,
            discovered_by="manual",
            severity=severity
        )
        
        logger.info(f"Registered new pattern: {pattern_id}")
        
        return pattern_id
    
    async def deprecate_vector(
        self,
        vector_id: str,
        reason: str
    ) -> bool:
        """
        Manually deprecate a vector.
        
        Args:
            vector_id: Vector to deprecate
            reason: Reason for deprecation
            
        Returns:
            True if deprecated successfully
        """
        from .auto_deprecation import DeprecationReason
        
        # Flag for review
        await self.auto_deprecation.flag_for_review(
            vector_id,
            DeprecationReason.MANUAL_DEPRECATION
        )
        
        # Approve immediately for manual deprecations
        success = await self.auto_deprecation.approve_deprecation(
            vector_id,
            reason
        )
        
        if success and vector_id in self.attack_registry:
            del self.attack_registry[vector_id]
        
        return success
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get current registry status."""
        latest_health = self.health_history[-1] if self.health_history else None
        
        return {
            "total_active_vectors": len(self.attack_registry),
            "registration_stats": self.auto_registration.get_statistics(),
            "deprecation_stats": self.auto_deprecation.get_statistics(),
            "latest_health": {
                "overall_health": latest_health.overall_health if latest_health else "UNKNOWN",
                "timestamp": latest_health.timestamp.isoformat() if latest_health else None,
            } if latest_health else None,
            "maintenance_enabled": self.maintenance_enabled,
        }
    
    def get_health_trend(self) -> List[Dict[str, Any]]:
        """Get historical health trend."""
        return [
            {
                "timestamp": health.timestamp.isoformat(),
                "overall_health": health.overall_health,
                "active_vectors": health.total_active_vectors,
                "pending_registration": health.pending_registration,
                "pending_deprecation": health.pending_deprecation,
            }
            for health in self.health_history
        ]
    
    def enable_maintenance(self) -> None:
        """Enable automatic maintenance."""
        self.maintenance_enabled = True
        logger.info("Automatic maintenance enabled")
    
    def disable_maintenance(self) -> None:
        """Disable automatic maintenance."""
        self.maintenance_enabled = False
        logger.warning("Automatic maintenance disabled")
    
    async def integrate_with_attack_registry(
        self,
        registry_module: Any
    ) -> None:
        """
        Integrate with existing attack_registry module.
        
        Args:
            registry_module: The nethical.core.attack_registry module
        """
        # Import existing vectors
        if hasattr(registry_module, 'ATTACK_VECTORS'):
            existing_vectors = registry_module.ATTACK_VECTORS
            
            # Add to managed registry
            for vector_id, vector_obj in existing_vectors.items():
                self.attack_registry[vector_id] = {
                    "id": vector_id,
                    "category": vector_obj.category.value if hasattr(vector_obj.category, 'value') else str(vector_obj.category),
                    "name": vector_obj.name,
                    "description": vector_obj.description,
                    "severity": vector_obj.severity,
                    "detector_class": vector_obj.detector_class,
                }
            
            logger.info(f"Integrated {len(existing_vectors)} existing vectors")
    
    def generate_registry_report(self) -> Dict[str, Any]:
        """Generate comprehensive registry report."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overview": self.get_registry_status(),
            "registration_report": {
                "statistics": self.auto_registration.get_statistics(),
                "pending_approval": [
                    p_id for p_id in self.auto_registration.pending_approval
                ],
            },
            "deprecation_report": self.auto_deprecation.generate_deprecation_report(),
            "health_trend": self.get_health_trend()[-10:],  # Last 10 entries
        }
