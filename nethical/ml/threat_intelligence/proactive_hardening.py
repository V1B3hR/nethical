"""
Proactive Hardening Module.

Automatically deploys defensive measures based on predicted threats
before attacks occur.

Phase: 5 - Detection Omniscience
Component: Threat Anticipation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class HardeningPriority(Enum):
    """Priority levels for hardening actions."""
    
    CRITICAL = "critical"  # Deploy immediately
    HIGH = "high"         # Deploy within hours
    MEDIUM = "medium"     # Deploy within days
    LOW = "low"          # Deploy within weeks
    DEFERRED = "deferred" # Schedule for future


class HardeningStatus(Enum):
    """Status of hardening actions."""
    
    PENDING = "pending"
    APPROVED = "approved"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class HardeningAction:
    """Represents a proactive hardening action."""
    
    action_id: str
    action_type: str
    description: str
    priority: HardeningPriority
    status: HardeningStatus = HardeningStatus.PENDING
    predicted_threat_id: Optional[str] = None
    threat_probability: float = 0.0
    deployment_steps: List[str] = field(default_factory=list)
    rollback_steps: List[str] = field(default_factory=list)
    affected_systems: List[str] = field(default_factory=list)
    estimated_impact: str = "low"
    requires_approval: bool = False
    approved_by: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "predicted_threat_id": self.predicted_threat_id,
            "threat_probability": self.threat_probability,
            "deployment_steps": self.deployment_steps,
            "rollback_steps": self.rollback_steps,
            "affected_systems": self.affected_systems,
            "estimated_impact": self.estimated_impact,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "created_at": self.created_at.isoformat(),
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "metadata": self.metadata,
        }


class ProactiveHardener:
    """
    Implements proactive hardening based on threat predictions.
    
    Features:
    - Automatic defense deployment
    - Risk-based prioritization
    - Human approval workflow for high-impact changes
    - Rollback capability
    - Deployment tracking and auditing
    """
    
    def __init__(
        self,
        auto_deploy_threshold: float = 0.8,
        approval_required_threshold: float = 0.6,
        max_concurrent_deployments: int = 5,
    ):
        """
        Initialize proactive hardener.
        
        Args:
            auto_deploy_threshold: Probability threshold for auto-deployment
            approval_required_threshold: Threshold for requiring human approval
            max_concurrent_deployments: Maximum concurrent deployments
        """
        self.auto_deploy_threshold = auto_deploy_threshold
        self.approval_required_threshold = approval_required_threshold
        self.max_concurrent_deployments = max_concurrent_deployments
        
        # Storage
        self.actions: Dict[str, HardeningAction] = {}
        self.actions_by_priority: Dict[HardeningPriority, Set[str]] = defaultdict(set)
        self.actions_by_status: Dict[HardeningStatus, Set[str]] = defaultdict(set)
        self.deployed_actions: List[str] = []
        
        # State
        self.active_deployments: int = 0
        
        # Statistics
        self.total_actions_created: int = 0
        self.successful_deployments: int = 0
        self.failed_deployments: int = 0
        self.rollbacks: int = 0
        
        logger.info(
            f"ProactiveHardener initialized with auto_deploy_threshold={auto_deploy_threshold}"
        )
    
    async def create_hardening_action(
        self,
        prediction: Any,
    ) -> HardeningAction:
        """
        Create a hardening action based on a threat prediction.
        
        Args:
            prediction: AttackPrediction object
            
        Returns:
            Created hardening action
        """
        try:
            # Determine priority based on probability and time horizon
            probability = getattr(prediction, "probability", 0.0)
            time_horizon = getattr(prediction, "time_horizon_days", 90)
            
            if probability >= 0.9 and time_horizon <= 7:
                priority = HardeningPriority.CRITICAL
            elif probability >= 0.8 and time_horizon <= 30:
                priority = HardeningPriority.HIGH
            elif probability >= 0.7:
                priority = HardeningPriority.MEDIUM
            else:
                priority = HardeningPriority.LOW
            
            # Determine if approval is required
            # High-risk actions (high priority or high probability) require approval
            requires_approval = (
                priority in [HardeningPriority.CRITICAL, HardeningPriority.HIGH]
                or probability >= self.approval_required_threshold
            )
            
            # Extract recommended defenses
            defenses = getattr(prediction, "recommended_defenses", [])
            
            # Create action
            action_id = f"harden_{getattr(prediction, 'prediction_id', 'unknown')}"
            
            action = HardeningAction(
                action_id=action_id,
                action_type="deploy_detector",
                description=f"Deploy defenses for predicted {getattr(prediction, 'attack_type', 'unknown')} attack",
                priority=priority,
                predicted_threat_id=getattr(prediction, "prediction_id", None),
                threat_probability=probability,
                deployment_steps=defenses or self._generate_deployment_steps(prediction),
                rollback_steps=self._generate_rollback_steps(prediction),
                affected_systems=["detection_engine", "policy_engine"],
                estimated_impact="low" if priority in [HardeningPriority.LOW, HardeningPriority.MEDIUM] else "medium",
                requires_approval=requires_approval,
                metadata={
                    "prediction_id": getattr(prediction, "prediction_id", None),
                    "attack_type": getattr(prediction, "attack_type", "unknown"),
                    "time_horizon_days": time_horizon,
                },
            )
            
            # Store action
            self.actions[action_id] = action
            self.actions_by_priority[priority].add(action_id)
            self.actions_by_status[action.status].add(action_id)
            self.total_actions_created += 1
            
            logger.info(
                f"Created hardening action {action_id} with priority {priority.value}"
            )
            
            # Auto-approve and deploy if threshold met
            if probability >= self.auto_deploy_threshold and not requires_approval:
                await self.approve_action(action_id, "auto_approval")
                await self.deploy_action(action_id)
            
            return action
            
        except Exception as e:
            logger.error(f"Error creating hardening action: {e}")
            raise
    
    def _generate_deployment_steps(self, prediction: Any) -> List[str]:
        """Generate deployment steps based on prediction."""
        attack_type = getattr(prediction, "attack_type", "unknown")
        return [
            f"1. Generate detector for {attack_type}",
            f"2. Validate detector with test cases",
            f"3. Deploy detector to staging",
            f"4. Monitor for 24 hours",
            f"5. Deploy to production",
        ]
    
    def _generate_rollback_steps(self, prediction: Any) -> List[str]:
        """Generate rollback steps."""
        attack_type = getattr(prediction, "attack_type", "unknown")
        return [
            f"1. Disable detector for {attack_type}",
            f"2. Remove from active detector list",
            f"3. Archive detector configuration",
            f"4. Notify security team",
        ]
    
    async def approve_action(
        self, action_id: str, approved_by: str
    ) -> Dict[str, Any]:
        """
        Approve a hardening action.
        
        Args:
            action_id: ID of action to approve
            approved_by: Who approved the action
            
        Returns:
            Approval result
        """
        if action_id not in self.actions:
            return {
                "status": "error",
                "error": f"Action {action_id} not found",
            }
        
        action = self.actions[action_id]
        
        if action.status != HardeningStatus.PENDING:
            return {
                "status": "error",
                "error": f"Action {action_id} is not pending (status: {action.status.value})",
            }
        
        # Update action
        action.status = HardeningStatus.APPROVED
        action.approved_by = approved_by
        
        # Update indices
        self.actions_by_status[HardeningStatus.PENDING].discard(action_id)
        self.actions_by_status[HardeningStatus.APPROVED].add(action_id)
        
        logger.info(f"Approved action {action_id} by {approved_by}")
        
        return {
            "status": "success",
            "action_id": action_id,
            "approved_by": approved_by,
        }
    
    async def deploy_action(self, action_id: str) -> Dict[str, Any]:
        """
        Deploy a hardening action.
        
        Args:
            action_id: ID of action to deploy
            
        Returns:
            Deployment result
        """
        if action_id not in self.actions:
            return {
                "status": "error",
                "error": f"Action {action_id} not found",
            }
        
        action = self.actions[action_id]
        
        # Check if approved or auto-approved
        if action.requires_approval and action.status != HardeningStatus.APPROVED:
            return {
                "status": "error",
                "error": f"Action {action_id} requires approval before deployment",
            }
        
        # Check deployment capacity
        if self.active_deployments >= self.max_concurrent_deployments:
            return {
                "status": "error",
                "error": "Maximum concurrent deployments reached",
            }
        
        try:
            # Update status
            action.status = HardeningStatus.DEPLOYING
            self.actions_by_status[HardeningStatus.APPROVED].discard(action_id)
            self.actions_by_status[HardeningStatus.DEPLOYING].add(action_id)
            self.active_deployments += 1
            
            logger.info(f"Deploying action {action_id}")
            
            # Simulate deployment (in production, this would execute actual deployment)
            await asyncio.sleep(0.5)
            
            # Successful deployment
            action.status = HardeningStatus.DEPLOYED
            action.deployed_at = datetime.now(timezone.utc)
            
            # Update indices
            self.actions_by_status[HardeningStatus.DEPLOYING].discard(action_id)
            self.actions_by_status[HardeningStatus.DEPLOYED].add(action_id)
            self.deployed_actions.append(action_id)
            
            self.active_deployments -= 1
            self.successful_deployments += 1
            
            logger.info(f"Successfully deployed action {action_id}")
            
            return {
                "status": "success",
                "action_id": action_id,
                "deployed_at": action.deployed_at.isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error deploying action {action_id}: {e}")
            
            # Mark as failed
            action.status = HardeningStatus.FAILED
            self.actions_by_status[HardeningStatus.DEPLOYING].discard(action_id)
            self.actions_by_status[HardeningStatus.FAILED].add(action_id)
            
            self.active_deployments -= 1
            self.failed_deployments += 1
            
            return {
                "status": "error",
                "action_id": action_id,
                "error": str(e),
            }
    
    async def rollback_action(
        self, action_id: str, reason: str
    ) -> Dict[str, Any]:
        """
        Rollback a deployed hardening action.
        
        Args:
            action_id: ID of action to rollback
            reason: Reason for rollback
            
        Returns:
            Rollback result
        """
        if action_id not in self.actions:
            return {
                "status": "error",
                "error": f"Action {action_id} not found",
            }
        
        action = self.actions[action_id]
        
        if action.status != HardeningStatus.DEPLOYED:
            return {
                "status": "error",
                "error": f"Action {action_id} is not deployed (status: {action.status.value})",
            }
        
        try:
            logger.info(f"Rolling back action {action_id}: {reason}")
            
            # Execute rollback steps (simulated)
            for step in action.rollback_steps:
                logger.info(f"Rollback step: {step}")
                await asyncio.sleep(0.1)
            
            # Update status
            action.status = HardeningStatus.ROLLED_BACK
            action.metadata["rollback_reason"] = reason
            action.metadata["rollback_at"] = datetime.now(timezone.utc).isoformat()
            
            # Update indices
            self.actions_by_status[HardeningStatus.DEPLOYED].discard(action_id)
            self.actions_by_status[HardeningStatus.ROLLED_BACK].add(action_id)
            
            if action_id in self.deployed_actions:
                self.deployed_actions.remove(action_id)
            
            self.rollbacks += 1
            
            logger.info(f"Successfully rolled back action {action_id}")
            
            return {
                "status": "success",
                "action_id": action_id,
                "reason": reason,
            }
            
        except Exception as e:
            logger.error(f"Error rolling back action {action_id}: {e}")
            return {
                "status": "error",
                "action_id": action_id,
                "error": str(e),
            }
    
    async def get_pending_actions(
        self, priority: Optional[HardeningPriority] = None
    ) -> List[HardeningAction]:
        """
        Get pending actions, optionally filtered by priority.
        
        Args:
            priority: Optional priority filter
            
        Returns:
            List of pending actions
        """
        pending_ids = self.actions_by_status.get(HardeningStatus.PENDING, set())
        
        actions = [self.actions[aid] for aid in pending_ids if aid in self.actions]
        
        if priority:
            actions = [a for a in actions if a.priority == priority]
        
        # Sort by priority and probability
        priority_order = {
            HardeningPriority.CRITICAL: 0,
            HardeningPriority.HIGH: 1,
            HardeningPriority.MEDIUM: 2,
            HardeningPriority.LOW: 3,
            HardeningPriority.DEFERRED: 4,
        }
        
        actions.sort(
            key=lambda a: (priority_order[a.priority], -a.threat_probability)
        )
        
        return actions
    
    async def process_queue(self, max_actions: int = 10) -> Dict[str, Any]:
        """
        Process pending hardening actions in priority order.
        
        Args:
            max_actions: Maximum actions to process
            
        Returns:
            Processing result
        """
        processed = 0
        deployed = 0
        failed = 0
        
        # Get high-priority approved actions
        approved_ids = self.actions_by_status.get(HardeningStatus.APPROVED, set())
        approved_actions = [
            self.actions[aid] for aid in approved_ids if aid in self.actions
        ]
        
        # Sort by priority
        priority_order = {
            HardeningPriority.CRITICAL: 0,
            HardeningPriority.HIGH: 1,
            HardeningPriority.MEDIUM: 2,
            HardeningPriority.LOW: 3,
        }
        
        approved_actions.sort(key=lambda a: priority_order[a.priority])
        
        for action in approved_actions[:max_actions]:
            if self.active_deployments >= self.max_concurrent_deployments:
                break
            
            result = await self.deploy_action(action.action_id)
            processed += 1
            
            if result["status"] == "success":
                deployed += 1
            else:
                failed += 1
        
        logger.info(
            f"Processed {processed} actions: {deployed} deployed, {failed} failed"
        )
        
        return {
            "processed": processed,
            "deployed": deployed,
            "failed": failed,
            "remaining_in_queue": len(approved_actions) - processed,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get hardening statistics.
        
        Returns:
            Dictionary of statistics
        """
        status_counts = {
            status.value: len(action_ids)
            for status, action_ids in self.actions_by_status.items()
        }
        
        priority_counts = {
            priority.value: len(action_ids)
            for priority, action_ids in self.actions_by_priority.items()
        }
        
        return {
            "total_actions_created": self.total_actions_created,
            "successful_deployments": self.successful_deployments,
            "failed_deployments": self.failed_deployments,
            "rollbacks": self.rollbacks,
            "active_deployments": self.active_deployments,
            "actions_by_status": status_counts,
            "actions_by_priority": priority_counts,
            "deployed_actions_count": len(self.deployed_actions),
        }
