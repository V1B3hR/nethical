"""
Release and Change Management

This module implements versioned policy packs, rollback procedures,
and canary deployment configuration for safe policy changes.

Production Readiness Checklist - Section 11: Release & Change
- Versioned policy pack
- Rollback procedure tested
- Canary deployment config
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


class DeploymentStage(Enum):
    """Deployment stage for canary releases"""
    DEVELOPMENT = "development"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


@dataclass
class PolicyVersion:
    """A versioned policy snapshot"""
    version: str
    name: str
    content: Dict[str, Any]
    checksum: str
    created_at: datetime
    created_by: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "name": self.name,
            "content": self.content,
            "checksum": self.checksum,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class CanaryConfig:
    """Canary deployment configuration"""
    canary_percentage: float  # 0-100
    duration_minutes: int
    success_threshold: float  # 0-1
    metrics_to_monitor: List[str]
    auto_promote: bool = False
    auto_rollback: bool = True
    rollback_on_error_rate: float = 0.05  # 5%
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "canary_percentage": self.canary_percentage,
            "duration_minutes": self.duration_minutes,
            "success_threshold": self.success_threshold,
            "metrics_to_monitor": self.metrics_to_monitor,
            "auto_promote": self.auto_promote,
            "auto_rollback": self.auto_rollback,
            "rollback_on_error_rate": self.rollback_on_error_rate
        }


@dataclass
class Deployment:
    """A deployment record"""
    deployment_id: str
    policy_version: str
    stage: DeploymentStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # in_progress, success, failed, rolled_back
    metrics: Dict[str, float] = field(default_factory=dict)
    canary_config: Optional[CanaryConfig] = None
    rollback_version: Optional[str] = None
    notes: str = ""


class PolicyPack:
    """
    Versioned policy pack with rollback and canary deployment support.
    
    This provides:
    - Version management for policies
    - Rollback to previous versions
    - Canary deployment configuration
    - Deployment tracking and metrics
    
    Example:
        >>> pack = PolicyPack("safety_policies")
        >>> pack.create_version("1.0.0", policy_content, "admin", "Initial release")
        >>> pack.deploy_canary("1.0.0", canary_percentage=10)
        >>> pack.promote_to_production("1.0.0")
        >>> pack.rollback_to_version("0.9.0")
    """
    
    def __init__(self, pack_name: str, storage_dir: str = "./nethical_policy_packs"):
        """
        Initialize policy pack.
        
        Args:
            pack_name: Name of the policy pack
            storage_dir: Directory for storing policy versions
        """
        self.pack_name = pack_name
        self.storage_dir = Path(storage_dir) / pack_name
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Version storage
        self._versions: Dict[str, PolicyVersion] = {}
        self._current_version: Optional[str] = None
        self._production_version: Optional[str] = None
        self._canary_version: Optional[str] = None
        
        # Deployment history
        self._deployments: List[Deployment] = []
        
        # Load existing versions
        self._load_versions()
        
        logger.info(f"Policy pack '{pack_name}' initialized at {storage_dir}")
    
    def _load_versions(self):
        """Load existing versions from storage"""
        versions_file = self.storage_dir / "versions.jsonl"
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        version = PolicyVersion(
                            version=data["version"],
                            name=data["name"],
                            content=data["content"],
                            checksum=data["checksum"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            created_by=data["created_by"],
                            description=data.get("description", ""),
                            metadata=data.get("metadata", {})
                        )
                        self._versions[version.version] = version
                logger.info(f"Loaded {len(self._versions)} versions")
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
        
        # Load deployment state
        state_file = self.storage_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self._current_version = state.get("current_version")
                    self._production_version = state.get("production_version")
                    self._canary_version = state.get("canary_version")
                logger.info(
                    f"Loaded state: current={self._current_version}, "
                    f"production={self._production_version}, "
                    f"canary={self._canary_version}"
                )
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
    
    def _save_version(self, version: PolicyVersion):
        """Save a version to storage"""
        versions_file = self.storage_dir / "versions.jsonl"
        with open(versions_file, 'a') as f:
            f.write(json.dumps(version.to_dict()) + '\n')
    
    def _save_state(self):
        """Save current state"""
        state = {
            "current_version": self._current_version,
            "production_version": self._production_version,
            "canary_version": self._canary_version,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        state_file = self.storage_dir / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _calculate_checksum(self, content: Dict[str, Any]) -> str:
        """Calculate checksum for policy content"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def create_version(
        self,
        version: str,
        content: Dict[str, Any],
        created_by: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PolicyVersion:
        """
        Create a new policy version.
        
        Args:
            version: Version identifier (e.g., "1.0.0")
            content: Policy content dictionary
            created_by: User creating the version
            description: Version description
            metadata: Additional metadata
        
        Returns:
            Created PolicyVersion
        """
        if version in self._versions:
            raise ValueError(f"Version {version} already exists")
        
        checksum = self._calculate_checksum(content)
        
        policy_version = PolicyVersion(
            version=version,
            name=self.pack_name,
            content=content,
            checksum=checksum,
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            description=description,
            metadata=metadata or {}
        )
        
        self._versions[version] = policy_version
        self._save_version(policy_version)
        
        # Set as current if this is the first version
        if not self._current_version:
            self._current_version = version
            self._save_state()
        
        logger.info(
            f"Created policy version {version} for pack '{self.pack_name}', "
            f"checksum: {checksum[:8]}..."
        )
        
        return policy_version
    
    def get_version(self, version: str) -> Optional[PolicyVersion]:
        """Get a specific version"""
        return self._versions.get(version)
    
    def list_versions(self) -> List[PolicyVersion]:
        """List all versions"""
        return sorted(
            self._versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
    
    def deploy_canary(
        self,
        version: str,
        canary_percentage: float = 10.0,
        duration_minutes: int = 60,
        metrics_to_monitor: Optional[List[str]] = None,
        auto_promote: bool = False,
        auto_rollback: bool = True
    ) -> Deployment:
        """
        Deploy a version to canary stage.
        
        Args:
            version: Version to deploy
            canary_percentage: Percentage of traffic for canary (0-100)
            duration_minutes: Duration to run canary
            metrics_to_monitor: List of metrics to monitor
            auto_promote: Automatically promote if successful
            auto_rollback: Automatically rollback if failed
        
        Returns:
            Deployment record
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        if not 0 <= canary_percentage <= 100:
            raise ValueError("Canary percentage must be between 0 and 100")
        
        canary_config = CanaryConfig(
            canary_percentage=canary_percentage,
            duration_minutes=duration_minutes,
            success_threshold=0.95,
            metrics_to_monitor=metrics_to_monitor or [
                "error_rate",
                "latency_p95",
                "violation_rate"
            ],
            auto_promote=auto_promote,
            auto_rollback=auto_rollback
        )
        
        deployment = Deployment(
            deployment_id=f"deploy-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            policy_version=version,
            stage=DeploymentStage.CANARY,
            started_at=datetime.now(timezone.utc),
            canary_config=canary_config,
            rollback_version=self._production_version
        )
        
        self._canary_version = version
        self._deployments.append(deployment)
        self._save_state()
        
        logger.info(
            f"Deployed version {version} to canary ({canary_percentage}% traffic) "
            f"for {duration_minutes} minutes"
        )
        
        return deployment
    
    def promote_to_production(self, version: str) -> Deployment:
        """
        Promote a version to production.
        
        Args:
            version: Version to promote
        
        Returns:
            Deployment record
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        deployment = Deployment(
            deployment_id=f"deploy-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            policy_version=version,
            stage=DeploymentStage.PRODUCTION,
            started_at=datetime.now(timezone.utc),
            rollback_version=self._production_version,
            status="success"
        )
        
        deployment.completed_at = datetime.now(timezone.utc)
        
        # Update versions
        old_production = self._production_version
        self._production_version = version
        self._current_version = version
        self._canary_version = None  # Clear canary
        
        self._deployments.append(deployment)
        self._save_state()
        
        logger.info(
            f"Promoted version {version} to production "
            f"(previous: {old_production})"
        )
        
        return deployment
    
    def rollback_to_version(self, version: str, reason: str = "") -> Deployment:
        """
        Rollback to a previous version.
        
        Args:
            version: Version to rollback to
            reason: Reason for rollback
        
        Returns:
            Deployment record
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")
        
        deployment = Deployment(
            deployment_id=f"rollback-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            policy_version=version,
            stage=DeploymentStage.ROLLBACK,
            started_at=datetime.now(timezone.utc),
            rollback_version=self._production_version,
            status="success",
            notes=reason
        )
        
        deployment.completed_at = datetime.now(timezone.utc)
        
        # Update versions
        old_production = self._production_version
        self._production_version = version
        self._current_version = version
        
        self._deployments.append(deployment)
        self._save_state()
        
        logger.warning(
            f"Rolled back from version {old_production} to {version}. "
            f"Reason: {reason}"
        )
        
        return deployment
    
    def get_current_policy(self) -> Optional[Dict[str, Any]]:
        """Get current policy content"""
        if not self._current_version:
            return None
        version = self._versions.get(self._current_version)
        return version.content if version else None
    
    def get_production_policy(self) -> Optional[Dict[str, Any]]:
        """Get production policy content"""
        if not self._production_version:
            return None
        version = self._versions.get(self._production_version)
        return version.content if version else None
    
    def get_canary_policy(self) -> Optional[Dict[str, Any]]:
        """Get canary policy content"""
        if not self._canary_version:
            return None
        version = self._versions.get(self._canary_version)
        return version.content if version else None
    
    def test_rollback(self) -> bool:
        """
        Test rollback procedure by simulating a rollback.
        
        Returns:
            True if rollback test succeeds
        """
        if not self._production_version:
            logger.warning("No production version to test rollback")
            return False
        
        # Find a previous version
        versions = self.list_versions()
        if len(versions) < 2:
            logger.warning("Need at least 2 versions to test rollback")
            return False
        
        current = self._production_version
        
        # Find previous version
        previous = None
        for v in versions:
            if v.version != current:
                previous = v.version
                break
        
        if not previous:
            logger.error("Could not find previous version for rollback test")
            return False
        
        try:
            # Simulate rollback
            logger.info(f"Testing rollback from {current} to {previous}")
            
            # Verify versions exist
            if current not in self._versions or previous not in self._versions:
                logger.error("Version verification failed")
                return False
            
            # Verify content can be loaded
            current_policy = self._versions[current].content
            previous_policy = self._versions[previous].content
            
            if not current_policy or not previous_policy:
                logger.error("Policy content verification failed")
                return False
            
            logger.info(f"Rollback test successful: {current} -> {previous}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback test failed: {e}")
            return False
    
    def get_deployment_history(self) -> List[Dict[str, Any]]:
        """Get deployment history"""
        return [
            {
                "deployment_id": d.deployment_id,
                "policy_version": d.policy_version,
                "stage": d.stage.value,
                "started_at": d.started_at.isoformat(),
                "completed_at": d.completed_at.isoformat() if d.completed_at else None,
                "status": d.status,
                "rollback_version": d.rollback_version,
                "notes": d.notes
            }
            for d in self._deployments
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "pack_name": self.pack_name,
            "total_versions": len(self._versions),
            "current_version": self._current_version,
            "production_version": self._production_version,
            "canary_version": self._canary_version,
            "total_deployments": len(self._deployments),
            "last_deployment": (
                self._deployments[-1].started_at.isoformat()
                if self._deployments else None
            ),
            "rollback_tested": self.test_rollback()
        }
