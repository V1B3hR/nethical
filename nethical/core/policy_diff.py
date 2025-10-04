"""Policy Diff Auditing for Phase 4.2: Policy Auditing & Diff.

This module implements:
- Policy diff audit (semantic diff computation)
- Risk scoring for policy changes
- Policy version management
- CLI support
"""

import json
import yaml
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum


class ChangeType(str, Enum):
    """Type of policy change."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class RiskLevel(str, Enum):
    """Risk level for policy changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PolicyChange:
    """Individual policy change."""
    path: str  # JSON path to changed field
    change_type: ChangeType
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    risk_level: RiskLevel = RiskLevel.LOW
    description: str = ""
    impact_score: float = 0.0


@dataclass
class PolicyDiffResult:
    """Result of policy diff comparison."""
    old_version: str
    new_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    changes: List[PolicyChange] = field(default_factory=list)
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PolicyDiffAuditor:
    """Policy diff auditing and risk assessment."""
    
    def __init__(
        self,
        storage_path: str = "policy_history",
        high_risk_fields: Optional[List[str]] = None
    ):
        """Initialize policy diff auditor.
        
        Args:
            storage_path: Path to store policy history
            high_risk_fields: List of high-risk field paths
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # High-risk fields that require extra scrutiny
        self.high_risk_fields = high_risk_fields or [
            'security',
            'authentication',
            'authorization',
            'permissions',
            'access_control',
            'rate_limits',
            'thresholds',
            'elevated_threshold',
            'quarantine'
        ]
        
        # Risk weights for different change types
        self.risk_weights = {
            ChangeType.REMOVED: 0.8,
            ChangeType.MODIFIED: 0.5,
            ChangeType.ADDED: 0.3,
            ChangeType.UNCHANGED: 0.0
        }
        
        # Policy version history
        self.version_history: List[Dict[str, Any]] = []
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _is_high_risk_field(self, field_path: str) -> bool:
        """Check if field is high-risk.
        
        Args:
            field_path: Field path
            
        Returns:
            True if high-risk
        """
        for risk_field in self.high_risk_fields:
            if risk_field in field_path.lower():
                return True
        return False
    
    def _calculate_change_risk(self, change: PolicyChange) -> float:
        """Calculate risk score for a change.
        
        Args:
            change: Policy change
            
        Returns:
            Risk score (0-1)
        """
        # Base risk from change type
        base_risk = self.risk_weights.get(change.change_type, 0.5)
        
        # Amplify if high-risk field
        if self._is_high_risk_field(change.path):
            base_risk *= 1.5
        
        # Additional risk factors
        if change.change_type == ChangeType.REMOVED:
            # Removing fields is risky
            base_risk *= 1.2
        elif change.change_type == ChangeType.MODIFIED:
            # Check magnitude of change
            if isinstance(change.old_value, (int, float)) and isinstance(change.new_value, (int, float)):
                try:
                    old_val = float(change.old_value)
                    new_val = float(change.new_value)
                    if old_val != 0:
                        percent_change = abs((new_val - old_val) / old_val)
                        # Large changes are riskier
                        if percent_change > 0.5:
                            base_risk *= 1.3
                        elif percent_change > 1.0:
                            base_risk *= 1.5
                except:
                    pass
        
        return min(base_risk, 1.0)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score.
        
        Args:
            risk_score: Risk score (0-1)
            
        Returns:
            Risk level
        """
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def compare_policies(
        self,
        old_policy: Dict[str, Any],
        new_policy: Dict[str, Any],
        old_version: str = "previous",
        new_version: str = "current"
    ) -> PolicyDiffResult:
        """Compare two policy versions.
        
        Args:
            old_policy: Old policy dictionary
            new_policy: New policy dictionary
            old_version: Old version identifier
            new_version: New version identifier
            
        Returns:
            Policy diff result
        """
        # Flatten policies for comparison
        old_flat = self._flatten_dict(old_policy)
        new_flat = self._flatten_dict(new_policy)
        
        changes = []
        
        # Find all keys
        all_keys = set(old_flat.keys()) | set(new_flat.keys())
        
        for key in all_keys:
            old_val = old_flat.get(key)
            new_val = new_flat.get(key)
            
            if key not in old_flat:
                # Added
                change = PolicyChange(
                    path=key,
                    change_type=ChangeType.ADDED,
                    new_value=new_val,
                    description=f"Added new field: {key}"
                )
            elif key not in new_flat:
                # Removed
                change = PolicyChange(
                    path=key,
                    change_type=ChangeType.REMOVED,
                    old_value=old_val,
                    description=f"Removed field: {key}"
                )
            elif old_val != new_val:
                # Modified
                change = PolicyChange(
                    path=key,
                    change_type=ChangeType.MODIFIED,
                    old_value=old_val,
                    new_value=new_val,
                    description=f"Modified {key}: {old_val} → {new_val}"
                )
            else:
                continue
            
            # Calculate risk for this change
            change.impact_score = self._calculate_change_risk(change)
            change.risk_level = self._determine_risk_level(change.impact_score)
            
            changes.append(change)
        
        # Calculate overall risk score
        if changes:
            total_risk = sum(c.impact_score for c in changes)
            risk_score = total_risk / len(changes)
        else:
            risk_score = 0.0
        
        # Determine overall risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Generate summary
        summary = {
            'total_changes': len(changes),
            'added': sum(1 for c in changes if c.change_type == ChangeType.ADDED),
            'removed': sum(1 for c in changes if c.change_type == ChangeType.REMOVED),
            'modified': sum(1 for c in changes if c.change_type == ChangeType.MODIFIED),
            'high_risk': sum(1 for c in changes if c.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(changes, risk_level)
        
        return PolicyDiffResult(
            old_version=old_version,
            new_version=new_version,
            changes=changes,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        changes: List[PolicyChange],
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate recommendations based on changes.
        
        Args:
            changes: List of policy changes
            risk_level: Overall risk level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("⚠️ High-risk policy change detected - requires review")
        
        # Check for removed fields
        removed = [c for c in changes if c.change_type == ChangeType.REMOVED]
        if removed:
            recommendations.append(f"Review {len(removed)} removed field(s) for backward compatibility")
        
        # Check for high-risk field changes
        high_risk = [c for c in changes if self._is_high_risk_field(c.path)]
        if high_risk:
            recommendations.append(f"Review {len(high_risk)} security-critical field(s)")
        
        # Check for threshold changes
        threshold_changes = [c for c in changes if 'threshold' in c.path.lower()]
        if threshold_changes:
            recommendations.append("Threshold changes may affect detection sensitivity")
        
        if not recommendations:
            recommendations.append("✓ Changes appear low-risk, but review recommended")
        
        return recommendations
    
    def save_policy_version(
        self,
        policy: Dict[str, Any],
        version: str,
        description: str = ""
    ):
        """Save policy version to history.
        
        Args:
            policy: Policy dictionary
            version: Version identifier
            description: Version description
        """
        version_file = self.storage_path / f"policy_{version}.json"
        
        version_data = {
            'version': version,
            'timestamp': datetime.utcnow().isoformat(),
            'description': description,
            'policy': policy
        }
        
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        self.version_history.append(version_data)
    
    def load_policy_version(self, version: str) -> Optional[Dict[str, Any]]:
        """Load policy version from history.
        
        Args:
            version: Version identifier
            
        Returns:
            Policy dictionary or None
        """
        version_file = self.storage_path / f"policy_{version}.json"
        
        if not version_file.exists():
            return None
        
        with open(version_file, 'r') as f:
            version_data = json.load(f)
        
        return version_data.get('policy')
    
    def format_diff_report(self, diff_result: PolicyDiffResult) -> str:
        """Format diff result as human-readable report.
        
        Args:
            diff_result: Policy diff result
            
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Policy Diff Report: {diff_result.old_version} → {diff_result.new_version}")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {diff_result.timestamp.isoformat()}")
        lines.append(f"Risk Level: {diff_result.risk_level.value.upper()}")
        lines.append(f"Risk Score: {diff_result.risk_score:.3f}")
        lines.append("")
        
        # Summary
        lines.append("Summary:")
        for key, value in diff_result.summary.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Changes by risk level
        for risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            level_changes = [c for c in diff_result.changes if c.risk_level == risk_level]
            if level_changes:
                lines.append(f"{risk_level.value.upper()} Risk Changes ({len(level_changes)}):")
                for change in level_changes:
                    symbol = {
                        ChangeType.ADDED: "+",
                        ChangeType.REMOVED: "-",
                        ChangeType.MODIFIED: "~"
                    }.get(change.change_type, "?")
                    lines.append(f"  {symbol} {change.path}")
                    if change.old_value is not None:
                        lines.append(f"      old: {change.old_value}")
                    if change.new_value is not None:
                        lines.append(f"      new: {change.new_value}")
                lines.append("")
        
        # Recommendations
        if diff_result.recommendations:
            lines.append("Recommendations:")
            for rec in diff_result.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
