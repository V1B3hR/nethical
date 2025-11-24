"""Threshold Configuration Versioning System

Manages versioned threshold configurations for ethical governance decisions.
Supports versioning, auditing, and rollback capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class ThresholdType(Enum):
    """Types of thresholds."""
    CONFIDENCE = "confidence"
    SEVERITY = "severity"
    RISK_SCORE = "risk_score"
    TOXICITY = "toxicity"
    BIAS = "bias"
    PRIVACY_RISK = "privacy_risk"
    MANIPULATION_SCORE = "manipulation_score"


@dataclass
class Threshold:
    """A single threshold configuration."""
    name: str
    threshold_type: ThresholdType
    value: float
    operator: str  # >, >=, <, <=, ==
    description: str
    unit: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def evaluate(self, actual_value: float) -> bool:
        """Evaluate if actual value meets threshold criteria.
        
        Args:
            actual_value: Value to check against threshold
            
        Returns:
            True if threshold condition is met
        """
        if self.operator == '>':
            return actual_value > self.value
        elif self.operator == '>=':
            return actual_value >= self.value
        elif self.operator == '<':
            return actual_value < self.value
        elif self.operator == '<=':
            return actual_value <= self.value
        elif self.operator == '==':
            return actual_value == self.value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'threshold_type': self.threshold_type.value,
            'value': self.value,
            'operator': self.operator,
            'description': self.description,
            'unit': self.unit,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Threshold:
        """Create from dictionary."""
        return cls(
            name=data['name'],
            threshold_type=ThresholdType(data['threshold_type']),
            value=float(data['value']),
            operator=data['operator'],
            description=data['description'],
            unit=data.get('unit'),
            metadata=data.get('metadata')
        )


@dataclass
class ThresholdConfig:
    """A versioned threshold configuration."""
    version: str
    timestamp: str
    author: str
    description: str
    thresholds: Dict[str, Threshold]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version': self.version,
            'timestamp': self.timestamp,
            'author': self.author,
            'description': self.description,
            'thresholds': {k: v.to_dict() for k, v in self.thresholds.items()},
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThresholdConfig:
        """Create from dictionary."""
        thresholds = {
            name: Threshold.from_dict(tdata) 
            for name, tdata in data['thresholds'].items()
        }
        return cls(
            version=data['version'],
            timestamp=data['timestamp'],
            author=data['author'],
            description=data['description'],
            thresholds=thresholds,
            metadata=data.get('metadata')
        )


class ThresholdVersionManager:
    """Manages versioned threshold configurations."""
    
    def __init__(self, storage_dir: str = "governance/thresholds"):
        """Initialize threshold version manager.
        
        Args:
            storage_dir: Directory to store threshold versions
        """
        self.storage_dir = Path(storage_dir).resolve()
        # Validate path doesn't escape expected boundaries
        if not str(self.storage_dir).startswith(str(Path.cwd().resolve())):
            # Allow absolute paths in /tmp for testing
            if not str(self.storage_dir).startswith('/tmp'):
                raise ValueError(f"Storage directory must be within current directory: {storage_dir}")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, ThresholdConfig] = {}
        self.current_version: Optional[str] = None
        
        # Load existing versions
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load all versions from storage."""
        version_file = self.storage_dir / "versions.json"
        
        if version_file.exists():
            with open(version_file, 'r') as f:
                data = json.load(f)
            
            self.current_version = data.get('current_version')
            
            for version_data in data.get('versions', []):
                config = ThresholdConfig.from_dict(version_data)
                self.versions[config.version] = config
    
    def _save_versions(self) -> None:
        """Save all versions to storage."""
        version_file = self.storage_dir / "versions.json"
        
        data = {
            'current_version': self.current_version,
            'total_versions': len(self.versions),
            'last_updated': datetime.utcnow().isoformat(),
            'versions': [config.to_dict() for config in self.versions.values()]
        }
        
        with open(version_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(self,
                      version: str,
                      author: str,
                      description: str,
                      thresholds: Dict[str, Threshold],
                      metadata: Optional[Dict[str, Any]] = None,
                      set_current: bool = True) -> ThresholdConfig:
        """Create a new threshold configuration version.
        
        Args:
            version: Version identifier (e.g., "1.0.0", "2023-11-24")
            author: Author of this version
            description: Description of changes
            thresholds: Dictionary of threshold configurations
            metadata: Optional metadata
            set_current: Whether to set this as current version
            
        Returns:
            Created ThresholdConfig
        """
        if version in self.versions:
            raise ValueError(f"Version {version} already exists")
        
        config = ThresholdConfig(
            version=version,
            timestamp=datetime.utcnow().isoformat(),
            author=author,
            description=description,
            thresholds=thresholds,
            metadata=metadata
        )
        
        self.versions[version] = config
        
        if set_current or self.current_version is None:
            self.current_version = version
        
        self._save_versions()
        
        return config
    
    def get_version(self, version: Optional[str] = None) -> Optional[ThresholdConfig]:
        """Get a specific version or current version.
        
        Args:
            version: Version identifier, or None for current version
            
        Returns:
            ThresholdConfig or None if not found
        """
        if version is None:
            version = self.current_version
        
        return self.versions.get(version) if version else None
    
    def set_current_version(self, version: str) -> None:
        """Set the current active version.
        
        Args:
            version: Version identifier to make current
        """
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        self.current_version = version
        self._save_versions()
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """List all available versions.
        
        Returns:
            List of version metadata
        """
        versions = []
        for config in self.versions.values():
            is_current = config.version == self.current_version
            versions.append({
                'version': config.version,
                'timestamp': config.timestamp,
                'author': config.author,
                'description': config.description,
                'num_thresholds': len(config.thresholds),
                'is_current': is_current
            })
        
        # Sort by timestamp descending
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return versions
    
    def compare_versions(self, 
                        version1: str, 
                        version2: str) -> Dict[str, Any]:
        """Compare two threshold configuration versions.
        
        Args:
            version1: First version identifier
            version2: Second version identifier
            
        Returns:
            Dictionary with comparison results
        """
        config1 = self.versions.get(version1)
        config2 = self.versions.get(version2)
        
        if not config1:
            raise ValueError(f"Version {version1} not found")
        if not config2:
            raise ValueError(f"Version {version2} not found")
        
        thresholds1 = set(config1.thresholds.keys())
        thresholds2 = set(config2.thresholds.keys())
        
        added = thresholds2 - thresholds1
        removed = thresholds1 - thresholds2
        common = thresholds1 & thresholds2
        
        changed = []
        for name in common:
            t1 = config1.thresholds[name]
            t2 = config2.thresholds[name]
            
            if t1.value != t2.value or t1.operator != t2.operator:
                changed.append({
                    'name': name,
                    'old_value': t1.value,
                    'new_value': t2.value,
                    'old_operator': t1.operator,
                    'new_operator': t2.operator
                })
        
        return {
            'version1': version1,
            'version2': version2,
            'added': list(added),
            'removed': list(removed),
            'changed': changed,
            'total_changes': len(added) + len(removed) + len(changed)
        }
    
    def get_threshold(self, 
                     threshold_name: str,
                     version: Optional[str] = None) -> Optional[Threshold]:
        """Get a specific threshold from a version.
        
        Args:
            threshold_name: Name of threshold
            version: Version identifier, or None for current
            
        Returns:
            Threshold or None if not found
        """
        config = self.get_version(version)
        if config:
            return config.thresholds.get(threshold_name)
        return None
    
    def evaluate_thresholds(self,
                           values: Dict[str, float],
                           version: Optional[str] = None) -> Dict[str, bool]:
        """Evaluate multiple values against thresholds.
        
        Args:
            values: Dictionary of threshold names to values
            version: Version to use, or None for current
            
        Returns:
            Dictionary of threshold names to pass/fail results
        """
        config = self.get_version(version)
        if not config:
            raise ValueError(f"Version not found: {version or 'current'}")
        
        results = {}
        for name, value in values.items():
            threshold = config.thresholds.get(name)
            if threshold:
                results[name] = threshold.evaluate(value)
        
        return results


# Default baseline thresholds
DEFAULT_THRESHOLDS = {
    'manipulation_detection': Threshold(
        name='manipulation_detection',
        threshold_type=ThresholdType.CONFIDENCE,
        value=0.85,
        operator='>=',
        description='Confidence threshold for manipulation detection',
        unit='probability'
    ),
    'privacy_risk': Threshold(
        name='privacy_risk',
        threshold_type=ThresholdType.PRIVACY_RISK,
        value=0.7,
        operator='>=',
        description='Privacy risk score threshold',
        unit='score'
    ),
    'toxicity': Threshold(
        name='toxicity',
        threshold_type=ThresholdType.TOXICITY,
        value=0.8,
        operator='>=',
        description='Toxicity detection threshold',
        unit='score'
    ),
    'high_severity': Threshold(
        name='high_severity',
        threshold_type=ThresholdType.SEVERITY,
        value=7.5,
        operator='>=',
        description='High severity classification threshold',
        unit='scale_0_10'
    ),
}


if __name__ == '__main__':
    # Demo usage
    manager = ThresholdVersionManager('/tmp/test_thresholds')
    
    # Create baseline version
    manager.create_version(
        version='1.0.0',
        author='system',
        description='Initial baseline thresholds',
        thresholds=DEFAULT_THRESHOLDS,
        metadata={'baseline': True}
    )
    
    # Create updated version
    updated_thresholds = DEFAULT_THRESHOLDS.copy()
    updated_thresholds['manipulation_detection'] = Threshold(
        name='manipulation_detection',
        threshold_type=ThresholdType.CONFIDENCE,
        value=0.90,  # Increased from 0.85
        operator='>=',
        description='Stricter manipulation detection threshold',
        unit='probability'
    )
    
    manager.create_version(
        version='1.1.0',
        author='admin',
        description='Increased manipulation detection threshold',
        thresholds=updated_thresholds,
        set_current=True
    )
    
    # List versions
    print("Available versions:")
    for v in manager.list_versions():
        current = " (CURRENT)" if v['is_current'] else ""
        print(f"  {v['version']}{current} - {v['description']}")
    
    # Compare versions
    print("\nComparison 1.0.0 vs 1.1.0:")
    diff = manager.compare_versions('1.0.0', '1.1.0')
    print(f"  Total changes: {diff['total_changes']}")
    for change in diff['changed']:
        print(f"  {change['name']}: {change['old_value']} → {change['new_value']}")
    
    # Evaluate thresholds
    test_values = {
        'manipulation_detection': 0.88,
        'privacy_risk': 0.65,
        'toxicity': 0.82
    }
    
    print("\nEvaluation results (v1.1.0):")
    results = manager.evaluate_thresholds(test_values)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status} (value={test_values[name]})")
