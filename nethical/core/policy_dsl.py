"""
Policy DSL (Domain-Specific Language) for F2: Detector & Policy Extensibility

This module implements a YAML/JSON-based policy specification language
for defining custom detection rules without code changes.

Features:
- YAML/JSON policy parsing
- Rule engine for policy evaluation
- Hot-reload of policy changes
- Policy versioning and rollback
- Compiled rule engine for performance

Policy DSL Example:
    policies:
      - name: "financial_compliance"
        version: "1.0.0"
        enabled: true
        rules:
          - condition: "action.context.contains('financial_data')"
            severity: HIGH
            actions:
              - "require_encryption"
              - "audit_log"
              - "escalate_to_compliance_team"
      
      - name: "pii_protection"
        version: "1.0.0"
        enabled: true
        rules:
          - condition: "action.contains_pii() AND NOT action.has_consent()"
            severity: CRITICAL
            actions:
              - "block_action"
              - "alert_dpo"
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

import yaml

from ..detectors.base_detector import SafetyViolation

logger = logging.getLogger(__name__)


class PolicyAction(str, Enum):
    """Actions that can be taken when a policy rule matches."""
    BLOCK_ACTION = "block_action"
    AUDIT_LOG = "audit_log"
    ALERT = "alert"
    REQUIRE_ENCRYPTION = "require_encryption"
    ESCALATE = "escalate"
    ALERT_DPO = "alert_dpo"
    QUARANTINE = "quarantine"
    NOTIFY_USER = "notify_user"


class RuleSeverity(str, Enum):
    """Severity levels for policy rules."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class PolicyRule:
    """A single rule within a policy."""
    condition: str
    severity: RuleSeverity
    actions: List[PolicyAction]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition,
            "severity": self.severity.value,
            "actions": [a.value for a in self.actions],
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class Policy:
    """A policy containing multiple rules."""
    name: str
    version: str
    enabled: bool
    rules: List[PolicyRule]
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "rules": [rule.to_dict() for rule in self.rules],
            "description": self.description,
            "tags": list(self.tags),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class PolicyParser:
    """Parser for YAML/JSON policy files."""
    
    @staticmethod
    def parse_yaml(yaml_content: str) -> List[Policy]:
        """
        Parse YAML policy content.
        
        Args:
            yaml_content: YAML string containing policy definitions
            
        Returns:
            List of Policy objects
        """
        try:
            data = yaml.safe_load(yaml_content)
            return PolicyParser._parse_policy_data(data)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise ValueError(f"Invalid YAML: {e}")
    
    @staticmethod
    def parse_json(json_content: str) -> List[Policy]:
        """
        Parse JSON policy content.
        
        Args:
            json_content: JSON string containing policy definitions
            
        Returns:
            List of Policy objects
        """
        try:
            data = json.loads(json_content)
            return PolicyParser._parse_policy_data(data)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Invalid JSON: {e}")
    
    @staticmethod
    def parse_file(file_path: str) -> List[Policy]:
        """
        Parse policy file (YAML or JSON based on extension).
        
        Args:
            file_path: Path to policy file
            
        Returns:
            List of Policy objects
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {file_path}")
        
        content = path.read_text()
        
        if path.suffix in ['.yaml', '.yml']:
            return PolicyParser.parse_yaml(content)
        elif path.suffix == '.json':
            return PolicyParser.parse_json(content)
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Use .yaml, .yml, or .json")
    
    @staticmethod
    def _parse_policy_data(data: Dict[str, Any]) -> List[Policy]:
        """Parse policy data dictionary into Policy objects."""
        if not isinstance(data, dict):
            raise ValueError("Policy data must be a dictionary")
        
        if "policies" not in data:
            raise ValueError("Policy data must contain 'policies' key")
        
        policies = []
        for policy_data in data["policies"]:
            policy = PolicyParser._parse_single_policy(policy_data)
            policies.append(policy)
        
        return policies
    
    @staticmethod
    def _parse_single_policy(policy_data: Dict[str, Any]) -> Policy:
        """Parse a single policy dictionary."""
        # Required fields
        name = policy_data.get("name")
        if not name:
            raise ValueError("Policy must have a 'name'")
        
        version = policy_data.get("version", "1.0.0")
        enabled = policy_data.get("enabled", True)
        
        # Parse rules
        rules_data = policy_data.get("rules", [])
        rules = [PolicyParser._parse_rule(rule_data) for rule_data in rules_data]
        
        # Optional fields
        description = policy_data.get("description", "")
        tags = set(policy_data.get("tags", []))
        
        return Policy(
            name=name,
            version=version,
            enabled=enabled,
            rules=rules,
            description=description,
            tags=tags,
            created_at=datetime.now(timezone.utc)
        )
    
    @staticmethod
    def _parse_rule(rule_data: Dict[str, Any]) -> PolicyRule:
        """Parse a single rule dictionary."""
        condition = rule_data.get("condition")
        if not condition:
            raise ValueError("Rule must have a 'condition'")
        
        severity_str = rule_data.get("severity", "MEDIUM")
        try:
            severity = RuleSeverity[severity_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid severity: {severity_str}")
        
        actions_data = rule_data.get("actions", [])
        actions = []
        for action_str in actions_data:
            # Convert snake_case or kebab-case to UPPER_SNAKE_CASE for enum lookup
            action_upper = action_str.upper().replace('-', '_')
            try:
                actions.append(PolicyAction[action_upper])
            except KeyError:
                logger.warning(f"Unknown action: {action_str}, skipping")
        
        description = rule_data.get("description", "")
        metadata = rule_data.get("metadata", {})
        
        return PolicyRule(
            condition=condition,
            severity=severity,
            actions=actions,
            description=description,
            metadata=metadata
        )


class RuleEvaluator:
    """
    Evaluates policy rules against actions.
    
    Provides a safe evaluation environment for policy conditions.
    """
    
    def __init__(self):
        """Initialize the rule evaluator."""
        self._builtin_functions = {
            "contains": self._contains,
            "matches_regex": self._matches_regex,
            "length": len,
            "upper": str.upper,
            "lower": str.lower,
            "startswith": str.startswith,
            "endswith": str.endswith,
        }
    
    def evaluate_condition(self, condition: str, action: Any, 
                          context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evaluate a condition against an action.
        
        Args:
            condition: Condition string (e.g., "action.content.contains('secret')")
            action: Action object to evaluate
            context: Optional context dictionary
            
        Returns:
            True if condition matches, False otherwise
        """
        try:
            # Create evaluation namespace
            namespace = {
                "action": action,
                "context": context or {},
                **self._builtin_functions
            }
            
            # Add action attribute accessors
            if hasattr(action, '__dict__'):
                for key, value in action.__dict__.items():
                    if not key.startswith('_'):
                        namespace[key] = value
            
            # Sanitize and evaluate condition
            sanitized_condition = self._sanitize_condition(condition)
            result = eval(sanitized_condition, {"__builtins__": {}}, namespace)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _sanitize_condition(self, condition: str) -> str:
        """
        Sanitize condition to prevent code injection.
        
        Removes dangerous patterns while allowing safe operations.
        """
        # Remove potentially dangerous keywords
        dangerous_patterns = [
            r'\b(import|exec|eval|compile|open|file)\b',
            r'__\w+__',  # Dunder methods
            r'globals\(\)',
            r'locals\(\)',
            r'vars\(\)',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                logger.warning(f"Potentially dangerous pattern detected in condition: {pattern}")
                raise ValueError(f"Unsafe condition: contains '{pattern}'")
        
        return condition
    
    @staticmethod
    def _contains(text: str, substring: str) -> bool:
        """Check if text contains substring (case-insensitive)."""
        return substring.lower() in str(text).lower()
    
    @staticmethod
    def _matches_regex(text: str, pattern: str) -> bool:
        """Check if text matches regex pattern."""
        try:
            return bool(re.search(pattern, str(text)))
        except re.error:
            return False


class PolicyEngine:
    """
    Policy engine that loads, manages, and evaluates policies.
    
    Supports:
    - Loading policies from files
    - Hot-reloading of policy changes
    - Policy versioning and rollback
    - Rule evaluation and action execution
    """
    
    def __init__(self):
        """Initialize the policy engine."""
        self.policies: Dict[str, Policy] = {}
        self.policy_history: Dict[str, List[Policy]] = {}
        self._policy_file_paths: Dict[str, str] = {}
        self._file_timestamps: Dict[str, float] = {}
        self.evaluator = RuleEvaluator()
        
        logger.info("PolicyEngine initialized")
    
    def load_policy_file(self, file_path: str) -> List[str]:
        """
        Load policies from a file.
        
        Args:
            file_path: Path to policy file
            
        Returns:
            List of loaded policy names
        """
        try:
            policies = PolicyParser.parse_file(file_path)
            loaded_names = []
            
            for policy in policies:
                self.add_policy(policy)
                self._policy_file_paths[policy.name] = file_path
                loaded_names.append(policy.name)
            
            # Track file timestamp for hot-reload
            self._file_timestamps[file_path] = os.path.getmtime(file_path)
            
            logger.info(f"Loaded {len(policies)} policy/policies from {file_path}")
            return loaded_names
            
        except Exception as e:
            logger.error(f"Failed to load policy file {file_path}: {e}")
            raise
    
    def add_policy(self, policy: Policy) -> None:
        """
        Add or update a policy.
        
        Args:
            policy: Policy object to add
        """
        # Store previous version in history
        if policy.name in self.policies:
            old_policy = self.policies[policy.name]
            if policy.name not in self.policy_history:
                self.policy_history[policy.name] = []
            self.policy_history[policy.name].append(old_policy)
            logger.info(f"Updating policy '{policy.name}' from v{old_policy.version} to v{policy.version}")
        else:
            logger.info(f"Adding new policy '{policy.name}' v{policy.version}")
        
        policy.updated_at = datetime.now(timezone.utc)
        self.policies[policy.name] = policy
    
    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a policy.
        
        Args:
            policy_name: Name of policy to remove
            
        Returns:
            True if removed, False if not found
        """
        if policy_name in self.policies:
            del self.policies[policy_name]
            logger.info(f"Removed policy '{policy_name}'")
            return True
        return False
    
    def get_policy(self, policy_name: str) -> Optional[Policy]:
        """Get a policy by name."""
        return self.policies.get(policy_name)
    
    def list_policies(self) -> Dict[str, Dict[str, Any]]:
        """
        List all policies with their info.
        
        Returns:
            Dictionary mapping policy names to their info
        """
        return {
            name: policy.to_dict()
            for name, policy in self.policies.items()
        }
    
    def rollback_policy(self, policy_name: str, version: Optional[str] = None) -> bool:
        """
        Rollback a policy to a previous version.
        
        Args:
            policy_name: Name of policy to rollback
            version: Specific version to rollback to (default: previous version)
            
        Returns:
            True if rolled back, False if no history available
        """
        if policy_name not in self.policy_history or not self.policy_history[policy_name]:
            logger.warning(f"No history available for policy '{policy_name}'")
            return False
        
        history = self.policy_history[policy_name]
        
        if version:
            # Find specific version
            for old_policy in reversed(history):
                if old_policy.version == version:
                    self.policies[policy_name] = old_policy
                    logger.info(f"Rolled back policy '{policy_name}' to version {version}")
                    return True
            logger.warning(f"Version {version} not found in history for '{policy_name}'")
            return False
        else:
            # Rollback to previous version
            old_policy = history.pop()
            self.policies[policy_name] = old_policy
            logger.info(f"Rolled back policy '{policy_name}' to version {old_policy.version}")
            return True
    
    def check_for_updates(self) -> List[str]:
        """
        Check if any policy files have been modified and reload them.
        
        Returns:
            List of reloaded policy file paths
        """
        reloaded = []
        
        for file_path, last_mtime in list(self._file_timestamps.items()):
            try:
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > last_mtime:
                    logger.info(f"Policy file {file_path} has been modified, reloading...")
                    self.load_policy_file(file_path)
                    reloaded.append(file_path)
            except FileNotFoundError:
                logger.warning(f"Policy file {file_path} no longer exists")
                del self._file_timestamps[file_path]
            except Exception as e:
                logger.error(f"Error checking for updates on {file_path}: {e}")
        
        return reloaded
    
    def evaluate_policies(self, action: Any, 
                         context: Optional[Dict[str, Any]] = None) -> List[SafetyViolation]:
        """
        Evaluate all enabled policies against an action.
        
        Args:
            action: Action to evaluate
            context: Optional context dictionary
            
        Returns:
            List of SafetyViolation objects for matched rules
        """
        violations = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            policy_violations = self.evaluate_policy(policy, action, context)
            violations.extend(policy_violations)
        
        return violations
    
    def evaluate_policy(self, policy: Policy, action: Any,
                       context: Optional[Dict[str, Any]] = None) -> List[SafetyViolation]:
        """
        Evaluate a single policy against an action.
        
        Args:
            policy: Policy to evaluate
            action: Action to evaluate
            context: Optional context dictionary
            
        Returns:
            List of SafetyViolation objects for matched rules
        """
        violations = []
        
        for rule in policy.rules:
            try:
                if self.evaluator.evaluate_condition(rule.condition, action, context):
                    # Rule matched, create violation
                    violation = SafetyViolation(
                        detector=f"PolicyEngine:{policy.name}",
                        severity=rule.severity.value.lower(),
                        description=rule.description or f"Policy rule matched: {rule.condition}",
                        category="policy_violation",
                        explanation=f"Action violated policy '{policy.name}' (v{policy.version})",
                        confidence=1.0,
                        recommendations=[action.value for action in rule.actions],
                        metadata={
                            "policy_name": policy.name,
                            "policy_version": policy.version,
                            "rule_condition": rule.condition,
                            "rule_actions": [action.value for action in rule.actions],
                            **rule.metadata
                        }
                    )
                    violations.append(violation)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule in policy '{policy.name}': {e}")
        
        return violations


# Global policy engine instance
_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Get the global policy engine instance."""
    global _policy_engine
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    return _policy_engine
