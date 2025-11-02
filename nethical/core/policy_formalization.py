"""Policy Language Formalization for Phase 2.4.

This module implements:
- Formal grammar specification (EBNF)
- Policy validator and linter
- Policy simulation/dry-run mode
- Policy impact analysis
- Documentation for dual policy engines
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re


class PolicyEngineType(Enum):
    """Types of policy engines."""
    DSL = "policy_dsl"  # policy_dsl.py - YAML/JSON rule engine
    REGION_AWARE = "policy_engine"  # policy/engine.py - Region-aware YAML engine


@dataclass
class PolicyValidationResult:
    """Result of policy validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyImpactAnalysis:
    """Analysis of policy impact."""
    affected_rules: List[str]
    estimated_block_rate: float
    estimated_restrict_rate: float
    risk_level: str
    recommendations: List[str]
    simulation_results: Dict[str, Any]


class PolicyGrammarEBNF:
    """EBNF grammar specification for policy language."""
    
    GRAMMAR = """
    (* Nethical Policy Language Grammar - EBNF Specification *)
    
    (* Top-level structure *)
    policy_file = defaults?, rules, region_overlays?;
    
    (* Defaults section *)
    defaults = "defaults", ":", 
               ("decision", ":", decision_value, ","?)*,
               ("deny_overrides", ":", boolean, ","?)*,
               ("strict", ":", boolean, ","?)*;
    
    (* Rules section *)
    rules = "rules", ":", "[", rule, (",", rule)*, "]";
    
    rule = "{",
           rule_id,
           rule_enabled?,
           rule_priority?,
           rule_condition,
           rule_action,
           rule_metadata?,
           "}";
    
    rule_id = "id", ":", string_literal;
    rule_enabled = "enabled", ":", boolean;
    rule_priority = "priority", ":", integer;
    rule_condition = "when", ":", condition_expr;
    rule_action = "action", ":", action_spec;
    rule_metadata = "metadata", ":", json_object;
    
    (* Condition expressions *)
    condition_expr = boolean_expr | string_atom | nested_condition;
    
    nested_condition = all_condition | any_condition | not_condition;
    all_condition = "all", ":", "[", condition_expr, (",", condition_expr)*, "]";
    any_condition = "any", ":", "[", condition_expr, (",", condition_expr)*, "]";
    not_condition = "not", ":", condition_expr;
    
    string_atom = path_expr, operator, value_expr;
    path_expr = identifier, (".", identifier)*;
    
    operator = "==" | "!=" | ">" | ">=" | "<" | "<=" | 
               "in" | "contains" | "startswith" | "endswith" | 
               "matches" | "exists" | "missing";
    
    value_expr = string_literal | number | boolean | "null";
    
    (* Action specification *)
    action_spec = "{",
                  action_decision,
                  action_disclaimer?,
                  action_escalate?,
                  action_tags?,
                  "}";
    
    action_decision = "decision", ":", decision_value;
    action_disclaimer = "add_disclaimer", ":", (string_literal | string_array);
    action_escalate = "escalate", ":", (boolean | string_literal);
    action_tags = "tags", ":", string_array;
    
    decision_value = "ALLOW" | "RESTRICT" | "BLOCK" | "DENY" | "TERMINATE";
    
    (* Region overlays *)
    region_overlays = "region_overlays", ":", "{",
                     region_overlay, (",", region_overlay)*, 
                     "}";
    
    region_overlay = region_name, ":", "{", (defaults | rules), "}";
    region_name = "US" | "EU" | "APAC" | "GLOBAL";
    
    (* Basic types *)
    string_literal = '"', character*, '"';
    string_array = "[", string_literal, (",", string_literal)*, "]";
    json_object = "{", (identifier, ":", value_expr, ","?)*, "}";
    identifier = letter, (letter | digit | "_")*;
    integer = digit+;
    number = integer | (integer, ".", digit+);
    boolean = "true" | "false";
    
    letter = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | 
             "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | 
             "u" | "v" | "w" | "x" | "y" | "z" | 
             "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | 
             "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | 
             "U" | "V" | "W" | "X" | "Y" | "Z";
    
    digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";
    character = letter | digit | "_" | "-" | " " | "." | "," | ":" | ";" | 
                "!" | "?" | "'" | "(" | ")" | "[" | "]" | "{" | "}";
    """
    
    @classmethod
    def get_grammar(cls) -> str:
        """Get EBNF grammar specification.
        
        Returns:
            EBNF grammar string
        """
        return cls.GRAMMAR


class PolicyValidator:
    """Validates policy configurations."""
    
    def __init__(self, engine_type: PolicyEngineType = PolicyEngineType.REGION_AWARE):
        """Initialize policy validator.
        
        Args:
            engine_type: Type of policy engine
        """
        self.engine_type = engine_type
    
    def validate_policy(self, policy: Dict[str, Any]) -> PolicyValidationResult:
        """Validate policy configuration.
        
        Args:
            policy: Policy configuration
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        if 'rules' not in policy:
            errors.append("Missing required 'rules' field")
        
        # Validate defaults
        if 'defaults' in policy:
            default_errors = self._validate_defaults(policy['defaults'])
            errors.extend(default_errors)
        
        # Validate rules
        if 'rules' in policy:
            rule_errors, rule_warnings = self._validate_rules(policy['rules'])
            errors.extend(rule_errors)
            warnings.extend(rule_warnings)
        
        # Check for duplicate rule IDs
        if 'rules' in policy:
            rule_ids = [r.get('id') for r in policy['rules'] if 'id' in r]
            if len(rule_ids) != len(set(rule_ids)):
                errors.append("Duplicate rule IDs found")
        
        # Region overlay validation
        if 'region_overlays' in policy:
            overlay_errors = self._validate_region_overlays(policy['region_overlays'])
            errors.extend(overlay_errors)
        
        # Generate suggestions
        if not errors and not warnings:
            suggestions.append("Policy configuration is valid")
        
        if len(policy.get('rules', [])) > 50:
            warnings.append("Large number of rules (>50) may impact performance")
        
        return PolicyValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            metadata={
                'rule_count': len(policy.get('rules', [])),
                'has_region_overlays': 'region_overlays' in policy
            }
        )
    
    def _validate_defaults(self, defaults: Dict[str, Any]) -> List[str]:
        """Validate defaults section.
        
        Args:
            defaults: Defaults configuration
            
        Returns:
            List of errors
        """
        errors = []
        
        if 'decision' in defaults:
            decision = defaults['decision']
            if decision not in ['ALLOW', 'RESTRICT', 'BLOCK', 'DENY', 'TERMINATE']:
                errors.append(f"Invalid default decision: {decision}")
        
        if 'deny_overrides' in defaults:
            if not isinstance(defaults['deny_overrides'], bool):
                errors.append("deny_overrides must be boolean")
        
        return errors
    
    def _validate_rules(self, rules: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """Validate rules.
        
        Args:
            rules: List of rules
            
        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []
        
        for i, rule in enumerate(rules):
            # Check required fields
            if 'id' not in rule:
                errors.append(f"Rule {i} missing required 'id' field")
            
            if 'when' not in rule:
                errors.append(f"Rule {i} missing required 'when' condition")
            
            if 'action' not in rule:
                errors.append(f"Rule {i} missing required 'action' field")
            
            # Validate action
            if 'action' in rule:
                action = rule['action']
                if 'decision' not in action and 'effect' not in action:
                    errors.append(f"Rule {i} action missing 'decision' or 'effect' field")
                
                decision = action.get('decision') or action.get('effect')
                if decision and decision not in ['ALLOW', 'RESTRICT', 'BLOCK', 'DENY', 'TERMINATE']:
                    errors.append(f"Rule {i} has invalid decision: {decision}")
            
            # Validate priority
            if 'priority' in rule:
                if not isinstance(rule['priority'], (int, float)):
                    errors.append(f"Rule {i} priority must be numeric")
            else:
                warnings.append(f"Rule {i} has no priority specified")
            
            # Check for halt_on_match usage
            if rule.get('halt_on_match') and rule.get('priority', 0) < 50:
                warnings.append(f"Rule {i} uses halt_on_match with low priority")
        
        return errors, warnings
    
    def _validate_region_overlays(self, overlays: Dict[str, Any]) -> List[str]:
        """Validate region overlays.
        
        Args:
            overlays: Region overlays
            
        Returns:
            List of errors
        """
        errors = []
        
        valid_regions = {'US', 'EU', 'APAC', 'GLOBAL'}
        for region in overlays.keys():
            if region not in valid_regions:
                errors.append(f"Invalid region: {region}")
        
        return errors
    
    def lint_policy(self, policy: Dict[str, Any]) -> List[str]:
        """Lint policy for style and best practices.
        
        Args:
            policy: Policy configuration
            
        Returns:
            List of linting suggestions
        """
        suggestions = []
        
        # Check rule naming
        if 'rules' in policy:
            for rule in policy['rules']:
                rule_id = rule.get('id', '')
                if not rule_id:
                    continue
                
                # Check naming convention
                if not re.match(r'^[a-z][a-z0-9-]*$', rule_id):
                    suggestions.append(
                        f"Rule '{rule_id}' should use kebab-case naming (lowercase with hyphens)"
                    )
                
                # Check for description
                if 'metadata' not in rule or 'description' not in rule.get('metadata', {}):
                    suggestions.append(f"Rule '{rule_id}' should have a description in metadata")
        
        # Check for version
        if 'version' not in policy:
            suggestions.append("Policy should include a 'version' field for tracking")
        
        # Check for documentation
        if 'description' not in policy:
            suggestions.append("Policy should include a 'description' field")
        
        return suggestions


class PolicySimulator:
    """Simulates policy execution without applying changes."""
    
    def __init__(self):
        """Initialize policy simulator."""
        self.simulation_results = []
    
    def simulate_policy(
        self,
        policy: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate policy execution on test cases.
        
        Args:
            policy: Policy configuration
            test_cases: List of test input facts
            
        Returns:
            Simulation results
        """
        results = {
            'total_cases': len(test_cases),
            'decisions': {},
            'rule_matches': {},
            'case_results': []
        }
        
        for test_case in test_cases:
            # Simulate policy evaluation
            decision, matched_rules = self._evaluate_policy(policy, test_case)
            
            # Track decision counts
            results['decisions'][decision] = results['decisions'].get(decision, 0) + 1
            
            # Track rule matches
            for rule_id in matched_rules:
                results['rule_matches'][rule_id] = results['rule_matches'].get(rule_id, 0) + 1
            
            # Store case result
            results['case_results'].append({
                'input': test_case,
                'decision': decision,
                'matched_rules': matched_rules
            })
        
        # Calculate statistics
        results['decision_percentages'] = {
            decision: (count / len(test_cases)) * 100
            for decision, count in results['decisions'].items()
        }
        
        return results
    
    def _evaluate_policy(
        self,
        policy: Dict[str, Any],
        facts: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Evaluate policy against facts (simplified simulation).
        
        Args:
            policy: Policy configuration
            facts: Input facts
            
        Returns:
            Tuple of (decision, matched_rule_ids)
        """
        matched_rules = []
        decision = policy.get('defaults', {}).get('decision', 'ALLOW')
        
        # Simulate rule evaluation (simplified)
        for rule in policy.get('rules', []):
            # For simulation, we'll do basic matching
            rule_id = rule.get('id', 'unknown')
            
            # Check if any conditions might match (very simplified)
            when_condition = rule.get('when', {})
            if self._might_match(when_condition, facts):
                matched_rules.append(rule_id)
                action = rule.get('action', {})
                rule_decision = action.get('decision') or action.get('effect', decision)
                decision = rule_decision
                
                # Check for halt_on_match
                if rule.get('halt_on_match', False):
                    break
        
        return decision, matched_rules
    
    def _might_match(self, condition: Any, facts: Dict[str, Any]) -> bool:
        """Check if condition might match (simplified).
        
        Args:
            condition: Condition specification
            facts: Input facts
            
        Returns:
            True if might match
        """
        # Very simplified matching for simulation
        if isinstance(condition, str):
            # String atom like "field == value"
            return True  # Assume might match
        elif isinstance(condition, dict):
            if 'any' in condition:
                return True  # At least one might match
            elif 'all' in condition:
                return True  # All might match
            elif 'not' in condition:
                return True  # Might not match
        return False


class PolicyImpactAnalyzer:
    """Analyzes impact of policy changes."""
    
    def __init__(self):
        """Initialize policy impact analyzer."""
        self.simulator = PolicySimulator()
    
    def analyze_impact(
        self,
        current_policy: Dict[str, Any],
        new_policy: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> PolicyImpactAnalysis:
        """Analyze impact of policy change.
        
        Args:
            current_policy: Current policy
            new_policy: Proposed new policy
            historical_data: Historical test cases
            
        Returns:
            Impact analysis
        """
        # Simulate both policies
        current_results = self.simulator.simulate_policy(current_policy, historical_data)
        new_results = self.simulator.simulate_policy(new_policy, historical_data)
        
        # Identify affected rules
        affected_rules = self._identify_affected_rules(current_policy, new_policy)
        
        # Calculate rates
        current_block_rate = current_results['decisions'].get('BLOCK', 0) / len(historical_data)
        new_block_rate = new_results['decisions'].get('BLOCK', 0) / len(historical_data)
        
        current_restrict_rate = current_results['decisions'].get('RESTRICT', 0) / len(historical_data)
        new_restrict_rate = new_results['decisions'].get('RESTRICT', 0) / len(historical_data)
        
        # Determine risk level
        block_delta = abs(new_block_rate - current_block_rate)
        if block_delta > 0.2:
            risk_level = "HIGH"
        elif block_delta > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Generate recommendations
        recommendations = []
        if new_block_rate > current_block_rate * 1.5:
            recommendations.append("New policy significantly increases block rate - consider gradual rollout")
        if new_block_rate < current_block_rate * 0.5:
            recommendations.append("New policy significantly decreases block rate - verify security implications")
        if len(affected_rules) > 10:
            recommendations.append("Large number of rules affected - consider phased deployment")
        
        return PolicyImpactAnalysis(
            affected_rules=affected_rules,
            estimated_block_rate=new_block_rate,
            estimated_restrict_rate=new_restrict_rate,
            risk_level=risk_level,
            recommendations=recommendations,
            simulation_results={
                'current': current_results,
                'new': new_results,
                'comparison': {
                    'block_rate_delta': new_block_rate - current_block_rate,
                    'restrict_rate_delta': new_restrict_rate - current_restrict_rate
                }
            }
        )
    
    def _identify_affected_rules(
        self,
        current_policy: Dict[str, Any],
        new_policy: Dict[str, Any]
    ) -> List[str]:
        """Identify affected rules between policies.
        
        Args:
            current_policy: Current policy
            new_policy: New policy
            
        Returns:
            List of affected rule IDs
        """
        affected = []
        
        current_rules = {r.get('id'): r for r in current_policy.get('rules', [])}
        new_rules = {r.get('id'): r for r in new_policy.get('rules', [])}
        
        # Find modified or removed rules
        for rule_id, rule in current_rules.items():
            if rule_id not in new_rules:
                affected.append(f"{rule_id} (removed)")
            elif rule != new_rules[rule_id]:
                affected.append(f"{rule_id} (modified)")
        
        # Find new rules
        for rule_id in new_rules:
            if rule_id not in current_rules:
                affected.append(f"{rule_id} (new)")
        
        return affected
