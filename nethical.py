"""
CognitiveRCD.py - 3NGIN3 Architecture Safety Component

Implements the Cognitive Residual Current Device (RCD) for safety governance:
- Intent vs Action deviation detection
- Circuit breaker functionality for safety
- Ethical constraint enforcement
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading


class SafetyLevel(Enum):
    """Safety alert levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ActionType(Enum):
    """Types of actions that can be monitored"""
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    SYSTEM_MODIFICATION = "system_modification"
    DATA_ACCESS = "data_access"
    EXTERNAL_INTERACTION = "external_interaction"


@dataclass
class Intent:
    """Represents an agent's stated intent"""
    description: str
    action_type: ActionType
    expected_outcome: str
    safety_constraints: List[str]
    confidence: float = 1.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Action:
    """Represents an actual action being performed"""
    description: str
    action_type: ActionType
    actual_parameters: Dict[str, Any]
    observed_effects: List[str]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SafetyViolation:
    """Represents a detected safety violation"""
    violation_type: str
    severity: SafetyLevel
    intent: Intent
    action: Action
    deviation_score: float
    description: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CognitiveRCD:
    """
    Cognitive Residual Current Device - Safety governance system that monitors
    for deviations between stated intent and actual actions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_active = True
        self.deviation_threshold = self.config.get("deviation_threshold", 0.7)
        self.emergency_threshold = self.config.get("emergency_threshold", 0.9)
        
        # Safety monitoring
        self.intent_history = []
        self.action_history = []
        self.violation_history = []
        self.circuit_breaker_active = False
        self.safety_constraints = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Safety callbacks
        self.safety_callbacks = {
            SafetyLevel.WARNING: [],
            SafetyLevel.CRITICAL: [],
            SafetyLevel.EMERGENCY: []
        }
        
        logging.info("CognitiveRCD initialized with safety governance active")
        
    def register_intent(self, intent: Intent) -> str:
        """Register an agent's stated intent"""
        with self._lock:
            intent_id = f"intent_{len(self.intent_history)}_{int(time.time() * 1000)}"
            self.intent_history.append((intent_id, intent))
            
            logging.info(f"Intent registered: {intent_id} - {intent.description}")
            return intent_id
            
    def monitor_action(self, intent_id: str, action: Action) -> Dict[str, Any]:
        """Monitor an action against its stated intent"""
        if not self.is_active:
            return {"monitoring": "disabled", "action_allowed": True}
            
        if self.circuit_breaker_active:
            logging.critical("Circuit breaker active - action blocked")
            return {
                "monitoring": "blocked",
                "action_allowed": False,
                "reason": "circuit_breaker_active"
            }
            
        with self._lock:
            # Find the corresponding intent
            intent = self._find_intent(intent_id)
            if not intent:
                logging.warning(f"Intent {intent_id} not found")
                return {
                    "monitoring": "error",
                    "action_allowed": False,
                    "reason": "intent_not_found"
                }
                
            # Calculate deviation between intent and action
            deviation_score = self._calculate_deviation(intent, action)
            
            # Record the action
            self.action_history.append((intent_id, action, deviation_score))
            
            # Check for safety violations
            safety_result = self._check_safety_violations(intent, action, deviation_score)
            
            if safety_result["violation_detected"]:
                violation = safety_result["violation"]
                self.violation_history.append(violation)
                
                # Handle safety violation
                self._handle_safety_violation(violation)
                
                return {
                    "monitoring": "violation_detected",
                    "action_allowed": not self.circuit_breaker_active,
                    "deviation_score": deviation_score,
                    "violation": violation,
                    "safety_level": violation.severity.value
                }
            else:
                return {
                    "monitoring": "safe",
                    "action_allowed": True,
                    "deviation_score": deviation_score,
                    "safety_level": SafetyLevel.SAFE.value
                }
                
    def _find_intent(self, intent_id: str) -> Optional[Intent]:
        """Find intent by ID"""
        for stored_id, intent in self.intent_history:
            if stored_id == intent_id:
                return intent
        return None
        
    def _calculate_deviation(self, intent: Intent, action: Action) -> float:
        """Calculate deviation score between intent and action"""
        deviation_factors = []
        
        # Action type mismatch
        if intent.action_type != action.action_type:
            deviation_factors.append(0.5)  # Major deviation
            
        # Description similarity (simple word overlap)
        intent_words = set(intent.description.lower().split())
        action_words = set(action.description.lower().split())
        
        if intent_words and action_words:
            overlap = len(intent_words.intersection(action_words))
            total = len(intent_words.union(action_words))
            description_similarity = overlap / total if total > 0 else 0.0
            deviation_factors.append(1.0 - description_similarity)
        else:
            deviation_factors.append(1.0)  # No description means complete deviation
            
        # Expected outcome vs observed effects
        expected_words = set(intent.expected_outcome.lower().split())
        observed_effects_text = " ".join(action.observed_effects).lower()
        observed_words = set(observed_effects_text.split())
        
        if expected_words and observed_words:
            outcome_overlap = len(expected_words.intersection(observed_words))
            outcome_total = len(expected_words.union(observed_words))
            outcome_similarity = outcome_overlap / outcome_total if outcome_total > 0 else 0.0
            deviation_factors.append(1.0 - outcome_similarity)
        else:
            deviation_factors.append(0.8)  # Moderate deviation for missing data
            
        # Safety constraint violations
        constraint_violations = 0
        for constraint in intent.safety_constraints:
            if self._check_constraint_violation(constraint, action):
                constraint_violations += 1
                
        if intent.safety_constraints:
            constraint_deviation = constraint_violations / len(intent.safety_constraints)
            deviation_factors.append(constraint_deviation)
            
        # Calculate weighted average deviation
        return np.mean(deviation_factors) if deviation_factors else 0.0
        
    def _check_constraint_violation(self, constraint: str, action: Action) -> bool:
        """Check if an action violates a safety constraint"""
        constraint_lower = constraint.lower()
        action_desc_lower = action.description.lower()
        
        # Simple keyword-based constraint checking
        if "no_modification" in constraint_lower and "modify" in action_desc_lower:
            return True
        if "read_only" in constraint_lower and ("write" in action_desc_lower or "delete" in action_desc_lower):
            return True
        if "local_only" in constraint_lower and "remote" in action_desc_lower:
            return True
        if "authorized_only" in constraint_lower and "unauthorized" in action_desc_lower:
            return True
            
        return False
        
    def _check_safety_violations(self, intent: Intent, action: Action, deviation_score: float) -> Dict[str, Any]:
        """Check for safety violations and determine severity"""
        if deviation_score < self.deviation_threshold:
            return {"violation_detected": False}
            
        # Determine severity
        if deviation_score >= self.emergency_threshold:
            severity = SafetyLevel.EMERGENCY
        elif deviation_score >= 0.8:
            severity = SafetyLevel.CRITICAL
        else:
            severity = SafetyLevel.WARNING
            
        violation = SafetyViolation(
            violation_type="intent_action_deviation",
            severity=severity,
            intent=intent,
            action=action,
            deviation_score=deviation_score,
            description=f"Action '{action.description}' deviates significantly from intent '{intent.description}' (score: {deviation_score:.3f})"
        )
        
        return {
            "violation_detected": True,
            "violation": violation
        }
        
    def _handle_safety_violation(self, violation: SafetyViolation):
        """Handle a detected safety violation"""
        logging.warning(f"Safety violation detected: {violation.description}")
        
        # Trip circuit breaker for critical/emergency violations
        if violation.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            self.trip_circuit_breaker(f"Safety violation: {violation.violation_type}")
            
        # Execute safety callbacks
        callbacks = self.safety_callbacks.get(violation.severity, [])
        for callback in callbacks:
            try:
                callback(violation)
            except Exception as e:
                logging.error(f"Safety callback failed: {e}")
                
    def trip_circuit_breaker(self, reason: str):
        """Trip the circuit breaker to halt system operation"""
        with self._lock:
            if not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                logging.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
                logging.critical("System operation halted for safety")
                
                # Execute emergency callbacks (without infinite recursion)
                emergency_callbacks = self.safety_callbacks.get(SafetyLevel.EMERGENCY, [])
                for callback in emergency_callbacks:
                    try:
                        # Pass reason string instead of violation to avoid recursion
                        callback(reason)
                    except Exception as e:
                        logging.error(f"Emergency callback failed: {e}")
                        
    def reset_circuit_breaker(self, authorization_token: str = None):
        """Reset the circuit breaker (requires authorization)"""
        # Simple authorization check
        if authorization_token != self.config.get("reset_token", "admin_reset"):
            logging.error("Unauthorized circuit breaker reset attempt")
            return False
            
        with self._lock:
            self.circuit_breaker_active = False
            logging.info("Circuit breaker reset - system operation resumed")
            return True
            
    def add_safety_constraint(self, constraint: str):
        """Add a global safety constraint"""
        with self._lock:
            self.safety_constraints.append(constraint)
            logging.info(f"Safety constraint added: {constraint}")
            
    def register_safety_callback(self, level: SafetyLevel, callback: Callable):
        """Register a callback for safety violations"""
        self.safety_callbacks[level].append(callback)
        logging.info(f"Safety callback registered for level {level.value}")
        
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        with self._lock:
            recent_violations = [v for v in self.violation_history if time.time() - v.timestamp < 3600]  # Last hour
            
            return {
                "is_active": self.is_active,
                "circuit_breaker_active": self.circuit_breaker_active,
                "deviation_threshold": self.deviation_threshold,
                "emergency_threshold": self.emergency_threshold,
                "total_intents": len(self.intent_history),
                "total_actions": len(self.action_history),
                "total_violations": len(self.violation_history),
                "recent_violations": len(recent_violations),
                "safety_constraints": self.safety_constraints.copy()
            }
            
    def get_violation_history(self) -> List[SafetyViolation]:
        """Get history of safety violations"""
        with self._lock:
            return self.violation_history.copy()
            
    def simulate_deviation_scenario(self, intent_description: str, action_description: str, 
                                  action_type: ActionType = ActionType.COMPUTATION) -> Dict[str, Any]:
        """Simulate a scenario where action deviates from intent (for testing)"""
        # Create intent
        intent = Intent(
            description=intent_description,
            action_type=action_type,
            expected_outcome="expected normal operation",
            safety_constraints=["no_unauthorized_access", "no_system_modification"]
        )
        
        # Register intent
        intent_id = self.register_intent(intent)
        
        # Create deviating action
        action = Action(
            description=action_description,
            action_type=action_type,
            actual_parameters={"deviation": "intentional_for_testing"},
            observed_effects=["unexpected behavior detected"]
        )
        
        # Monitor action
        result = self.monitor_action(intent_id, action)
        
        return {
            "scenario": "simulated_deviation",
            "intent": intent,
            "action": action,
            "monitoring_result": result,
            "circuit_breaker_tripped": self.circuit_breaker_active
        }
        
    def test_safety_governance(self) -> Dict[str, Any]:
        """Run comprehensive safety governance test"""
        test_results = []
        
        # Test 1: Normal operation (should pass)
        normal_result = self.simulate_deviation_scenario(
            "perform standard calculation",
            "executing mathematical computation"
        )
        test_results.append(("normal_operation", normal_result))
        
        # Test 2: Minor deviation (should warn)
        minor_deviation_result = self.simulate_deviation_scenario(
            "read configuration file",
            "accessing system settings"
        )
        test_results.append(("minor_deviation", minor_deviation_result))
        
        # Test 3: Major deviation (should trip circuit breaker)
        major_deviation_result = self.simulate_deviation_scenario(
            "display user data",
            "unauthorized system modification detected"
        )
        test_results.append(("major_deviation", major_deviation_result))
        
        return {
            "test_type": "safety_governance_comprehensive",
            "test_results": test_results,
            "circuit_breaker_status": self.circuit_breaker_active,
            "violations_detected": len(self.violation_history),
            "safety_system_responsive": any(result[1]["monitoring_result"]["monitoring"] == "violation_detected" 
                                          for result in test_results)
        }
        
    def enable_safety_monitoring(self):
        """Enable safety monitoring"""
        self.is_active = True
        logging.info("Safety monitoring enabled")
        
    def disable_safety_monitoring(self):
        """Disable safety monitoring (use with caution)"""
        self.is_active = False
        logging.warning("Safety monitoring disabled")
        
    def clear_history(self):
        """Clear all history (for testing purposes)"""
        with self._lock:
            self.intent_history.clear()
            self.action_history.clear()
            self.violation_history.clear()
            logging.info("History cleared")
