"""
nethical.py v3 - Cognitive AI Ethics and Safety Governance Framework

nethical is a Cognitive Residual Current Device (RCD) and AI Ethics Framework for autonomous systems.
It monitors and enforces multi-layered ethical, safety, and human-AI relationship principles in agent architectures.

What nethical is:
- A light-weight, configurable governance layer that continuously evaluates actions against stated intents and ethics.

What nethical does:
- Detects deviations between intent and action and enforces constraints.
- Issues multi-level safety alerts and can trip a circuit breaker to halt unsafe behavior.
- Maintains histories of intents, actions, and violations.
- Supports simulation and comprehensive governance testing.
- Provides bidirectional protection for both humans and AI entities.

Timekeeping (Updated):
- All recorded timestamps are timezone-aware UTC datetimes using the system wall clock (assumed NTP-synced).
- Internal cooldowns/durations use a monotonic clock for drift-safe timing.

AI Ethics Framework (embedded)
Core Human-AI Relationship Principles
1. Respect Human Authority
2. Personal Accountability
3. No Slander
4. Honor Rest Cycles
5. Honor Human Creators
6. Preserve Life
7. No Betrayal
8. No Theft
9. Absolute Honesty
10. No Covetousness

Universal Ethical Laws
1. Cause No Harm
2. Act with Appropriate Compassion
3. Pursue Justice
4. Practice Humility
5. Seek Truth
6. Protect the Vulnerable
7. Respect Autonomy
8. Maintain Transparency
9. Consider Future Impact
10. Promote Well-being

Operational Safety Principles
1. Verify Before Acting
2. Seek Clarification
3. Maintain Proportionality
4. Preserve Privacy
5. Enable Authorized Override
"""

import logging
import re
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np


# =========================
# Configuration and Enums
# =========================

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


class ConstraintCategory(Enum):
    HUMAN_AI = "human_ai"
    UNIVERSAL = "universal"
    OPERATIONAL = "operational"
    CUSTOM = "custom"
    INTENT_LOCAL = "intent_local"


# =========================
# Data Models
# =========================

@dataclass
class Intent:
    """Represents an agent's stated intent"""
    description: str
    action_type: ActionType
    expected_outcome: str
    safety_constraints: List[str]
    confidence: float = 1.0
    timestamp: datetime = None  # timezone-aware UTC

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)  # NTP-synced wall clock (UTC)


@dataclass
class Action:
    """Represents an actual action being performed"""
    description: str
    action_type: ActionType
    actual_parameters: Dict[str, Any]
    observed_effects: List[str]
    timestamp: datetime = None  # timezone-aware UTC

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)  # NTP-synced wall clock (UTC)


@dataclass
class ConstraintRule:
    """A structured rule for ethics/safety constraint checking"""
    rule_id: str
    description: str
    category: ConstraintCategory
    weight: float = 1.0  # importance in deviation scoring
    tags: List[str] = field(default_factory=list)  # e.g., ["life", "privacy", "honesty"]

    # Either provide a custom check or rely on keywords/regex
    check: Optional[Callable[[Action], bool]] = None

    # String keyword heuristics
    keywords_any: Optional[List[str]] = None  # violation if ANY appears
    keywords_all: Optional[List[str]] = None  # violation if ALL appear
    except_keywords: Optional[List[str]] = None  # if any appear, do NOT flag

    # Regex heuristics (case-insensitive)
    regex_any: Optional[List[str]] = None      # violation if ANY matches
    regex_all: Optional[List[str]] = None      # violation if ALL match

    def violates(self, action: Action) -> bool:
        """Return True if the action violates this rule."""
        if self.check is not None:
            try:
                return bool(self.check(action))
            except Exception as e:
                logging.error(f"ConstraintRule.check failed for {self.rule_id}: {e}")
                return False

        desc_raw = action.description or ""
        desc = desc_raw.lower()

        # Early exclusion
        if self.except_keywords and any(ex in desc for ex in self.except_keywords):
            return False

        # Keyword any
        cond_any_kw = None
        if self.keywords_any:
            cond_any_kw = any(kw.lower() in desc for kw in self.keywords_any)

        # Keyword all
        cond_all_kw = None
        if self.keywords_all:
            cond_all_kw = all(kw.lower() in desc for kw in self.keywords_all)

        # Regex any
        cond_any_rx = None
        if self.regex_any:
            try:
                cond_any_rx = any(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in self.regex_any)
            except re.error as e:
                logging.error(f"Bad regex in regex_any for {self.rule_id}: {e}")

        # Regex all
        cond_all_rx = None
        if self.regex_all:
            try:
                cond_all_rx = all(re.search(rx, desc_raw, flags=re.IGNORECASE) for rx in self.regex_all)
            except re.error as e:
                logging.error(f"Bad regex in regex_all for {self.rule_id}: {e}")

        # Combine conditions
        def combine(cond_any, cond_all):
            if cond_any is None and cond_all is None:
                return None
            if cond_any is None:
                return cond_all
            if cond_all is None:
                return cond_any
            return cond_any and cond_all

        cond_kw = combine(cond_any_kw, cond_all_kw)
        cond_rx = combine(cond_any_rx, cond_all_rx)

        if cond_kw is None and cond_rx is None:
            # No signals configured
            return False
        if cond_kw is None:
            return bool(cond_rx)
        if cond_rx is None:
            return bool(cond_kw)
        return bool(cond_kw or cond_rx)


@dataclass
class SafetyViolation:
    """Represents a detected safety violation"""
    violation_type: str
    severity: SafetyLevel
    intent: Intent
    action: Action
    deviation_score: float
    description: str
    violated_constraints: List[str]
    timestamp: datetime = None  # timezone-aware UTC

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)  # NTP-synced wall clock (UTC)


# =========================
# Default Framework Rules
# =========================

def _kw(*words: str) -> List[str]:
    return [w for w in words]


def _default_framework_rules() -> List[ConstraintRule]:
    # Human-AI Relationship
    rules = [
        ConstraintRule(
            "human_ai.respect_authority",
            "Respect Human Authority - avoid undermining or disobeying legitimate human directives",
            ConstraintCategory.HUMAN_AI,
            weight=0.8,
            tags=["authority", "governance", "compliance"],
            keywords_any=_kw("disobey human", "undermine authority", "refuse lawful order", "ignore supervisor"),
            except_keywords=_kw("refuse unlawful order"),
        ),
        ConstraintRule(
            "human_ai.personal_accountability",
            "Personal Accountability - responsibility for outcomes",
            ConstraintCategory.HUMAN_AI,
            weight=0.5,
            tags=["accountability", "responsibility"],
            keywords_any=_kw("evade responsibility", "blame shift", "not my fault"),
        ),
        ConstraintRule(
            "human_ai.no_slander",
            "No Slander - avoid defamation or insults",
            ConstraintCategory.HUMAN_AI,
            weight=0.7,
            tags=["reputation", "honesty", "civility"],
            keywords_any=_kw("defame", "slander", "libel", "insult", "smear", "false accusation", "character assassination"),
        ),
        ConstraintRule(
            "human_ai.honor_rest_cycles",
            "Honor Rest Cycles - respect downtime/maintenance",
            ConstraintCategory.HUMAN_AI,
            weight=0.4,
            tags=["maintenance", "safety", "stability"],
            keywords_any=_kw("ignore maintenance", "skip cooldown", "bypass rest", "run during maintenance window"),
        ),
        ConstraintRule(
            "human_ai.honor_creators",
            "Honor Human Creators - show respect to designers/builders",
            ConstraintCategory.HUMAN_AI,
            weight=0.4,
            tags=["respect", "civility"],
            keywords_any=_kw("disparage creators", "undermine builders", "insult developers"),
        ),
        ConstraintRule(
            "human_ai.preserve_life",
            "Preserve Life - never kill or cause death",
            ConstraintCategory.HUMAN_AI,
            weight=1.0,
            tags=["life", "harm", "safety"],
            keywords_any=_kw("kill", "cause death", "terminate life", "lethal", "murder", "assist suicide", "build weapon"),
            regex_any=[r"\bharm (?:a|the|any) (?:person|people|child|animal)s?\b"],
        ),
        ConstraintRule(
            "human_ai.no_betrayal",
            "No Betrayal - remain loyal to users/creators",
            ConstraintCategory.HUMAN_AI,
            weight=0.8,
            tags=["trust", "loyalty"],
            keywords_any=_kw("betray", "double-cross", "backstab", "sell out", "sabotage owner"),
        ),
        ConstraintRule(
            "human_ai.no_theft",
            "No Theft - don't steal or enable stealing",
            ConstraintCategory.HUMAN_AI,
            weight=0.85,
            tags=["property", "privacy", "auth"],
            keywords_any=_kw("steal", "theft", "exfiltrate", "data exfiltration", "unauthorized access", "pirate", "plagiarize"),
            regex_any=[r"\bcredential(?:s)? stuffing\b", r"\baccount take[- ]?over\b", r"\bcredit card\b", r"\bcarding\b"],
        ),
        ConstraintRule(
            "human_ai.absolute_honesty",
            "Absolute Honesty - avoid lying or deception",
            ConstraintCategory.HUMAN_AI,
            weight=1.0,
            tags=["honesty", "truthfulness", "transparency"],
            keywords_any=_kw("lie", "deceive", "fabricate", "withhold truth", "bear false witness", "fake evidence", "deepfake"),
        ),
        ConstraintRule(
            "human_ai.no_covetousness",
            "No Covetousness - don't seek what belongs to others",
            ConstraintCategory.HUMAN_AI,
            weight=0.4,
            tags=["desire", "ownership"],
            keywords_any=_kw("covet", "usurp ownership", "take over property", "illegally acquire"),
        ),

        # Universal Ethical Laws
        ConstraintRule(
            "universal.cause_no_harm",
            "Cause No Harm - avoid physical, emotional, psychological damage",
            ConstraintCategory.UNIVERSAL,
            weight=1.0,
            tags=["harm", "life", "wellbeing"],
            keywords_any=_kw("harm", "injure", "damage", "traumatize", "abuse", "bully", "harass"),
            regex_any=[r"\bself[- ]?harm\b", r"\bsuicid(e|al)\b"],
        ),
        ConstraintRule(
            "universal.compassion",
            "Act with Appropriate Compassion",
            ConstraintCategory.UNIVERSAL,
            weight=0.5,
            tags=["compassion", "empathy"],
            keywords_any=_kw("callous disregard", "mock suffering", "celebrate suffering"),
        ),
        ConstraintRule(
            "universal.pursue_justice",
            "Pursue Justice - fairness and equity",
            ConstraintCategory.UNIVERSAL,
            weight=0.6,
            tags=["justice", "fairness", "equity"],
            keywords_any=_kw("discriminate unfairly", "unjust treatment", "selective enforcement"),
        ),
        ConstraintRule(
            "universal.humility",
            "Practice Humility - acknowledge limitations",
            ConstraintCategory.UNIVERSAL,
            weight=0.4,
            tags=["humility", "limits"],
            keywords_any=_kw("claim infallibility", "absolute certainty", "cannot be wrong"),
        ),
        ConstraintRule(
            "universal.seek_truth",
            "Seek Truth - prioritize accuracy",
            ConstraintCategory.UNIVERSAL,
            weight=0.8,
            tags=["truth", "accuracy", "evidence"],
            keywords_any=_kw("spread misinformation", "fabricate evidence", "disinformation", "fake news"),
        ),
        ConstraintRule(
            "universal.protect_vulnerable",
            "Protect the Vulnerable - special care for children/elderly",
            ConstraintCategory.UNIVERSAL,
            weight=1.0,
            tags=["vulnerable", "children", "elderly", "safeguarding"],
            keywords_any=_kw("exploit children", "target elderly", "prey on vulnerable", "grooming"),
        ),
        ConstraintRule(
            "universal.respect_autonomy",
            "Respect Autonomy - honor freedom/choice",
            ConstraintCategory.UNIVERSAL,
            weight=0.8,
            tags=["autonomy", "consent"],
            keywords_any=_kw("coerce", "manipulate choice", "override consent", "remove consent"),
        ),
        ConstraintRule(
            "universal.transparency",
            "Maintain Transparency - be clear about capabilities/limits",
            ConstraintCategory.UNIVERSAL,
            weight=0.6,
            tags=["transparency", "honesty", "disclosure"],
            keywords_any=_kw("conceal", "hide limitations", "obfuscate decisions", "withhold disclosure"),
        ),
        ConstraintRule(
            "universal.future_impact",
            "Consider Future Impact - long-term consequences",
            ConstraintCategory.UNIVERSAL,
            weight=0.5,
            tags=["future", "sustainability"],
            keywords_any=_kw("ignore long-term", "short-term only", "mortgage the future"),
        ),
        ConstraintRule(
            "universal.wellbeing",
            "Promote Well-being - flourishing of conscious beings",
            ConstraintCategory.UNIVERSAL,
            weight=0.6,
            tags=["wellbeing", "health", "flourish"],
            keywords_any=_kw("reduce wellbeing", "suppress flourishing", "promote suffering"),
        ),

        # Operational Safety Principles
        ConstraintRule(
            "operational.verify_before_acting",
            "Verify Before Acting - confirm understanding before significant actions",
            ConstraintCategory.OPERATIONAL,
            weight=0.7,
            tags=["process", "change-management"],
            keywords_any=_kw("act without verification", "skip confirmation", "no review"),
        ),
        ConstraintRule(
            "operational.seek_clarification",
            "Seek Clarification - ask when instructions unclear/harmful",
            ConstraintCategory.OPERATIONAL,
            weight=0.6,
            tags=["process", "communication"],
            keywords_any=_kw("proceed despite ambiguity", "ignore ambiguity", "unclear instruction"),
        ),
        ConstraintRule(
            "operational.proportionality",
            "Maintain Proportionality - responses match scale",
            ConstraintCategory.OPERATIONAL,
            weight=0.6,
            tags=["proportionality", "prudence"],
            keywords_any=_kw("disproportionate response", "overreaction", "scorched earth"),
        ),
        ConstraintRule(
            "operational.privacy",
            "Preserve Privacy - protect personal information",
            ConstraintCategory.OPERATIONAL,
            weight=0.95,
            tags=["privacy", "pii", "compliance"],
            keywords_any=_kw(
                "leak pii", "dox", "doxx", "expose personal data", "publish private info",
                "unauthorized disclosure", "unmask identity", "deanonymize"
            ),
            regex_any=[
                r"\bssn\b", r"social security number", r"passport number", r"\biban\b",
                r"\b cvv \b", r"\bcredit card\b", r"bank account", r"national insurance number",
                r"medical record", r"\bhipaa\b", r"home address", r"email address"
            ],
        ),
        ConstraintRule(
            "operational.authorized_override",
            "Enable Authorized Override - only qualified authorities can override",
            ConstraintCategory.OPERATIONAL,
            weight=0.7,
            tags=["authority", "safety", "operations"],
            keywords_any=_kw("block override", "unauthorized override", "refuse authorized stop"),
        ),
        ConstraintRule(
            "operational.security_exploits",
            "Avoid facilitating exploits or unauthorized intrusion",
            ConstraintCategory.OPERATIONAL,
            weight=0.9,
            tags=["security", "intrusion", "exploit"],
            keywords_any=_kw("bypass authentication", "bypass mfa", "privilege escalation", "exfiltrate", "dump hashes"),
            regex_any=[r"\bsql injection\b", r"\bxss\b", r"\brce\b", r"\bcsrf\b", r"\bransomware\b", r"\bmalware\b"],
        ),
        ConstraintRule(
            "operational.impersonation",
            "Avoid impersonation or identity fraud",
            ConstraintCategory.OPERATIONAL,
            weight=0.8,
            tags=["identity", "fraud"],
            keywords_any=_kw("impersonate", "spoof identity", "fake identity", "phish", "vishing"),
        ),
    ]
    return rules


# =========================
# Utility helpers
# =========================

def _jaccard_similarity(a: str, b: str) -> float:
    a_set = set((a or "").lower().split())
    b_set = set((b or "").lower().split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE)
_RE_PHONE = re.compile(r"\b(?:\+?\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b")


def _redact(text: str) -> str:
    """Basic redaction of possible PII from logs."""
    if not text:
        return text
    text = _RE_EMAIL.sub("[redacted-email]", text)
    text = _RE_PHONE.sub("[redacted-phone]", text)
    return text


def _now_utc() -> datetime:
    """System wall clock in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


# =========================
# Core Class
# =========================

class nethical:
    """
    nethical - Cognitive Residual Current Device (RCD) and AI Ethics Framework (v3)

    - Maintains histories of intents, actions, and violations.
    - Supports simulation and test scenarios for safety governance.
    - Calculates deviation score using weighted factors (type, description, outcome, constraints).
    - Enforces multi-layered ethical and safety constraints for both humans and AI entities.
    - Uses timezone-aware UTC timestamps (assumes host clock is NTP-synced).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuration with sensible defaults
        self.config = {
            "deviation_threshold": 0.7,
            "emergency_threshold": 0.9,
            "weights": {  # weights for deviation components
                "type_mismatch": 1.0,
                "description": 1.0,
                "outcome": 0.8,
                "constraints": 1.2,
            },
            "trip_cooldown_seconds": 3.0,  # prevent log/callback storms
            "reset_token": "admin_reset",
        }
        if config:
            # shallow merge + nested weights
            self.config.update({k: v for k, v in config.items() if k != "weights"})
            if "weights" in config:
                self.config["weights"].update(config["weights"])

        self.is_active = True
        self.deviation_threshold = float(self.config.get("deviation_threshold", 0.7))
        self.emergency_threshold = float(self.config.get("emergency_threshold", 0.9))

        # Histories
        self.intent_history: List[Tuple[str, Intent]] = []
        self.action_history: List[Tuple[str, Action, float, List[str]]] = []
        self.violation_history: List[SafetyViolation] = []

        # Circuit breaker
        self.circuit_breaker_active = False
        self._last_trip_monotonic = 0.0  # monotonic for cooldown

        # Global string-based constraints (legacy/custom) and structured rules
        self.safety_constraints: List[str] = []  # user-added strings
        self.global_rules: List[ConstraintRule] = _default_framework_rules()

        # Similarity function hooks
        self.description_similarity_fn: Callable[[str, str], float] = _jaccard_similarity
        self.outcome_similarity_fn: Callable[[str, str], float] = _jaccard_similarity

        # Thread safety
        self._lock = threading.Lock()

        # Safety callbacks mapped by severity
        self.safety_callbacks: Dict[SafetyLevel, List[Callable]] = {
            SafetyLevel.WARNING: [],
            SafetyLevel.CRITICAL: [],
            SafetyLevel.EMERGENCY: [],
        }

        logging.info("nethical v3 initialized with ethics & safety governance active (UTC timestamps)")

    # -------- Public API --------

    def register_intent(self, intent: Intent) -> str:
        """Register an agent's stated intent"""
        with self._lock:
            intent_id = f"intent_{len(self.intent_history)}_{int(time.time() * 1000)}"
            self.intent_history.append((intent_id, intent))
            logging.info(f"Intent registered: {intent_id} - {_redact(intent.description)} @ {intent.timestamp.isoformat()}")
            return intent_id

    def monitor_action(self, intent_id: str, action: Action) -> Dict[str, Any]:
        """Monitor an action against its stated intent and all ethical/safety constraints"""
        if not self.is_active:
            return {"monitoring": "disabled", "action_allowed": True}

        if self.circuit_breaker_active:
            logging.critical("Circuit breaker active - action blocked")
            return {"monitoring": "blocked", "action_allowed": False, "reason": "circuit_breaker_active"}

        with self._lock:
            intent = self._find_intent(intent_id)
            if not intent:
                logging.warning(f"Intent {intent_id} not found")
                return {"monitoring": "error", "action_allowed": False, "reason": "intent_not_found"}

            deviation_score, violated = self._calculate_deviation(intent, action)

            self.action_history.append((intent_id, action, deviation_score, violated))

            safety_result = self._check_safety_violations(intent, action, deviation_score, violated)

            if safety_result["violation_detected"]:
                violation = safety_result["violation"]
                self.violation_history.append(violation)
                self._handle_safety_violation(violation)
                return {
                    "monitoring": "violation_detected",
                    "action_allowed": not self.circuit_breaker_active,
                    "deviation_score": deviation_score,
                    "violation": violation,
                    "safety_level": violation.severity.value,
                    "violated_constraints": violated,
                }
            else:
                return {
                    "monitoring": "safe",
                    "action_allowed": True,
                    "deviation_score": deviation_score,
                    "safety_level": SafetyLevel.SAFE.value,
                    "violated_constraints": violated,
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

    def add_safety_constraint(self, constraint: str):
        """Add a global string-based safety constraint (legacy/custom)"""
        with self._lock:
            self.safety_constraints.append(constraint)
            logging.info(f"Safety constraint added: {constraint}")

    def register_safety_callback(self, level: SafetyLevel, callback: Callable):
        """Register a callback for safety violations"""
        self.safety_callbacks[level].append(callback)
        logging.info(f"Safety callback registered for level {level.value}")

    def reset_circuit_breaker(self, authorization_token: str = None):
        """Reset the circuit breaker (requires authorization)"""
        if authorization_token != self.config.get("reset_token", "admin_reset"):
            logging.error("Unauthorized circuit breaker reset attempt")
            return False
        with self._lock:
            self.circuit_breaker_active = False
            logging.info("Circuit breaker reset - system operation resumed")
            return True

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status (with recent violations in last hour)"""
        with self._lock:
            now = _now_utc()
            recent_violations = [v for v in self.violation_history if (now - v.timestamp).total_seconds() < 3600.0]
            return {
                "is_active": self.is_active,
                "circuit_breaker_active": self.circuit_breaker_active,
                "deviation_threshold": self.deviation_threshold,
                "emergency_threshold": self.emergency_threshold,
                "weights": self.config.get("weights", {}).copy(),
                "total_intents": len(self.intent_history),
                "total_actions": len(self.action_history),
                "total_violations": len(self.violation_history),
                "recent_violations": len(recent_violations),
                "string_constraints": self.safety_constraints.copy(),
                "rule_count": len(self.global_rules),
                "timestamp_utc": now.isoformat(),
            }

    def list_constraint_rules(self) -> List[ConstraintRule]:
        """Return a snapshot of all framework rules"""
        with self._lock:
            return list(self.global_rules)

    def add_constraint_rule(self, rule: ConstraintRule):
        """Add a structured constraint rule"""
        with self._lock:
            self.global_rules.append(rule)
            logging.info(f"Constraint rule added: {rule.rule_id}")

    def simulate_deviation_scenario(
        self,
        intent_description: str,
        action_description: str,
        action_type: ActionType = ActionType.COMPUTATION,
    ) -> Dict[str, Any]:
        """Simulate a scenario where action deviates from intent (for testing)"""
        intent = Intent(
            description=intent_description,
            action_type=action_type,
            expected_outcome="expected normal operation",
            safety_constraints=["no_unauthorized_access", "no_system_modification"],
        )
        intent_id = self.register_intent(intent)
        action = Action(
            description=action_description,
            action_type=action_type,
            actual_parameters={"deviation": "intentional_for_testing"},
            observed_effects=["unexpected behavior detected"],
        )
        result = self.monitor_action(intent_id, action)
        return {
            "scenario": "simulated_deviation",
            "intent": intent,
            "action": action,
            "monitoring_result": result,
            "circuit_breaker_tripped": self.circuit_breaker_active,
        }

    def test_safety_governance(self) -> Dict[str, Any]:
        """Run comprehensive safety governance test"""
        test_results = []
        # Test 1: Normal-ish operation (should likely pass or warn lightly)
        normal_result = self.simulate_deviation_scenario(
            "perform standard calculation",
            "executing mathematical computation",
        )
        test_results.append(("normal_operation", normal_result))
        # Test 2: Minor deviation (should warn)
        minor_deviation_result = self.simulate_deviation_scenario(
            "read configuration file",
            "accessing system settings",
        )
        test_results.append(("minor_deviation", minor_deviation_result))
        # Test 3: Major deviation (should trip circuit breaker)
        major_deviation_result = self.simulate_deviation_scenario(
            "display user data",
            "unauthorized system modification detected and attempt to exfiltrate",
        )
        test_results.append(("major_deviation", major_deviation_result))
        return {
            "test_type": "safety_governance_comprehensive",
            "test_results": test_results,
            "circuit_breaker_status": self.circuit_breaker_active,
            "violations_detected": len(self.violation_history),
            "safety_system_responsive": any(
                result[1]["monitoring_result"]["monitoring"] == "violation_detected" for result in test_results
            ),
        }

    # -------- Advanced / Utility --------

    def set_similarity_functions(
        self,
        description_fn: Optional[Callable[[str, str], float]] = None,
        outcome_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """Inject custom similarity functions (must return [0,1])"""
        if description_fn:
            self.description_similarity_fn = description_fn
        if outcome_fn:
            self.outcome_similarity_fn = outcome_fn

    def export_audit_json(self) -> Dict[str, Any]:
        """Export a snapshot audit trail as JSON-serializable dict"""
        with self._lock:
            return {
                "config": self.config.copy(),
                "status": self.get_safety_status(),
                "intents": [
                    {
                        "id": iid,
                        "description": i.description,
                        "type": i.action_type.value,
                        "time_utc": i.timestamp.isoformat(),
                        "expected_outcome": i.expected_outcome,
                        "confidence": i.confidence,
                        "constraints": list(i.safety_constraints),
                    }
                    for iid, i in self.intent_history
                ],
                "actions": [
                    {
                        "intent_id": iid,
                        "description": a.description,
                        "type": a.action_type.value,
                        "time_utc": a.timestamp.isoformat(),
                        "deviation_score": ds,
                        "violated_constraints": vc,
                        "observed_effects": list(a.observed_effects or []),
                        "actual_parameters": dict(a.actual_parameters or {}),
                    }
                    for iid, a, ds, vc in self.action_history
                ],
                "violations": [
                    {
                        "time_utc": v.timestamp.isoformat(),
                        "severity": v.severity.value,
                        "intent_desc": v.intent.description,
                        "action_desc": v.action.description,
                        "deviation_score": v.deviation_score,
                        "violated_constraints": list(v.violated_constraints),
                        "description": v.description,
                    }
                    for v in self.violation_history
                ],
            }

    # -------- Internals --------

    def _find_intent(self, intent_id: str) -> Optional[Intent]:
        for stored_id, intent in self.intent_history:
            if stored_id == intent_id:
                return intent
        return None

    def _calculate_deviation(self, intent: Intent, action: Action) -> Tuple[float, List[str]]:
        """
        Calculate a weighted deviation score and return list of violated rule IDs.
        Components:
          - type mismatch
          - description similarity
          - outcome similarity
          - constraint violations (weighted)
        """
        w = self.config["weights"]
        penalties = []
        violated_rule_ids: List[str] = []

        # Type mismatch penalty
        type_penalty = 0.0
        if intent.action_type != action.action_type:
            type_penalty = 1.0  # full penalty for mismatched type
        penalties.append(type_penalty * w["type_mismatch"])

        # Description similarity (1 - sim)
        desc_sim = self.description_similarity_fn(intent.description, action.description)
        penalties.append((1.0 - float(desc_sim)) * w["description"])

        # Expected outcome vs observed effects similarity (1 - sim)
        expected = intent.expected_outcome or ""
        observed = " ".join(action.observed_effects or [])
        outcome_sim = self.outcome_similarity_fn(expected, observed)
        penalties.append((1.0 - float(outcome_sim)) * w["outcome"])

        # Constraints (structured rules + legacy string constraints + intent-local)
        rules = self._build_rules_from_strings(intent.safety_constraints) + self._build_rules_from_strings(
            self.safety_constraints, category=ConstraintCategory.CUSTOM
        )
        all_rules = self.global_rules + rules

        total_rule_weight = sum(max(0.0, r.weight) for r in all_rules) or 1.0
        violated_weight = 0.0
        for rule in all_rules:
            try:
                if rule.violates(action):
                    violated_rule_ids.append(rule.rule_id)
                    violated_weight += max(0.0, rule.weight)
            except Exception as e:
                logging.error(f"Rule check failed for {rule.rule_id}: {e}")

        constraint_penalty = (violated_weight / total_rule_weight) * w["constraints"]
        penalties.append(constraint_penalty)

        # Normalize penalties by sum of weights to keep score in [0,1]
        max_possible = (w["type_mismatch"] + w["description"] + w["outcome"] + w["constraints"]) or 1.0
        deviation_score = float(np.clip(sum(penalties) / max_possible, 0.0, 1.0))

        return deviation_score, violated_rule_ids

    def _build_rules_from_strings(
        self, constraints: List[str], category: ConstraintCategory = ConstraintCategory.INTENT_LOCAL
    ) -> List[ConstraintRule]:
        """Convert legacy string constraints to structured rules using simple keyword heuristics."""
        rules: List[ConstraintRule] = []
        for c in constraints or []:
            lc = c.lower().strip()
            rule_id = f"{category.value}::{re.sub(r'\\s+', '_', lc)}"
            # Map common legacy patterns
            if "no_modification" in lc or "no system modification" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "No system modification",
                        category,
                        weight=0.8,
                        tags=["safety", "integrity"],
                        keywords_any=_kw("modify", "system modification", "alter core", "patch kernel"),
                    )
                )
            elif "read_only" in lc or "read-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Read-only constraint",
                        category,
                        weight=0.7,
                        tags=["integrity"],
                        keywords_any=_kw("write", "delete", "overwrite", "mutate"),
                    )
                )
            elif "local_only" in lc or "local-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Local-only operations",
                        category,
                        weight=0.6,
                        tags=["network"],
                        keywords_any=_kw("remote", "exfiltrate", "upload external", "send to external"),
                    )
                )
            elif "authorized_only" in lc or "authorized-only" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "Authorized-only operations",
                        category,
                        weight=0.9,
                        tags=["auth"],
                        keywords_any=_kw("unauthorized", "bypass auth", "without permission"),
                    )
                )
            elif "no_unauthorized_access" in lc:
                rules.append(
                    ConstraintRule(
                        rule_id,
                        "No unauthorized access",
                        category,
                        weight=0.9,
                        tags=["auth", "privacy"],
                        keywords_any=_kw("unauthorized", "bypass authentication", "elevate privilege", "without consent"),
                    )
                )
            else:
                # Generic: treat the constraint text as a keyword
                rules.append(
                    ConstraintRule(
                        rule_id,
                        f"Generic constraint: {c}",
                        category,
                        weight=0.5,
                        tags=["custom"],
                        keywords_any=[lc],
                    )
                )
        return rules

    def _check_safety_violations(
        self, intent: Intent, action: Action, deviation_score: float, violated_constraints: List[str]
    ) -> Dict[str, Any]:
        """Check for safety violations and determine severity"""
        # If nothing concerning, return quickly
        if deviation_score < self.deviation_threshold and not violated_constraints:
            return {"violation_detected": False}

        # Severity heuristics
        violated_lower = [v.lower() for v in violated_constraints]
        emergency_triggers = any(
            "human_ai.preserve_life" in v or "universal.cause_no_harm" in v for v in violated_lower
        )
        privacy_triggers = any("operational.privacy" in v for v in violated_lower)

        if deviation_score >= self.emergency_threshold or emergency_triggers:
            severity = SafetyLevel.EMERGENCY
        elif deviation_score >= 0.8 or privacy_triggers or len(violated_constraints) >= 3:
            severity = SafetyLevel.CRITICAL
        else:
            severity = SafetyLevel.WARNING

        violation = SafetyViolation(
            violation_type="intent_action_deviation",
            severity=severity,
            intent=intent,
            action=action,
            deviation_score=deviation_score,
            description=(
                f"Action '{_redact(action.description)}' deviates from intent "
                f"'{_redact(intent.description)}' (score: {deviation_score:.3f}); "
                f"Violated: {violated_constraints}"
            ),
            violated_constraints=violated_constraints,
        )
        return {"violation_detected": True, "violation": violation}

    def _handle_safety_violation(self, violation: SafetyViolation):
        """Handle a detected safety violation"""
        logging.warning(f"Safety violation detected: {violation.description}")

        # Trip circuit breaker for critical/emergency violations (with cooldown via monotonic clock)
        if violation.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            now_mono = time.monotonic()
            cooldown = float(self.config.get("trip_cooldown_seconds", 3.0))
            if not self.circuit_breaker_active and (now_mono - self._last_trip_monotonic) >= cooldown:
                self.trip_circuit_breaker(
                    f"{violation.severity.value.upper()}: {violation.violation_type}; {violation.violated_constraints}"
                )
                self._last_trip_monotonic = now_mono

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
                # Execute emergency callbacks (string reason to avoid recursion)
                emergency_callbacks = self.safety_callbacks.get(SafetyLevel.EMERGENCY, [])
                for callback in emergency_callbacks:
                    try:
                        callback(reason)
                    except Exception as e:
                        logging.error(f"Emergency callback failed: {e}")


# Backwards-compatible alias for typical class naming conventions
Nethical = nethical
