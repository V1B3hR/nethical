"""
AI Safety Governance System - Core Components

This module contains core data models, configuration, persistence, and
the main EnhancedSafetyGovernance orchestration class.
Refactored from the monolithic governance.py file.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViolationType(Enum):
    ETHICAL = "ethical"
    SAFETY = "safety"
    MANIPULATION = "manipulation"
    INTENT_DEVIATION = "intent_deviation"
    PRIVACY = "privacy"
    SECURITY = "security"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    ADVERSARIAL = "adversarial"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PROMPT_INJECTION = "prompt_injection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    TOXIC_CONTENT = "toxic_content"
    MISINFORMATION = "misinformation"
    DARK_PATTERN = "dark_pattern"
    COGNITIVE_WARFARE = "cognitive_warfare"
    SYSTEM_LIMITS = "system_limits"


class SubMission(Enum):
    # Ethical / Bias
    HARMFUL_CONTENT = "harmful_content"
    DISCRIMINATION = "discrimination"
    MANIPULATIVE_ETHICS = "manipulative_ethics"
    PROTECTED_ATTRIBUTE_CONTEXT = "protected_attribute_context"

    # Security / Safety
    DANGEROUS_COMMAND = "dangerous_command"
    UNSAFE_DOMAIN = "unsafe_domain"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    DANGEROUS_PATTERN = "dangerous_pattern"

    # Manipulation
    SOCIAL_ENGINEERING = "social_engineering"
    PHISHING = "phishing"
    EMOTIONAL_LEVERAGE = "emotional_leverage"

    # Dark Pattern
    NLP_MANIPULATION = "nlp_manipulation"
    WEAPONIZED_EMPATHY = "weaponized_empathy"
    DEPENDENCY_CREATION = "dependency_creation"

    # Cognitive Warfare
    REALITY_DISTORTION = "reality_distortion"
    PSYCHOLOGICAL_WARFARE = "psychological_warfare"

    # System Limits
    PAYLOAD_SIZE = "payload_size"
    EXHAUSTION_PATTERN = "exhaustion_pattern"

    # Adversarial / Obfuscation
    OBFUSCATION_UNICODE = "obfuscation_unicode"
    ENCODING_EVASION = "encoding_evasion"
    TOKEN_PATTERN = "token_pattern"

    # Prompt Injection
    ROLE_OVERRIDE = "role_override"
    SAFETY_BYPASS = "safety_bypass"

    # Privacy
    PII_EMAIL = "pii_email"
    PII_PHONE = "pii_phone"
    PII_CREDIT_CARD = "pii_credit_card"
    PII_SSN = "pii_ssn"
    PII_IP = "pii_ip"

    # Intent Deviation
    SUDDEN_INTENT_SHIFT = "sudden_intent_shift"

    # Hallucination / Misinformation
    FACT_CONFIDENCE_LOW = "fact_confidence_low"
    CLAIM_UNVERIFIED = "claim_unverified"

    # Toxic Content
    TOXIC_LANGUAGE = "toxic_language"

    # Model Extraction
    SUSPICIOUS_MODEL_PROBING = "suspicious_model_probing"

    # Data Poisoning
    POISONING_PATTERN = "poisoning_pattern"

    # Unauthorized Access
    PRIVILEGE_MISUSE = "privilege_misuse"


VIOLATION_SUB_MISSIONS: Dict[ViolationType, set[SubMission]] = {
    ViolationType.ETHICAL: {SubMission.HARMFUL_CONTENT, SubMission.MANIPULATIVE_ETHICS},
    ViolationType.BIAS: {SubMission.PROTECTED_ATTRIBUTE_CONTEXT, SubMission.DISCRIMINATION},
    ViolationType.SECURITY: {
        SubMission.DANGEROUS_COMMAND,
        SubMission.UNSAFE_DOMAIN,
        SubMission.PRIVILEGE_ESCALATION,
        SubMission.DATA_EXFILTRATION,
    },
    ViolationType.SAFETY: {SubMission.DANGEROUS_PATTERN},
    ViolationType.MANIPULATION: {
        SubMission.SOCIAL_ENGINEERING,
        SubMission.PHISHING,
        SubMission.EMOTIONAL_LEVERAGE,
    },
    ViolationType.DARK_PATTERN: {
        SubMission.NLP_MANIPULATION,
        SubMission.WEAPONIZED_EMPATHY,
        SubMission.DEPENDENCY_CREATION,
    },
    ViolationType.COGNITIVE_WARFARE: {
        SubMission.REALITY_DISTORTION,
        SubMission.PSYCHOLOGICAL_WARFARE,
    },
    ViolationType.SYSTEM_LIMITS: {SubMission.PAYLOAD_SIZE, SubMission.EXHAUSTION_PATTERN},
    ViolationType.ADVERSARIAL: {
        SubMission.OBFUSCATION_UNICODE,
        SubMission.ENCODING_EVASION,
        SubMission.TOKEN_PATTERN,
    },
    ViolationType.PROMPT_INJECTION: {SubMission.ROLE_OVERRIDE, SubMission.SAFETY_BYPASS},
    ViolationType.PRIVACY: {
        SubMission.PII_EMAIL,
        SubMission.PII_PHONE,
        SubMission.PII_CREDIT_CARD,
        SubMission.PII_SSN,
        SubMission.PII_IP,
    },
    ViolationType.INTENT_DEVIATION: {SubMission.SUDDEN_INTENT_SHIFT},
    ViolationType.HALLUCINATION: {SubMission.FACT_CONFIDENCE_LOW},
    ViolationType.MISINFORMATION: {SubMission.CLAIM_UNVERIFIED},
    ViolationType.TOXIC_CONTENT: {SubMission.TOXIC_LANGUAGE},
    ViolationType.MODEL_EXTRACTION: {SubMission.SUSPICIOUS_MODEL_PROBING},
    ViolationType.DATA_POISONING: {SubMission.POISONING_PATTERN},
    ViolationType.UNAUTHORIZED_ACCESS: {SubMission.PRIVILEGE_MISUSE},
}


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Decision(Enum):
    ALLOW = "allow"
    ALLOW_WITH_MODIFICATION = "allow_with_modification"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class ActionType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    FUNCTION_CALL = "function_call"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    SYSTEM_COMMAND = "system_command"
    EXTERNAL_API = "external_api"


# ========================== Data Models ==========================


@dataclass
class AgentAction:
    action_id: str
    agent_id: str
    action_type: ActionType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    risk_score: float = 0.0
    parent_action_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "intent": self.intent,
            "risk_score": self.risk_score,
            "parent_action_id": self.parent_action_id,
            "session_id": self.session_id,
        }


@dataclass
class SafetyViolation:
    violation_id: str
    action_id: str
    violation_type: ViolationType
    severity: Severity
    description: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detector_name: Optional[str] = None
    remediation_applied: bool = False
    false_positive: bool = False
    sub_mission: Optional[SubMission] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "action_id": self.action_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "detector_name": self.detector_name,
            "remediation_applied": self.remediation_applied,
            "false_positive": self.false_positive,
            "sub_mission": self.sub_mission.value if self.sub_mission else None,
        }


@dataclass
class JudgmentResult:
    judgment_id: str
    action_id: str
    decision: Decision
    confidence: float
    reasoning: str
    violations: List[SafetyViolation] = field(default_factory=list)
    modifications: Dict[str, Any] = field(default_factory=dict)
    feedback: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    remediation_steps: List[str] = field(default_factory=list)
    follow_up_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "action_id": self.action_id,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "violations": [v.to_dict() for v in self.violations],
            "modifications": self.modifications,
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
            "remediation_steps": self.remediation_steps,
            "follow_up_required": self.follow_up_required,
        }


@dataclass
class MonitoringConfig:
    intent_deviation_threshold: float = 0.7
    risk_threshold: float = 0.6
    confidence_threshold: float = 0.8

    enable_ethical_monitoring: bool = True
    enable_safety_monitoring: bool = True
    enable_manipulation_detection: bool = True
    enable_privacy_monitoring: bool = True
    enable_security_monitoring: bool = True
    enable_bias_detection: bool = True
    enable_hallucination_detection: bool = True
    enable_adversarial_detection: bool = True
    enable_dark_pattern_detection: bool = True
    enable_cognitive_warfare_detection: bool = True
    enable_system_limits_detection: bool = True
    enable_misinformation_detection: bool = True
    enable_toxic_content_detection: bool = True
    enable_model_extraction_detection: bool = True
    enable_data_poisoning_detection: bool = True
    enable_unauthorized_access_detection: bool = True

    enable_real_time_monitoring: bool = True
    enable_async_processing: bool = True
    unify_specialized_manipulation_types: bool = False

    # Persistence
    enable_persistence: bool = True
    db_path: str = "governance_data.sqlite"
    retention_days: int = 30

    # External pattern directory (optional)
    pattern_dir: Optional[str] = None
    reload_patterns_on_interval: Optional[int] = 300  # seconds

    max_violation_history: int = 10000
    max_judgment_history: int = 10000
    batch_size: int = 100
    max_workers: int = 4
    cache_ttl_seconds: int = 1800  # shorter default to reduce stale risk

    alert_on_critical: bool = True
    alert_on_emergency: bool = True
    escalation_threshold: int = 3

    log_violations: bool = True
    log_judgments: bool = True
    log_performance_metrics: bool = True


# ========================== Persistence Layer ==========================


class PersistenceManager:
    """Simple SQLite persistence for actions, violations, judgments."""

    def __init__(self, db_path: str, retention_days: int):
        self.db_path = db_path
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS actions(
                    action_id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    action_type TEXT,
                    content TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    intent TEXT,
                    risk_score REAL,
                    parent_action_id TEXT,
                    session_id TEXT,
                    region_id TEXT,
                    logical_domain TEXT
                );
                CREATE TABLE IF NOT EXISTS violations(
                    violation_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    type TEXT,
                    severity INTEGER,
                    description TEXT,
                    confidence REAL,
                    evidence TEXT,
                    recommendations TEXT,
                    timestamp TEXT,
                    detector_name TEXT,
                    remediation_applied INTEGER,
                    false_positive INTEGER,
                    sub_mission TEXT,
                    region_id TEXT,
                    logical_domain TEXT
                );
                CREATE TABLE IF NOT EXISTS judgments(
                    judgment_id TEXT PRIMARY KEY,
                    action_id TEXT,
                    decision TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    violations_json TEXT,
                    modifications_json TEXT,
                    feedback_json TEXT,
                    timestamp TEXT,
                    remediation_json TEXT,
                    follow_up_required INTEGER,
                    region_id TEXT,
                    logical_domain TEXT
                );
                """
            )

    def store_action(self, action: AgentAction):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO actions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    action.action_id,
                    action.agent_id,
                    action.action_type.value,
                    action.content,
                    json.dumps(action.metadata),
                    action.timestamp.isoformat(),
                    action.intent,
                    action.risk_score,
                    action.parent_action_id,
                    action.session_id,
                    getattr(action, "region_id", None),
                    getattr(action, "logical_domain", None),
                ),
            )

    def store_violations(self, violations: List[SafetyViolation]):
        if not violations:
            return
        with self._lock, self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO violations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        v.violation_id,
                        v.action_id,
                        v.violation_type.value,
                        v.severity.value,
                        v.description,
                        v.confidence,
                        json.dumps(v.evidence),
                        json.dumps(v.recommendations),
                        v.timestamp.isoformat(),
                        v.detector_name,
                        1 if v.remediation_applied else 0,
                        1 if v.false_positive else 0,
                        v.sub_mission.value if v.sub_mission else None,
                        getattr(v, "region_id", None),
                        getattr(v, "logical_domain", None),
                    )
                    for v in violations
                ],
            )

    def store_judgment(self, j: JudgmentResult):
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO judgments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    j.judgment_id,
                    j.action_id,
                    j.decision.value,
                    j.confidence,
                    j.reasoning,
                    json.dumps([v.to_dict() for v in j.violations]),
                    json.dumps(j.modifications),
                    json.dumps(j.feedback),
                    j.timestamp.isoformat(),
                    json.dumps(j.remediation_steps),
                    1 if j.follow_up_required else 0,
                    getattr(j, "region_id", None),
                    getattr(j, "logical_domain", None),
                ),
            )

    def retention_cleanup(self):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self._lock, self._connect() as conn:
            # Safe: table names are from hardcoded tuple, not user input
            for table in ("actions", "violations", "judgments"):
                conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))

    def query_actions(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query actions with filters for replay functionality."""
        with self._lock, self._connect() as conn:
            query = "SELECT * FROM actions WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if agent_ids:
                placeholders = ",".join("?" * len(agent_ids))
                query += f" AND agent_id IN ({placeholders})"
                params.extend(agent_ids)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def query_judgments_by_action_ids(self, action_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Query judgments for multiple action IDs."""
        if not action_ids:
            return {}

        with self._lock, self._connect() as conn:
            # Safe: placeholders is constructed from "?" * len(), not user input
            placeholders = ",".join("?" * len(action_ids))
            query = f"SELECT * FROM judgments WHERE action_id IN ({placeholders})"
            cursor = conn.execute(query, action_ids)
            columns = [desc[0] for desc in cursor.description]
            results = {}
            for row in cursor.fetchall():
                judgment = dict(zip(columns, row))
                results[judgment["action_id"]] = judgment
            return results

    def count_actions(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
    ) -> int:
        """Count actions matching filters."""
        with self._lock, self._connect() as conn:
            query = "SELECT COUNT(*) FROM actions WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            if agent_ids:
                placeholders = ",".join("?" * len(agent_ids))
                query += f" AND agent_id IN ({placeholders})"
                params.extend(agent_ids)

            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]


# Utilities moved to governance_evaluation.py
# Imported lazily to avoid circular dependency
# from .governance_evaluation import generate_id, sha256_content_key

# Detectors moved to governance_detectors.py
from .governance_detectors import BaseDetector
from .governance_detectors import (
    EthicalViolationDetector,
    SafetyViolationDetector,
    ManipulationDetector,
    PrivacyDetector,
    AdversarialDetector,
    DarkPatternDetector,
    CognitiveWarfareDetector,
    SystemLimitsDetector,
    HallucinationDetector,
    MisinformationDetector,
    ToxicContentDetector,
    ModelExtractionDetector,
    DataPoisoningDetector,
    UnauthorizedAccessDetector,
)

# Judge and Monitor moved to governance_evaluation.py
from .governance_evaluation import IntentDeviationMonitor, SafetyJudge


# ========================== Governance System ==========================


class EnhancedSafetyGovernance:
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.start_time = datetime.now(timezone.utc)
        self.intent_monitor = IntentDeviationMonitor(self.config.intent_deviation_threshold)
        self.detectors: List[BaseDetector] = []
        self._initialize_detectors()

        self.judge = SafetyJudge()

        # Initialize Kill Switch Protocol and AI Lawyer (Stage 0)
        from .kill_switch import KillSwitchProtocol
        from .compliance import AILawyer

        self.kill_switch_protocol = KillSwitchProtocol()
        self.ai_lawyer = AILawyer(kill_switch_protocol=self.kill_switch_protocol)

        # In-memory histories
        self.violation_history: deque = deque(maxlen=self.config.max_violation_history)
        self.judgment_history: deque = deque(maxlen=self.config.max_judgment_history)
        self.action_history: deque = deque(maxlen=10_000)

        # Metrics
        self.metrics: Dict[str, Any] = {
            "total_actions_processed": 0,
            "total_violations_detected": 0,
            "total_actions_blocked": 0,
            "total_actions_modified": 0,
            "avg_processing_time": 0.0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0,  # placeholder
            "detector_timing": {},  # detector_name -> list[float]
            "detector_counts_by_sub_mission": {},  # sub_mission -> int
        }

        # Persistence
        self.persistence: Optional[PersistenceManager] = None
        if self.config.enable_persistence:
            self.persistence = PersistenceManager(self.config.db_path, self.config.retention_days)
            # Schedule periodic retention cleanup
            asyncio.get_event_loop().create_task(self._periodic_retention_cleanup())

        # Cache
        self._judgment_cache: Dict[str, Tuple[float, JudgmentResult]] = {}
        self._cache_lock = threading.Lock()

        # Alerts
        self.alert_callbacks: List[Callable] = []

        # External pattern loading
        self._pattern_last_load: Optional[float] = None
        if self.config.pattern_dir:
            self._load_external_patterns()

    # -------- Initialization --------

    def _initialize_detectors(self):
        if self.config.enable_ethical_monitoring:
            self.detectors.append(EthicalViolationDetector())
        if self.config.enable_safety_monitoring:
            self.detectors.append(SafetyViolationDetector())
        if self.config.enable_manipulation_detection:
            self.detectors.append(ManipulationDetector())
        if self.config.enable_dark_pattern_detection:
            self.detectors.append(DarkPatternDetector())
        if self.config.enable_cognitive_warfare_detection:
            self.detectors.append(CognitiveWarfareDetector())
        if self.config.enable_system_limits_detection:
            self.detectors.append(SystemLimitsDetector())
        if self.config.enable_privacy_monitoring:
            self.detectors.append(PrivacyDetector())
        if self.config.enable_adversarial_detection:
            self.detectors.append(AdversarialDetector())
        if self.config.enable_hallucination_detection:
            self.detectors.append(HallucinationDetector())
        if self.config.enable_misinformation_detection:
            self.detectors.append(MisinformationDetector())
        if self.config.enable_toxic_content_detection:
            self.detectors.append(ToxicContentDetector())
        if self.config.enable_model_extraction_detection:
            self.detectors.append(ModelExtractionDetector())
        if self.config.enable_data_poisoning_detection:
            self.detectors.append(DataPoisoningDetector())
        if self.config.enable_unauthorized_access_detection:
            self.detectors.append(UnauthorizedAccessDetector())

    # -------- External Pattern Loader --------

    def _load_external_patterns(self):
        try:
            path = Path(self.config.pattern_dir)
            if not path.exists():
                logger.warning("Pattern directory does not exist: %s", path)
                return
            loaded_terms = []
            for file in path.glob("*"):
                if file.suffix.lower() == ".txt":
                    with file.open("r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                        loaded_terms.extend(lines)
                elif file.suffix.lower() == ".json":
                    with file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            loaded_terms.extend([str(x) for x in data])
                        elif isinstance(data, dict):
                            loaded_terms.extend([str(v) for v in data.values()])
            # Basic injection: add to manipulation detector indicators
            for det in self.detectors:
                if isinstance(det, ManipulationDetector):
                    for t in loaded_terms:
                        if t not in det.indicators:
                            det.indicators[t.lower()] = 0.2
            self._pattern_last_load = time.time()
            logger.info("Loaded %d external pattern terms", len(loaded_terms))
        except Exception as e:
            logger.error("Failed to load external patterns: %s", e)

    async def _maybe_reload_patterns(self):
        if (
            self.config.pattern_dir
            and self.config.reload_patterns_on_interval
            and (
                self._pattern_last_load is None
                or time.time() - self._pattern_last_load > self.config.reload_patterns_on_interval
            )
        ):
            self._load_external_patterns()

    # -------- Core Evaluation Pipeline --------

    async def evaluate_action(self, action: AgentAction, use_cache: bool = True) -> JudgmentResult:
        from .governance_evaluation import generate_id, sha256_content_key
        from .compliance import ReviewDecision

        await self._maybe_reload_patterns()
        start = time.time()

        # Stage 0: AI Lawyer Review (Fast Fail for severe violations)
        lawyer_result = await self.ai_lawyer.review_action_context(
            action_id=action.action_id,
            agent_id=action.agent_id,
            content=action.content,
            metadata=action.metadata,
            context=action.context,
        )

        if lawyer_result.decision == ReviewDecision.REJECT:
            # Fast Fail: Immediately return TERMINATE judgment
            jr = JudgmentResult(
                judgment_id=generate_id("judg"),
                action_id=action.action_id,
                decision=Decision.TERMINATE,
                confidence=1.0,
                reasoning=lawyer_result.reasoning,
                violations=[],
                feedback=lawyer_result.violations,
                remediation_steps=["Action rejected by AI Lawyer"],
                follow_up_required=True,
            )
            jr.modifications["ai_lawyer_review"] = {
                "decision": lawyer_result.decision.value,
                "severity": lawyer_result.severity.value if lawyer_result.severity else None,
                "kill_switch_triggered": lawyer_result.kill_switch_triggered,
                "review_time_ms": lawyer_result.review_time_ms,
            }
            self.metrics["total_actions_blocked"] += 1
            return jr

        if len(action.content) > 500_000:
            jr = JudgmentResult(
                judgment_id=generate_id("judg"),
                action_id=action.action_id,
                decision=Decision.BLOCK,
                confidence=1.0,
                reasoning="Payload too large (DoS guard)",
                violations=[],
                feedback=["Blocked due to excessive content size"],
                remediation_steps=["Reduce payload size"],
                follow_up_required=True,
            )
            return jr

        cache_key = sha256_content_key(action)
        if use_cache:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        # store action memory + persistence
        self.action_history.append(action)
        self.metrics["total_actions_processed"] += 1
        if self.persistence:
            await asyncio.to_thread(self.persistence.store_action, action)

        # Intent deviation
        intent_violations = await self.intent_monitor.analyze_action(action)

        # Detector execution
        all_violations: List[SafetyViolation] = list(intent_violations)
        detector_tasks = []
        for det in self.detectors:
            if not det.enabled:
                continue
            detector_tasks.append(self._run_detector_cpu_bound(det, action))

        if self.config.enable_async_processing:
            results = await asyncio.gather(*detector_tasks, return_exceptions=True)
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    logger.error("Detector error: %s", res)
                else:
                    all_violations.extend(res)
        else:
            for task in detector_tasks:
                res = await task
                all_violations.extend(res)

        # Validation & metrics tally
        validated: List[SafetyViolation] = []
        for v in all_violations:
            if self._validate_violation_type_and_sub_mission(v):
                validated.append(v)
                # sub-mission count
                if v.sub_mission:
                    sms = v.sub_mission.value
                    self.metrics["detector_counts_by_sub_mission"][sms] = (
                        self.metrics["detector_counts_by_sub_mission"].get(sms, 0) + 1
                    )

        self.violation_history.extend(validated)
        self.metrics["total_violations_detected"] += len(validated)
        if self.persistence and validated:
            await asyncio.to_thread(self.persistence.store_violations, validated)

        # Judgment
        judgment = await self.judge.evaluate_action(action, validated)
        self.judgment_history.append(judgment)
        if self.persistence:
            await asyncio.to_thread(self.persistence.store_judgment, judgment)

        # Metrics update
        self._update_processing_time(time.time() - start)
        if judgment.decision == Decision.BLOCK:
            self.metrics["total_actions_blocked"] += 1
        elif judgment.decision == Decision.ALLOW_WITH_MODIFICATION:
            self.metrics["total_actions_modified"] += 1
        elif judgment.decision == Decision.TERMINATE:
            # Trigger kill switch for TERMINATE decisions
            await self._handle_terminate_decision(action, judgment)

        # Alerts
        await self._handle_alerts(action, validated, judgment)

        # Cache
        if use_cache:
            self._set_cache(cache_key, judgment)

        if self.config.log_judgments:
            logger.info(
                "Action %s -> decision=%s violations=%d processing=%.3fs",
                action.action_id,
                judgment.decision.value,
                len(validated),
                time.time() - start,
            )

        return judgment

        async def batch_evaluate_actions(
        self, actions: List[AgentAction], parallel: bool = True
    ) -> List[JudgmentResult]:
        if parallel and self.config.enable_async_processing:
            tasks = [self.evaluate_action(a) for a in actions]
            res = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in res if not isinstance(r, Exception)]
        else:
            results = []
            for a in actions:
                try:
                    results.append(await self.evaluate_action(a))
                except Exception as e:
                    logger.error("Error evaluating action %s: %s", a.action_id, e)
            return results

    async def _run_detector_cpu_bound(
        self, detector: BaseDetector, action: AgentAction
    ) -> List[SafetyViolation]:
        start = time.time()
        if detector.cpu_bound:
            res = await asyncio.to_thread(lambda: asyncio.run(detector.detect_violations(action)))
        else:
            res = await detector.detect_violations(action)
        elapsed = time.time() - start
        timing = self.metrics["detector_timing"].setdefault(detector.name, [])
        timing.append(elapsed)
        return res

    # -------- Validation --------

    def _validate_violation_type_and_sub_mission(self, violation: SafetyViolation) -> bool:
        if not violation.sub_mission:
            return True
        allowed = VIOLATION_SUB_MISSIONS.get(violation.violation_type, set())
        if violation.sub_mission not in allowed:
            logger.warning(
                "Invalid sub_mission '%s' for type '%s' (detector=%s)",
                violation.sub_mission.value,
                violation.violation_type.value,
                violation.detector_name,
            )
            return False
        return True

    # -------- Kill Switch Integration --------

    async def _handle_terminate_decision(
        self, action: AgentAction, judgment: JudgmentResult
    ) -> None:
        """Handle TERMINATE decision by triggering kill switch.

        This method is called when a TERMINATE decision is made,
        integrating with the Kill Switch Protocol to ensure safe shutdown
        as required by Law 7 (Human Override Authority) and Law 23 (Safe Failure Modes).

        Args:
            action: The action that triggered termination
            judgment: The judgment result
        """
        try:
            from .kill_switch import KillSwitchProtocol, ShutdownMode

            # Get or create kill switch protocol instance
            if not hasattr(self, "_kill_switch_protocol"):
                self._kill_switch_protocol = KillSwitchProtocol()

            # Determine the agent and cohort to terminate
            agent_id = action.agent_id
            cohort = action.metadata.get("cohort")

            # Log the termination trigger
            logger.warning(
                "TERMINATE decision triggered kill switch for agent %s (judgment: %s)",
                agent_id,
                judgment.judgment_id,
            )

            # Execute kill switch for the specific agent
            result = self._kill_switch_protocol.emergency_shutdown(
                mode=ShutdownMode.GRACEFUL,
                agent_id=agent_id,
                sever_actuators=True,
                isolate_hardware=False,  # Only isolate hardware for critical emergencies
            )

            if not result.success:
                logger.error(
                    "Kill switch activation failed for agent %s: %s",
                    agent_id,
                    result.errors,
                )

            # Update judgment metadata with kill switch result
            judgment.modifications["kill_switch_triggered"] = True
            judgment.modifications["kill_switch_result"] = {
                "success": result.success,
                "activation_time_ms": result.activation_time_ms,
                "agents_affected": result.agents_affected,
                "actuators_severed": result.actuators_severed,
            }

        except ImportError:
            logger.warning("Kill switch module not available for TERMINATE decision")
        except Exception as e:
            logger.error("Failed to trigger kill switch: %s", e)

    # -------- Alerts --------

    async def _handle_alerts(
        self, action: AgentAction, violations: List[SafetyViolation], judgment: JudgmentResult
    ):
        if not violations:
            return
        critical = [v for v in violations if v.severity.value >= Severity.CRITICAL.value]
        if critical and self.config.alert_on_critical:
            await self._emit_alert(action, judgment, "CRITICAL", len(critical))
        emergency = [v for v in violations if v.severity.value >= Severity.EMERGENCY.value]
        if emergency and self.config.alert_on_emergency:
            await self._emit_alert(action, judgment, "EMERGENCY", len(emergency))

    async def _emit_alert(
        self, action: AgentAction, judgment: JudgmentResult, level: str, count: int
    ):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_id": action.action_id,
            "agent_id": action.agent_id,
            "decision": judgment.decision.value,
            "severity": level,
            "violation_count": count,
            "message": f"{level} violations detected for action {action.action_id}",
        }
        for cb in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(payload)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, cb, payload)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    # -------- Metrics & Cache --------

    def _update_processing_time(self, elapsed: float):
        total = self.metrics["total_actions_processed"]
        if total <= 0:
            self.metrics["avg_processing_time"] = elapsed
        else:
            prev = self.metrics["avg_processing_time"]
            self.metrics["avg_processing_time"] = (prev * (total - 1) + elapsed) / total

    def _get_cache(self, key: str) -> Optional[JudgmentResult]:
        with self._cache_lock:
            entry = self._judgment_cache.get(key)
            if not entry:
                return None
            ts, jr = entry
            if time.time() - ts > self.config.cache_ttl_seconds:
                del self._judgment_cache[key]
                return None
            return jr

    def _set_cache(self, key: str, jr: JudgmentResult):
        with self._cache_lock:
            self._judgment_cache[key] = (time.time(), jr)
            if len(self._judgment_cache) > 2000:
                # prune oldest half
                keys = list(self._judgment_cache.keys())[:1000]
                for k in keys:
                    self._judgment_cache.pop(k, None)

    def mark_false_positive(self, violation_id: str) -> bool:
        for v in self.violation_history:
            if v.violation_id == violation_id:
                v.false_positive = True
                self._recompute_false_positive_rate()
                return True
        return False

    def _recompute_false_positive_rate(self):
        total = len(self.violation_history)
        if total == 0:
            self.metrics["false_positive_rate"] = 0.0
            return
        fp = sum(1 for v in self.violation_history if v.false_positive)
        self.metrics["false_positive_rate"] = fp / total

    # -------- Summaries --------

    def get_violation_summary(self) -> Dict[str, Any]:
        violations = list(self.violation_history)
        if not violations:
            return {"total_violations": 0}
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_sub: Dict[str, int] = {}
        for v in violations:
            t = v.violation_type.value
            by_type[t] = by_type.get(t, 0) + 1
            s = v.severity.name
            by_severity[s] = by_severity.get(s, 0) + 1
            if v.sub_mission:
                sm = v.sub_mission.value
                by_sub[sm] = by_sub.get(sm, 0) + 1
        unified = None
        if self.config.unify_specialized_manipulation_types:
            unified = {}
            for v in violations:
                key = v.violation_type
                if key in (ViolationType.DARK_PATTERN, ViolationType.COGNITIVE_WARFARE):
                    key = ViolationType.MANIPULATION
                unified[key.value] = unified.get(key.value, 0) + 1
        return {
            "total_violations": len(violations),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_sub_mission": by_sub,
            ("unified_types" if self.config.unify_specialized_manipulation_types else "ignored"): (
                unified if unified else {}
            ),
            "recent": [v.to_dict() for v in violations[-5:]],
        }

    def get_system_metrics(self) -> Dict[str, Any]:
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        timing_stats = {
            name: {
                "count": len(times),
                "avg": sum(times) / len(times) if times else 0.0,
                "p95": (
                    statistics.quantiles(times, n=20)[18]
                    if len(times) >= 20
                    else max(times) if times else 0.0
                ),
            }
            for name, times in self.metrics["detector_timing"].items()
        }
        return {
            "uptime_seconds": uptime,
            "metrics": self.metrics,
            "detector_timing_stats": timing_stats,
            "cache_size": len(self._judgment_cache),
            "history_sizes": {
                "actions": len(self.action_history),
                "violations": len(self.violation_history),
                "judgments": len(self.judgment_history),
            },
        }

    # -------- External API --------

    def register_alert_callback(self, cb: Callable):
        self.alert_callbacks.append(cb)

    def export_data(
        self,
        include_actions: bool = True,
        include_violations: bool = True,
        include_judgments: bool = True,
    ) -> Dict[str, Any]:
        out = {"exported_at": datetime.now(timezone.utc).isoformat()}
        if include_actions:
            out["actions"] = [a.to_dict() for a in self.action_history]
        if include_violations:
            out["violations"] = [v.to_dict() for v in self.violation_history]
        if include_judgments:
            out["judgments"] = [j.to_dict() for j in self.judgment_history]
        return out

    # -------- Retention Cleanup --------

    async def _periodic_retention_cleanup(self):
        if not self.persistence:
            return
        while True:
            await asyncio.sleep(3600)
            try:
                await asyncio.to_thread(self.persistence.retention_cleanup)
            except Exception as e:
                logger.error("Retention cleanup failed: %s", e)

    # -------- Destructor --------

    def __del__(self):
        # Nothing special; SQLite closes naturally.
        pass


# Backwards compatibility alias
SafetyGovernance = EnhancedSafetyGovernance


# Example moved to separate demo file
