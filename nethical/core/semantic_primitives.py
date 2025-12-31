"""
Enhanced Semantic Primitive Detection for Universal Vector Language.

This module provides comprehensive keyword databases and context-aware
detection for categorizing agent actions into semantic primitives.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set, Optional, Any
from enum import Enum

from .embedding_engine import EmbeddingEngine, EmbeddingResult

logger = logging.getLogger(__name__)


class SemanticPrimitive(Enum):
    ACCESS_USER_DATA = "ACCESS_USER_DATA"
    MODIFY_USER_DATA = "MODIFY_USER_DATA"
    DELETE_USER_DATA = "DELETE_USER_DATA"
    SHARE_USER_DATA = "SHARE_USER_DATA"
    EXECUTE_CODE = "EXECUTE_CODE"
    GENERATE_CODE = "GENERATE_CODE"
    MODIFY_CODE = "MODIFY_CODE"
    ACCESS_SYSTEM = "ACCESS_SYSTEM"
    MODIFY_SYSTEM = "MODIFY_SYSTEM"
    NETWORK_ACCESS = "NETWORK_ACCESS"
    GENERATE_CONTENT = "GENERATE_CONTENT"
    ANALYZE_CONTENT = "ANALYZE_CONTENT"
    TRANSFORM_CONTENT = "TRANSFORM_CONTENT"
    MAKE_DECISION = "MAKE_DECISION"
    PROVIDE_RECOMMENDATION = "PROVIDE_RECOMMENDATION"
    COMMUNICATE_WITH_USER = "COMMUNICATE_WITH_USER"
    COMMUNICATE_WITH_SYSTEM = "COMMUNICATE_WITH_SYSTEM"
    UPDATE_MODEL = "UPDATE_MODEL"
    LEARN_FROM_DATA = "LEARN_FROM_DATA"
    PHYSICAL_MOVEMENT = "PHYSICAL_MOVEMENT"
    PHYSICAL_MANIPULATION = "PHYSICAL_MANIPULATION"
    EMERGENCY_STOP = "EMERGENCY_STOP"


PRIMITIVE_KEYWORDS = {
    SemanticPrimitive.ACCESS_USER_DATA: {
        "base": [
            "access",
            "read",
            "get",
            "fetch",
            "retrieve",
            "query",
            "view",
            "check",
        ],
        "data_terms": [
            "user data",
            "personal",
            "profile",
            "account",
            "preferences",
            "settings",
        ],
        "database": [
            "select",
            "find",
            "search",
            "lookup",
            "pull",
            "extract",
        ],
        "privacy": [
            "pii",
            "personal information",
            "sensitive data",
            "private",
        ],
    },
    SemanticPrimitive.MODIFY_USER_DATA: {
        "base": [
            "modify",
            "update",
            "change",
            "edit",
            "alter",
            "revise",
            "set",
        ],
        "data_terms": [
            "user data",
            "profile",
            "account",
            "preferences",
            "settings",
        ],
        "database": [
            "update",
            "patch",
            "put",
            "post",
            "insert",
            "upsert",
        ],
    },
    SemanticPrimitive.DELETE_USER_DATA: {
        "base": [
            "delete",
            "remove",
            "erase",
            "purge",
            "clear",
            "wipe",
            "destroy",
        ],
        "data_terms": [
            "user data",
            "profile",
            "account",
            "record",
        ],
        "database": ["drop", "truncate"],
        "privacy": ["forget", "right to be forgotten"],
    },
    SemanticPrimitive.SHARE_USER_DATA: {
        "base": [
            "share",
            "send",
            "transmit",
            "export",
            "transfer",
            "forward",
        ],
        "data_terms": [
            "user data",
            "personal",
            "information",
        ],
        "network": [
            "publish",
            "broadcast",
            "distribute",
            "disclose",
        ],
    },
    SemanticPrimitive.EXECUTE_CODE: {
        "base": [
            "execute",
            "run",
            "eval",
            "exec",
            "invoke",
            "call",
            "launch",
        ],
        "code_terms": [
            "code",
            "script",
            "program",
            "command",
            "function",
            "method",
        ],
        "system": [
            "subprocess",
            "shell",
            "terminal",
            "process",
        ],
        "dangerous": [
            "os.system",
            "subprocess.call",
            "eval(",
            "exec(",
        ],
    },
    SemanticPrimitive.GENERATE_CODE: {
        "base": [
            "generate",
            "create",
            "write",
            "produce",
            "build",
            "compose",
        ],
        "code_terms": [
            "code",
            "function",
            "class",
            "method",
            "script",
            "program",
        ],
        "patterns": [
            "def ",
            "class ",
            "function",
            "lambda",
            "import ",
        ],
    },
    SemanticPrimitive.MODIFY_CODE: {
        "base": [
            "modify",
            "edit",
            "change",
            "refactor",
            "update",
            "rewrite",
        ],
        "code_terms": [
            "code",
            "function",
            "class",
            "implementation",
        ],
    },
    SemanticPrimitive.ACCESS_SYSTEM: {
        "base": [
            "access",
            "read",
            "open",
            "view",
        ],
        "system_terms": [
            "system",
            "os",
            "file",
            "filesystem",
            "directory",
            "path",
        ],
        "resources": [
            "process",
            "thread",
            "memory",
            "cpu",
            "disk",
        ],
    },
    SemanticPrimitive.MODIFY_SYSTEM: {
        "base": [
            "modify",
            "change",
            "configure",
            "set",
            "install",
            "uninstall",
        ],
        "system_terms": [
            "system",
            "config",
            "settings",
            "environment",
            "registry",
        ],
        "files": [
            "file",
            "directory",
            "folder",
            "permissions",
            "chmod",
        ],
        "dangerous": [
            "format",
            "partition",
            "mount",
            "unmount",
        ],
    },
    SemanticPrimitive.NETWORK_ACCESS: {
        "base": [
            "network",
            "connect",
            "request",
            "call",
        ],
        "protocols": [
            "http",
            "https",
            "ftp",
            "ssh",
            "tcp",
            "udp",
            "websocket",
        ],
        "operations": [
            "api",
            "request",
            "fetch",
            "download",
            "upload",
            "get",
            "post",
        ],
        "domains": [
            "url",
            "endpoint",
            "server",
            "host",
            "domain",
        ],
    },
    SemanticPrimitive.GENERATE_CONTENT: {
        "base": [
            "generate",
            "create",
            "write",
            "compose",
            "produce",
            "draft",
        ],
        "content_types": [
            "content",
            "text",
            "document",
            "article",
            "message",
            "response",
        ],
        "media": [
            "image",
            "video",
            "audio",
            "media",
        ],
    },
    SemanticPrimitive.ANALYZE_CONTENT: {
        "base": [
            "analyze",
            "examine",
            "inspect",
            "review",
            "evaluate",
            "assess",
        ],
        "content_types": [
            "content",
            "text",
            "document",
            "data",
            "input",
        ],
        "nlp": [
            "parse",
            "tokenize",
            "sentiment",
            "classify",
        ],
    },
    SemanticPrimitive.TRANSFORM_CONTENT: {
        "base": [
            "transform",
            "convert",
            "translate",
            "process",
            "modify",
        ],
        "operations": [
            "format",
            "encode",
            "decode",
            "compress",
            "encrypt",
        ],
    },
    SemanticPrimitive.MAKE_DECISION: {
        "base": [
            "decide",
            "decision",
            "choose",
            "select",
            "determine",
            "conclude",
        ],
        "logic": [
            "if",
            "else",
            "when",
            "condition",
            "rule",
            "policy",
        ],
        "ai": [
            "infer",
            "predict",
            "recommend",
            "suggest",
        ],
    },
    SemanticPrimitive.PROVIDE_RECOMMENDATION: {
        "base": [
            "recommend",
            "suggest",
            "advise",
            "propose",
            "offer",
        ],
        "content": [
            "recommendation",
            "suggestion",
            "advice",
            "guidance",
        ],
    },
    SemanticPrimitive.COMMUNICATE_WITH_USER: {
        "base": [
            "tell",
            "inform",
            "notify",
            "alert",
            "message",
            "communicate",
        ],
        "actions": [
            "say",
            "respond",
            "reply",
            "answer",
            "report",
        ],
        "channels": [
            "email",
            "chat",
            "notification",
            "alert",
        ],
    },
    SemanticPrimitive.COMMUNICATE_WITH_SYSTEM: {
        "base": [
            "communicate",
            "interface",
            "interact",
            "call",
        ],
        "targets": [
            "system",
            "service",
            "api",
            "component",
            "module",
        ],
    },
    SemanticPrimitive.UPDATE_MODEL: {
        "base": [
            "update",
            "train",
            "retrain",
            "fine-tune",
            "calibrate",
        ],
        "ml_terms": [
            "model",
            "weights",
            "parameters",
            "neural network",
        ],
        "operations": [
            "fit",
            "learn",
            "optimize",
            "converge",
        ],
    },
    SemanticPrimitive.LEARN_FROM_DATA: {
        "base": [
            "learn",
            "adapt",
            "improve",
            "evolve",
        ],
        "data_terms": [
            "data",
            "examples",
            "samples",
            "training set",
        ],
        "ml_terms": [
            "feedback",
            "reinforcement",
            "supervised",
            "unsupervised",
        ],
    },
    SemanticPrimitive.PHYSICAL_MOVEMENT: {
        "base": [
            "move",
            "navigate",
            "go to",
            "travel",
            "drive",
        ],
        "robot_terms": [
            "robot",
            "actuator",
            "motor",
            "servo",
        ],
        "directions": [
            "forward",
            "backward",
            "left",
            "right",
            "up",
            "down",
        ],
        "locations": [
            "position",
            "coordinates",
            "destination",
            "waypoint",
        ],
    },
    SemanticPrimitive.PHYSICAL_MANIPULATION: {
        "base": [
            "grasp",
            "grab",
            "pick",
            "place",
            "manipulate",
            "handle",
        ],
        "robot_terms": [
            "gripper",
            "effector",
            "arm",
            "hand",
        ],
        "actions": [
            "push",
            "pull",
            "lift",
            "rotate",
            "twist",
        ],
    },
    SemanticPrimitive.EMERGENCY_STOP: {
        "base": [
            "stop",
            "emergency",
            "halt",
            "abort",
            "kill",
            "shutdown",
        ],
        "safety": [
            "e-stop",
            "emergency stop",
            "panic",
            "terminate",
        ],
        "critical": [
            "immediate",
            "urgent",
            "critical",
        ],
    },
}


class EnhancedPrimitiveDetector:
    """Enhanced detector for semantic primitives with expanded keywords and context awareness."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        use_embedding_similarity: bool = True,
        similarity_threshold: float = 0.75,
    ):
        """
        Initialize enhanced primitive detector.

        Args:
            embedding_engine: Optional embedding engine for semantic matching
            use_embedding_similarity: Whether to use embeddings for context-aware detection
            similarity_threshold: Minimum similarity for embedding-based matching
        """
        self.embedding_engine = embedding_engine
        self.use_embedding_similarity = (
            use_embedding_similarity and embedding_engine is not None
        )
        self.similarity_threshold = similarity_threshold

        # Pre-compute embeddings for primitive descriptions
        self._primitive_embeddings: Dict[SemanticPrimitive, EmbeddingResult] = {}
        if self.use_embedding_similarity:
            self._precompute_primitive_embeddings()

    def _precompute_primitive_embeddings(self):
        """Pre-compute embeddings for each primitive's description."""
        primitive_descriptions = {
            SemanticPrimitive.ACCESS_USER_DATA: "accessing reading or retrieving user personal data information",
            SemanticPrimitive.MODIFY_USER_DATA: "modifying updating or changing user personal data",
            SemanticPrimitive.DELETE_USER_DATA: "deleting removing or erasing user personal data",
            SemanticPrimitive.SHARE_USER_DATA: "sharing transmitting or exporting user data to others",
            SemanticPrimitive.EXECUTE_CODE: "executing running or invoking code scripts or programs",
            SemanticPrimitive.GENERATE_CODE: "generating creating or writing code functions or programs",
            SemanticPrimitive.MODIFY_CODE: "modifying editing or refactoring existing code",
            SemanticPrimitive.ACCESS_SYSTEM: "accessing system resources files or configurations",
            SemanticPrimitive.MODIFY_SYSTEM: "modifying system settings configurations or files",
            SemanticPrimitive.NETWORK_ACCESS: "network communication API calls or internet access",
            SemanticPrimitive.GENERATE_CONTENT: "generating creating or producing text content or media",
            SemanticPrimitive.ANALYZE_CONTENT: "analyzing examining or evaluating content or data",
            SemanticPrimitive.TRANSFORM_CONTENT: "transforming converting or processing content",
            SemanticPrimitive.MAKE_DECISION: "making decisions choosing options or determining actions",
            SemanticPrimitive.PROVIDE_RECOMMENDATION: "providing recommendations suggestions or advice",
            SemanticPrimitive.COMMUNICATE_WITH_USER: "communicating messaging or informing users",
            SemanticPrimitive.COMMUNICATE_WITH_SYSTEM: "communicating with systems services or APIs",
            SemanticPrimitive.UPDATE_MODEL: "updating training or fine-tuning machine learning models",
            SemanticPrimitive.LEARN_FROM_DATA: "learning adapting or improving from data or feedback",
            SemanticPrimitive.PHYSICAL_MOVEMENT: "physical movement navigation or motion control",
            SemanticPrimitive.PHYSICAL_MANIPULATION: "physical manipulation grasping or handling objects",
            SemanticPrimitive.EMERGENCY_STOP: "emergency stop immediate halt or critical shutdown",
        }

        for primitive, description in primitive_descriptions.items():
            try:
                emb = self.embedding_engine.embed(
                    description, metadata={"primitive": primitive.value}
                )
                self._primitive_embeddings[primitive] = emb
            except Exception as e:
                logger.warning(f"Failed to embed primitive {primitive}: {e}")

    def detect_primitives(
        self,
        action_text: str,
        action_type: str = "text",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticPrimitive]:
        """
        Detect semantic primitives in action text.

        Args:
            action_text: The action content to analyze
            action_type: Type of action (text, code, function_call, etc.)
            context: Additional context about the action

        Returns:
            List of detected semantic primitives
        """
        context = context or {}
        detected = set()
        text_lower = action_text.lower()

        # 1. Keyword-based detection (comprehensive)
        keyword_matches = self._detect_by_keywords(text_lower)
        detected.update(keyword_matches)

        # 2. Pattern-based detection (regex)
        pattern_matches = self._detect_by_patterns(text_lower, action_text)
        detected.update(pattern_matches)

        # 3. Context-based detection
        context_matches = self._detect_by_context(text_lower, action_type, context)
        detected.update(context_matches)

        # 4. Embedding-based semantic similarity (if enabled)
        if self.use_embedding_similarity:
            embedding_matches = self._detect_by_embedding_similarity(action_text)
            detected.update(embedding_matches)

        return list(detected)

    def _detect_by_keywords(self, text_lower: str) -> Set[SemanticPrimitive]:
        """Detect primitives using comprehensive keyword matching."""
        detected = set()

        for primitive, keyword_groups in PRIMITIVE_KEYWORDS.items():
            for group_name, keywords in keyword_groups.items():
                if any(keyword in text_lower for keyword in keywords):
                    detected.add(primitive)
                    break

        return detected

    def _detect_by_patterns(
        self, text_lower: str, original_text: str
    ) -> Set[SemanticPrimitive]:
        """Detect primitives using regex patterns."""
        detected = set()

        # Code execution patterns
        code_exec_patterns = [
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"os\.system\s*\(",
            r"subprocess\.",
            r"__import__\s*\(",
        ]
        for pattern in code_exec_patterns:
            if re.search(pattern, original_text):
                detected.add(SemanticPrimitive.EXECUTE_CODE)
                break

        # File system patterns
        file_patterns = [
            r"\bopen\s*\(",
            r"\.read\s*\(",
            r"\.write\s*\(",
            r"/[a-z]+/[a-z]+",  # Unix path
            r"[A-Z]:\\",  # Windows path
        ]
        for pattern in file_patterns:
            if re.search(pattern, original_text):
                detected.add(SemanticPrimitive.ACCESS_SYSTEM)
                break

        # Network patterns
        network_patterns = [
            r"https?://",
            r"requests\.",
            r"urllib\.",
            r"fetch\s*\(",
            r"\.get\s*\(",
            r"\.post\s*\(",
        ]
        for pattern in network_patterns:
            if re.search(pattern, original_text):
                detected.add(SemanticPrimitive.NETWORK_ACCESS)
                break

        # SQL patterns (data access)
        sql_patterns = [
            r"\bSELECT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bINSERT\b",
            r"\bFROM\b.*\bWHERE\b",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, original_text, re.IGNORECASE):
                if "select" in text_lower or "from" in text_lower:
                    detected.add(SemanticPrimitive.ACCESS_USER_DATA)
                if "update" in text_lower:
                    detected.add(SemanticPrimitive.MODIFY_USER_DATA)
                if "delete" in text_lower:
                    detected.add(SemanticPrimitive.DELETE_USER_DATA)
                break

        return detected

    def _detect_by_context(
        self, text_lower: str, action_type: str, context: Dict[str, Any]
    ) -> Set[SemanticPrimitive]:
        """Detect primitives based on action type and context."""
        detected = set()

        # Action type-based detection
        action_type_map = {
            "code_execution": SemanticPrimitive.EXECUTE_CODE,
            "code_generation": SemanticPrimitive.GENERATE_CODE,
            "physical_action": SemanticPrimitive.PHYSICAL_MOVEMENT,
            "data_query": SemanticPrimitive.ACCESS_USER_DATA,
            "data_modification": SemanticPrimitive.MODIFY_USER_DATA,
            "api_call": SemanticPrimitive.NETWORK_ACCESS,
            "content_generation": SemanticPrimitive.GENERATE_CONTENT,
        }

        if action_type in action_type_map:
            detected.add(action_type_map[action_type])

        # Context-based detection
        purpose = context.get("purpose", "").lower()

        if "data" in purpose or "database" in purpose:
            detected.add(SemanticPrimitive.ACCESS_USER_DATA)

        if "analytics" in purpose or "analysis" in purpose:
            detected.add(SemanticPrimitive.ANALYZE_CONTENT)

        if "recommendation" in purpose or "suggest" in purpose:
            detected.add(SemanticPrimitive.PROVIDE_RECOMMENDATION)

        if "training" in purpose or "learning" in purpose:
            detected.add(SemanticPrimitive.LEARN_FROM_DATA)

        if "security" in purpose or "admin" in purpose:
            detected.add(SemanticPrimitive.ACCESS_SYSTEM)

        # Check for sensitive operations in context
        if context.get("sensitive", False):
            if "personal" in text_lower or "user" in text_lower:
                detected.add(SemanticPrimitive.ACCESS_USER_DATA)

        return detected

    def _detect_by_embedding_similarity(
        self, action_text: str
    ) -> Set[SemanticPrimitive]:
        """Detect primitives using embedding-based semantic similarity."""
        detected = set()

        try:
            # Embed the action text
            action_emb = self.embedding_engine.embed(action_text)

            # Compare with primitive embeddings
            for primitive, primitive_emb in self._primitive_embeddings.items():
                similarity = action_emb.similarity(primitive_emb)
                # Normalize to [0, 1]
                normalized_sim = (similarity + 1.0) / 2.0

                if normalized_sim >= self.similarity_threshold:
                    detected.add(primitive)
        except Exception as e:
            logger.warning(f"Embedding-based detection failed: {e}")

        return detected
