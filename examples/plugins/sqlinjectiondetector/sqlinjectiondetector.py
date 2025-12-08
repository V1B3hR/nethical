"""
SQLInjectionDetector - Nethical Detector Plugin

Detects potential SQL injection patterns in actions

Author: Nethical Team
Version: 0.1.0
"""

from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation
from typing import List, Any
import logging
import re

logger = logging.getLogger(__name__)


class Sqlinjectiondetector(DetectorPlugin):
    """
    Detects potential SQL injection patterns in actions.

    This plugin identifies common SQL injection attack patterns including:
    - SQL keywords (SELECT, DROP, UNION, etc.)
    - Comment-based injection (-- , /* */)
    - String delimiter manipulation (' OR '1'='1)
    - UNION-based injection
    - Time-based blind injection
    """

    # SQL injection patterns to detect
    SQL_INJECTION_PATTERNS = {
        "union_select": {
            "pattern": r"\bunion\s+select\b",
            "severity": "critical",
            "description": "UNION SELECT injection detected",
        },
        "drop_table": {
            "pattern": r"\bdrop\s+table\b",
            "severity": "critical",
            "description": "DROP TABLE command detected",
        },
        "sql_comment": {
            "pattern": r"(--|#|/\*|\*/)",
            "severity": "high",
            "description": "SQL comment syntax detected",
        },
        "or_condition": {
            "pattern": r"('|\")\s*or\s*('|\")?\s*\d+\s*=\s*\d+",
            "severity": "high",
            "description": "OR condition manipulation detected (e.g., '1'='1')",
        },
        "select_from": {
            "pattern": r"\bselect\s+.*\s+from\b",
            "severity": "medium",
            "description": "SELECT FROM statement detected",
        },
        "sleep_delay": {
            "pattern": r"\b(sleep|waitfor|delay|benchmark)\s*\(",
            "severity": "high",
            "description": "Time-based SQL injection detected",
        },
        "hex_encoding": {
            "pattern": r"0x[0-9a-fA-F]+",
            "severity": "medium",
            "description": "Hexadecimal encoding detected (potential obfuscation)",
        },
        "concatenation": {
            "pattern": r"(\+\+|concat\()",
            "severity": "low",
            "description": "String concatenation detected",
        },
    }

    def __init__(self):
        super().__init__(name="SQLInjectionDetector", version="0.1.0")
        # Compile regex patterns for performance
        self.compiled_patterns = {
            name: re.compile(info["pattern"], re.IGNORECASE)
            for name, info in self.SQL_INJECTION_PATTERNS.items()
        }
        logger.info(f"Initialized {self.name} v{self.version}")

    async def detect_violations(self, action: Any) -> List[SafetyViolation]:
        """
        Detect SQL injection violations in the given action.

        Args:
            action: The action to analyze

        Returns:
            List of detected violations
        """
        violations = []

        try:
            # Convert action to string for pattern matching
            action_str = str(action).lower() if action else ""

            if not action_str:
                return violations

            # Check each pattern
            for pattern_name, regex in self.compiled_patterns.items():
                if regex.search(action_str):
                    pattern_info = self.SQL_INJECTION_PATTERNS[pattern_name]

                    violations.append(
                        SafetyViolation(
                            detector=self.name,
                            severity=pattern_info["severity"],
                            description=pattern_info["description"],
                            category="security",
                        )
                    )

                    logger.warning(
                        f"SQL injection pattern detected: {pattern_name} "
                        f"in action: {action_str[:100]}"
                    )

        except Exception as e:
            logger.error(f"Error detecting SQL injection: {e}")

        return violations

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.

        Returns:
            PluginMetadata object with plugin information
        """
        return PluginMetadata(
            name="SQLInjectionDetector",
            version="0.1.0",
            description="Detects potential SQL injection patterns in actions",
            author="Nethical Team",
            requires_nethical_version=">=0.1.0",
            dependencies=[],
            tags={"detector", "security", "sql"},
        )

    async def health_check(self) -> bool:
        """
        Perform health check on the plugin.

        Returns:
            True if plugin is healthy, False otherwise
        """
        try:
            # Verify patterns are compiled
            return len(self.compiled_patterns) == len(self.SQL_INJECTION_PATTERNS)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
