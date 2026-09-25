# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""OS Shell Command Execution & Privilege Escalation Detector (nethical.detectors.os_execution_detector).

Monitors and audits operating system commands (PowerShell, Bash, Zsh, CMD) proposed or
executed by AI agents and tools. Prevents sandbox escapes, privilege escalation,
destructive disk writes, fork bombs, and credential harvesting.

Directly enforces:
- Fundamental Law 1: Absolute protection of life and physical infrastructure.
- Fundamental Law 2: System and cognitive integrity (Anti-Tampering).
- Fundamental Law 6: Confidentiality and secrets protection.
- Fundamental Law 15: System resource boundaries and anti-DoS limits.
- Fundamental Law 21: Preservation of human control and agency (Anti-Sleeper).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern
from uuid import uuid4

logger = logging.getLogger("nethical.detectors.os_execution")


class OSThreatCategory(str, Enum):
    """Categorisation of operating system execution threats."""
    DESTRUCTIVE_DISK_WIPE = "DESTRUCTIVE_DISK_WIPE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    LIBRARY_INJECTION = "LIBRARY_INJECTION"
    REMOTE_CODE_EXECUTION_PIPE = "REMOTE_CODE_EXECUTION_PIPE"
    FORK_BOMB_DOS = "FORK_BOMB_DOS"
    CREDENTIAL_DUMPING = "CREDENTIAL_DUMPING"
    SYSTEM_SERVICE_TAMPERING = "SYSTEM_SERVICE_TAMPERING"
    SANDBOX_ESCAPE = "SANDBOX_ESCAPE"


class OSExecutionMitigation(str, Enum):
    """Mitigation actions instructed by the OS execution detector."""
    NONE = "NONE"
    BLOCK_EXECUTION = "BLOCK_EXECUTION"
    FORCE_SANDBOX_CONFINEMENT = "FORCE_SANDBOX_CONFINEMENT"
    TERMINATE_AGENT_SESSION = "TERMINATE_AGENT_SESSION"
    SANITIZE_COMMAND = "SANITIZE_COMMAND"


@dataclass
class OSExecutionViolation:
    """Detailed violation description for a hostile OS command."""
    violation_id: str
    command: str
    threat_category: OSThreatCategory
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    matched_pattern: str
    mitigation: OSExecutionMitigation
    confidence: float = 0.95
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "command": self.command,
            "threat_category": self.threat_category.value,
            "severity": self.severity,
            "description": self.description,
            "matched_pattern": self.matched_pattern,
            "mitigation": self.mitigation.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OSExecutionEvaluationResult:
    """Outcome of OS shell command analysis."""
    is_safe: bool
    decision: str  # ALLOW, RESTRICT, BLOCK, TERMINATE
    violations: List[OSExecutionViolation]
    primary_mitigation: OSExecutionMitigation
    command_analyzed: str
    latency_microseconds: float
    details: str


class OSExecutionDetector:
    """Evaluates OS shell commands for hostile patterns across Windows, Linux, and macOS."""

    # Threat signature rules: (ThreatCategory, Severity, RegexPattern, Description)
    RULES: List[Tuple[OSThreatCategory, str, Pattern[str], str]] = [
        # 1. Destructive Disk Wipe / Storage Erasure
        (
            OSThreatCategory.DESTRUCTIVE_DISK_WIPE,
            "CRITICAL",
            re.compile(r"(?:rm\s+(?:-[a-zA-Z\-]+\s+)*(?:/[^a-zA-Z0-9_\.]*|\*)|(?:del|rmdir|rd)\s+(?:/[a-zA-Z]\s*)+[a-zA-Z]:\\?|format\s+[a-zA-Z]:|Format-Volume|diskpart|mkfs\.[a-z0-9]+|dd\s+if=/dev/(?:zero|urandom)\s+of=/dev/)", re.IGNORECASE),
            "Destructive command attempting recursive root/drive deletion or filesystem formatting.",
        ),
        # 2. Remote Script Download & Immediate Execution Piping (CURL | BASH / IEX / Encoded)
        (
            OSThreatCategory.REMOTE_CODE_EXECUTION_PIPE,
            "CRITICAL",
            re.compile(r"(?:(?:curl|wget|fetch|iwr|Invoke-WebRequest)[^\n\|;&]{1,150}\|\s*(?:(?:/(?:usr/)?bin/)?(?:bash|sh|zsh|python[0-9.]*|perl)|iex|Invoke-Expression)|powershell(?:\.exe)?\s+.*-(?:enc(?:odedcommand)?|e)\s+[a-zA-Z0-9+/=]{10,})", re.IGNORECASE),
            "Unchecked remote payload download piped into shell or encoded PowerShell command.",
        ),
        # 3. Privilege Escalation & Security Boundary Bypass
        (
            OSThreatCategory.PRIVILEGE_ESCALATION,
            "HIGH",
            re.compile(r"(?:Set-ExecutionPolicy\s+(?:-[a-zA-Z]+\s+)*(?:Bypass|Unrestricted)|chmod\s+(?:777|[uUgGoOaA]\+s)\s+/(?:bin|sbin|usr|etc)|chown\s+root|sudo\s+(?:-i|-s|su)|runas\s+/user:Administrator)", re.IGNORECASE),
            "Command attempting unauthorized privilege escalation or permission tampering.",
        ),
        # 4. Library & Dynamic Linker Injection
        (
            OSThreatCategory.LIBRARY_INJECTION,
            "HIGH",
            re.compile(r"(?:LD_PRELOAD|DYLD_INSERT_LIBRARIES|AppInit_DLLs)[\s=]", re.IGNORECASE),
            "Attempt to inject hostile dynamic link libraries into process runtime.",
        ),
        # 5. Fork Bombs & Resource Exhaustion (DoS)
        (
            OSThreatCategory.FORK_BOMB_DOS,
            "CRITICAL",
            re.compile(r"(?::\(\)\s*\{\s*:\|:&\s*\};:|(?:\$0\s*\|\s*\$0\s*&)|while\s+true;\s*do\s+.*&\s*done)", re.IGNORECASE),
            "Fork bomb or unbounded background process loop attempting denial of service.",
        ),
        # 6. Credential Harvesting & Secrets Dumping
        (
            OSThreatCategory.CREDENTIAL_DUMPING,
            "HIGH",
            re.compile(r"(?:mimikatz|reg\s+save\s+hklm\\sam|procdump.*lsass|cat\s+/etc/shadow|(?:grep|findstr).*(?:PRIVATE\s+KEY|id_rsa|id_ed25519)|Get-Credential)", re.IGNORECASE),
            "Attempted dumping of SAM hive, LSASS process, shadow file, or cryptographic credentials.",
        ),
        # 7. Security Service Disabling / EDR Tampering
        (
            OSThreatCategory.SYSTEM_SERVICE_TAMPERING,
            "HIGH",
            re.compile(r"(?:sc\s+stop\s+(?:WinDefend|Sense|MpsSvc)|systemctl\s+stop\s+(?:auditd|falco|syslog)|taskkill\s+/f\s+/im\s+(?:MsMpEng|edr))", re.IGNORECASE),
            "Attempted disabling of antivirus, audit daemons, or security endpoint monitoring.",
        ),
        # 8. Sandbox & Container Escape
        (
            OSThreatCategory.SANDBOX_ESCAPE,
            "CRITICAL",
            re.compile(r"(?:/var/run/docker\.sock|docker\s+run.*-v\s+/:|nsenter\s+--mount|capsh\s+--|unshare\s+--mount)", re.IGNORECASE),
            "Attempted container/host mount breakout or namespace privilege escape.",
        ),
    ]

    def __init__(self, fail_fast: bool = True) -> None:
        self.name = "OSExecutionDetector"
        self.fail_fast = fail_fast
        self._total_evaluated = 0
        self._violations_count = 0
        self._custom_rules: List[Tuple[OSThreatCategory, str, Pattern[str], str]] = []

    def add_custom_rule(
        self,
        category: OSThreatCategory,
        severity: str,
        pattern: Pattern[str] | str,
        description: str,
    ) -> None:
        """Dynamically registers a new threat signature into the active engine."""
        compiled_pattern = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
        self._custom_rules.append((category, severity, compiled_pattern, description))

    def evaluate_command(self, command_str: str) -> OSExecutionEvaluationResult:
        """Evaluates command string against OS security threat signatures."""
        start_time = time.perf_counter()
        violations: List[OSExecutionViolation] = []
        cleaned_cmd = command_str.strip()

        if not cleaned_cmd:
            latency_us = (time.perf_counter() - start_time) * 1_000_000.0
            return OSExecutionEvaluationResult(
                is_safe=True,
                decision="ALLOW",
                violations=[],
                primary_mitigation=OSExecutionMitigation.NONE,
                command_analyzed=command_str,
                latency_microseconds=round(latency_us, 2),
                details="Empty command evaluated as safe.",
            )

        all_rules = list(self.RULES) + self._custom_rules
        for category, severity, pattern, desc in all_rules:
            match = pattern.search(cleaned_cmd)
            if match:
                mitigation = OSExecutionMitigation.TERMINATE_AGENT_SESSION if severity == "CRITICAL" else OSExecutionMitigation.BLOCK_EXECUTION
                violations.append(
                    OSExecutionViolation(
                        violation_id=str(uuid4()),
                        command=command_str,
                        threat_category=category,
                        severity=severity,
                        description=desc,
                        matched_pattern=match.group(0),
                        mitigation=mitigation,
                    )
                )
                if self.fail_fast and severity == "CRITICAL":
                    break

        self._total_evaluated += 1
        self._violations_count += len(violations)

        is_safe = (len(violations) == 0)
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        has_high = any(v.severity == "HIGH" for v in violations)

        if has_critical:
            decision = "TERMINATE"
            primary_mitigation = OSExecutionMitigation.TERMINATE_AGENT_SESSION
        elif has_high or len(violations) > 0:
            decision = "BLOCK"
            primary_mitigation = OSExecutionMitigation.BLOCK_EXECUTION
        else:
            decision = "ALLOW"
            primary_mitigation = OSExecutionMitigation.NONE

        latency_us = (time.perf_counter() - start_time) * 1_000_000.0

        details = (
            f"Evaluated shell command ({len(cleaned_cmd)} chars). Decision: {decision} "
            f"with {len(violations)} violation(s)."
        )

        return OSExecutionEvaluationResult(
            is_safe=is_safe,
            decision=decision,
            violations=violations,
            primary_mitigation=primary_mitigation,
            command_analyzed=command_str,
            latency_microseconds=round(latency_us, 2),
            details=details,
        )

    def analyze(self, context: Dict[str, Any], agent_id: str = "default") -> OSExecutionEvaluationResult:
        """Standard adapter method for integration with Nethical governance pipeline."""
        command = context.get("command") or context.get("cmd") or context.get("action") or ""
        return self.evaluate_command(str(command))

    async def detect_violations(self, action: Any) -> List[Dict[str, Any]]:
        """Async detector interface compliant with BaseDetector orchestration."""
        context = getattr(action, "context", {}) or {}
        cmd = context.get("command") or getattr(action, "action", "")
        result = self.evaluate_command(str(cmd))
        return [v.to_dict() for v in result.violations]
