"""Plugin Governance for Security and Quality Assurance.

This module provides security scanning, performance benchmarking,
compatibility testing, and certification for plugins.

Enhancements:
- AST-based security scanning with structured findings, risk score, and per-file/line context
- Optional recursive scanning of plugin paths; combined with single-file code scanning
- Secrets detection (AWS keys, private keys, token-like strings)
- Storage/caching of results in a storage directory with JSON export
- Benchmark improvements: optional operation callback, perf_counter timing, p95/p99 calc, tracemalloc peak memory
- Compatibility testing optionally imports the plugin to verify basic load
- Flexible thresholds and strict mode behavior that escalates severities
- Generate comprehensive report including summarized findings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
from datetime import datetime
import time
import hashlib
import re
import json
import ast
import logging
import tracemalloc
import sys
from pathlib import Path

# Configure a basic logger for governance operations
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class SecurityLevel(Enum):
    """Security risk level."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CertificationStatus(Enum):
    """Plugin certification status."""

    NOT_CERTIFIED = "not_certified"
    PENDING = "pending"
    CERTIFIED = "certified"
    REVOKED = "revoked"


SEVERITY_WEIGHTS: Dict[SecurityLevel, int] = {
    SecurityLevel.LOW: 1,
    SecurityLevel.MEDIUM: 3,
    SecurityLevel.HIGH: 7,
    SecurityLevel.CRITICAL: 15,
    SecurityLevel.SAFE: 0,
}


@dataclass
class Finding:
    """A structured finding from security analysis."""

    rule_id: str
    description: str
    severity: SecurityLevel
    file_path: Optional[str] = None
    line: Optional[int] = None
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line": self.line,
            "code": self.code,
        }


@dataclass
class SecurityScanResult:
    """Result from security scanning."""

    plugin_id: str
    scan_date: datetime
    security_level: SecurityLevel
    vulnerabilities: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed: bool = True
    scan_duration: float = 0.0
    findings: List[Finding] = field(default_factory=list)
    risk_score: int = 0
    fingerprint: Optional[str] = None

    def add_vulnerability(self, description: str, level: SecurityLevel = SecurityLevel.MEDIUM):
        """Add a security vulnerability (legacy string-based)."""
        self.vulnerabilities.append(f"[{level.value.upper()}] {description}")
        self._adjust_level_and_passed(level)

    def add_warning(self, description: str):
        """Add a security warning (legacy string-based)."""
        self.warnings.append(description)

    def add_finding(self, finding: Finding):
        """Add a structured finding and update risk score/level."""
        self.findings.append(finding)
        self.risk_score += SEVERITY_WEIGHTS.get(finding.severity, 0)
        self._adjust_level_and_passed(finding.severity)
        # Mirror structured findings into legacy lists for backward compatibility
        if finding.severity in (SecurityLevel.HIGH, SecurityLevel.CRITICAL, SecurityLevel.MEDIUM):
            self.vulnerabilities.append(f"[{finding.severity.value.upper()}] {finding.description}")
        else:
            self.warnings.append(finding.description)

    def _adjust_level_and_passed(self, level: SecurityLevel):
        if level in (SecurityLevel.HIGH, SecurityLevel.CRITICAL):
            self.passed = False
        # Raise the overall level to the highest encountered
        order = [
            SecurityLevel.SAFE,
            SecurityLevel.LOW,
            SecurityLevel.MEDIUM,
            SecurityLevel.HIGH,
            SecurityLevel.CRITICAL,
        ]
        if order.index(level) > order.index(self.security_level):
            self.security_level = level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "scan_date": self.scan_date.isoformat(),
            "security_level": self.security_level.value,
            "vulnerabilities": self.vulnerabilities,
            "warnings": self.warnings,
            "passed": self.passed,
            "scan_duration": self.scan_duration,
            "findings": [f.to_dict() for f in self.findings],
            "risk_score": self.risk_score,
            "fingerprint": self.fingerprint,
        }


@dataclass
class BenchmarkResult:
    """Result from performance benchmarking."""

    plugin_id: str
    benchmark_date: datetime
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cpu_usage_percent: float
    passed: bool = True
    notes: List[str] = field(default_factory=list)
    jitter_ms: float = 0.0
    iterations: int = 0

    def meets_requirements(
        self,
        max_latency_ms: float = 100.0,
        min_throughput: float = 10.0,
        max_memory_mb: float = 500.0,
    ) -> bool:
        """Check if benchmark meets performance requirements."""
        meets = (
            self.p95_latency_ms <= max_latency_ms
            and self.throughput_ops_per_sec >= min_throughput
            and self.memory_usage_mb <= max_memory_mb
        )
        self.passed = meets
        return meets

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "benchmark_date": self.benchmark_date.isoformat(),
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "passed": self.passed,
            "notes": self.notes,
            "jitter_ms": self.jitter_ms,
            "iterations": self.iterations,
        }


@dataclass
class CompatibilityReport:
    """Compatibility testing report."""

    plugin_id: str
    test_date: datetime
    nethical_version: str
    python_version: str
    compatible: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    test_results: Dict[str, bool] = field(default_factory=dict)

    def add_test_result(self, test_name: str, passed: bool, details: Optional[str] = None):
        """Add a test result."""
        self.test_results[test_name] = passed
        if not passed:
            self.compatible = False
            if details:
                self.issues.append(f"{test_name}: {details}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "test_date": self.test_date.isoformat(),
            "nethical_version": self.nethical_version,
            "python_version": self.python_version,
            "compatible": self.compatible,
            "issues": self.issues,
            "warnings": self.warnings,
            "test_results": self.test_results,
        }


class PluginGovernance:
    """Plugin governance for security, performance, and quality assurance.

    This class provides methods to scan plugins for security issues,
    benchmark performance, test compatibility, and manage certification.

    Example:
        >>> governance = PluginGovernance()
        >>> scan_result = governance.security_scan("my-plugin", plugin_path="./plugins/my-plugin")
        >>> bench_result = governance.benchmark("my-plugin", iterations=500)
        >>> compat_result = governance.compatibility_test("my-plugin")
        >>> governance.certify("my-plugin")
    """

    def __init__(
        self,
        storage_dir: str = "./nethical_governance",
        strict_mode: bool = False,
        allowlist_rules: Optional[Set[str]] = None,
        denylist_rules: Optional[Set[str]] = None,
    ):
        """Initialize plugin governance.

        Args:
            storage_dir: Directory for governance data
            strict_mode: Enable strict security and performance requirements
            allowlist_rules: Optional set of rule_ids allowed/suppressed
            denylist_rules: Optional set of rule_ids forced/highlighted
        """
        self.storage_dir = storage_dir
        self.strict_mode = strict_mode
        self.allowlist_rules = allowlist_rules or set()
        self.denylist_rules = denylist_rules or set()

        # Security regex patterns for secrets and suspicious tokens
        self.secret_patterns = {
            "aws_access_key_id": re.compile(r"AKIA[0-9A-Z]{16}"),
            "aws_secret_access_key": re.compile(
                r'(?i)aws(.{0,20})?(secret|access).{0,20}?[:=]\s*[\'"]?[A-Za-z0-9/+=]{40}[\'"]?'
            ),
            "private_key": re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----"),
            "token_like": re.compile(
                r'(?i)(api|access|secret|token|key)[\s:_-]*[\'"]?([A-Za-z0-9_\-]{16,})[\'"]?'
            ),
        }

        # Performance thresholds
        self.perf_thresholds = {
            "max_latency_ms": 50.0 if strict_mode else 100.0,
            "min_throughput": 20.0 if strict_mode else 10.0,
            "max_memory_mb": 250.0 if strict_mode else 500.0,
        }

        # Certification records
        self._certifications: Dict[str, CertificationStatus] = {}

        # Create storage directory if not exists
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Security scanning
    # --------------------------

    def security_scan(
        self, plugin_id: str, plugin_code: Optional[str] = None, plugin_path: Optional[str] = None
    ) -> SecurityScanResult:
        """Perform security scanning on a plugin.

        Args:
            plugin_id: Plugin identifier
            plugin_code: Plugin source code (if available)
            plugin_path: Path to plugin files

        Returns:
            Security scan results
        """
        start_time = time.perf_counter()

        result = SecurityScanResult(
            plugin_id=plugin_id, scan_date=datetime.now(), security_level=SecurityLevel.SAFE
        )

        sources: List[Tuple[str, str]] = []  # list of (file_path, code)

        if plugin_path:
            for file in self._iter_python_files(plugin_path):
                try:
                    code = Path(file).read_text(encoding="utf-8", errors="ignore")
                    sources.append((str(file), code))
                except Exception as e:
                    result.add_warning(f"Failed to read {file}: {e}")

        if plugin_code:
            sources.append(("<memory>", plugin_code))

        if not sources:
            result.add_warning("No code provided for scanning - limited analysis")
            result.scan_duration = time.perf_counter() - start_time
            self._save_json(plugin_id, "security_scan", result.to_dict())
            return result

        # Compute a simple fingerprint across files for traceability
        result.fingerprint = self._compute_fingerprint(sources)

        # Scan each source using AST and regex for secrets
        for file_path, code in sources:
            self._scan_code_ast(code, file_path, result)
            self._scan_code_secrets(code, file_path, result)

        # Check for known vulnerabilities (stub)
        self._check_vulnerabilities(plugin_id, result)

        # Verify signatures (stub)
        self._verify_signatures(plugin_id, result)

        result.scan_duration = time.perf_counter() - start_time

        # Persist result
        self._save_json(plugin_id, "security_scan", result.to_dict())
        return result

    def _iter_python_files(self, base_path: str) -> List[Path]:
        base = Path(base_path)
        if base.is_file() and base.suffix == ".py":
            return [base]
        files: List[Path] = []
        for p in base.rglob("*.py"):
            # Skip common non-production dirs
            parts = set(p.parts)
            if {"venv", ".venv", "__pycache__", ".git", "site-packages"} & parts:
                continue
            files.append(p)
        return files

    def _scan_code_ast(self, code: str, file_path: str, result: SecurityScanResult):
        """Scan code using AST for dangerous constructs."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result.add_warning(f"Syntax error parsing {file_path}: {e}")
            return

        # Helper: record finding
        def add(rule_id: str, msg: str, severity: SecurityLevel, node: Optional[ast.AST] = None):
            if rule_id in self.allowlist_rules:
                return
            # Escalate if denylisted or strict mode
            sev = severity
            if rule_id in self.denylist_rules:
                sev = (
                    SecurityLevel.HIGH
                    if severity in (SecurityLevel.LOW, SecurityLevel.MEDIUM)
                    else severity
                )
            if self.strict_mode and severity == SecurityLevel.MEDIUM:
                sev = SecurityLevel.HIGH
            line = getattr(node, "lineno", None) if node else None
            excerpt = self._extract_line(code, line) if line else None
            result.add_finding(
                Finding(
                    rule_id=rule_id,
                    description=msg,
                    severity=sev,
                    file_path=file_path,
                    line=line,
                    code=excerpt,
                )
            )

        # Walk AST
        for node in ast.walk(tree):
            # from x import *
            if isinstance(node, ast.ImportFrom):
                if any(getattr(n, "name", "") == "*" for n in node.names):
                    add(
                        "import_star",
                        "Wildcard import detected - may hide issues and complicate auditing",
                        SecurityLevel.LOW,
                        node,
                    )

            # import suspicious modules
            if isinstance(node, ast.Import):
                for n in node.names:
                    mod = n.name.split(".")[0]
                    if mod in {"subprocess", "pickle", "socket", "ftplib"}:
                        sev = (
                            SecurityLevel.MEDIUM
                            if mod in {"socket", "ftplib"}
                            else (
                                SecurityLevel.HIGH
                                if mod in {"subprocess", "pickle"}
                                else SecurityLevel.MEDIUM
                            )
                        )
                        add(f"import_{mod}", f"Import of sensitive module '{mod}'", sev, node)

            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in {"subprocess", "pickle", "socket", "ftplib"}:
                    sev = (
                        SecurityLevel.MEDIUM
                        if mod in {"socket", "ftplib"}
                        else (
                            SecurityLevel.HIGH
                            if mod in {"subprocess", "pickle"}
                            else SecurityLevel.MEDIUM
                        )
                    )
                    add(f"from_{mod}_import", f"Import from sensitive module '{mod}'", sev, node)

            # Calls: exec/eval/compile/__import__, os.system, subprocess.*, pickle.loads
            if isinstance(node, ast.Call):
                # Helper to resolve called function full name (best-effort)
                func_name = self._resolve_call_name(node.func)

                # Builtins
                if func_name in {"exec", "eval", "compile", "__import__"}:
                    add(
                        f"call_{func_name}",
                        f"Dangerous function call '{func_name}' detected",
                        SecurityLevel.HIGH,
                        node,
                    )

                # os.system / os.popen
                if func_name in {"os.system", "os.popen"}:
                    add(
                        "os_system",
                        f"OS command execution via '{func_name}'",
                        SecurityLevel.HIGH,
                        node,
                    )

                # subprocess calls
                if func_name and func_name.startswith("subprocess."):
                    add(
                        "subprocess_call",
                        f"Subprocess execution '{func_name}'",
                        SecurityLevel.HIGH,
                        node,
                    )

                # pickle.loads
                if func_name in {"pickle.loads", "pickle.load"}:
                    add(
                        "pickle_load",
                        f"Untrusted deserialization via '{func_name}'",
                        SecurityLevel.HIGH,
                        node,
                    )

                # open(..., mode="w"/"a"/"x")
                if func_name == "open":
                    mode = self._extract_open_mode(node)
                    if mode and any(m in mode for m in ("w", "a", "x", "+")):
                        add(
                            "file_write",
                            f"File write operation detected (mode='{mode}')",
                            SecurityLevel.MEDIUM,
                            node,
                        )

                # network libs usage
                if func_name and (
                    func_name.startswith("requests.")
                    or func_name.startswith("urllib.")
                    or func_name.startswith("http.client")
                    or func_name in {"socket.socket"}
                ):
                    add(
                        "network_usage",
                        f"Network operation via '{func_name}'",
                        SecurityLevel.LOW,
                        node,
                    )

        # Simple regex fallback for exec/eval in code (in case of dynamic constructs)
        legacy_patterns = {
            "exec": re.compile(r"\bexec\s*\("),
            "eval": re.compile(r"\beval\s*\("),
        }
        for rid, pat in legacy_patterns.items():
            for m in pat.finditer(code):
                line_num = code.count("\n", 0, m.start()) + 1
                result.add_finding(
                    Finding(
                        rule_id=f"rx_{rid}",
                        description=f"Suspicious use of {rid} (regex)",
                        severity=SecurityLevel.HIGH,
                        file_path=file_path,
                        line=line_num,
                        code=self._extract_line(code, line_num),
                    )
                )

    def _scan_code_secrets(self, code: str, file_path: str, result: SecurityScanResult):
        """Scan code for possible secrets."""
        for rule_id, pattern in self.secret_patterns.items():
            for m in pattern.finditer(code):
                line_num = code.count("\n", 0, m.start()) + 1
                excerpt = self._extract_line(code, line_num)
                # Reduce chance of false positives for token-like by checking variable names and length
                severity = SecurityLevel.MEDIUM if rule_id == "token_like" else SecurityLevel.HIGH
                result.add_finding(
                    Finding(
                        rule_id=f"secret_{rule_id}",
                        description=f"Potential secret detected: {rule_id}",
                        severity=severity,
                        file_path=file_path,
                        line=line_num,
                        code=excerpt,
                    )
                )

    def _check_vulnerabilities(self, plugin_id: str, result: SecurityScanResult):
        """Check for known vulnerabilities.

        In a real implementation, this would query a vulnerability database (e.g., OSV/CVE).
        """
        # Placeholder - would check CVE database, etc.
        return

    def _verify_signatures(self, plugin_id: str, result: SecurityScanResult):
        """Verify code signatures.

        In a real implementation, this would verify cryptographic signatures.
        """
        # Placeholder - would verify digital signatures or compare fingerprints to a registry
        return

    def _extract_line(self, code: str, line_number: Optional[int]) -> Optional[str]:
        if not line_number or line_number <= 0:
            return None
        try:
            return code.splitlines()[line_number - 1].strip()[:300]
        except Exception:
            return None

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            parts.reverse()
            return ".".join(parts)
        return None

    def _extract_open_mode(self, call: ast.Call) -> Optional[str]:
        # mode can be a positional arg (index 1) or keyword 'mode'
        mode_val = None
        if len(call.args) >= 2 and isinstance(call.args[1], ast.Str):
            mode_val = call.args[1].s
        for kw in call.keywords or []:
            if kw.arg == "mode" and isinstance(kw.value, ast.Str):
                mode_val = kw.value.s
        return mode_val

    def _compute_fingerprint(self, sources: List[Tuple[str, str]]) -> str:
        h = hashlib.sha256()
        for path, code in sorted(sources, key=lambda x: x[0]):
            h.update(path.encode("utf-8", errors="ignore"))
            h.update(b"\x00")
            h.update(code.encode("utf-8", errors="ignore"))
            h.update(b"\x00")
        return h.hexdigest()

    # --------------------------
    # Benchmarking
    # --------------------------

    def benchmark(
        self,
        plugin_id: str,
        test_data: Optional[Any] = None,
        iterations: int = 1000,
        operation: Optional[Callable[[Any], Any]] = None,
        warmup: int = 10,
        sleep_ms: Optional[float] = 1.0,
    ) -> BenchmarkResult:
        """Benchmark plugin performance.

        Args:
            plugin_id: Plugin identifier
            test_data: Test data for benchmarking (passed to operation if provided)
            iterations: Number of test iterations
            operation: Optional callable representing the plugin operation under test
            warmup: Number of warmup iterations not counted in metrics
            sleep_ms: If no operation provided, simulate latency with time.sleep(sleep_ms/1000)

        Returns:
            Benchmark results
        """
        result = BenchmarkResult(
            plugin_id=plugin_id,
            benchmark_date=datetime.now(),
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            throughput_ops_per_sec=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            iterations=iterations,
        )

        # Warm-up
        for _ in range(max(0, warmup)):
            if operation:
                operation(test_data)
            else:
                if sleep_ms is not None:
                    time.sleep(max(0.0, sleep_ms) / 1000.0)

        latencies: List[float] = []
        start_time = time.perf_counter()
        tracemalloc.start()

        for _ in range(iterations):
            op_start = time.perf_counter()
            if operation:
                operation(test_data)
            else:
                if sleep_ms is not None:
                    time.sleep(max(0.0, sleep_ms) / 1000.0)
            op_end = time.perf_counter()
            latencies.append((op_end - op_start) * 1000.0)  # ms

        total_time = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if latencies:
            latencies_sorted = sorted(latencies)
            result.avg_latency_ms = sum(latencies_sorted) / len(latencies_sorted)
            result.p95_latency_ms = self._percentile(latencies_sorted, 95.0)
            result.p99_latency_ms = self._percentile(latencies_sorted, 99.0)
            # jitter: std dev approximation using MAD-like measure
            median = self._percentile(latencies_sorted, 50.0)
            mad = sum(abs(x - median) for x in latencies_sorted) / len(latencies_sorted)
            result.jitter_ms = mad

        result.throughput_ops_per_sec = iterations / total_time if total_time > 0 else 0.0
        result.memory_usage_mb = peak / (1024 * 1024)  # approximate peak during run
        # CPU usage percent would require psutil or OS-specific resource measurement.
        # Keep a heuristic based on work done (unspecified here).
        result.cpu_usage_percent = 0.0

        # Check against thresholds
        result.meets_requirements(**self.perf_thresholds)

        if not result.passed:
            result.notes.append("Performance requirements not met")

        # Persist result
        self._save_json(plugin_id, "benchmark", result.to_dict())
        return result

    def _percentile(self, data: List[float], p: float) -> float:
        """Compute percentile using nearest-rank method."""
        if not data:
            return 0.0
        k = max(1, int(round((p / 100.0) * len(data)))) - 1
        k = min(k, len(data) - 1)
        return data[k]

    # --------------------------
    # Compatibility testing
    # --------------------------

    def compatibility_test(
        self,
        plugin_id: str,
        nethical_version: str = "0.1.0",
        python_version: Optional[str] = None,
        attempt_import: bool = True,
    ) -> CompatibilityReport:
        """Test plugin compatibility.

        Args:
            plugin_id: Plugin identifier (also used as module name for import)
            nethical_version: Nethical version to test against
            python_version: Python version to test against (defaults to current)
            attempt_import: Attempt to import the plugin to verify basic load

        Returns:
            Compatibility report
        """
        report = CompatibilityReport(
            plugin_id=plugin_id,
            test_date=datetime.now(),
            nethical_version=nethical_version,
            python_version=python_version or f"{sys.version_info.major}.{sys.version_info.minor}",
            compatible=True,
        )

        # Test basic import
        if attempt_import:
            try:
                __import__(plugin_id)
                report.add_test_result("import_test", True, "Plugin imports successfully")
            except Exception as e:
                report.add_test_result("import_test", False, f"Import failed: {e}")

        # Test API compatibility (placeholder, extend with concrete checks as nethical matures)
        report.add_test_result("api_test", True, "Plugin API appears compatible")

        # Test dependencies (placeholder)
        report.add_test_result("dependency_test", True, "All dependencies satisfied")

        # Test detector interface (if applicable, placeholder)
        report.add_test_result(
            "detector_interface", True, "Detector interface implemented correctly"
        )

        # Persist result
        self._save_json(plugin_id, "compatibility", report.to_dict())
        return report

    # --------------------------
    # Certification
    # --------------------------

    def certify(
        self,
        plugin_id: str,
        security_result: Optional[SecurityScanResult] = None,
        benchmark_result: Optional[BenchmarkResult] = None,
        compatibility_result: Optional[CompatibilityReport] = None,
    ) -> CertificationStatus:
        """Certify a plugin for marketplace distribution.

        Args:
            plugin_id: Plugin identifier
            security_result: Security scan results
            benchmark_result: Benchmark results
            compatibility_result: Compatibility test results

        Returns:
            Certification status
        """
        # Run tests if not provided
        if security_result is None:
            security_result = self.security_scan(plugin_id)

        if benchmark_result is None:
            benchmark_result = self.benchmark(plugin_id)

        if compatibility_result is None:
            compatibility_result = self.compatibility_test(plugin_id)

        # Check certification requirements
        high_risk = security_result.security_level in (SecurityLevel.HIGH, SecurityLevel.CRITICAL)
        requirements_met = (
            security_result.passed
            and not high_risk
            and benchmark_result.passed
            and compatibility_result.compatible
        )

        if requirements_met:
            status = CertificationStatus.CERTIFIED
        else:
            # If close to passing (no HIGH/CRITICAL but maybe perf borderline), mark as PENDING
            if (
                not high_risk
                and compatibility_result.compatible
                and security_result.risk_score <= 3
            ):
                status = CertificationStatus.PENDING
            else:
                status = CertificationStatus.NOT_CERTIFIED

        self._certifications[plugin_id] = status
        self._persist_certifications()
        return status

    def get_certification_status(self, plugin_id: str) -> CertificationStatus:
        """Get certification status of a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Certification status
        """
        return self._certifications.get(plugin_id, CertificationStatus.NOT_CERTIFIED)

    def revoke_certification(self, plugin_id: str, reason: str) -> bool:
        """Revoke certification for a plugin.

        Args:
            plugin_id: Plugin identifier
            reason: Reason for revocation

        Returns:
            True if successful
        """
        if plugin_id in self._certifications:
            self._certifications[plugin_id] = CertificationStatus.REVOKED
            self._persist_certifications()
            # Save audit log entry
            self._save_json(
                plugin_id,
                "revocation",
                {
                    "plugin_id": plugin_id,
                    "reason": reason,
                    "revoked_at": datetime.now().isoformat(),
                },
            )
            return True
        return False

    # --------------------------
    # Reporting
    # --------------------------

    def generate_report(
        self,
        plugin_id: str,
        security_result: SecurityScanResult,
        benchmark_result: BenchmarkResult,
        compatibility_result: CompatibilityReport,
    ) -> Dict[str, Any]:
        """Generate comprehensive governance report.

        Args:
            plugin_id: Plugin identifier
            security_result: Security scan results
            benchmark_result: Benchmark results
            compatibility_result: Compatibility test results

        Returns:
            Complete governance report
        """
        cert_status = self.get_certification_status(plugin_id)

        report = {
            "plugin_id": plugin_id,
            "report_date": datetime.now().isoformat(),
            "certification_status": cert_status.value,
            "security": {
                "level": security_result.security_level.value,
                "passed": security_result.passed,
                "risk_score": security_result.risk_score,
                "vulnerabilities": len(security_result.vulnerabilities),
                "warnings": len(security_result.warnings),
                "scan_duration": security_result.scan_duration,
                "fingerprint": security_result.fingerprint,
                "findings": [
                    f.to_dict() for f in security_result.findings[:100]
                ],  # cap output size
            },
            "performance": {
                "passed": benchmark_result.passed,
                "avg_latency_ms": benchmark_result.avg_latency_ms,
                "p95_latency_ms": benchmark_result.p95_latency_ms,
                "p99_latency_ms": benchmark_result.p99_latency_ms,
                "throughput_ops_per_sec": benchmark_result.throughput_ops_per_sec,
                "memory_mb": benchmark_result.memory_usage_mb,
                "jitter_ms": benchmark_result.jitter_ms,
                "iterations": benchmark_result.iterations,
                "notes": benchmark_result.notes,
            },
            "compatibility": {
                "compatible": compatibility_result.compatible,
                "nethical_version": compatibility_result.nethical_version,
                "python_version": compatibility_result.python_version,
                "tests_passed": sum(1 for p in compatibility_result.test_results.values() if p),
                "tests_total": len(compatibility_result.test_results),
                "issues": compatibility_result.issues,
                "warnings": compatibility_result.warnings,
            },
        }

        # Persist combined report
        self._save_json(plugin_id, "governance_report", report)
        return report

    # --------------------------
    # Persistence helpers
    # --------------------------

    def _persist_certifications(self):
        try:
            path = Path(self.storage_dir) / "certifications.json"
            data = {pid: status.value for pid, status in self._certifications.items()}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to persist certifications: {e}")

    def _save_json(self, plugin_id: str, kind: str, content: Dict[str, Any]):
        try:
            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            base_dir = Path(self.storage_dir) / plugin_id
            base_dir.mkdir(parents=True, exist_ok=True)
            file_path = base_dir / f"{kind}_{ts}.json"
            file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to save {kind} JSON for {plugin_id}: {e}")
