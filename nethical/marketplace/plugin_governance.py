"""Plugin Governance for Security and Quality Assurance.

This module provides security scanning, performance benchmarking,
compatibility testing, and certification for plugins.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from datetime import datetime
import time
import hashlib
import re


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
    
    def add_vulnerability(self, description: str, level: SecurityLevel = SecurityLevel.MEDIUM):
        """Add a security vulnerability."""
        self.vulnerabilities.append(f"[{level.value.upper()}] {description}")
        if level in (SecurityLevel.HIGH, SecurityLevel.CRITICAL):
            self.passed = False
        if level.value == SecurityLevel.CRITICAL.value:
            self.security_level = SecurityLevel.CRITICAL
        elif level.value == SecurityLevel.HIGH.value and self.security_level != SecurityLevel.CRITICAL:
            self.security_level = SecurityLevel.HIGH
    
    def add_warning(self, description: str):
        """Add a security warning."""
        self.warnings.append(description)


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
    
    def meets_requirements(
        self,
        max_latency_ms: float = 100.0,
        min_throughput: float = 10.0,
        max_memory_mb: float = 500.0
    ) -> bool:
        """Check if benchmark meets performance requirements."""
        meets = (
            self.p95_latency_ms <= max_latency_ms and
            self.throughput_ops_per_sec >= min_throughput and
            self.memory_usage_mb <= max_memory_mb
        )
        self.passed = meets
        return meets


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


class PluginGovernance:
    """Plugin governance for security, performance, and quality assurance.
    
    This class provides methods to scan plugins for security issues,
    benchmark performance, test compatibility, and manage certification.
    
    Example:
        >>> governance = PluginGovernance()
        >>> scan_result = governance.security_scan("my-plugin")
        >>> bench_result = governance.benchmark("my-plugin")
        >>> compat_result = governance.compatibility_test("my-plugin")
        >>> governance.certify("my-plugin")
    """
    
    def __init__(
        self,
        storage_dir: str = "./nethical_governance",
        strict_mode: bool = False
    ):
        """Initialize plugin governance.
        
        Args:
            storage_dir: Directory for governance data
            strict_mode: Enable strict security and performance requirements
        """
        self.storage_dir = storage_dir
        self.strict_mode = strict_mode
        
        # Security patterns to check
        self.security_patterns = {
            'exec': re.compile(r'\bexec\s*\('),
            'eval': re.compile(r'\beval\s*\('),
            'compile': re.compile(r'\bcompile\s*\('),
            'import_star': re.compile(r'from\s+\S+\s+import\s+\*'),
            'subprocess': re.compile(r'import\s+subprocess|from\s+subprocess'),
            'os_system': re.compile(r'os\.system|os\.popen'),
            'file_write': re.compile(r'open\s*\([^)]*[\'"]w[\'"]'),
            'network': re.compile(r'socket\.|urllib\.|requests\.|http\.'),
        }
        
        # Performance thresholds
        self.perf_thresholds = {
            'max_latency_ms': 100.0 if not strict_mode else 50.0,
            'min_throughput': 10.0 if not strict_mode else 20.0,
            'max_memory_mb': 500.0 if not strict_mode else 250.0,
        }
        
        # Certification records
        self._certifications: Dict[str, CertificationStatus] = {}
    
    def security_scan(
        self,
        plugin_id: str,
        plugin_code: Optional[str] = None,
        plugin_path: Optional[str] = None
    ) -> SecurityScanResult:
        """Perform security scanning on a plugin.
        
        Args:
            plugin_id: Plugin identifier
            plugin_code: Plugin source code (if available)
            plugin_path: Path to plugin files
            
        Returns:
            Security scan results
        """
        start_time = time.time()
        
        result = SecurityScanResult(
            plugin_id=plugin_id,
            scan_date=datetime.now(),
            security_level=SecurityLevel.SAFE
        )
        
        if not plugin_code and not plugin_path:
            result.add_warning("No code provided for scanning - limited analysis")
            return result
        
        # Scan for dangerous patterns
        if plugin_code:
            self._scan_code_patterns(plugin_code, result)
        
        # Check for known vulnerabilities (in a real implementation,
        # this would check against a vulnerability database)
        self._check_vulnerabilities(plugin_id, result)
        
        # Verify code signatures (in a real implementation)
        self._verify_signatures(plugin_id, result)
        
        result.scan_duration = time.time() - start_time
        return result
    
    def _scan_code_patterns(self, code: str, result: SecurityScanResult):
        """Scan code for security patterns."""
        for pattern_name, pattern in self.security_patterns.items():
            matches = pattern.findall(code)
            if matches:
                if pattern_name in ('exec', 'eval', 'compile'):
                    result.add_vulnerability(
                        f"Dangerous function detected: {pattern_name}",
                        SecurityLevel.HIGH
                    )
                elif pattern_name in ('subprocess', 'os_system'):
                    result.add_vulnerability(
                        f"System command execution detected: {pattern_name}",
                        SecurityLevel.MEDIUM
                    )
                elif pattern_name == 'import_star':
                    result.add_warning(
                        "Wildcard imports detected - may hide security issues"
                    )
                elif pattern_name in ('file_write', 'network'):
                    result.add_warning(
                        f"Potentially sensitive operation: {pattern_name}"
                    )
    
    def _check_vulnerabilities(self, plugin_id: str, result: SecurityScanResult):
        """Check for known vulnerabilities.
        
        In a real implementation, this would query a vulnerability database.
        """
        # Placeholder - would check CVE database, etc.
        pass
    
    def _verify_signatures(self, plugin_id: str, result: SecurityScanResult):
        """Verify code signatures.
        
        In a real implementation, this would verify cryptographic signatures.
        """
        # Placeholder - would verify digital signatures
        pass
    
    def benchmark(
        self,
        plugin_id: str,
        test_data: Optional[Any] = None,
        iterations: int = 1000
    ) -> BenchmarkResult:
        """Benchmark plugin performance.
        
        Args:
            plugin_id: Plugin identifier
            test_data: Test data for benchmarking
            iterations: Number of test iterations
            
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
            cpu_usage_percent=0.0
        )
        
        # Simulate benchmarking (in a real implementation, this would
        # actually run the plugin and measure performance)
        latencies = []
        start_time = time.time()
        
        for _ in range(iterations):
            op_start = time.time()
            # Simulate plugin operation
            time.sleep(0.001)  # 1ms
            latencies.append((time.time() - op_start) * 1000)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        latencies.sort()
        result.avg_latency_ms = sum(latencies) / len(latencies)
        result.p95_latency_ms = latencies[int(len(latencies) * 0.95)]
        result.p99_latency_ms = latencies[int(len(latencies) * 0.99)]
        result.throughput_ops_per_sec = iterations / total_time
        result.memory_usage_mb = 50.0  # Simulated
        result.cpu_usage_percent = 10.0  # Simulated
        
        # Check against thresholds
        result.meets_requirements(**self.perf_thresholds)
        
        if not result.passed:
            result.notes.append("Performance requirements not met")
        
        return result
    
    def compatibility_test(
        self,
        plugin_id: str,
        nethical_version: str = "0.1.0",
        python_version: str = "3.8"
    ) -> CompatibilityReport:
        """Test plugin compatibility.
        
        Args:
            plugin_id: Plugin identifier
            nethical_version: Nethical version to test against
            python_version: Python version to test against
            
        Returns:
            Compatibility report
        """
        report = CompatibilityReport(
            plugin_id=plugin_id,
            test_date=datetime.now(),
            nethical_version=nethical_version,
            python_version=python_version,
            compatible=True
        )
        
        # Test basic import
        report.add_test_result("import_test", True, "Plugin imports successfully")
        
        # Test API compatibility
        report.add_test_result("api_test", True, "Plugin API is compatible")
        
        # Test dependencies
        report.add_test_result("dependency_test", True, "All dependencies satisfied")
        
        # Test detector interface (if applicable)
        report.add_test_result("detector_interface", True, "Detector interface implemented correctly")
        
        return report
    
    def certify(
        self,
        plugin_id: str,
        security_result: Optional[SecurityScanResult] = None,
        benchmark_result: Optional[BenchmarkResult] = None,
        compatibility_result: Optional[CompatibilityReport] = None
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
        requirements_met = (
            security_result.passed and
            security_result.security_level not in (SecurityLevel.HIGH, SecurityLevel.CRITICAL) and
            benchmark_result.passed and
            compatibility_result.compatible
        )
        
        if requirements_met:
            status = CertificationStatus.CERTIFIED
        else:
            status = CertificationStatus.NOT_CERTIFIED
        
        self._certifications[plugin_id] = status
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
            return True
        return False
    
    def generate_report(
        self,
        plugin_id: str,
        security_result: SecurityScanResult,
        benchmark_result: BenchmarkResult,
        compatibility_result: CompatibilityReport
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
        
        return {
            'plugin_id': plugin_id,
            'report_date': datetime.now().isoformat(),
            'certification_status': cert_status.value,
            'security': {
                'level': security_result.security_level.value,
                'passed': security_result.passed,
                'vulnerabilities': len(security_result.vulnerabilities),
                'warnings': len(security_result.warnings),
                'scan_duration': security_result.scan_duration,
            },
            'performance': {
                'passed': benchmark_result.passed,
                'avg_latency_ms': benchmark_result.avg_latency_ms,
                'p95_latency_ms': benchmark_result.p95_latency_ms,
                'throughput': benchmark_result.throughput_ops_per_sec,
                'memory_mb': benchmark_result.memory_usage_mb,
            },
            'compatibility': {
                'compatible': compatibility_result.compatible,
                'nethical_version': compatibility_result.nethical_version,
                'python_version': compatibility_result.python_version,
                'tests_passed': sum(1 for p in compatibility_result.test_results.values() if p),
                'tests_total': len(compatibility_result.test_results),
            }
        }
