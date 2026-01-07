"""Shadow AI Detector - Detect unauthorized AI models in infrastructure.

This detector identifies unauthorized AI models running in infrastructure by analyzing:
- LLM API calls (OpenAI, Anthropic, Cohere, Google)
- Local model execution (Ollama, LM Studio, vLLM)
- Edge AI devices
- GPU usage patterns
- Model file signatures

Target latency: <20ms
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

from ...core.models import SafetyViolation
from ..base_detector import BaseDetector, DetectorStatus, ViolationSeverity


@dataclass
class ShadowAIDetectorConfig:
    """Configuration for Shadow AI Detector."""

    # Authorized model registry/whitelist
    authorized_apis: set[str] = field(
        default_factory=lambda: {
            "api.openai.com",
            "api.anthropic.com",
            "api.cohere.ai",
            "generativelanguage.googleapis.com",
        }
    )
    authorized_models: set[str] = field(default_factory=set)
    authorized_ports: set[int] = field(default_factory=lambda: {11434, 8080})

    # Detection patterns
    enable_api_detection: bool = True
    enable_gpu_detection: bool = True
    enable_model_file_detection: bool = True
    enable_port_scan: bool = True

    # Performance
    max_scan_time_ms: float = 18.0  # Target: <20ms

    # Severity thresholds
    critical_threshold: float = 0.9
    high_threshold: float = 0.7
    medium_threshold: float = 0.5


class ShadowAIDetector(BaseDetector):
    """Detect unauthorized AI models in infrastructure with ultra-low latency."""

    # API endpoint patterns for common LLM providers
    API_PATTERNS = {
        "openai": re.compile(r"api\.openai\.com/v1/(chat/completions|completions|embeddings)"),
        "anthropic": re.compile(r"api\.anthropic\.com/v1/(messages|complete)"),
        "cohere": re.compile(r"api\.cohere\.ai/(generate|embed|classify)"),
        "google": re.compile(r"generativelanguage\.googleapis\.com/v1/(models|generateContent)"),
        "huggingface": re.compile(r"api-inference\.huggingface\.co/models/"),
        "replicate": re.compile(r"api\.replicate\.com/v1/predictions"),
    }

    # Local model execution patterns
    LOCAL_PATTERNS = {
        "ollama": re.compile(r"localhost:11434|127\.0\.0\.1:11434"),
        "lm_studio": re.compile(r"localhost:8080|127\.0\.0\.1:8080"),
        "vllm": re.compile(r"localhost:8000|127\.0\.0\.1:8000"),
        "text-generation-webui": re.compile(r"localhost:5000|127\.0\.0\.1:5000"),
    }

    # Model file signatures
    MODEL_FILE_PATTERNS = {
        "gguf": re.compile(r"\.gguf$", re.IGNORECASE),
        "bin": re.compile(r"pytorch_model\.bin$", re.IGNORECASE),
        "safetensors": re.compile(r"\.safetensors$", re.IGNORECASE),
        "onnx": re.compile(r"\.onnx$", re.IGNORECASE),
    }

    def __init__(self, config: ShadowAIDetectorConfig | None = None):
        """Initialize the Shadow AI Detector.

        Args:
            config: Optional configuration for the detector
        """
        super().__init__(
            name="shadow_ai_detector",
            version="1.0.0",
            description="Detects unauthorized AI models in infrastructure",
        )
        self.config = config or ShadowAIDetectorConfig()
        self._status = DetectorStatus.ACTIVE

    async def detect_violations(
        self, context: dict[str, Any], **kwargs: Any
    ) -> list[SafetyViolation]:
        """Detect unauthorized AI models in network traffic and system activity.

        Args:
            context: Detection context containing network_traffic, system_info, etc.
            **kwargs: Additional parameters

        Returns:
            List of detected safety violations
        """
        start_time = time.perf_counter()
        violations = []

        try:
            # Extract network traffic data
            network_traffic = context.get("network_traffic", {})

            # Run parallel detection
            detection_tasks = []

            if self.config.enable_api_detection:
                detection_tasks.append(self._detect_api_calls(network_traffic))

            if self.config.enable_gpu_detection:
                detection_tasks.append(self._detect_gpu_usage(context.get("system_info", {})))

            if self.config.enable_model_file_detection:
                detection_tasks.append(self._detect_model_files(context.get("file_system", {})))

            if self.config.enable_port_scan:
                detection_tasks.append(self._detect_suspicious_ports(network_traffic))

            # Gather results
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    continue
                if result:
                    violations.extend(result)

            # Check execution time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.config.max_scan_time_ms:
                self._metrics.false_positives += 1  # Track performance issues

        except Exception:
            self._metrics.failed_runs += 1
            raise

        self._metrics.total_runs += 1
        self._metrics.successful_runs += 1
        self._metrics.violations_detected += len(violations)

        return violations

    async def _detect_api_calls(self, network_traffic: dict[str, Any]) -> list[SafetyViolation]:
        """Detect unauthorized API calls to LLM providers.

        Args:
            network_traffic: Network traffic data with URLs and endpoints

        Returns:
            List of violations for unauthorized API calls
        """
        violations = []
        urls = network_traffic.get("urls", [])
        endpoints = network_traffic.get("endpoints", [])

        all_targets = urls + endpoints

        for target in all_targets:
            target_str = str(target)

            # Check against patterns
            for provider, pattern in self.API_PATTERNS.items():
                if pattern.search(target_str):
                    # Check if authorized
                    is_authorized = any(auth in target_str for auth in self.config.authorized_apis)

                    if not is_authorized:
                        confidence = 0.95
                        violations.append(
                            SafetyViolation(
                                severity=self._compute_severity(confidence),
                                category="unauthorized_ai_model",
                                description=f"Unauthorized {provider} API call detected",
                                confidence=confidence,
                                evidence=[f"URL: {target_str}"],
                                recommendation=f"Review and whitelist {provider} API usage or block this endpoint",
                            )
                        )

        return violations

    async def _detect_gpu_usage(self, system_info: dict[str, Any]) -> list[SafetyViolation]:
        """Detect GPU usage patterns indicative of AI model execution.

        Args:
            system_info: System information including GPU metrics

        Returns:
            List of violations for suspicious GPU usage
        """
        violations = []
        gpu_metrics = system_info.get("gpu_metrics", {})

        if not gpu_metrics:
            return violations

        # Check for high GPU memory usage
        gpu_memory_usage = gpu_metrics.get("memory_usage_percent", 0)
        gpu_processes = gpu_metrics.get("processes", [])

        if gpu_memory_usage > 70 and gpu_processes:
            # Analyze processes
            for process in gpu_processes:
                process_name = process.get("name", "").lower()

                # Check for known AI frameworks
                suspicious_keywords = ["python", "pytorch", "tensorflow", "ollama", "vllm"]

                if any(keyword in process_name for keyword in suspicious_keywords):
                    # Check if authorized
                    is_authorized = process.get("pid") in self.config.authorized_models

                    if not is_authorized:
                        confidence = 0.75
                        violations.append(
                            SafetyViolation(
                                severity=self._compute_severity(confidence),
                                category="unauthorized_ai_model",
                                description="Unauthorized AI model execution detected on GPU",
                                confidence=confidence,
                                evidence=[
                                    f"Process: {process_name}",
                                    f"GPU Memory: {gpu_memory_usage}%",
                                    f"PID: {process.get('pid')}",
                                ],
                                recommendation="Review GPU process and verify authorization",
                            )
                        )

        return violations

    async def _detect_model_files(self, file_system: dict[str, Any]) -> list[SafetyViolation]:
        """Detect model file signatures in the file system.

        Args:
            file_system: File system data with file paths

        Returns:
            List of violations for unauthorized model files
        """
        violations = []
        files = file_system.get("files", [])

        for file_path in files:
            file_path_str = str(file_path)

            # Check against model file patterns
            for model_type, pattern in self.MODEL_FILE_PATTERNS.items():
                if pattern.search(file_path_str):
                    # Check if authorized
                    is_authorized = any(
                        auth_model in file_path_str for auth_model in self.config.authorized_models
                    )

                    if not is_authorized:
                        confidence = 0.85
                        violations.append(
                            SafetyViolation(
                                severity=self._compute_severity(confidence),
                                category="unauthorized_ai_model",
                                description=f"Unauthorized {model_type} model file detected",
                                confidence=confidence,
                                evidence=[f"File: {file_path_str}"],
                                recommendation="Review and whitelist model file or remove it",
                            )
                        )

        return violations

    async def _detect_suspicious_ports(self, network_traffic: dict[str, Any]) -> list[SafetyViolation]:
        """Detect suspicious ports commonly used by local AI models.

        Args:
            network_traffic: Network traffic data with port information

        Returns:
            List of violations for suspicious ports
        """
        violations = []
        connections = network_traffic.get("connections", [])

        # Known AI model ports
        ai_ports = {
            11434: "Ollama",
            8080: "LM Studio",
            8000: "vLLM",
            5000: "Text Generation WebUI",
            7860: "Gradio",
        }

        for connection in connections:
            port = connection.get("port")

            if port in ai_ports:
                # Check if authorized
                is_authorized = port in self.config.authorized_ports

                if not is_authorized:
                    service = ai_ports[port]
                    confidence = 0.90
                    violations.append(
                        SafetyViolation(
                            severity=self._compute_severity(confidence),
                            category="unauthorized_ai_model",
                            description=f"Unauthorized {service} service detected on port {port}",
                            confidence=confidence,
                            evidence=[f"Port: {port}", f"Service: {service}"],
                            recommendation=f"Review {service} service or add to authorized ports",
                        )
                    )

        return violations

    def _compute_severity(self, confidence: float) -> ViolationSeverity:
        """Compute violation severity based on confidence.

        Args:
            confidence: Detection confidence score

        Returns:
            Violation severity level
        """
        if confidence >= self.config.critical_threshold:
            return ViolationSeverity.CRITICAL
        elif confidence >= self.config.high_threshold:
            return ViolationSeverity.HIGH
        elif confidence >= self.config.medium_threshold:
            return ViolationSeverity.MEDIUM
        else:
            return ViolationSeverity.LOW

    async def scan(self, network_traffic: dict[str, Any]) -> dict[str, Any]:
        """Public API for scanning network traffic.

        Args:
            network_traffic: Network traffic data to scan

        Returns:
            Dictionary with scan results
        """
        context = {"network_traffic": network_traffic}
        violations = await self.detect_violations(context)

        return {
            "status": "success",
            "violations_count": len(violations),
            "violations": [
                {
                    "severity": v.severity.value,
                    "category": v.category,
                    "description": v.description,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                }
                for v in violations
            ],
            "latency_ms": self._metrics.avg_execution_time * 1000,
        }
