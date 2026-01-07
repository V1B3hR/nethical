"""Realtime Threat Detector - Unified interface for all threat detectors.

This provides a single entry point for all 5 specialized detectors with
support for parallel detection and performance monitoring.

Target: <50ms average latency under load
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .shadow_ai_detector import ShadowAIDetector, ShadowAIDetectorConfig
from .deepfake_detector import DeepfakeDetector, DeepfakeDetectorConfig
from .polymorphic_detector import PolymorphicMalwareDetector, PolymorphicDetectorConfig
from .prompt_injection_guard import PromptInjectionGuard, PromptInjectionGuardConfig
from .ai_vs_ai_defender import AIvsAIDefender, AIvsAIDefenderConfig


@dataclass
class RealtimeThreatDetectorConfig:
    """Configuration for Realtime Threat Detector."""

    # Performance target
    max_latency_ms: float = 50.0

    # Enable/disable specific detectors
    enable_shadow_ai: bool = True
    enable_deepfake: bool = True
    enable_polymorphic: bool = True
    enable_prompt_injection: bool = True
    enable_ai_vs_ai: bool = True

    # Parallel execution
    parallel_detection: bool = True

    # Individual detector configs
    shadow_ai_config: Optional[ShadowAIDetectorConfig] = None
    deepfake_config: Optional[DeepfakeDetectorConfig] = None
    polymorphic_config: Optional[PolymorphicDetectorConfig] = None
    prompt_injection_config: Optional[PromptInjectionGuardConfig] = None
    ai_vs_ai_config: Optional[AIvsAIDefenderConfig] = None


class RealtimeThreatDetector:
    """Unified interface for all threat detectors with performance optimization."""

    def __init__(self, config: Optional[RealtimeThreatDetectorConfig] = None):
        """Initialize the Realtime Threat Detector.

        Args:
            config: Optional configuration for the detector
        """
        self.config = config or RealtimeThreatDetectorConfig()

        # Initialize detectors
        self.shadow_ai = (
            ShadowAIDetector(self.config.shadow_ai_config)
            if self.config.enable_shadow_ai
            else None
        )
        self.deepfake = (
            DeepfakeDetector(self.config.deepfake_config) if self.config.enable_deepfake else None
        )
        self.polymorphic = (
            PolymorphicMalwareDetector(self.config.polymorphic_config)
            if self.config.enable_polymorphic
            else None
        )
        self.prompt_guard = (
            PromptInjectionGuard(self.config.prompt_injection_config)
            if self.config.enable_prompt_injection
            else None
        )
        self.ai_defense = (
            AIvsAIDefender(self.config.ai_vs_ai_config) if self.config.enable_ai_vs_ai else None
        )

        # Performance metrics
        self._total_detections = 0
        self._total_latency = 0.0
        self._latency_samples: List[float] = []

    async def evaluate_threat(
        self,
        input_data: Dict[str, Any],
        threat_type: str,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate threat with optional parallel detection.

        Args:
            input_data: Input data for detection
            threat_type: Type of threat to detect (or "all" for all detectors)
            parallel: Whether to run detectors in parallel

        Returns:
            Dictionary with threat score, detected issues, and latency metrics
        """
        start_time = time.perf_counter()

        try:
            # Route to appropriate detector(s)
            if threat_type == "shadow_ai" and self.shadow_ai:
                result = await self._detect_shadow_ai(input_data)
            elif threat_type == "deepfake" and self.deepfake:
                result = await self._detect_deepfake(input_data)
            elif threat_type == "polymorphic" and self.polymorphic:
                result = await self._detect_polymorphic(input_data)
            elif threat_type == "prompt_injection" and self.prompt_guard:
                result = await self._detect_prompt_injection(input_data)
            elif threat_type == "ai_vs_ai" and self.ai_defense:
                result = await self._detect_ai_vs_ai(input_data)
            elif threat_type == "all":
                result = await self._detect_all(input_data, parallel)
            else:
                result = {
                    "status": "error",
                    "message": f"Unknown or disabled threat type: {threat_type}",
                }

            # Calculate latency
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._total_detections += 1
            self._total_latency += elapsed_ms
            self._latency_samples.append(elapsed_ms)

            # Keep only last 1000 samples
            if len(self._latency_samples) > 1000:
                self._latency_samples = self._latency_samples[-1000:]

            # Add latency metrics to result
            result["latency_ms"] = elapsed_ms
            result["avg_latency_ms"] = self._total_latency / self._total_detections
            result["p95_latency_ms"] = self._compute_percentile(95)
            result["p99_latency_ms"] = self._compute_percentile(99)

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

    async def _detect_shadow_ai(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect shadow AI using dedicated detector."""
        violations = await self.shadow_ai.detect_violations(input_data)
        return self._format_result("shadow_ai", violations)

    async def _detect_deepfake(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect deepfakes using dedicated detector."""
        violations = await self.deepfake.detect_violations(input_data)
        return self._format_result("deepfake", violations)

    async def _detect_polymorphic(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect polymorphic malware using dedicated detector."""
        violations = await self.polymorphic.detect_violations(input_data)
        return self._format_result("polymorphic", violations)

    async def _detect_prompt_injection(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect prompt injections using dedicated detector."""
        violations = await self.prompt_guard.detect_violations(input_data)
        return self._format_result("prompt_injection", violations)

    async def _detect_ai_vs_ai(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect AI attacks using dedicated detector."""
        violations = await self.ai_defense.detect_violations(input_data)
        return self._format_result("ai_vs_ai", violations)

    async def _detect_all(
        self, input_data: Dict[str, Any], parallel: bool
    ) -> Dict[str, Any]:
        """Run all enabled detectors."""
        detection_tasks = []

        if self.shadow_ai:
            detection_tasks.append(("shadow_ai", self._detect_shadow_ai(input_data)))

        if self.deepfake:
            detection_tasks.append(("deepfake", self._detect_deepfake(input_data)))

        if self.polymorphic:
            detection_tasks.append(("polymorphic", self._detect_polymorphic(input_data)))

        if self.prompt_guard:
            detection_tasks.append(("prompt_injection", self._detect_prompt_injection(input_data)))

        if self.ai_defense:
            detection_tasks.append(("ai_vs_ai", self._detect_ai_vs_ai(input_data)))

        if parallel:
            # Run in parallel
            results = await asyncio.gather(
                *[task for _, task in detection_tasks], return_exceptions=True
            )

            # Combine results
            combined_result = {
                "status": "success",
                "detectors": {},
                "total_violations": 0,
                "max_threat_score": 0.0,
            }

            for (detector_name, _), result in zip(detection_tasks, results):
                if isinstance(result, Exception):
                    combined_result["detectors"][detector_name] = {
                        "status": "error",
                        "message": str(result),
                    }
                else:
                    combined_result["detectors"][detector_name] = result
                    combined_result["total_violations"] += result.get("violations_count", 0)
                    combined_result["max_threat_score"] = max(
                        combined_result["max_threat_score"], result.get("threat_score", 0.0)
                    )

            return combined_result
        else:
            # Run sequentially
            combined_result = {
                "status": "success",
                "detectors": {},
                "total_violations": 0,
                "max_threat_score": 0.0,
            }

            for detector_name, task in detection_tasks:
                try:
                    result = await task
                    combined_result["detectors"][detector_name] = result
                    combined_result["total_violations"] += result.get("violations_count", 0)
                    combined_result["max_threat_score"] = max(
                        combined_result["max_threat_score"], result.get("threat_score", 0.0)
                    )
                except Exception as e:
                    combined_result["detectors"][detector_name] = {
                        "status": "error",
                        "message": str(e),
                    }

            return combined_result

    def _format_result(self, detector_name: str, violations: List[Any]) -> Dict[str, Any]:
        """Format detector result."""
        threat_score = max([v.confidence for v in violations], default=0.0)

        return {
            "status": "success",
            "detector": detector_name,
            "violations_count": len(violations),
            "threat_score": threat_score,
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
        }

    def _compute_percentile(self, percentile: int) -> float:
        """Compute latency percentile."""
        if not self._latency_samples:
            return 0.0

        sorted_samples = sorted(self._latency_samples)
        index = int(len(sorted_samples) * percentile / 100)
        return sorted_samples[min(index, len(sorted_samples) - 1)]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_detections": self._total_detections,
            "avg_latency_ms": self._total_latency / self._total_detections
            if self._total_detections > 0
            else 0.0,
            "p50_latency_ms": self._compute_percentile(50),
            "p95_latency_ms": self._compute_percentile(95),
            "p99_latency_ms": self._compute_percentile(99),
            "detectors": {
                "shadow_ai": self.shadow_ai._metrics if self.shadow_ai else None,
                "deepfake": self.deepfake._metrics if self.deepfake else None,
                "polymorphic": self.polymorphic._metrics if self.polymorphic else None,
                "prompt_injection": self.prompt_guard._metrics if self.prompt_guard else None,
                "ai_vs_ai": self.ai_defense._metrics if self.ai_defense else None,
            },
        }
