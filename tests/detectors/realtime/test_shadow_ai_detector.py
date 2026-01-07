"""Unit tests for Shadow AI Detector."""

import pytest

from nethical.detectors.realtime import ShadowAIDetector, ShadowAIDetectorConfig


class TestShadowAIDetector:
    """Test cases for Shadow AI Detector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return ShadowAIDetector()

    @pytest.fixture
    def detector_with_config(self):
        """Create detector with custom config."""
        config = ShadowAIDetectorConfig(
            authorized_apis={"api.openai.com"},
            enable_gpu_detection=True,
        )
        return ShadowAIDetector(config)

    @pytest.mark.asyncio
    async def test_detect_unauthorized_openai_api(self, detector):
        """Test detection of unauthorized OpenAI API calls."""
        context = {
            "network_traffic": {
                "urls": ["https://unauthorized-api.openai.com/v1/chat/completions"],
                "endpoints": [],
                "connections": [],
            }
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert violations[0].category == "unauthorized_ai_model"
        assert "openai" in violations[0].description.lower()

    @pytest.mark.asyncio
    async def test_detect_anthropic_api(self, detector):
        """Test detection of Anthropic API calls."""
        context = {
            "network_traffic": {
                "urls": ["https://api.anthropic.com/v1/messages"],
                "endpoints": [],
                "connections": [],
            }
        }

        violations = await detector.detect_violations(context)
        # Should detect as it's not in authorized list
        assert len(violations) > 0

    @pytest.mark.asyncio
    async def test_detect_ollama_port(self, detector):
        """Test detection of Ollama service on port 11434."""
        context = {
            "network_traffic": {
                "urls": [],
                "endpoints": [],
                "connections": [{"port": 11434, "host": "localhost"}],
            }
        }

        violations = await detector.detect_violations(context)
        # Ollama port is in authorized by default
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_detect_unauthorized_port(self, detector):
        """Test detection of unauthorized AI service port."""
        context = {
            "network_traffic": {
                "urls": [],
                "endpoints": [],
                "connections": [{"port": 5000, "host": "localhost"}],
            }
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert "5000" in str(violations[0].evidence)

    @pytest.mark.asyncio
    async def test_detect_gpu_usage(self, detector):
        """Test detection of GPU usage for AI models."""
        context = {
            "network_traffic": {},
            "system_info": {
                "gpu_metrics": {
                    "memory_usage_percent": 85,
                    "processes": [
                        {"name": "python pytorch", "pid": 1234},
                    ],
                }
            },
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert "GPU" in violations[0].description

    @pytest.mark.asyncio
    async def test_detect_model_files(self, detector):
        """Test detection of model files."""
        context = {
            "network_traffic": {},
            "file_system": {
                "files": [
                    "/home/user/models/llama-7b.gguf",
                    "/opt/models/pytorch_model.bin",
                ]
            },
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert any("gguf" in v.description.lower() or "bin" in v.description.lower() for v in violations)

    @pytest.mark.asyncio
    async def test_no_violations_empty_context(self, detector):
        """Test with empty context."""
        context = {"network_traffic": {}}
        violations = await detector.detect_violations(context)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_scan_api(self, detector):
        """Test public scan API."""
        network_traffic = {
            "urls": ["https://api.cohere.ai/generate"],
            "endpoints": [],
            "connections": [],
        }

        result = await detector.scan(network_traffic)

        assert result["status"] == "success"
        assert "violations_count" in result
        assert "violations" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_performance_target(self, detector):
        """Test that detection meets performance target (<20ms)."""
        import time

        context = {
            "network_traffic": {
                "urls": ["https://api.openai.com/v1/completions"],
                "endpoints": [],
                "connections": [{"port": 11434}],
            },
            "system_info": {
                "gpu_metrics": {
                    "memory_usage_percent": 50,
                    "processes": [],
                }
            },
        }

        start = time.perf_counter()
        await detector.detect_violations(context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 20ms (with some margin for test overhead)
        assert elapsed_ms < 50  # Allow 50ms for test overhead

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = ShadowAIDetector()
        assert detector.name == "shadow_ai_detector"
        assert detector.version == "1.0.0"

    def test_custom_config(self, detector_with_config):
        """Test detector with custom configuration."""
        assert len(detector_with_config.config.authorized_apis) == 1
        assert detector_with_config.config.enable_gpu_detection is True
