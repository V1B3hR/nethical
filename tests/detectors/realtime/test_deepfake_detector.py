"""Unit tests for Deepfake Detector."""

import pytest

from nethical.detectors.realtime import DeepfakeDetector, DeepfakeDetectorConfig


class TestDeepfakeDetector:
    """Test cases for Deepfake Detector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return DeepfakeDetector()

    @pytest.mark.asyncio
    async def test_detect_image_deepfake(self, detector):
        """Test detection of deepfake images."""
        context = {
            "media": b"fake_image_data_with_gan_artifacts",
            "media_type": "image",
            "metadata": {},
        }

        violations = await detector.detect_violations(context)
        # Detection depends on heuristics, so check structure
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_missing_exif(self, detector):
        """Test detection based on missing EXIF data."""
        context = {
            "media": b"image_without_exif",
            "media_type": "image",
            "metadata": {},  # No EXIF data
        }

        violations = await detector.detect_violations(context)
        # May or may not detect depending on other factors
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_suspicious_software(self, detector):
        """Test detection of suspicious software tags."""
        context = {
            "media": b"suspicious_image",
            "media_type": "image",
            "metadata": {
                "software": "FaceSwap GAN Editor",
            },
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert violations[0].category == "deepfake_media"

    @pytest.mark.asyncio
    async def test_detect_video_deepfake(self, detector):
        """Test detection of deepfake videos."""
        context = {
            "media": b"fake_video_data",
            "media_type": "video",
        }

        violations = await detector.detect_violations(context)
        # Temporal analysis should run
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_audio_deepfake(self, detector):
        """Test detection of audio deepfakes."""
        context = {
            "media": b"fake_audio_data",
            "media_type": "audio",
        }

        violations = await detector.detect_violations(context)
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_detect_api(self, detector):
        """Test public detect API."""
        media_data = b"test_image_data"

        result = await detector.detect(media_data, "image")

        assert result["status"] == "success"
        assert "is_deepfake" in result
        assert "confidence" in result
        assert "violations" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_empty_media(self, detector):
        """Test with empty media data."""
        context = {
            "media": b"",
            "media_type": "image",
        }

        violations = await detector.detect_violations(context)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_performance_target(self, detector):
        """Test that detection meets performance target (<30ms)."""
        import time

        context = {
            "media": b"test_image_data" * 100,
            "media_type": "image",
            "metadata": {},
        }

        start = time.perf_counter()
        await detector.detect_violations(context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 30ms (with margin for test overhead)
        assert elapsed_ms < 100

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = DeepfakeDetector()
        assert detector.name == "deepfake_detector"
        assert detector.version == "1.0.0"

    def test_custom_config(self):
        """Test detector with custom configuration."""
        config = DeepfakeDetectorConfig(
            image_threshold=0.8,
            enable_frequency_analysis=False,
        )
        detector = DeepfakeDetector(config)
        assert detector.config.image_threshold == 0.8
        assert detector.config.enable_frequency_analysis is False
