"""Unit tests for Polymorphic Malware Detector."""

import pytest

from nethical.detectors.realtime import PolymorphicMalwareDetector


class TestPolymorphicMalwareDetector:
    """Test cases for Polymorphic Malware Detector."""

    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return PolymorphicMalwareDetector()

    @pytest.mark.asyncio
    async def test_detect_high_entropy(self, detector):
        """Test detection of high entropy (encrypted/packed) data."""
        # Generate high entropy data
        import random
        high_entropy_data = bytes([random.randint(0, 255) for _ in range(1000)])

        context = {
            "executable_data": high_entropy_data,
        }

        violations = await detector.detect_violations(context)
        # High entropy should be detected
        assert len(violations) > 0
        assert violations[0].category == "polymorphic_malware"

    @pytest.mark.asyncio
    async def test_detect_suspicious_syscalls(self, detector):
        """Test detection of suspicious syscall patterns."""
        context = {
            "executable_data": b"test_executable",
            "syscall_trace": ["mprotect", "mmap", "execve", "ptrace", "fork"],
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert any("syscall" in str(v.evidence).lower() for v in violations)

    @pytest.mark.asyncio
    async def test_detect_code_injection(self, detector):
        """Test detection of code injection behavior."""
        context = {
            "executable_data": b"test_data",
            "behavior_log": [
                {"type": "code_injection"},
                {"type": "privilege_escalation"},
            ],
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0
        assert any("injection" in str(v.evidence).lower() for v in violations)

    @pytest.mark.asyncio
    async def test_detect_memory_patterns(self, detector):
        """Test detection of suspicious memory access patterns."""
        context = {
            "executable_data": b"test_data",
            "memory_access": [
                {"type": "write_execute", "region": "0x1000"},
                {"type": "write_execute", "region": "0x2000"},
            ],
        }

        violations = await detector.detect_violations(context)
        assert len(violations) > 0

    @pytest.mark.asyncio
    async def test_analyze_api(self, detector):
        """Test public analyze API."""
        executable_data = b"test_executable_data"

        result = await detector.analyze(executable_data)

        assert result["status"] == "success"
        assert "is_malware" in result
        assert "confidence" in result
        assert "violations" in result
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_empty_executable(self, detector):
        """Test with empty executable data."""
        context = {"executable_data": b""}
        violations = await detector.detect_violations(context)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_performance_target(self, detector):
        """Test that detection meets performance target (<50ms)."""
        import time

        context = {
            "executable_data": b"test_data" * 100,
            "syscall_trace": ["open", "read", "write"],
            "behavior_log": [],
        }

        start = time.perf_counter()
        await detector.detect_violations(context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in under 50ms (with margin)
        assert elapsed_ms < 150

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = PolymorphicMalwareDetector()
        assert detector.name == "polymorphic_detector"
        assert detector.version == "1.0.0"

    def test_signature_database(self, detector):
        """Test signature database initialization."""
        assert len(detector._signature_db) > 0
        assert "polymorphic_packer" in detector._signature_db
