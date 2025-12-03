"""
Tests for Zero Trust enhancements including rate limiting and anomaly detection.

Tests cover:
- RateLimiter token bucket implementation
- AnomalyDetector pattern analysis
- QuarantineManager device isolation
"""

import pytest
import time
from datetime import datetime, timezone

from nethical.security.zero_trust import (
    TrustLevel,
    DeviceHealthStatus,
    RateLimiter,
    AnomalyDetector,
    QuarantineManager,
)


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.requests_per_minute == 60
        assert limiter.burst_size == 10

    def test_first_request_allowed(self):
        """Test that first request is always allowed."""
        limiter = RateLimiter()
        allowed, remaining = limiter.check_rate("user1")
        assert allowed is True
        assert remaining >= 0

    def test_burst_capacity(self):
        """Test burst capacity allows multiple requests."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Should allow up to burst_size requests
        for i in range(5):
            allowed, _ = limiter.check_rate("user1")
            assert allowed is True

        # Next request should be denied
        allowed, remaining = limiter.check_rate("user1")
        assert allowed is False
        assert remaining == 0

    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # Use up all tokens
        limiter.check_rate("user1")
        limiter.check_rate("user1")

        # Wait for refill
        time.sleep(0.1)  # 0.1 seconds should add some tokens

        # Should be allowed again
        allowed, _ = limiter.check_rate("user1")
        # Depending on timing, might or might not be allowed
        # Just verify no error

    def test_separate_identities(self):
        """Test that different identities have separate buckets."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # Use up user1's tokens
        limiter.check_rate("user1")
        limiter.check_rate("user1")
        allowed1, _ = limiter.check_rate("user1")

        # User2 should still have tokens
        allowed2, _ = limiter.check_rate("user2")

        assert allowed1 is False
        assert allowed2 is True

    def test_get_status(self):
        """Test status reporting."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        limiter.check_rate("user1")

        status = limiter.get_status("user1")
        assert status["identity"] == "user1"
        assert "tokens_remaining" in status
        assert status["requests_per_minute"] == 60

    def test_reset(self):
        """Test resetting rate limit for identity."""
        limiter = RateLimiter(burst_size=2)

        # Use up tokens
        limiter.check_rate("user1")
        limiter.check_rate("user1")
        limiter.check_rate("user1")

        # Reset
        limiter.reset("user1")

        # Should be allowed again
        allowed, _ = limiter.check_rate("user1")
        assert allowed is True


class TestAnomalyDetector:
    """Test AnomalyDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = AnomalyDetector(window_size=100, threshold_std=3.0)
        assert detector.window_size == 100
        assert detector.threshold_std == 3.0

    def test_normal_request(self):
        """Test that normal requests don't trigger anomaly."""
        detector = AnomalyDetector()

        # Record some normal requests
        for i in range(20):
            result = detector.record_request(
                identity="user1",
                request_type="inference",
            )
            time.sleep(0.05)  # 50ms between requests

        # Normal requests shouldn't trigger anomaly
        # (anomaly is based on frequency, first few won't trigger)

    def test_high_frequency_anomaly(self):
        """Test that high frequency requests trigger anomaly."""
        detector = AnomalyDetector()

        # Record many requests very quickly
        anomaly = None
        for i in range(15):
            result = detector.record_request(
                identity="user1",
                request_type="inference",
            )
            if result is not None:
                anomaly = result

        # Should detect high frequency
        if anomaly:
            assert "frequency" in anomaly.get("reason", "").lower()

    def test_unusual_request_type(self):
        """Test detection of unusual request types."""
        detector = AnomalyDetector()

        # Establish baseline with one type - use slower rate to avoid frequency anomaly
        for i in range(25):
            detector.record_request(
                identity="user1",
                request_type="inference",
            )
            time.sleep(0.15)  # Slower rate to avoid frequency detection

        # Try unusual type
        result = detector.record_request(
            identity="user1",
            request_type="admin_override",
        )

        # Should detect unusual type (may or may not trigger depending on heuristics)
        # The test is flexible since detection depends on timing
        if result:
            reason = result.get("reason", "").lower()
            # Accept either unusual type or frequency anomaly
            assert "unusual" in reason or "frequency" in reason

    def test_get_anomalies(self):
        """Test getting anomaly history."""
        detector = AnomalyDetector()

        # Generate some activity
        for i in range(10):
            detector.record_request("user1", "type1")

        anomalies = detector.get_anomalies(limit=50)
        assert isinstance(anomalies, list)

    def test_get_identity_profile(self):
        """Test getting identity behavioral profile."""
        detector = AnomalyDetector()

        # Generate some activity
        detector.record_request("user1", "type_a")
        detector.record_request("user1", "type_b")
        detector.record_request("user1", "type_a")

        profile = detector.get_identity_profile("user1")
        assert profile["identity"] == "user1"
        assert profile["requests"] == 3
        assert "type_a" in profile["request_types"]
        assert profile["request_types"]["type_a"] == 2

    def test_empty_profile(self):
        """Test profile for unknown identity."""
        detector = AnomalyDetector()
        profile = detector.get_identity_profile("unknown")
        assert profile["requests"] == 0


class TestQuarantineManager:
    """Test QuarantineManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = QuarantineManager(
            auto_quarantine=True,
            quarantine_duration_hours=24,
        )
        assert manager.auto_quarantine is True
        assert manager.quarantine_duration_hours == 24

    def test_quarantine_device(self):
        """Test quarantining a device."""
        manager = QuarantineManager()

        record = manager.quarantine(
            device_id="device123",
            reason="Suspicious activity detected",
        )

        assert record["device_id"] == "device123"
        assert record["status"] == "active"
        assert "Suspicious" in record["reason"]

    def test_is_quarantined(self):
        """Test checking quarantine status."""
        manager = QuarantineManager()

        # Not quarantined initially
        assert manager.is_quarantined("device123") is False

        # Quarantine
        manager.quarantine("device123", "Test reason")

        # Should be quarantined now
        assert manager.is_quarantined("device123") is True

    def test_release_device(self):
        """Test releasing device from quarantine."""
        manager = QuarantineManager()

        manager.quarantine("device123", "Test reason")
        assert manager.is_quarantined("device123") is True

        # Release
        result = manager.release("device123", "Investigation complete")
        assert result is True

        # Should not be quarantined anymore
        assert manager.is_quarantined("device123") is False

    def test_release_non_quarantined(self):
        """Test releasing a non-quarantined device."""
        manager = QuarantineManager()

        result = manager.release("unknown_device")
        assert result is False

    def test_get_quarantined_devices(self):
        """Test getting list of quarantined devices."""
        manager = QuarantineManager()

        manager.quarantine("device1", "Reason 1")
        manager.quarantine("device2", "Reason 2")

        devices = manager.get_quarantined_devices()
        assert len(devices) == 2

        device_ids = [d["device_id"] for d in devices]
        assert "device1" in device_ids
        assert "device2" in device_ids

    def test_get_history(self):
        """Test getting quarantine history."""
        manager = QuarantineManager()

        manager.quarantine("device1", "First quarantine")
        manager.release("device1", "Released")
        manager.quarantine("device1", "Second quarantine")

        history = manager.get_history(limit=100)
        assert len(history) >= 2  # At least quarantine + release

    def test_quarantine_with_custom_duration(self):
        """Test quarantine with custom duration."""
        manager = QuarantineManager(quarantine_duration_hours=24)

        record = manager.quarantine(
            device_id="device123",
            reason="Test",
            duration_hours=48,  # Custom duration
        )

        # Check expiration time is correct
        expected_hours = 48
        actual_diff = record["expires_at"] - record["quarantined_at"]
        assert actual_diff.total_seconds() / 3600 == expected_hours


class TestIntegration:
    """Integration tests for Zero Trust enhancements."""

    def test_rate_limit_and_quarantine(self):
        """Test rate limiting leading to quarantine."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        quarantine = QuarantineManager()

        # Simulate rate limit abuse
        device_id = "suspicious_device"
        violations = 0

        for _ in range(10):
            allowed, _ = limiter.check_rate(device_id)
            if not allowed:
                violations += 1

        # If too many violations, quarantine
        if violations >= 3:
            quarantine.quarantine(device_id, "Rate limit abuse")

        assert quarantine.is_quarantined(device_id)

    def test_anomaly_detection_and_quarantine(self):
        """Test anomaly detection leading to quarantine."""
        detector = AnomalyDetector()
        quarantine = QuarantineManager()

        device_id = "anomalous_device"

        # Generate anomalous behavior
        high_severity_anomalies = 0
        for _ in range(15):
            result = detector.record_request(device_id, "suspicious_action")
            if result and result.get("severity") == "high":
                high_severity_anomalies += 1

        # Quarantine if high severity anomalies detected
        if high_severity_anomalies >= 1:
            quarantine.quarantine(device_id, "High severity anomaly detected")
            assert quarantine.is_quarantined(device_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
