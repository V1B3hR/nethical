"""
Unit tests for Shadow Traffic Replay Tool

Tests cover:
- HAR and JSON parsing
- Dry-run behavior
- Header rewriting
- Method filtering
- Report generation
- Safety checks
- Rate limiting
- Retry logic
"""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

# Import the module to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from shadow_replay import (  # noqa: E402
    ShadowReplayTool,
    ReplayRequest,
    ReplayReport,
)

# Use requests_mock for mocking HTTP requests
try:
    import requests_mock
except ImportError:
    pytest.skip("requests_mock not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_har_data():
    """Sample HAR data for testing."""
    return {
        "log": {
            "version": "1.2",
            "creator": {"name": "test", "version": "1.0"},
            "entries": [
                {
                    "request": {
                        "method": "GET",
                        "url": "https://api.example.com/api/users",
                        "headers": [
                            {"name": "Accept", "value": "application/json"},
                            {"name": "User-Agent", "value": "TestAgent/1.0"},
                        ],
                        "queryString": [
                            {"name": "page", "value": "1"},
                        ],
                        "cookies": [],
                    }
                },
                {
                    "request": {
                        "method": "POST",
                        "url": "https://api.example.com/api/users",
                        "headers": [
                            {"name": "Content-Type", "value": "application/json"},
                        ],
                        "queryString": [],
                        "postData": {
                            "mimeType": "application/json",
                            "text": '{"name": "Test User"}',
                        },
                    }
                },
            ],
        }
    }


@pytest.fixture
def sample_json_data():
    """Sample JSON traffic data for testing."""
    return [
        {
            "method": "GET",
            "url": "/api/health",
            "headers": {"Accept": "application/json"},
            "body": None,
            "query_params": {},
        },
        {
            "method": "PUT",
            "url": "/api/settings",
            "headers": {"Content-Type": "application/json"},
            "body": '{"setting": "value"}',
            "query_params": {},
        },
    ]


@pytest.fixture
def temp_har_file(sample_har_data):
    """Create a temporary HAR file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".har", delete=False) as f:
        json.dump(sample_har_data, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    try:
        os.unlink(temp_path)
    except OSError:
        # Ignore if file is already deleted or cannot be deleted during cleanup
        pass

@pytest.fixture
def temp_json_file(sample_json_data):
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_json_data, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    try:
        # Ignore if file is already deleted or cannot be deleted during cleanup
        os.unlink(temp_path)
    except OSError:
        pass

@pytest.fixture
def staging_url():
    """Staging URL for testing."""
    return "https://staging.example.com"


# =============================================================================
# Test HAR Parsing
# =============================================================================


def test_parse_har_basic(temp_har_file, staging_url):
    """Test basic HAR file parsing."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    requests = tool.parse_har(temp_har_file)

    assert len(requests) == 2
    assert requests[0].method == "GET"
    assert requests[0].url == "https://api.example.com/api/users"
    assert requests[0].headers["Accept"] == "application/json"
    assert requests[0].query_params["page"] == "1"

    assert requests[1].method == "POST"
    assert requests[1].body == '{"name": "Test User"}'


def test_parse_har_empty_entries(staging_url):
    """Test parsing HAR with no entries."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".har", delete=False) as f:
        json.dump({"log": {"entries": []}}, f)
        temp_path = f.name

    try:
        tool = ShadowReplayTool(staging_base_url=staging_url, dry_run=True)
        requests = tool.parse_har(Path(temp_path))
        assert len(requests) == 0
    finally:
        os.unlink(temp_path)


def test_parse_har_missing_fields(staging_url):
    """Test parsing HAR with missing optional fields."""
    har_data = {
        "log": {
            "entries": [
                {
                    "request": {
                        "method": "GET",
                        "url": "https://example.com/test",
                        # Missing headers, queryString, postData
                    }
                }
            ]
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".har", delete=False) as f:
        json.dump(har_data, f)
        temp_path = f.name

    try:
        tool = ShadowReplayTool(staging_base_url=staging_url, dry_run=True)
        requests = tool.parse_har(Path(temp_path))
        assert len(requests) == 1
        assert requests[0].method == "GET"
        assert requests[0].headers == {}
        assert requests[0].query_params == {}
        assert requests[0].body is None
    finally:
        os.unlink(temp_path)


# =============================================================================
# Test JSON Parsing
# =============================================================================


def test_parse_json_basic(temp_json_file, staging_url):
    """Test basic JSON file parsing."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    requests = tool.parse_json(temp_json_file)

    assert len(requests) == 2
    assert requests[0].method == "GET"
    assert requests[0].url == "/api/health"
    assert requests[0].headers["Accept"] == "application/json"

    assert requests[1].method == "PUT"
    assert requests[1].body == '{"setting": "value"}'


def test_parse_json_invalid_format(staging_url):
    """Test parsing invalid JSON format (not a list)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"not": "a list"}, f)
        temp_path = f.name

    try:
        tool = ShadowReplayTool(staging_base_url=staging_url, dry_run=True)
        with pytest.raises(ValueError, match="must contain a list"):
            tool.parse_json(Path(temp_path))
    finally:
        os.unlink(temp_path)


# =============================================================================
# Test Request Rewriting
# =============================================================================


def test_rewrite_request_basic(staging_url):
    """Test basic request rewriting."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    original = ReplayRequest(
        method="GET",
        url="https://api.example.com/api/users/123",
        headers={"Accept": "application/json"},
    )

    rewritten = tool._rewrite_request(original)

    assert rewritten.url == "https://staging.example.com/api/users/123"
    assert rewritten.headers["X-Nethical-Shadow"] == "true"
    assert rewritten.headers["Accept"] == "application/json"
    assert "Host" not in rewritten.headers


def test_rewrite_request_with_auth(staging_url):
    """Test request rewriting with authentication."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        auth="test-token-123",
    )

    original = ReplayRequest(
        method="GET",
        url="https://api.example.com/api/secure",
        headers={},
    )

    rewritten = tool._rewrite_request(original)

    assert rewritten.headers["Authorization"] == "Bearer test-token-123"
    assert rewritten.headers["X-Nethical-Shadow"] == "true"


def test_rewrite_request_removes_host_header(staging_url):
    """Test that Host header is removed during rewriting."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    original = ReplayRequest(
        method="GET",
        url="https://api.example.com/test",
        headers={
            "Host": "api.example.com",
            "Accept": "application/json",
        },
    )

    rewritten = tool._rewrite_request(original)

    assert "Host" not in rewritten.headers
    assert rewritten.headers["Accept"] == "application/json"


# =============================================================================
# Test Method Filtering
# =============================================================================


def test_skip_methods_default(staging_url):
    """Test that POST, PUT, PATCH, DELETE are skipped by default."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    # Should skip these methods by default
    for method in ["POST", "PUT", "PATCH", "DELETE"]:
        request = ReplayRequest(method=method, url="/test")
        should_skip, reason = tool._should_skip_request(request)
        assert should_skip is True
        assert method in reason

    # Should NOT skip GET
    request = ReplayRequest(method="GET", url="/test")
    should_skip, reason = tool._should_skip_request(request)
    assert should_skip is False


def test_allow_modifying_flag(staging_url):
    """Test that --allow-modifying enables state-changing methods."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        allow_modifying=True,
    )

    # Should NOT skip any methods when allow_modifying is True
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        request = ReplayRequest(method=method, url="/test")
        should_skip, reason = tool._should_skip_request(request)
        assert should_skip is False


def test_custom_skip_methods(staging_url):
    """Test custom skip methods configuration."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        skip_methods=["DELETE", "PATCH"],
    )

    # Only DELETE and PATCH should be skipped
    request = ReplayRequest(method="DELETE", url="/test")
    should_skip, _ = tool._should_skip_request(request)
    assert should_skip is True

    request = ReplayRequest(method="POST", url="/test")
    should_skip, skip_reason = tool._should_skip_request(request)
    assert should_skip is False
    assert skip_reason is None


# =============================================================================
# Test Dry-Run Mode
# =============================================================================


def test_dry_run_no_requests_sent(temp_json_file, staging_url):
    """Test that dry-run mode doesn't send actual requests."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    requests = tool.parse_json(temp_json_file)

    # Mock the _send_request method to ensure it's never called
    with patch.object(tool, "_send_request") as mock_send:
        report = tool.replay_traffic(requests)

        # _send_request should never be called in dry-run mode
        mock_send.assert_not_called()

    # Report should show processed but not sent
    assert report.processed >= 0  # Some may be skipped
    assert report.sent == 0


# =============================================================================
# Test Sending Requests
# =============================================================================


def test_send_request_success(staging_url):
    """Test successful request sending with mocked response."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
    )

    request = ReplayRequest(
        method="GET",
        url="https://staging.example.com/api/test",
        headers={"Accept": "application/json"},
    )

    with requests_mock.Mocker() as m:
        m.get(
            "https://staging.example.com/api/test",
            status_code=200,
            text='{"status": "ok"}',
        )

        response = tool._send_request(request)

        assert response.status_code == 200
        assert response.error is None
        assert response.latency_ms > 0


def test_send_request_error(staging_url):
    """Test request sending with network error."""
    import requests

    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
    )

    request = ReplayRequest(
        method="GET",
        url="https://staging.example.com/api/test",
        headers={},
    )

    with requests_mock.Mocker() as m:
        m.get(
            "https://staging.example.com/api/test",
            exc=requests.exceptions.ConnectionError("Network error"),
        )

        response = tool._send_request(request)

        assert response.error is not None
        assert "Network error" in response.error


# =============================================================================
# Test Replay Traffic
# =============================================================================


def test_replay_traffic_with_mocked_responses(temp_json_file, staging_url):
    """Test full replay with mocked HTTP responses."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
        skip_methods=[],  # Don't skip any methods for this test
    )

    requests = tool.parse_json(temp_json_file)

    with requests_mock.Mocker() as m:
        # Mock all requests
        m.register_uri(
            requests_mock.ANY,
            requests_mock.ANY,
            status_code=200,
            text='{"result": "ok"}',
        )

        report = tool.replay_traffic(requests)

        assert report.total_requests == len(requests)
        assert report.sent == len(requests)
        assert report.successful == len(requests)
        assert report.errors == 0


def test_replay_traffic_with_skipped_methods(temp_json_file, staging_url):
    """Test replay with method filtering."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
        skip_methods=["PUT"],  # Skip PUT methods
    )

    requests = tool.parse_json(temp_json_file)

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=200)

        report = tool.replay_traffic(requests)

        # One PUT request should be skipped
        assert report.skipped > 0
        assert "PUT" in str(report.skipped_reasons)


# =============================================================================
# Test Report Generation
# =============================================================================


def test_report_to_dict():
    """Test report serialization to dictionary."""
    report = ReplayReport(
        total_requests=10,
        processed=8,
        skipped=2,
        sent=8,
        successful=7,
        failed=1,
        errors=0,
    )

    report_dict = report.to_dict()

    assert isinstance(report_dict, dict)
    assert report_dict["total_requests"] == 10
    assert report_dict["processed"] == 8
    assert report_dict["skipped"] == 2
    assert report_dict["sent"] == 8
    assert report_dict["successful"] == 7


def test_save_report(staging_url):
    """Test saving report to JSON file."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    tool.report = ReplayReport(
        total_requests=5,
        processed=5,
        sent=0,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        tool.save_report(temp_path)

        # Verify file was created and contains valid JSON
        with open(temp_path, "r") as f:
            saved_data = json.load(f)

        assert saved_data["total_requests"] == 5
        assert saved_data["processed"] == 5
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


# =============================================================================
# Test Safety Checks
# =============================================================================


def test_production_safety_check_blocks_production_url():
    """Test that production-like URLs are blocked."""
    with pytest.raises(SystemExit):
        ShadowReplayTool(
            staging_base_url="https://api.nethical.com",
            dry_run=True,
        )


def test_production_safety_check_with_force_flag():
    """Test that --force flag overrides production safety check."""
    # Should not raise SystemExit with force=True
    tool = ShadowReplayTool(
        staging_base_url="https://api.nethical.com",
        dry_run=True,
        force=True,
    )

    assert tool.staging_base_url == "https://api.nethical.com"


def test_production_safety_check_env_variable():
    """Test that production environment variables are checked."""
    original_env = os.environ.get("ENV")

    try:
        os.environ["ENV"] = "production"

        with pytest.raises(SystemExit):
            ShadowReplayTool(
                staging_base_url="https://staging.example.com",
                dry_run=True,
            )
    finally:
        # Restore original env
        if original_env:
            os.environ["ENV"] = original_env
        else:
            os.environ.pop("ENV", None)


# =============================================================================
# Test Rate Limiting
# =============================================================================


def test_rate_limiting_delays_requests(staging_url):
    """Test that rate limiting adds appropriate delays."""
    import time

    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        rps=10,  # 10 requests per second = 0.1s between requests
    )

    start_time = time.time()

    # Simulate processing 3 requests
    for i in range(3):
        tool._apply_rate_limit(i, start_time)

    elapsed = time.time() - start_time

    # Should take at least 0.2 seconds (2 intervals)
    # Give some tolerance for execution time
    assert elapsed >= 0.15


def test_rate_limiting_disabled_when_none(staging_url):
    """Test that rate limiting is disabled when rps is None."""
    import time

    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        rps=None,
    )

    start_time = time.time()

    # Should return immediately
    tool._apply_rate_limit(0, start_time)
    tool._apply_rate_limit(1, start_time)
    tool._apply_rate_limit(2, start_time)

    elapsed = time.time() - start_time

    # Should be very fast (< 0.01s)
    assert elapsed < 0.01


# =============================================================================
# Test Response Sampling
# =============================================================================


def test_response_sampling_limits_samples(temp_json_file, staging_url):
    """Test that response samples are limited to 10."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
        skip_methods=[],
    )

    # Create many requests
    many_requests = [
        ReplayRequest(method="GET", url=f"/api/test/{i}", headers={}) for i in range(20)
    ]

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=200)

        report = tool.replay_traffic(many_requests)

        # Should only sample up to 10 responses
        assert len(report.response_samples) <= 10


def test_error_sampling_limits_samples(staging_url):
    """Test that error samples are limited to 10."""
    import requests

    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
        skip_methods=[],
    )

    # Create many requests
    many_requests = [
        ReplayRequest(method="GET", url=f"/api/test/{i}", headers={}) for i in range(20)
    ]

    with requests_mock.Mocker() as m:
        # All requests fail
        m.register_uri(
            requests_mock.ANY,
            requests_mock.ANY,
            exc=requests.exceptions.ConnectionError("Network error"),
        )

        report = tool.replay_traffic(many_requests)

        # Should only sample up to 10 errors
        assert len(report.error_samples) <= 10


# =============================================================================
# Test Edge Cases
# =============================================================================


def test_empty_requests_list(staging_url):
    """Test replaying empty requests list."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
    )

    report = tool.replay_traffic([])

    assert report.total_requests == 0
    assert report.processed == 0
    assert report.sent == 0


def test_url_normalization(staging_url):
    """Test that staging URL is normalized (trailing slash removed)."""
    tool = ShadowReplayTool(
        staging_base_url="https://staging.example.com/",  # With trailing slash
        dry_run=True,
    )

    assert tool.staging_base_url == "https://staging.example.com"


def test_method_case_insensitive(staging_url):
    """Test that method filtering is case-insensitive."""
    # Test that lowercase input in skip_methods is converted to uppercase
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,
        skip_methods=["post", "Put"],  # mixed case input
    )

    # Verify skip_methods are stored as uppercase
    assert "POST" in tool.skip_methods
    assert "PUT" in tool.skip_methods

    # Should skip POST (uppercase request method)
    request = ReplayRequest(method="POST", url="/test")
    should_skip, reason = tool._should_skip_request(request)
    assert should_skip is True
    assert "POST" in reason

    # Should also skip post (lowercase request method)
    request = ReplayRequest(method="post", url="/test")
    should_skip, reason = tool._should_skip_request(request)
    assert should_skip is True
    assert "post" in reason

    # Should skip PUT with mixed case
    request = ReplayRequest(method="Put", url="/test")
    should_skip, reason = tool._should_skip_request(request)
    assert should_skip is True


# =============================================================================
# Integration Tests
# =============================================================================


def test_end_to_end_har_replay(temp_har_file, staging_url):
    """Test complete end-to-end HAR replay workflow."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=False,
        skip_methods=["POST"],  # Skip POST
        auth="test-token",
        rps=None,  # No rate limiting for test speed
    )

    with requests_mock.Mocker() as m:
        m.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=200)

        # Parse HAR
        requests = tool.parse_har(temp_har_file)
        assert len(requests) == 2

        # Replay traffic
        report = tool.replay_traffic(requests)

        # Verify report
        assert report.total_requests == 2
        assert report.skipped == 1  # POST should be skipped
        assert report.sent == 1  # Only GET should be sent

        # Save report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_report_path = Path(f.name)

        try:
            tool.save_report(temp_report_path)
            assert temp_report_path.exists()
        finally:
            try:
                os.unlink(temp_report_path)
            except OSError:
                pass


def test_end_to_end_json_replay_dry_run(temp_json_file, staging_url):
    """Test complete end-to-end JSON replay in dry-run mode."""
    tool = ShadowReplayTool(
        staging_base_url=staging_url,
        dry_run=True,  # Dry-run mode
    )

    # Parse JSON
    requests = tool.parse_json(temp_json_file)
    assert len(requests) == 2

    # Replay traffic (dry-run)
    report = tool.replay_traffic(requests)

    # In dry-run mode, no requests should be sent
    assert report.sent == 0
    # But requests should be processed (unless skipped by method filter)
    assert report.total_requests == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
