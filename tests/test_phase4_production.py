"""
Tests for Phase 4 Production Hardening Features

Tests for:
- Health check endpoints
- WebSocket streaming
- CLI commands
- Plugin security
- Security headers middleware
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import the modules we're testing
from nethical.core.plugin_security import (
    PluginVerifier,
    VerificationStatus,
    VerificationResult,
)
from nethical.middleware.security import SecurityHeadersMiddleware


class TestPluginSecurity:
    """Tests for plugin security module."""

    def test_plugin_verifier_initialization(self):
        """Test PluginVerifier can be initialized."""
        verifier = PluginVerifier()
        assert verifier is not None
        assert verifier.trusted_publishers == []
        assert verifier.revoked_plugins == {}

    def test_plugin_verifier_with_trusted_publishers(self):
        """Test PluginVerifier with trusted publishers."""
        publishers = ["nethical-org", "trusted-vendor"]
        verifier = PluginVerifier(trusted_publishers=publishers)
        assert verifier.trusted_publishers == publishers

    def test_add_trusted_publisher(self):
        """Test adding a trusted publisher."""
        verifier = PluginVerifier()
        verifier.add_trusted_publisher("new-publisher")
        assert "new-publisher" in verifier.trusted_publishers

    def test_remove_trusted_publisher(self):
        """Test removing a trusted publisher."""
        verifier = PluginVerifier(trusted_publishers=["publisher-1"])
        verifier.remove_trusted_publisher("publisher-1")
        assert "publisher-1" not in verifier.trusted_publishers

    def test_revoke_plugin(self):
        """Test revoking a plugin version."""
        verifier = PluginVerifier()
        verifier.revoke_plugin("test-plugin", "1.0.0", "security issue")
        assert "test-plugin" in verifier.revoked_plugins
        assert "1.0.0" in verifier.revoked_plugins["test-plugin"]

    def test_verify_nonexistent_plugin(self):
        """Test verification of non-existent plugin."""
        verifier = PluginVerifier()
        result = verifier.verify_plugin("/nonexistent/path")
        assert result.status == VerificationStatus.INVALID_SIGNATURE
        assert result.signature_valid is False

    def test_generate_manifest_hash(self, tmp_path):
        """Test manifest hash generation."""
        # Create a test file
        test_file = tmp_path / "test_plugin.py"
        test_file.write_text("# Test plugin content")

        verifier = PluginVerifier()
        hash_value = verifier.generate_manifest_hash(str(test_file))

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA-256 hex digest length

    def test_generate_manifest_hash_directory(self, tmp_path):
        """Test manifest hash generation for directory."""
        # Create test directory structure
        plugin_dir = tmp_path / "my_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "main.py").write_text("# Main module")
        (plugin_dir / "utils.py").write_text("# Utils module")

        verifier = PluginVerifier()
        hash_value = verifier.generate_manifest_hash(str(plugin_dir))

        assert hash_value is not None
        assert len(hash_value) == 64

    def test_verification_result_dataclass(self):
        """Test VerificationResult dataclass."""
        result = VerificationResult(
            status=VerificationStatus.VALID,
            plugin_name="test-plugin",
            version="1.0.0",
            publisher="nethical-org",
            verified_at="2024-01-01T00:00:00Z",
            manifest_hash="abc123",
            signature_valid=True,
            message="Verified successfully",
            metadata={"key": "value"},
        )
        assert result.status == VerificationStatus.VALID
        assert result.plugin_name == "test-plugin"
        assert result.signature_valid is True


class TestSecurityHeadersMiddleware:
    """Tests for security headers middleware."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock ASGI app."""
        async def app(scope, receive, send):
            pass
        return app

    def test_middleware_initialization(self, mock_app):
        """Test middleware can be initialized."""
        middleware = SecurityHeadersMiddleware(mock_app)
        assert middleware is not None
        assert middleware.enable_hsts is True

    def test_middleware_custom_config(self, mock_app):
        """Test middleware with custom configuration."""
        middleware = SecurityHeadersMiddleware(
            mock_app,
            enable_hsts=False,
            hsts_max_age=86400,
            cache_control="no-cache",
        )
        assert middleware.enable_hsts is False
        assert middleware.hsts_max_age == 86400
        assert middleware.cache_control == "no-cache"

    def test_hsts_header_building(self, mock_app):
        """Test HSTS header construction."""
        middleware = SecurityHeadersMiddleware(
            mock_app,
            hsts_max_age=31536000,
            hsts_include_subdomains=True,
            hsts_preload=True,
        )
        hsts_header = middleware._build_hsts_header()
        assert "max-age=31536000" in hsts_header
        assert "includeSubDomains" in hsts_header
        assert "preload" in hsts_header

    def test_default_csp(self, mock_app):
        """Test default Content-Security-Policy."""
        middleware = SecurityHeadersMiddleware(mock_app)
        csp = middleware._default_csp()
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_sensitive_endpoint_detection(self, mock_app):
        """Test sensitive endpoint detection."""
        middleware = SecurityHeadersMiddleware(mock_app)
        assert middleware._is_sensitive_endpoint("/evaluate") is True
        assert middleware._is_sensitive_endpoint("/health/ready") is True
        assert middleware._is_sensitive_endpoint("/status") is True
        assert middleware._is_sensitive_endpoint("/public/assets") is False


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client for the API."""
        from fastapi.testclient import TestClient
        from nethical.api import app
        return TestClient(app)

    def test_liveness_endpoint(self, client):
        """Test /health/live endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_startup_endpoint_structure(self, client):
        """Test /health/startup endpoint returns expected structure."""
        response = client.get("/health/startup")
        # May be 200 or 503 depending on startup state
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "version" in data
            assert data["version"] == "2.3.0"

    def test_readiness_endpoint_structure(self, client):
        """Test /health/ready endpoint returns expected structure."""
        response = client.get("/health/ready")
        # May be 200 or 503 depending on initialization state
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "checks" in data


class TestAPIUpdates:
    """Tests for API updates and new features."""

    @pytest.fixture
    def client(self):
        """Create test client for the API."""
        from fastapi.testclient import TestClient
        from nethical.api import app
        return TestClient(app)

    def test_root_endpoint_version(self, client):
        """Test root endpoint returns correct version."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.3.0"

    def test_root_endpoint_new_features(self, client):
        """Test root endpoint includes new features."""
        response = client.get("/")
        data = response.json()
        features = data.get("features", [])
        assert "WebSocket streaming" in features
        assert "Health checks" in features

    def test_root_endpoint_new_endpoints(self, client):
        """Test root endpoint lists new endpoints."""
        response = client.get("/")
        data = response.json()
        endpoints = data.get("endpoints", {})
        assert "health_live" in endpoints
        assert "health_ready" in endpoints
        assert "health_startup" in endpoints
        assert "ws_violations" in endpoints
        assert "ws_metrics" in endpoints


class TestCLI:
    """Tests for CLI module."""

    def test_cli_import(self):
        """Test CLI module can be imported."""
        from nethical.cli import cli
        assert cli is not None

    def test_cli_group(self):
        """Test CLI is a click group."""
        from nethical.cli import cli
        import click
        assert isinstance(cli, click.core.Group)

    def test_cli_commands_exist(self):
        """Test CLI has expected commands."""
        from nethical.cli import cli
        commands = cli.commands
        assert "init" in commands
        assert "evaluate" in commands
        assert "status" in commands
        assert "serve" in commands
        assert "verify-plugin" in commands

    def test_init_command(self, tmp_path):
        """Test init command creates configuration."""
        from click.testing import CliRunner
        from nethical.cli import cli

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert "Created configuration file" in result.output


class TestBenchmarkFramework:
    """Tests for benchmark framework."""

    def test_benchmark_config_dataclass(self):
        """Test BenchmarkConfig dataclass."""
        from benchmarks.runner import BenchmarkConfig

        config = BenchmarkConfig(
            name="test",
            iterations=100,
            warmup_iterations=10,
            concurrent_workers=5,
        )
        assert config.name == "test"
        assert config.iterations == 100
        assert config.warmup_iterations == 10
        assert config.concurrent_workers == 5

    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass."""
        from benchmarks.runner import BenchmarkResult

        result = BenchmarkResult(
            name="test",
            timestamp="2024-01-01T00:00:00Z",
            total_iterations=100,
            successful_iterations=98,
            failed_iterations=2,
            total_duration_seconds=10.0,
            avg_latency_ms=5.0,
            p50_latency_ms=4.0,
            p95_latency_ms=8.0,
            p99_latency_ms=12.0,
            min_latency_ms=1.0,
            max_latency_ms=20.0,
            throughput_rps=9.8,
        )
        assert result.name == "test"
        assert result.successful_iterations == 98

    def test_benchmark_runner_initialization(self):
        """Test BenchmarkRunner initialization."""
        from benchmarks.runner import BenchmarkRunner, BenchmarkConfig

        config = BenchmarkConfig(name="test")
        runner = BenchmarkRunner(config)
        assert runner.config.name == "test"
        assert runner.results == []

    def test_benchmark_comparer_initialization(self):
        """Test BenchmarkComparer initialization."""
        from benchmarks.compare import BenchmarkComparer

        comparer = BenchmarkComparer()
        assert comparer.baselines_dir is not None
        assert "avg_latency_ms" in comparer.thresholds

    def test_compare_metric(self):
        """Test metric comparison logic."""
        from benchmarks.compare import BenchmarkComparer

        comparer = BenchmarkComparer()

        # Test latency increase within threshold
        result = comparer.compare_metric("avg_latency_ms", 9.0, 10.0, 15.0)
        assert result.regression is False

        # Test latency increase beyond threshold
        result = comparer.compare_metric("avg_latency_ms", 12.0, 10.0, 10.0)
        assert result.regression is True

        # Test throughput decrease within threshold
        result = comparer.compare_metric("throughput_rps", 90.0, 100.0, -15.0)
        assert result.regression is False


class TestGrafanaDashboards:
    """Tests for Grafana dashboard files."""

    def test_overview_dashboard_exists(self):
        """Test overview dashboard file exists and is valid JSON."""
        dashboard_path = Path("deploy/grafana/dashboards/nethical-overview.json")
        assert dashboard_path.exists()
        
        with open(dashboard_path) as f:
            data = json.load(f)
        
        assert "title" in data
        assert data["title"] == "Nethical Overview"
        assert "panels" in data

    def test_violations_dashboard_exists(self):
        """Test violations dashboard file exists and is valid JSON."""
        dashboard_path = Path("deploy/grafana/dashboards/nethical-violations.json")
        assert dashboard_path.exists()
        
        with open(dashboard_path) as f:
            data = json.load(f)
        
        assert data["title"] == "Nethical Violations"

    def test_performance_dashboard_exists(self):
        """Test performance dashboard file exists and is valid JSON."""
        dashboard_path = Path("deploy/grafana/dashboards/nethical-performance.json")
        assert dashboard_path.exists()
        
        with open(dashboard_path) as f:
            data = json.load(f)
        
        assert data["title"] == "Nethical Performance"


class TestHelmChart:
    """Tests for Helm chart files."""

    def test_chart_yaml_version(self):
        """Test Chart.yaml has correct version."""
        chart_path = Path("deploy/helm/nethical/Chart.yaml")
        assert chart_path.exists()

        import yaml
        with open(chart_path) as f:
            data = yaml.safe_load(f)

        assert data["version"] == "1.0.0"
        assert data["appVersion"] == "2.3.0"
        assert data["name"] == "nethical"

    def test_values_yaml_structure(self):
        """Test values.yaml has expected structure."""
        values_path = Path("deploy/helm/nethical/values.yaml")
        assert values_path.exists()

        import yaml
        with open(values_path) as f:
            data = yaml.safe_load(f)

        assert "replicaCount" in data
        assert "image" in data
        assert "service" in data
        assert "autoscaling" in data
        assert "probes" in data

    def test_deployment_template_exists(self):
        """Test deployment.yaml template exists."""
        deployment_path = Path("deploy/helm/nethical/templates/deployment.yaml")
        assert deployment_path.exists()
