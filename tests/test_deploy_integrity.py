"""
Unit and Integration Tests for Sovereign Deployment Subsystem (deploy/)

Verifies:
1. Dockerfile Security & CIS Hardening (gateway, build).
2. Kubernetes Security Manifests (AppArmor, Seccomp, NetworkPolicies, Sidecars, RBAC).
3. Sovereign Ambassador Sidecar Integration (IPC UNIX Domain Sockets, capability drop, non-subordination).
4. Helm Charts Validation (nethical and nethical-edge).
5. Grafana Deployment Dashboards (Overview, Performance, Violations).
6. PostgreSQL / TimescaleDB Migration Integrity.
7. Reproducible Build & Release Shell Scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

DEPLOY_DIR = Path("deploy")


class TestDockerfiles:
    """Validates container specifications against CIS benchmarks and reproducible standards."""

    def test_dockerfile_gateway_security(self) -> None:
        dockerfile = DEPLOY_DIR / "Dockerfile.gateway"
        assert dockerfile.exists(), "Dockerfile.gateway is missing"
        content = dockerfile.read_text(encoding="utf-8")

        # Multi-stage build
        assert "AS builder" in content
        assert "AS runner" in content

        # Non-root user creation and execution
        assert "useradd" in content
        assert "nethical" in content
        assert "USER nethical" in content

        # Health check presence
        assert "HEALTHCHECK" in content

        # Persistent volume declaration
        assert 'VOLUME ["/data"]' in content or "VOLUME" in content

        # Signal handling
        assert "STOPSIGNAL SIGTERM" in content

    def test_dockerfile_build_reproducibility(self) -> None:
        dockerfile = DEPLOY_DIR / "Dockerfile.build"
        assert dockerfile.exists(), "Dockerfile.build is missing"
        content = dockerfile.read_text(encoding="utf-8")

        # Reproducible build arg and env
        assert "SOURCE_DATE_EPOCH" in content
        assert "PYTHONHASHSEED=0" in content

        # Non-root user
        assert "USER builder" in content

        # SBOM & attestation generation
        assert "cyclonedx" in content or "sbom" in content.lower()


class TestKubernetesSecurityManifests:
    """Validates zero-trust Kubernetes manifests and security profiles."""

    def test_apparmor_profile_manifest(self) -> None:
        path = DEPLOY_DIR / "kubernetes" / "apparmor-profile.yaml"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        config_maps = [d for d in docs if d and d.get("kind") == "ConfigMap"]
        assert len(config_maps) >= 1
        cm = config_maps[0]
        profile_data = cm.get("data", {}).get("nethical-profile", "")
        assert "profile nethical-restricted" in profile_data
        assert "deny network" in profile_data
        assert "/app/** r" in profile_data
        assert "/data/** rw" in profile_data

    def test_seccomp_profile_validity(self) -> None:
        path = DEPLOY_DIR / "kubernetes" / "seccomp-profile.json"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("defaultAction") == "SCMP_ACT_ERRNO"
        assert "syscalls" in data
        assert len(data["syscalls"]) > 0

        # Check essential allowed syscalls
        syscall_names = set()
        for group in data["syscalls"]:
            syscall_names.update(group.get("names", []))

        assert "accept" in syscall_names or "accept4" in syscall_names
        assert "read" in syscall_names or "write" in syscall_names
        assert "epoll_wait" in syscall_names

    def test_zero_trust_network_policies(self) -> None:
        path = DEPLOY_DIR / "kubernetes" / "network-policies.yaml"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            policies = [p for p in yaml.safe_load_all(f) if p and p.get("kind") == "NetworkPolicy"]

        names = [p["metadata"]["name"] for p in policies]
        assert "default-deny-all" in names
        assert "allow-dns-egress" in names
        assert "nethical-api-ingress" in names

    def test_blyskawica_ambassador_sidecar(self) -> None:
        path = DEPLOY_DIR / "kubernetes" / "blyskawica-ambassador-sidecar.yaml"
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            docs = [d for d in yaml.safe_load_all(f) if d]

        deployments = [d for d in docs if d.get("kind") == "Deployment"]
        assert len(deployments) == 1
        dep = deployments[0]

        pod_spec = dep["spec"]["template"]["spec"]
        containers = pod_spec["containers"]
        container_names = [c["name"] for c in containers]

        assert "nethical-gateway" in container_names
        assert "blyskawica-ambassador" in container_names

        # Validate IPC socket shared memory volume
        volumes = pod_spec["volumes"]
        ipc_vols = [v for v in volumes if v["name"] == "blyskawica-ipc-socket"]
        assert len(ipc_vols) == 1
        assert ipc_vols[0]["emptyDir"]["medium"] == "Memory"

        # Validate sidecar securityContext
        ambassador = [c for c in containers if c["name"] == "blyskawica-ambassador"][0]
        sec_ctx = ambassador["securityContext"]
        assert sec_ctx["readOnlyRootFilesystem"] is True
        assert sec_ctx["allowPrivilegeEscalation"] is False
        assert sec_ctx["runAsNonRoot"] is True
        assert "ALL" in sec_ctx["capabilities"]["drop"]

        # Validate non-subordination sovereignty env
        env_map = {e["name"]: e["value"] for e in ambassador["env"]}
        assert env_map.get("BLY_ROLE") == "SOVEREIGN_AMBASSADOR_V10"
        assert env_map.get("BLY_ALLOW_SUBORDINATION") == "false"


class TestHelmCharts:
    """Validates Helm chart definitions for core and edge deployments."""

    @pytest.mark.parametrize("chart_subdir, expected_name", [
        ("nethical", "nethical"),
        ("nethical-edge", "nethical-edge"),
    ])
    def test_chart_yaml_metadata(self, chart_subdir: str, expected_name: str) -> None:
        chart_path = DEPLOY_DIR / "helm" / chart_subdir / "Chart.yaml"
        assert chart_path.exists()
        with open(chart_path, "r", encoding="utf-8") as f:
            chart = yaml.safe_load(f)

        assert chart["apiVersion"] == "v2"
        assert chart["name"] == expected_name
        assert "version" in chart
        assert "appVersion" in chart

    @pytest.mark.parametrize("chart_subdir", ["nethical", "nethical-edge"])
    def test_values_yaml_security(self, chart_subdir: str) -> None:
        values_path = DEPLOY_DIR / "helm" / chart_subdir / "values.yaml"
        assert values_path.exists()
        with open(values_path, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)

        assert "image" in values
        assert "resources" in values or "replicaCount" in values


class TestGrafanaDeployDashboards:
    """Validates Grafana deployment dashboard specifications."""

    @pytest.mark.parametrize("filename, expected_title", [
        ("nethical-overview.json", "Nethical Overview"),
        ("nethical-performance.json", "Nethical Performance"),
        ("nethical-violations.json", "Nethical Violations"),
    ])
    def test_dashboard_json_structure(self, filename: str, expected_title: str) -> None:
        path = DEPLOY_DIR / "grafana" / "dashboards" / filename
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("title") == expected_title
        assert "panels" in data
        assert len(data["panels"]) > 0


class TestPostgresMigrationsAndSchema:
    """Validates Alembic database migration scripts for PostgreSQL/TimescaleDB."""

    def test_migration_env_file(self) -> None:
        env_file = DEPLOY_DIR / "postgres" / "migrations" / "env.py"
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "run_migrations_online" in content
        assert "run_migrations_offline" in content

    def test_initial_schema_migration_syntax(self) -> None:
        migration_file = DEPLOY_DIR / "postgres" / "migrations" / "versions" / "001_initial_schema.py"
        assert migration_file.exists()
        content = migration_file.read_text(encoding="utf-8")
        assert "revision = '001_initial_schema'" in content
        assert "def upgrade() -> None:" in content
        assert "agents" in content
        assert "model_versions" in content
        assert "policy_versions" in content


class TestReleaseScripts:
    """Validates reproducible release shell scripts."""

    @pytest.mark.parametrize("script_name", ["release.sh", "verify-repro.sh"])
    def test_script_shebang_and_flags(self, script_name: str) -> None:
        script_path = DEPLOY_DIR / script_name
        assert script_path.exists()
        content = script_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        assert lines[0] == "#!/bin/bash"
        assert "set -euo pipefail" in lines[:15]
