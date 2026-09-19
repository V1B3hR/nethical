"""Tests for Kubernetes and Docker Compose Sovereign Deployment Specs.

Validates IPC socket sharing, volume definitions, environment variables,
and health probes for Nethical Gateway and Błyskawica Ambassador Sidecar.
"""

from pathlib import Path
import yaml
import pytest


REPO_ROOT = Path(__file__).parent.parent
K8S_MANIFEST_PATH = REPO_ROOT / "deploy" / "kubernetes" / "blyskawica-ambassador-sidecar.yaml"
DOCKER_COMPOSE_PATH = REPO_ROOT / "docker-compose.sovereign.yml"


def test_k8s_manifest_structure_and_ipc() -> None:
    """Ensure Kubernetes sidecar manifest properly defines shared memory IPC and socket configuration."""
    assert K8S_MANIFEST_PATH.exists(), f"K8s manifest missing at {K8S_MANIFEST_PATH}"

    with open(K8S_MANIFEST_PATH, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    assert len(docs) >= 2, "Expected at least 2 manifests (Deployment and Service)"

    deployment = next((d for d in docs if d.get("kind") == "Deployment"), None)
    service = next((d for d in docs if d.get("kind") == "Service"), None)

    assert deployment is not None, "Deployment manifest not found"
    assert service is not None, "Service manifest not found"

    # Validate Deployment metadata
    assert deployment["metadata"]["name"] == "nethical-sovereign-node"
    assert deployment["metadata"]["namespace"] == "nethical-governance"

    pod_spec = deployment["spec"]["template"]["spec"]
    volumes = {v["name"]: v for v in pod_spec.get("volumes", [])}

    # Validate IPC Volume (Shared memory)
    assert "blyskawica-ipc-socket" in volumes, "blyskawica-ipc-socket volume is missing"
    assert volumes["blyskawica-ipc-socket"].get("emptyDir", {}).get("medium") == "Memory", \
        "IPC socket volume must use in-memory emptyDir for microsecond latency"

    containers = {c["name"]: c for c in pod_spec.get("containers", [])}
    assert "nethical-gateway" in containers, "nethical-gateway container missing"
    assert "blyskawica-ambassador" in containers, "blyskawica-ambassador container missing"

    # Validate Volume Mounts
    gw_mounts = {m["name"]: m["mountPath"] for m in containers["nethical-gateway"].get("volumeMounts", [])}
    amb_mounts = {m["name"]: m["mountPath"] for m in containers["blyskawica-ambassador"].get("volumeMounts", [])}

    assert gw_mounts.get("blyskawica-ipc-socket") == "/var/run/blyskawica", \
        "Gateway must mount IPC socket at /var/run/blyskawica"
    assert amb_mounts.get("blyskawica-ipc-socket") == "/var/run/blyskawica", \
        "Ambassador must mount IPC socket at /var/run/blyskawica"

    # Validate Environment Variables
    gw_env = {e["name"]: e["value"] for e in containers["nethical-gateway"].get("env", [])}
    amb_env = {e["name"]: e["value"] for e in containers["blyskawica-ambassador"].get("env", [])}

    assert gw_env.get("BLY_IPC_SOCKET_PATH") == "/var/run/blyskawica/ambassador.sock", \
        "Gateway BLY_IPC_SOCKET_PATH must point to shared socket"
    assert amb_env.get("BLY_AMBASSADOR_SOCKET") == "/var/run/blyskawica/ambassador.sock", \
        "Ambassador BLY_AMBASSADOR_SOCKET must match socket path"

    # Validate Ambassador Probes
    amb_liveness = containers["blyskawica-ambassador"].get("livenessProbe", {})
    amb_readiness = containers["blyskawica-ambassador"].get("readinessProbe", {})

    assert "exec" in amb_liveness, "Ambassador should have exec-based socket liveness probe"
    assert "test -S /var/run/blyskawica/ambassador.sock" in " ".join(amb_liveness["exec"]["command"])
    assert "exec" in amb_readiness, "Ambassador should have exec-based socket readiness probe"
    assert "test -S /var/run/blyskawica/ambassador.sock" in " ".join(amb_readiness["exec"]["command"])

    # Validate Service Selector and Ports
    svc_spec = service["spec"]
    assert svc_spec["selector"].get("app") == "nethical-sovereign-node"
    port_names = {p["name"]: p["port"] for p in svc_spec.get("ports", [])}
    assert "http-api" in port_names and port_names["http-api"] == 8000
    assert "metrics" in port_names and port_names["metrics"] == 9090


def test_docker_compose_sovereign_spec() -> None:
    """Ensure Docker Compose sovereign file correctly maps IPC volume and socket environment."""
    assert DOCKER_COMPOSE_PATH.exists(), f"Docker compose missing at {DOCKER_COMPOSE_PATH}"

    with open(DOCKER_COMPOSE_PATH, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "blyskawica-ambassador" in services, "blyskawica-ambassador service missing"
    assert "nethical-gateway" in services, "nethical-gateway service missing"

    volumes = compose_data.get("volumes", {})
    assert "blyskawica_ipc" in volumes, "Shared blyskawica_ipc volume missing"

    gw = services["nethical-gateway"]
    amb = services["blyskawica-ambassador"]

    # Check volume mounts
    assert any("blyskawica_ipc:/var/run/blyskawica" in v for v in gw.get("volumes", [])), \
        "Gateway must mount blyskawica_ipc to /var/run/blyskawica"
    assert any("blyskawica_ipc:/var/run/blyskawica" in v for v in amb.get("volumes", [])), \
        "Ambassador must mount blyskawica_ipc to /var/run/blyskawica"

    # Check env vars
    gw_env = {}
    for item in gw.get("environment", []):
        k, v = item.split("=", 1)
        gw_env[k] = v

    amb_env = {}
    for item in amb.get("environment", []):
        k, v = item.split("=", 1)
        amb_env[k] = v

    assert gw_env.get("BLY_IPC_SOCKET_PATH") == "/var/run/blyskawica/ambassador.sock"
    assert amb_env.get("BLY_AMBASSADOR_SOCKET") == "/var/run/blyskawica/ambassador.sock"
