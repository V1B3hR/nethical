# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit and integration tests for Sovereign Data Diode, Air-Gapped Node, and Multi-Tenant Gateway Routing."""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.auth.tenant_manager import TenantManager
from nethical.core.models import ClassificationLevel
from nethical.gateway.proxy import GovernanceGateway
from nethical.security.air_gapped_node import AirGappedSovereignNode, SecurityClassification
from nethical.security.data_diode import (
    DataDiodeBridge,
    SovereignPackage,
)
from nethical.security.merkle_ledger import MerkleLedger


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_data_diode_package_creation_and_pqc_verification() -> None:
    """Verifies that DataDiodeBridge creates a valid ML-DSA-65 signed package and verifies it offline."""
    bridge = DataDiodeBridge(node_id="sovereign_bridge_alpha")
    payload = {
        "policy_id": "POL-EU-AI-ACT-HIGH-RISK-2026",
        "rules": ["prohibit_subliminal_manipulation", "enforce_human_oversight"],
        "version": "3.4.0",
    }

    package = bridge.create_package(
        target_tenant_id="gov_pl_cyber",
        payload=payload,
        package_type="POLICY_UPDATE",
        classification=ClassificationLevel.SECRET,
    )

    assert package.header.package_id.startswith("PKG-")
    assert package.header.target_tenant_id == "gov_pl_cyber"
    assert package.header.classification == ClassificationLevel.SECRET.value
    assert len(package.header.sha3_512_digest) == 128  # SHA3-512 hex length
    assert len(package.header.signature_hex) > 0

    # Offline verification with trusted key
    is_valid, msg = bridge.verify_package(package)
    assert is_valid is True
    assert "Sygnatura ML-DSA-65 prawidłowa" in msg


def test_data_diode_anti_tampering_detection() -> None:
    """Verifies that any modification to payload or manifest breaks verification."""
    bridge = DataDiodeBridge(node_id="sovereign_bridge_tamper_test")
    payload = {"critical_threshold": 42}

    package = bridge.create_package(
        target_tenant_id="enterprise_fin_eu",
        payload=payload,
    )

    # 1. Tamper payload
    tampered_package = SovereignPackage(
        header=package.header,
        payload={"critical_threshold": 999},  # Tampered
    )
    is_valid, msg = bridge.verify_package(tampered_package)
    assert is_valid is False
    assert "Naruszenie integralności ładunku" in msg

    # 2. Tamper header classification without updating signature
    tampered_header = package.header.model_copy(update={"classification": "UNCLASSIFIED"})
    tampered_package2 = SovereignPackage(
        header=tampered_header,
        payload=payload,
    )
    is_valid2, msg2 = bridge.verify_package(tampered_package2)
    assert is_valid2 is False
    assert "Naruszenie integralności manifestu" in msg2 or "Błąd weryfikacji podpisu" in msg2


def test_data_diode_anti_replay_protection() -> None:
    """Verifies that replaying the same package nonce is rejected."""
    bridge = DataDiodeBridge(node_id="sovereign_bridge_replay_test")
    ledger = MerkleLedger()
    package = bridge.create_package(
        target_tenant_id="default_tenant",
        payload={"command": "activate_watchdog"},
    )

    # First import: should succeed
    success1, msg1, receipt1 = bridge.import_and_seal(package, ledger)
    assert success1 is True
    assert receipt1 is not None

    # Second import (replay): must fail
    success2, msg2, receipt2 = bridge.import_and_seal(package, ledger)
    assert success2 is False
    assert receipt2 is None
    assert "atak powtórzeniowy" in msg2.lower()


def test_data_diode_import_and_merkle_seal() -> None:
    """Verifies that an imported package is permanently sealed into the tenant's Merkle ledger."""
    bridge = DataDiodeBridge(node_id="sovereign_bridge_seal_test")
    ledger = MerkleLedger()
    init_blocks = ledger.total_blocks

    package = bridge.create_package(
        target_tenant_id="defense_airgap",
        payload={"mission_code": "ORZEŁ_BIAŁY_2026", "zones": ["SECTOR_4", "SECTOR_7"]},
        package_type="DEFENSE_DIRECTIVE",
    )

    success, msg, receipt = bridge.import_and_seal(package, ledger)
    assert success is True
    assert receipt is not None
    assert ledger.total_blocks == init_blocks + 1
    assert ledger.verify_receipt(receipt) is True
    assert ledger.blocks[-1].decision_payload["package_id"] == package.header.package_id


def test_governance_gateway_tenant_routing() -> None:
    """Verifies that GovernanceGateway routes decisions to the isolated tenant ledger."""
    tm = TenantManager()
    gateway = GovernanceGateway(tenant_manager=tm)

    # Tool call targeted at gov_pl_cyber tenant
    decision_gov = gateway.intercept_tool_call(
        agent_id="gov_agent_01",
        tool_name="verify_ksc_compliance",
        arguments={"audit_target": "router_01"},
        context={"tenant_id": "gov_pl_cyber"},
    )

    assert decision_gov.tenant_id == "gov_pl_cyber"
    assert decision_gov.receipt_id is not None

    # Check that it exists in gov_pl_cyber ledger
    gov_ledger = tm.get_tenant_ledger("gov_pl_cyber")
    assert decision_gov.receipt_id in gov_ledger.receipts

    # And check that it DOES NOT exist in enterprise_fin_eu ledger
    fin_ledger = tm.get_tenant_ledger("enterprise_fin_eu")
    assert decision_gov.receipt_id not in fin_ledger.receipts


def test_airgap_sovereign_node_zero_egress() -> None:
    """Verifies that AirGappedSovereignNode blocks unauthorized outbound connections."""
    node = AirGappedSovereignNode(
        node_id="test_bunker_node",
        classification=SecurityClassification.SECRET,
        strict_airgap=True,
    )

    res = node.intercept_network_egress("api.openai.com", 443)
    assert res["allowed"] is False
    assert "strictly forbids outbound traffic" in res["reason"]
    assert node.blocked_egress_attempts == 1

    dossier = node.export_defense_dossier()
    assert dossier.node_id == "test_bunker_node"
    assert dossier.classification == SecurityClassification.SECRET
    assert len(dossier.sha3_512_digest) == 128


def test_airgap_api_full_cycle(client: TestClient) -> None:
    """Tests the complete REST API lifecycle for AirGap and Data Diode synchronization."""
    # 1. GET status
    status_res = client.get("/api/v1/security/airgap/status")
    assert status_res.status_code == 200
    st_data = status_res.json()
    assert "node_status" in st_data
    assert "data_diode" in st_data
    assert st_data["data_diode"]["pqc_algorithm"] == "ML-DSA-65 (CRYSTALS-Dilithium Level 3)"

    # 2. Simulate Egress attempt (should be intercepted and blocked)
    egress_res = client.post(
        "/api/v1/security/airgap/simulate-egress",
        json={"target_host": "telemetry.cloud.external", "port": 443},
    )
    assert egress_res.status_code == 200
    egress_data = egress_res.json()
    assert egress_data["allowed"] is False
    assert "forbids outbound traffic" in egress_data["reason"]

    # 3. Export package
    export_res = client.post(
        "/api/v1/security/airgap/export-package",
        json={
            "target_tenant_id": "gov_pl_cyber",
            "payload": {"rule": "zero_trust_enforcement", "active": True},
            "package_type": "POLICY_UPDATE",
            "classification": "SECRET",
        },
    )
    assert export_res.status_code == 200
    package_data = export_res.json()
    assert "header" in package_data
    assert "payload" in package_data
    assert package_data["header"]["target_tenant_id"] == "gov_pl_cyber"

    # 4. Import package into sovereign tenant
    import_res = client.post(
        "/api/v1/security/airgap/import-package",
        json={"package": package_data},
    )
    assert import_res.status_code == 200
    import_result = import_res.json()
    assert import_result["success"] is True
    assert import_result["target_tenant"] == "gov_pl_cyber"
    assert import_result["receipt"] is not None
    assert "receipt_id" in import_result["receipt"]
