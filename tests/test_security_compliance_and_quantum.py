# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit tests for sovereign data compliance, residency mapping, and quantum cryptography."""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from nethical.security.data_compliance import (
    DataRegion,
    DataCategory,
    ProcessingPurpose,
    RequestType,
    RequestStatus,
    DataStore,
    DataFlow,
    DataSubjectRequest,
    DataResidencyMapper,
    DataSubjectRequestHandler,
)
from nethical.security.quantum_crypto import (
    PQCAlgorithm,
    SecurityLevel,
    QuantumThreatLevel,
    HybridMode,
    CRYSTALSKyber,
    CRYSTALSDilithium,
    HybridTLSManager,
    QuantumCryptoManager,
    QuantumThreatAnalyzer,
    PQCMigrationPlanner,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests and cleanup afterwards."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


def test_data_residency_mapper_persistence_and_cross_border(temp_dir):
    """Test DataResidencyMapper registration, persistence, and cross-border detection."""
    mapper1 = DataResidencyMapper(storage_dir=str(temp_dir))

    # Register EU and US stores
    store_eu = mapper1.register_data_store(
        store_id="store-eu-1",
        name="EU Primary Store",
        region=DataRegion.EU,
        data_categories={DataCategory.PII, DataCategory.FINANCIAL},
        retention_days=365,
    )
    store_us = mapper1.register_data_store(
        store_id="store-us-1",
        name="US Analytics Store",
        region=DataRegion.US,
        data_categories={DataCategory.TELEMETRY},
        retention_days=90,
    )
    assert store_eu.region == DataRegion.EU

    # Register intra-region flow
    flow_internal = mapper1.register_data_flow(
        flow_id="flow-eu-internal",
        source="store-eu-1",
        destination="store-eu-1",
        data_categories={DataCategory.FINANCIAL},
        purpose=ProcessingPurpose.CONTRACT,
    )
    assert flow_internal.cross_border is False

    # Register cross-border flow
    flow_cross = mapper1.register_data_flow(
        flow_id="flow-eu-us",
        source="store-eu-1",
        destination="store-us-1",
        data_categories={DataCategory.PII},
        purpose=ProcessingPurpose.CONSENT,
    )
    assert flow_cross.cross_border is True

    # Generate diagram
    diagram_path = temp_dir / "diagram.json"
    diagram = mapper1.generate_data_flow_diagram(output_file=str(diagram_path))
    assert diagram["metadata"]["total_stores"] == 2
    assert diagram["metadata"]["cross_border_flows"] == 1
    assert diagram_path.exists()

    # Re-initialise mapper from same storage directory and verify persistence
    mapper2 = DataResidencyMapper(storage_dir=str(temp_dir))
    assert len(mapper2._stores) == 2
    assert len(mapper2._flows) == 2
    assert "store-eu-1" in mapper2._stores
    assert len(mapper2.get_cross_border_flows()) == 1


def test_data_subject_request_handler(temp_dir):
    """Test DataSubjectRequestHandler submission, workflows, and persistence."""
    mapper = DataResidencyMapper(storage_dir=str(temp_dir / "mapper"))
    handler1 = DataSubjectRequestHandler(
        storage_dir=str(temp_dir / "requests"),
        residency_mapper=mapper,
        sla_hours=72,
    )

    # 1. Submit access request
    req = handler1.submit_request(
        request_type=RequestType.ACCESS,
        subject_id="user-999",
        data_categories=[DataCategory.PII],
        verification_method="oidc",
        notes="Audit request",
    )
    assert req.status == RequestStatus.PENDING

    # 2. Process access request
    access_result = handler1.process_access_request(req.request_id)
    assert access_result["subject_id"] == "user-999"
    assert req.status == RequestStatus.COMPLETED

    # 3. Test workflow method
    test_report = handler1.test_workflow()
    assert test_report["access_request"] is True
    assert test_report["deletion_request"] is True

    # 4. Verify request persistence on new instance
    handler2 = DataSubjectRequestHandler(
        storage_dir=str(temp_dir / "requests"),
        residency_mapper=mapper,
        sla_hours=72,
    )
    assert req.request_id in handler2._requests
    loaded_req = handler2._requests[req.request_id]
    assert loaded_req.subject_id == "user-999"
    assert loaded_req.status == RequestStatus.COMPLETED


def test_pqc_migration_planner():
    """Test PQC migration planning roadmap and sequential phase transitions."""
    planner = PQCMigrationPlanner(organization_name="Sovereign AI Corp")
    assert len(planner.phases) == 5
    assert planner.phases[0].status == "pending"

    # Start migration
    init_status = planner.start_migration()
    assert planner.phases[0].status == "in_progress"
    assert init_status["current_phase"]["phase_number"] == 1

    # Complete phase 1 -> triggers phase 2
    success = planner.complete_phase(1)
    assert success is True
    assert planner.phases[0].status == "completed"
    assert planner.phases[0].completion_date is not None
    assert planner.phases[1].status == "in_progress"
    assert planner.phases[1].start_date is not None

    status = planner.get_migration_status()
    assert status["completed_phases"] == 1
    assert status["total_phases"] == 5
    assert status["progress_percentage"] == 20.0

    roadmap = planner.export_roadmap()
    assert roadmap["organization"] == "Sovereign AI Corp"
    assert len(roadmap["phases"]) == 5


def test_crystals_kyber_kem():
    """Test CRYSTALS-Kyber keypair generation, encapsulation, and decapsulation."""
    kyber = CRYSTALSKyber()
    keypair = kyber.generate_keypair(PQCAlgorithm.KYBER_768)

    assert keypair.algorithm == PQCAlgorithm.KYBER_768
    assert keypair.security_level == SecurityLevel.LEVEL_3
    assert len(keypair.public_key) > 0
    assert len(keypair.private_key) > 0

    # Encapsulate
    encap = kyber.encapsulate(keypair.public_key, PQCAlgorithm.KYBER_768)
    assert len(encap.ciphertext) > 0
    assert len(encap.shared_secret) == 32

    # Decapsulate
    decapped_secret = kyber.decapsulate(
        encap.ciphertext, keypair.private_key, PQCAlgorithm.KYBER_768
    )
    assert decapped_secret == encap.shared_secret


def test_crystals_dilithium_signatures():
    """Test CRYSTALS-Dilithium signing, verification, and tampering detection."""
    dilithium = CRYSTALSDilithium()
    keypair = dilithium.generate_keypair(PQCAlgorithm.DILITHIUM_3)

    message = b"Sovereign AI Governance Policy Statement v1.0"
    sig = dilithium.sign(message, keypair.private_key, PQCAlgorithm.DILITHIUM_3)

    assert sig.algorithm == PQCAlgorithm.DILITHIUM_3
    assert len(sig.signature) > 0

    # Verify signature
    valid = dilithium.verify(message, sig, keypair.public_key)
    assert valid is True

    # Tampered message must fail
    tampered_msg = b"Tampered AI Policy Statement"
    invalid = dilithium.verify(tampered_msg, sig, keypair.public_key)
    assert invalid is False


def test_hybrid_tls_manager():
    """Test hybrid classical and post-quantum TLS handshake and key derivation."""
    tls_manager = HybridTLSManager(
        hybrid_mode=HybridMode.HYBRID_KDF,
        pqc_algorithm=PQCAlgorithm.KYBER_768,
    )
    kyber = CRYSTALSKyber()
    keypair = kyber.generate_keypair(PQCAlgorithm.KYBER_768)

    peer_classical = b"X25519_PUBLIC_KEY_SIMULATED_32B!"
    handshake = tls_manager.perform_hybrid_handshake(
        peer_public_key_classical=peer_classical,
        peer_public_key_quantum=keypair.public_key,
    )

    assert handshake["success"] is True
    assert handshake["hybrid_mode"] == HybridMode.HYBRID_KDF.value
    assert handshake["quantum_used"] is True
    assert handshake["classical_used"] is True
    assert len(handshake["combined_key"]) == 32
    assert handshake["ciphertext"] is not None

    stats = tls_manager.get_statistics()
    assert stats["handshake_count"] == 1
    assert stats["hybrid_success_count"] == 1


def test_quantum_crypto_manager_and_threat_assessment():
    """Test QuantumCryptoManager integrated security status and compliance export."""
    manager = QuantumCryptoManager(organization_name="Nethical Security")
    assessment = manager.assess_threat()
    assert assessment.threat_level is not None

    status = manager.get_security_status()

    assert status["organization"] == "Nethical Security"
    assert status["kyber"]["enabled"] is True
    assert status["dilithium"]["enabled"] is True
    assert status["hybrid_tls"]["enabled"] is True

    # Threat assessment
    assert status["threat_assessment"]["total_assessments"] >= 1

    # Export compliance
    report = manager.export_compliance_report()
    assert report["organization"] == "Nethical Security"
    assert "nist_compliance" in report
    assert "NIST FIPS 203 (ML-KEM/Kyber)" in report["nist_compliance"]["pqc_standards"]
    assert "NIST FIPS 204 (ML-DSA/Dilithium)" in report["nist_compliance"]["pqc_standards"]
    assert report["timestamp"] is not None
