"""
Unit tests for Phase 6.2: Quantum-Resistant Cryptography Framework
"""

import pytest
from datetime import datetime, timedelta

from nethical.security.quantum_crypto import (
    CRYSTALSKyber,
    CRYSTALSDilithium,
    HybridTLSManager,
    QuantumThreatAnalyzer,
    PQCMigrationPlanner,
    QuantumCryptoManager,
    PQCAlgorithm,
    SecurityLevel,
    QuantumThreatLevel,
    HybridMode,
    KyberKeyPair,
    DilithiumKeyPair
)


class TestCRYSTALSKyber:
    """Test CRYSTALS-Kyber key encapsulation."""
    
    def test_initialization_kyber512(self):
        """Test Kyber-512 initialization."""
        kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_512)
        
        assert kyber.algorithm == PQCAlgorithm.KYBER_512
        assert kyber.params['security_level'] == SecurityLevel.LEVEL_1
        assert kyber.params['public_key_size'] == 800
    
    def test_initialization_kyber768(self):
        """Test Kyber-768 initialization (recommended)."""
        kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_768)
        
        assert kyber.algorithm == PQCAlgorithm.KYBER_768
        assert kyber.params['security_level'] == SecurityLevel.LEVEL_3
        assert kyber.params['public_key_size'] == 1184
    
    def test_initialization_kyber1024(self):
        """Test Kyber-1024 initialization."""
        kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_1024)
        
        assert kyber.algorithm == PQCAlgorithm.KYBER_1024
        assert kyber.params['security_level'] == SecurityLevel.LEVEL_5
    
    def test_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError):
            CRYSTALSKyber(algorithm=PQCAlgorithm.DILITHIUM_2)
    
    def test_generate_keypair(self):
        """Test Kyber keypair generation."""
        kyber = CRYSTALSKyber()
        
        keypair = kyber.generate_keypair()
        
        assert isinstance(keypair, KyberKeyPair)
        assert len(keypair.public_key) > 0
        assert len(keypair.private_key) > 0
        assert keypair.algorithm == PQCAlgorithm.KYBER_768
        assert len(keypair.key_id) > 0
    
    def test_keypair_caching(self):
        """Test keypair caching."""
        kyber = CRYSTALSKyber(enable_key_caching=True)
        
        keypair1 = kyber.generate_keypair()
        keypair2 = kyber.generate_keypair()
        
        assert len(kyber.key_cache) == 2
        assert keypair1.key_id in kyber.key_cache
        assert keypair2.key_id in kyber.key_cache
    
    def test_encapsulate(self):
        """Test key encapsulation."""
        kyber = CRYSTALSKyber()
        keypair = kyber.generate_keypair()
        
        encapsulated = kyber.encapsulate(keypair.public_key)
        
        assert len(encapsulated.ciphertext) > 0
        assert len(encapsulated.shared_secret) == 32
        assert encapsulated.algorithm == PQCAlgorithm.KYBER_768
    
    def test_decapsulate(self):
        """Test key decapsulation."""
        kyber = CRYSTALSKyber()
        keypair = kyber.generate_keypair()
        
        encapsulated = kyber.encapsulate(keypair.public_key)
        shared_secret = kyber.decapsulate(
            encapsulated.ciphertext,
            keypair.private_key
        )
        
        assert len(shared_secret) > 0
        assert isinstance(shared_secret, bytes)
    
    def test_key_sizes(self):
        """Test that key sizes match algorithm parameters."""
        for algorithm in [PQCAlgorithm.KYBER_512, PQCAlgorithm.KYBER_768, PQCAlgorithm.KYBER_1024]:
            kyber = CRYSTALSKyber(algorithm=algorithm)
            keypair = kyber.generate_keypair()
            
            assert len(keypair.public_key) == kyber.params['public_key_size']
            assert len(keypair.private_key) == kyber.params['private_key_size']
    
    def test_statistics(self):
        """Test Kyber statistics."""
        kyber = CRYSTALSKyber()
        keypair = kyber.generate_keypair()
        
        for _ in range(5):
            kyber.encapsulate(keypair.public_key)
        
        stats = kyber.get_statistics()
        
        assert stats['encapsulation_count'] == 5
        assert stats['algorithm'] == PQCAlgorithm.KYBER_768.value
        assert 'parameters' in stats


class TestCRYSTALSDilithium:
    """Test CRYSTALS-Dilithium signatures."""
    
    def test_initialization_dilithium2(self):
        """Test Dilithium2 initialization."""
        dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_2)
        
        assert dilithium.algorithm == PQCAlgorithm.DILITHIUM_2
        assert dilithium.params['security_level'] == SecurityLevel.LEVEL_2
    
    def test_initialization_dilithium3(self):
        """Test Dilithium3 initialization (recommended)."""
        dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        
        assert dilithium.algorithm == PQCAlgorithm.DILITHIUM_3
        assert dilithium.params['security_level'] == SecurityLevel.LEVEL_3
    
    def test_initialization_dilithium5(self):
        """Test Dilithium5 initialization."""
        dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_5)
        
        assert dilithium.algorithm == PQCAlgorithm.DILITHIUM_5
        assert dilithium.params['security_level'] == SecurityLevel.LEVEL_5
    
    def test_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError):
            CRYSTALSDilithium(algorithm=PQCAlgorithm.KYBER_512)
    
    def test_generate_keypair(self):
        """Test Dilithium keypair generation."""
        dilithium = CRYSTALSDilithium()
        
        keypair = dilithium.generate_keypair()
        
        assert isinstance(keypair, DilithiumKeyPair)
        assert len(keypair.public_key) > 0
        assert len(keypair.private_key) > 0
        assert keypair.algorithm == PQCAlgorithm.DILITHIUM_3
    
    def test_sign_message(self):
        """Test message signing."""
        dilithium = CRYSTALSDilithium()
        keypair = dilithium.generate_keypair()
        
        message = b"Top secret military communication"
        signature = dilithium.sign(message, keypair.private_key, keypair.key_id)
        
        assert len(signature.signature) > 0
        assert signature.algorithm == PQCAlgorithm.DILITHIUM_3
        assert signature.signer_key_id == keypair.key_id
        assert len(signature.message_hash) > 0
    
    def test_verify_signature(self):
        """Test signature verification."""
        dilithium = CRYSTALSDilithium()
        keypair = dilithium.generate_keypair()
        
        message = b"Test message"
        signature = dilithium.sign(message, keypair.private_key, keypair.key_id)
        
        is_valid = dilithium.verify(message, signature, keypair.public_key)
        
        assert is_valid is True
    
    def test_verify_wrong_message(self):
        """Test signature verification with wrong message."""
        dilithium = CRYSTALSDilithium()
        keypair = dilithium.generate_keypair()
        
        message = b"Original message"
        signature = dilithium.sign(message, keypair.private_key, keypair.key_id)
        
        wrong_message = b"Modified message"
        is_valid = dilithium.verify(wrong_message, signature, keypair.public_key)
        
        assert is_valid is False
    
    def test_signature_sizes(self):
        """Test that signature sizes match algorithm parameters."""
        for algorithm in [PQCAlgorithm.DILITHIUM_2, PQCAlgorithm.DILITHIUM_3, PQCAlgorithm.DILITHIUM_5]:
            dilithium = CRYSTALSDilithium(algorithm=algorithm)
            keypair = dilithium.generate_keypair()
            
            message = b"Test"
            signature = dilithium.sign(message, keypair.private_key, keypair.key_id)
            
            assert len(signature.signature) == dilithium.params['signature_size']
    
    def test_statistics(self):
        """Test Dilithium statistics."""
        dilithium = CRYSTALSDilithium()
        keypair = dilithium.generate_keypair()
        
        message = b"Test"
        for _ in range(3):
            signature = dilithium.sign(message, keypair.private_key, keypair.key_id)
            dilithium.verify(message, signature, keypair.public_key)
        
        stats = dilithium.get_statistics()
        
        assert stats['signature_count'] == 3
        assert stats['verification_count'] == 3
        assert 'algorithm' in stats


class TestHybridTLSManager:
    """Test hybrid TLS implementation."""
    
    def test_initialization(self):
        """Test hybrid TLS manager initialization."""
        manager = HybridTLSManager(
            hybrid_mode=HybridMode.HYBRID_KDF,
            pqc_algorithm=PQCAlgorithm.KYBER_768
        )
        
        assert manager.hybrid_mode == HybridMode.HYBRID_KDF
        assert manager.pqc_algorithm == PQCAlgorithm.KYBER_768
        assert manager.kyber is not None
        assert manager.dilithium is not None
    
    def test_hybrid_handshake_full(self):
        """Test hybrid handshake with both classical and quantum keys."""
        manager = HybridTLSManager()
        
        # Generate quantum key
        keypair = manager.kyber.generate_keypair()
        
        result = manager.perform_hybrid_handshake(
            peer_public_key_classical=b"classical_key_data",
            peer_public_key_quantum=keypair.public_key
        )
        
        assert result['success'] is True
        assert result['quantum_used'] is True
        assert result['classical_used'] is True
        assert result['combined_key'] is not None
    
    def test_hybrid_handshake_quantum_only(self):
        """Test handshake with quantum key only."""
        manager = HybridTLSManager()
        keypair = manager.kyber.generate_keypair()
        
        result = manager.perform_hybrid_handshake(
            peer_public_key_quantum=keypair.public_key
        )
        
        assert result['success'] is True
        assert result['quantum_used'] is True
        assert result['classical_used'] is False
    
    def test_hybrid_handshake_classical_fallback(self):
        """Test fallback to classical crypto."""
        manager = HybridTLSManager(enable_classical_fallback=True)
        
        result = manager.perform_hybrid_handshake(
            peer_public_key_classical=b"classical_key"
        )
        
        assert result['success'] is True
        assert result['classical_used'] is True
    
    def test_hybrid_mode_concatenate(self):
        """Test concatenate hybrid mode."""
        manager = HybridTLSManager(hybrid_mode=HybridMode.HYBRID_CONCATENATE)
        
        classical = b"classical_secret"
        quantum = b"quantum_secret"
        
        combined = manager._combine_secrets(classical, quantum)
        
        assert combined == classical + quantum
    
    def test_hybrid_mode_xor(self):
        """Test XOR hybrid mode."""
        manager = HybridTLSManager(hybrid_mode=HybridMode.HYBRID_XOR)
        
        classical = b"\x01\x02\x03"
        quantum = b"\x04\x05\x06"
        
        combined = manager._combine_secrets(classical, quantum)
        
        assert len(combined) == 3
        assert combined[0] == 0x01 ^ 0x04
    
    def test_hybrid_mode_kdf(self):
        """Test KDF hybrid mode (recommended)."""
        manager = HybridTLSManager(hybrid_mode=HybridMode.HYBRID_KDF)
        
        classical = b"classical_secret"
        quantum = b"quantum_secret"
        
        combined = manager._combine_secrets(classical, quantum)
        
        assert len(combined) == 32  # SHA-256 output
        assert combined != classical
        assert combined != quantum
    
    def test_statistics(self):
        """Test hybrid TLS statistics."""
        manager = HybridTLSManager()
        keypair = manager.kyber.generate_keypair()
        
        for _ in range(5):
            manager.perform_hybrid_handshake(
                peer_public_key_quantum=keypair.public_key
            )
        
        stats = manager.get_statistics()
        
        assert stats['handshake_count'] == 5
        assert stats['hybrid_success_count'] > 0
        assert 'hybrid_rate' in stats


class TestQuantumThreatAnalyzer:
    """Test quantum threat assessment."""
    
    def test_initialization(self):
        """Test threat analyzer initialization."""
        analyzer = QuantumThreatAnalyzer(
            current_qubit_count=1000,
            error_correction_progress=0.3
        )
        
        assert analyzer.current_qubit_count == 1000
        assert analyzer.error_correction_progress == 0.3
    
    def test_assess_quantum_threat_critical(self):
        """Test critical threat assessment."""
        analyzer = QuantumThreatAnalyzer(
            current_qubit_count=3000,  # Close to breaking RSA
            error_correction_progress=0.8
        )
        
        assessment = analyzer.assess_quantum_threat(
            cryptographic_inventory=['RSA-2048', 'ECDH-256'],
            data_lifetime_years=20.0,
            criticality_level='critical'
        )
        
        assert assessment.threat_level in [
            QuantumThreatLevel.CRITICAL,
            QuantumThreatLevel.HIGH
        ]
        assert assessment.estimated_years_to_threat < 10
        assert len(assessment.recommended_algorithms) > 0
    
    def test_assess_quantum_threat_moderate(self):
        """Test moderate threat assessment."""
        analyzer = QuantumThreatAnalyzer(
            current_qubit_count=500,
            error_correction_progress=0.2
        )
        
        assessment = analyzer.assess_quantum_threat(
            cryptographic_inventory=['RSA-2048'],
            data_lifetime_years=5.0,
            criticality_level='medium'
        )
        
        assert assessment.estimated_years_to_threat > 0
        assert len(assessment.affected_systems) > 0
    
    def test_identify_affected_systems(self):
        """Test identification of vulnerable systems."""
        analyzer = QuantumThreatAnalyzer()
        
        inventory = ['RSA-2048', 'AES-256', 'ECDH-P256', 'SHA-256']
        affected = analyzer._identify_affected_systems(inventory)
        
        assert 'RSA-2048' in affected
        assert 'ECDH-P256' in affected
        assert 'AES-256' not in affected  # Symmetric crypto not affected
    
    def test_recommend_algorithms_high_criticality(self):
        """Test algorithm recommendations for high criticality."""
        analyzer = QuantumThreatAnalyzer()
        
        recommended = analyzer._recommend_algorithms(
            criticality='critical',
            affected_systems=['RSA-2048']
        )
        
        assert PQCAlgorithm.KYBER_1024 in recommended
        assert PQCAlgorithm.DILITHIUM_5 in recommended
    
    def test_recommend_algorithms_medium_criticality(self):
        """Test algorithm recommendations for medium criticality."""
        analyzer = QuantumThreatAnalyzer()
        
        recommended = analyzer._recommend_algorithms(
            criticality='medium',
            affected_systems=['RSA-2048']
        )
        
        assert PQCAlgorithm.KYBER_768 in recommended
        assert PQCAlgorithm.DILITHIUM_3 in recommended
    
    def test_cryptographic_agility_score(self):
        """Test cryptographic agility scoring."""
        analyzer = QuantumThreatAnalyzer()
        
        # Inventory with PQC algorithms
        score_with_pqc = analyzer._calculate_agility_score([
            'RSA-2048', 'Kyber-768', 'Dilithium-3'
        ])
        
        # Inventory without PQC
        score_without_pqc = analyzer._calculate_agility_score([
            'RSA-2048', 'ECDH-256'
        ])
        
        assert score_with_pqc > score_without_pqc
    
    def test_threat_summary(self):
        """Test threat assessment summary."""
        analyzer = QuantumThreatAnalyzer()
        
        for i in range(3):
            analyzer.assess_quantum_threat(
                cryptographic_inventory=['RSA-2048'],
                data_lifetime_years=10.0,
                criticality_level='high'
            )
        
        summary = analyzer.get_threat_summary()
        
        assert summary['total_assessments'] == 3
        assert 'average_years_to_threat' in summary
        assert 'threat_distribution' in summary


class TestPQCMigrationPlanner:
    """Test PQC migration planning."""
    
    def test_initialization(self):
        """Test migration planner initialization."""
        planner = PQCMigrationPlanner(
            organization_name="Test Organization"
        )
        
        assert planner.organization_name == "Test Organization"
        assert len(planner.phases) == 5
    
    def test_migration_phases(self):
        """Test migration phase structure."""
        planner = PQCMigrationPlanner(organization_name="Test")
        
        assert planner.phases[0].phase_name == "Assessment and Inventory"
        assert planner.phases[1].phase_name == "Algorithm Selection and Testing"
        assert planner.phases[2].phase_name == "Hybrid Deployment"
        assert planner.phases[3].phase_name == "Full PQC Migration"
        assert planner.phases[4].phase_name == "Optimization and Maintenance"
    
    def test_start_migration(self):
        """Test migration start."""
        planner = PQCMigrationPlanner(organization_name="Test")
        
        status = planner.start_migration()
        
        assert status['completed_phases'] == 0
        assert status['current_phase'] is not None
        assert status['current_phase']['status'] == 'in_progress'
    
    def test_complete_phase(self):
        """Test phase completion."""
        planner = PQCMigrationPlanner(organization_name="Test")
        planner.start_migration()
        
        success = planner.complete_phase(1)
        
        assert success is True
        assert planner.phases[0].status == 'completed'
        assert planner.phases[1].status == 'in_progress'
    
    def test_complete_invalid_phase(self):
        """Test completion of invalid phase."""
        planner = PQCMigrationPlanner(organization_name="Test")
        
        success = planner.complete_phase(99)
        
        assert success is False
    
    def test_migration_progress(self):
        """Test migration progress calculation."""
        planner = PQCMigrationPlanner(organization_name="Test")
        planner.start_migration()
        
        planner.complete_phase(1)
        planner.complete_phase(2)
        
        status = planner.get_migration_status()
        
        assert status['completed_phases'] == 2
        assert status['progress_percentage'] == 40.0  # 2/5 = 40%
    
    def test_export_roadmap(self):
        """Test roadmap export."""
        planner = PQCMigrationPlanner(organization_name="Test Org")
        
        roadmap = planner.export_roadmap()
        
        assert roadmap['organization'] == "Test Org"
        assert 'total_duration_months' in roadmap
        assert len(roadmap['phases']) == 5
        assert 'key_milestones' in roadmap


class TestQuantumCryptoManager:
    """Test comprehensive quantum crypto manager."""
    
    def test_initialization_all_enabled(self):
        """Test manager with all features enabled."""
        manager = QuantumCryptoManager(
            organization_name="Military Base Alpha",
            enable_kyber=True,
            enable_dilithium=True,
            enable_hybrid_tls=True
        )
        
        assert manager.kyber is not None
        assert manager.dilithium is not None
        assert manager.hybrid_tls is not None
        assert manager.threat_analyzer is not None
        assert manager.migration_planner is not None
    
    def test_initialization_selective(self):
        """Test manager with selective features."""
        manager = QuantumCryptoManager(
            enable_kyber=True,
            enable_dilithium=False,
            enable_hybrid_tls=False
        )
        
        assert manager.kyber is not None
        assert manager.dilithium is None
        assert manager.hybrid_tls is None
    
    def test_security_status(self):
        """Test comprehensive security status."""
        manager = QuantumCryptoManager(organization_name="Test Org")
        
        status = manager.get_security_status()
        
        assert 'organization' in status
        assert 'kyber' in status
        assert 'dilithium' in status
        assert 'hybrid_tls' in status
        assert 'threat_assessment' in status
        assert 'migration_status' in status
    
    def test_compliance_report(self):
        """Test compliance report export."""
        manager = QuantumCryptoManager(organization_name="Government Agency")
        
        report = manager.export_compliance_report()
        
        assert report['organization'] == "Government Agency"
        assert 'quantum_readiness' in report
        assert 'nist_compliance' in report
        assert 'migration_roadmap' in report
        
        # Check NIST compliance
        nist = report['nist_compliance']
        assert 'NIST FIPS 203' in nist['pqc_standards'][0]
        assert 'NIST FIPS 204' in nist['pqc_standards'][1]
    
    def test_algorithm_configuration(self):
        """Test custom algorithm configuration."""
        manager = QuantumCryptoManager(
            kyber_algorithm=PQCAlgorithm.KYBER_1024,
            dilithium_algorithm=PQCAlgorithm.DILITHIUM_5
        )
        
        assert manager.kyber.algorithm == PQCAlgorithm.KYBER_1024
        assert manager.dilithium.algorithm == PQCAlgorithm.DILITHIUM_5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
