# Quantum-Resistant Cryptography Guide

## Overview

The Quantum-Resistant Cryptography Framework provides post-quantum cryptographic protection against quantum computer attacks. This guide covers NIST-standardized algorithms, hybrid implementations, threat assessment, and migration planning for military, government, and healthcare deployments.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Quick Start](#quick-start)
4. [CRYSTALS-Kyber](#crystals-kyber)
5. [CRYSTALS-Dilithium](#crystals-dilithium)
6. [Hybrid TLS](#hybrid-tls)
7. [Quantum Threat Assessment](#quantum-threat-assessment)
8. [Migration Planning](#migration-planning)
9. [Best Practices](#best-practices)
10. [Compliance](#compliance)

## Architecture

The Quantum-Resistant Cryptography Framework consists of five integrated components:

```
┌────────────────────────────────────────────────────────────┐
│              QuantumCryptoManager                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────┐  ┌──────────────────────────┐      │
│  │  CRYSTALS-Kyber  │  │  CRYSTALS-Dilithium      │      │
│  │  (KEM)           │  │  (Signatures)            │      │
│  └──────────────────┘  └──────────────────────────┘      │
│                                                            │
│  ┌──────────────────┐  ┌──────────────────────────┐      │
│  │  Hybrid TLS      │  │  Quantum Threat          │      │
│  │  Manager         │  │  Analyzer                │      │
│  └──────────────────┘  └──────────────────────────┘      │
│                                                            │
│  ┌──────────────────────────────────────────┐            │
│  │  PQC Migration Planner                   │            │
│  └──────────────────────────────────────────┘            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Components

### 1. CRYSTALS-Kyber (Key Encapsulation)

NIST-standardized lattice-based KEM for secure key exchange:
- **Kyber-512**: NIST Level 1 (≈ AES-128)
- **Kyber-768**: NIST Level 3 (≈ AES-192) - **RECOMMENDED**
- **Kyber-1024**: NIST Level 5 (≈ AES-256)

### 2. CRYSTALS-Dilithium (Digital Signatures)

NIST-standardized lattice-based signatures:
- **Dilithium2**: NIST Level 2
- **Dilithium3**: NIST Level 3 - **RECOMMENDED**
- **Dilithium5**: NIST Level 5

### 3. Hybrid TLS Manager

Combines classical and quantum-resistant algorithms:
- Defense-in-depth approach
- Backward compatibility
- Cryptographic agility

### 4. Quantum Threat Analyzer

Assesses quantum computing threats:
- Timeline estimation
- Risk prioritization
- System inventory
- Recommendation engine

### 5. PQC Migration Planner

Structured migration approach:
- 5-phase roadmap
- Timeline management
- Progress tracking
- Deliverable tracking

## Quick Start

### Basic Setup

```python
from nethical.security.quantum_crypto import QuantumCryptoManager

# Initialize with all features enabled
manager = QuantumCryptoManager(
    organization_name="Military Base Alpha",
    enable_kyber=True,
    enable_dilithium=True,
    enable_hybrid_tls=True
)
```

### Selective Features

```python
# Enable only required features
manager = QuantumCryptoManager(
    enable_kyber=True,
    enable_dilithium=True,
    enable_hybrid_tls=False
)
```

## CRYSTALS-Kyber

### Key Encapsulation Mechanism

Kyber provides quantum-resistant key exchange for TLS, VPNs, and secure communications.

#### Algorithm Selection

```python
from nethical.security.quantum_crypto import (
    CRYSTALSKyber,
    PQCAlgorithm
)

# Recommended: Kyber-768 (NIST Level 3)
kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_768)

# High security: Kyber-1024 (NIST Level 5)
kyber_high = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_1024)

# Constrained devices: Kyber-512 (NIST Level 1)
kyber_lite = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_512)
```

#### Key Generation

```python
# Generate Kyber keypair
keypair = kyber.generate_keypair()

print(f"Key ID: {keypair.key_id}")
print(f"Algorithm: {keypair.algorithm.value}")
print(f"Security Level: {keypair.security_level.value}")
print(f"Public Key Size: {len(keypair.public_key)} bytes")
print(f"Private Key Size: {len(keypair.private_key)} bytes")
```

#### Key Encapsulation

```python
# Sender: Encapsulate shared secret
encapsulated = kyber.encapsulate(recipient_public_key)

print(f"Ciphertext Size: {len(encapsulated.ciphertext)} bytes")
print(f"Shared Secret Size: {len(encapsulated.shared_secret)} bytes")

# Send ciphertext to recipient
send_to_recipient(encapsulated.ciphertext)
```

#### Key Decapsulation

```python
# Recipient: Decapsulate shared secret
shared_secret = kyber.decapsulate(
    ciphertext=received_ciphertext,
    private_key=keypair.private_key
)

# Use shared secret for AES key
aes_key = derive_key(shared_secret)
```

#### Performance Metrics

```python
# Get Kyber statistics
stats = kyber.get_statistics()

print(f"Algorithm: {stats['algorithm']}")
print(f"Security Level: {stats['security_level']}")
print(f"Encapsulations: {stats['encapsulation_count']}")
print(f"Public Key Size: {stats['parameters']['public_key_size']} bytes")
print(f"Ciphertext Size: {stats['parameters']['ciphertext_size']} bytes")
```

**Kyber-768 Performance:**
- Public Key: 1,184 bytes
- Ciphertext: 1,088 bytes
- Shared Secret: 32 bytes
- Key Generation: ~100 μs
- Encapsulation: ~120 μs
- Decapsulation: ~140 μs

## CRYSTALS-Dilithium

### Digital Signatures

Dilithium provides quantum-resistant digital signatures for code signing, document authentication, and certificate authorities.

#### Algorithm Selection

```python
from nethical.security.quantum_crypto import CRYSTALSDilithium

# Recommended: Dilithium3 (NIST Level 3)
dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)

# High security: Dilithium5 (NIST Level 5)
dilithium_high = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_5)

# Balanced: Dilithium2 (NIST Level 2)
dilithium_fast = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_2)
```

#### Key Generation

```python
# Generate Dilithium keypair
keypair = dilithium.generate_keypair()

print(f"Key ID: {keypair.key_id}")
print(f"Algorithm: {keypair.algorithm.value}")
print(f"Public Key Size: {len(keypair.public_key)} bytes")
print(f"Private Key Size: {len(keypair.private_key)} bytes")
```

#### Message Signing

```python
# Sign sensitive document
message = b"Top Secret: Classified Military Communication"

signature = dilithium.sign(
    message=message,
    private_key=keypair.private_key,
    key_id=keypair.key_id
)

print(f"Signature Size: {len(signature.signature)} bytes")
print(f"Message Hash: {signature.message_hash}")
print(f"Signer: {signature.signer_key_id}")
```

#### Signature Verification

```python
# Verify signature
is_valid = dilithium.verify(
    message=message,
    signature=signature,
    public_key=keypair.public_key
)

if is_valid:
    print("✓ Signature verified - Message authentic")
else:
    print("✗ Signature verification failed - Message tampered")
```

#### Performance Metrics

```python
# Get Dilithium statistics
stats = dilithium.get_statistics()

print(f"Signatures Created: {stats['signature_count']}")
print(f"Verifications: {stats['verification_count']}")
```

**Dilithium3 Performance:**
- Public Key: 1,952 bytes
- Private Key: 4,000 bytes
- Signature: 3,293 bytes
- Signing: ~400 μs
- Verification: ~200 μs

## Hybrid TLS

### Combining Classical and Quantum-Resistant Crypto

Hybrid TLS provides defense-in-depth by using both classical (RSA/ECDH) and quantum-resistant (Kyber) algorithms simultaneously.

#### Hybrid Modes

```python
from nethical.security.quantum_crypto import (
    HybridTLSManager,
    HybridMode
)

# KDF mode (recommended) - combines via key derivation
hybrid_kdf = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    pqc_algorithm=PQCAlgorithm.KYBER_768
)

# Concatenate mode - simple concatenation
hybrid_concat = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_CONCATENATE
)

# XOR mode - XOR combination
hybrid_xor = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_XOR
)
```

#### Hybrid Handshake

```python
# Initialize hybrid TLS
manager = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    enable_classical_fallback=True
)

# Generate keys
kyber_keypair = manager.kyber.generate_keypair()
classical_key = generate_classical_keypair()  # RSA/ECDH

# Perform hybrid handshake
result = manager.perform_hybrid_handshake(
    peer_public_key_classical=peer_classical_key,
    peer_public_key_quantum=peer_quantum_key
)

if result['success']:
    print("✓ Hybrid handshake successful")
    print(f"Mode: {result['hybrid_mode']}")
    print(f"Quantum used: {result['quantum_used']}")
    print(f"Classical used: {result['classical_used']}")
    
    # Use combined key for encryption
    session_key = result['combined_key']
else:
    print("✗ Hybrid handshake failed")
```

#### Migration Strategy

**Phase 1: Preparation (Now)**
```python
# Test quantum-resistant algorithms
manager = HybridTLSManager(
    hybrid_mode=HybridMode.CLASSICAL_ONLY
)
```

**Phase 2: Hybrid Deployment (6-12 months)**
```python
# Deploy hybrid classical + quantum
manager = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    enable_classical_fallback=True
)
```

**Phase 3: Quantum-Only (18-24 months)**
```python
# Transition to quantum-only
manager = HybridTLSManager(
    hybrid_mode=HybridMode.QUANTUM_ONLY,
    enable_classical_fallback=False
)
```

## Quantum Threat Assessment

### Analyzing Quantum Computing Risks

```python
from nethical.security.quantum_crypto import QuantumThreatAnalyzer

# Initialize analyzer with current quantum computing status
analyzer = QuantumThreatAnalyzer(
    current_qubit_count=1000,
    error_correction_progress=0.3
)

# Perform threat assessment
assessment = analyzer.assess_quantum_threat(
    cryptographic_inventory=[
        'RSA-2048',
        'ECDH-P256',
        'AES-256',
        'SHA-256'
    ],
    data_lifetime_years=20.0,
    criticality_level='critical'
)

print(f"Threat Level: {assessment.threat_level.value}")
print(f"Years to Threat: {assessment.estimated_years_to_threat:.1f}")
print(f"Cryptographic Agility: {assessment.cryptographic_agility_score:.2%}")
print(f"Migration Urgency: {assessment.migration_urgency}")

print("\nAffected Systems:")
for system in assessment.affected_systems:
    print(f"  - {system}")

print("\nRecommended Algorithms:")
for algo in assessment.recommended_algorithms:
    print(f"  - {algo.value}")
```

### Threat Levels

- **MINIMAL**: >15 years to quantum threat
- **LOW**: 10-15 years
- **MODERATE**: 5-10 years
- **HIGH**: 2-5 years
- **CRITICAL**: <2 years or "harvest now, decrypt later" risk

### Harvest Now, Decrypt Later (HNDL)

If your data lifetime exceeds time to quantum threat, you're at CRITICAL risk:

```python
# Long-lived sensitive data
assessment = analyzer.assess_quantum_threat(
    cryptographic_inventory=['RSA-2048'],
    data_lifetime_years=30.0,  # Medical records, classified docs
    criticality_level='critical'
)

# Will likely return CRITICAL threat level
if assessment.threat_level == QuantumThreatLevel.CRITICAL:
    print("⚠️  IMMEDIATE ACTION REQUIRED")
    print("Adversaries may be harvesting encrypted data now")
    print("to decrypt when quantum computers are available")
```

## Migration Planning

### 5-Phase Migration Roadmap

```python
from nethical.security.quantum_crypto import PQCMigrationPlanner

# Initialize planner
planner = PQCMigrationPlanner(
    organization_name="Department of Defense"
)

# Start migration process
planner.start_migration()

# Check status
status = planner.get_migration_status()
print(f"Organization: {status['organization']}")
print(f"Progress: {status['progress_percentage']:.0f}%")
print(f"Current Phase: {status['current_phase']['phase_name']}")
```

### Migration Phases

**Phase 1: Assessment and Inventory (3 months)**
- Complete cryptographic inventory
- Risk assessment report
- Stakeholder analysis
- Budget and resource allocation

```python
# Complete phase 1
planner.complete_phase(1)
```

**Phase 2: Algorithm Selection and Testing (4 months)**
- Select PQC algorithms
- Performance benchmarks
- Compatibility testing
- Proof of concept

```python
# Complete phase 2
planner.complete_phase(2)
```

**Phase 3: Hybrid Deployment (6 months)**
- Deploy hybrid classical-PQC
- TLS/SSL upgrades
- API and service updates
- Monitoring and alerting

```python
# Complete phase 3
planner.complete_phase(3)
```

**Phase 4: Full PQC Migration (6 months)**
- All systems using PQC
- Classical crypto deprecated
- Security validation
- Compliance certification

```python
# Complete phase 4
planner.complete_phase(4)
```

**Phase 5: Optimization and Maintenance (12 months)**
- Performance optimization
- Continuous monitoring
- Staff training
- Documentation

```python
# Complete phase 5
planner.complete_phase(5)
```

### Export Roadmap

```python
# Export complete roadmap
roadmap = planner.export_roadmap()

print(f"Total Duration: {roadmap['total_duration_months']} months")
print("\nKey Milestones:")
for milestone in roadmap['key_milestones']:
    print(f"  - {milestone}")
```

## Best Practices

### 1. Start with Hybrid Deployment

```python
# Don't switch to quantum-only immediately
manager = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    enable_classical_fallback=True  # Keep fallback during migration
)
```

### 2. Use Recommended Security Levels

```python
# For most organizations: NIST Level 3
kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_768)
dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)

# For critical infrastructure: NIST Level 5
kyber_critical = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_1024)
dilithium_critical = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_5)
```

### 3. Regular Threat Assessments

```python
# Quarterly assessments
def quarterly_assessment():
    analyzer = QuantumThreatAnalyzer(
        current_qubit_count=get_latest_qubit_count(),
        error_correction_progress=get_qec_progress()
    )
    
    assessment = analyzer.assess_quantum_threat(
        cryptographic_inventory=get_crypto_inventory(),
        data_lifetime_years=30.0,
        criticality_level='high'
    )
    
    if assessment.threat_level >= QuantumThreatLevel.HIGH:
        alert_security_team(assessment)
```

### 4. Cryptographic Agility

```python
# Design for easy algorithm swaps
class CryptoProvider:
    def __init__(self):
        # Start with hybrid
        self.manager = HybridTLSManager()
    
    def switch_to_quantum_only(self):
        # Easy transition
        self.manager = HybridTLSManager(
            hybrid_mode=HybridMode.QUANTUM_ONLY
        )
```

### 5. Performance Monitoring

```python
# Monitor performance impact
manager = QuantumCryptoManager()

# Baseline metrics
baseline_latency = measure_handshake_latency()

# With PQC
pqc_latency = measure_pqc_handshake_latency()

overhead = (pqc_latency - baseline_latency) / baseline_latency
print(f"PQC overhead: {overhead:.2%}")

# Typical overhead: 10-30% for hybrid TLS
```

## Compliance

### NIST Post-Quantum Cryptography Standards

```python
# NIST FIPS 203 (ML-KEM / Kyber)
kyber = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_768)

# NIST FIPS 204 (ML-DSA / Dilithium)
dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)

# Export compliance report
manager = QuantumCryptoManager()
report = manager.export_compliance_report()

print("NIST Compliance:")
for standard in report['nist_compliance']['pqc_standards']:
    print(f"  ✓ {standard}")
```

### NSA Suite-B Quantum

Requirements for classified systems:

```python
# Use highest security level (NIST Level 5)
manager = QuantumCryptoManager(
    kyber_algorithm=PQCAlgorithm.KYBER_1024,
    dilithium_algorithm=PQCAlgorithm.DILITHIUM_5
)
```

### CNSA 2.0 (Commercial National Security Algorithm Suite)

```python
# Hybrid mode required initially
manager = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    pqc_algorithm=PQCAlgorithm.KYBER_1024
)
```

## Performance Optimization

### Key Caching

```python
# Enable key caching for better performance
kyber = CRYSTALSKyber(
    algorithm=PQCAlgorithm.KYBER_768,
    enable_key_caching=True
)

# Pre-generate keys
for _ in range(100):
    kyber.generate_keypair()

# Keys cached and ready for fast access
```

### Batch Operations

```python
# Batch signature verifications
signatures_to_verify = [(msg1, sig1), (msg2, sig2), ...]

dilithium = CRYSTALSDilithium()
results = []

for message, signature in signatures_to_verify:
    is_valid = dilithium.verify(message, signature, public_key)
    results.append(is_valid)
```

## Troubleshooting

### Large Key/Signature Sizes

**Problem**: PQC keys and signatures are larger than classical crypto

**Solution**:
```python
# Use appropriate security level for constraints
# Level 1 for IoT/embedded
kyber_lite = CRYSTALSKyber(algorithm=PQCAlgorithm.KYBER_512)

# Compress signatures (if supported by implementation)
# Use batch verification when possible
```

### Performance Overhead

**Problem**: TLS handshakes slower with PQC

**Solution**:
```python
# Use session resumption
# Cache established keys
# Consider hardware acceleration

# Monitor and optimize
stats = manager.get_security_status()
if stats['hybrid_tls']['statistics']['hybrid_rate'] < 0.9:
    optimize_hybrid_handshake()
```

### Compatibility Issues

**Problem**: Legacy systems don't support PQC

**Solution**:
```python
# Use hybrid mode with fallback
manager = HybridTLSManager(
    hybrid_mode=HybridMode.HYBRID_KDF,
    enable_classical_fallback=True  # Fallback for legacy
)
```

## Testing

### Unit Tests

```python
# Test quantum crypto implementation
pytest tests/unit/test_phase6_quantum_crypto.py -v

# Expected: 47 tests passing
```

### Integration Tests

```python
# Test hybrid TLS in realistic scenario
def test_hybrid_tls_integration():
    manager = HybridTLSManager()
    
    # Client side
    client_kyber = manager.kyber.generate_keypair()
    
    # Server side
    result = manager.perform_hybrid_handshake(
        peer_public_key_quantum=client_kyber.public_key
    )
    
    assert result['success']
    assert result['quantum_used']
```

## References

1. **NIST PQC Standards**: https://csrc.nist.gov/projects/post-quantum-cryptography
2. **CRYSTALS-Kyber**: https://pq-crystals.org/kyber/
3. **CRYSTALS-Dilithium**: https://pq-crystals.org/dilithium/
4. **NSA CNSA 2.0**: https://media.defense.gov/2022/Sep/07/2003071834/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS_.PDF
5. **IETF Hybrid Modes**: RFC drafts on hybrid key exchange

## Support

For implementation questions or security concerns:
- Review [SECURITY.md](../../SECURITY.md)
- Open an issue on GitHub
- Contact security team for classified environments
- Consult NIST PQC migration resources
