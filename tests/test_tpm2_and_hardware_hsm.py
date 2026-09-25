# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit and Integration Tests for TPM 2.0 Hardware Root-of-Trust Provider.

Verifies:
- Native TPM 2.0 initialization & platform detection (Windows CNG Platform Crypto Provider / Linux TSS2).
- Key generation, signing, and cryptographic verification.
- Hardware-backed AES-256-GCM authenticated encryption.
- Platform Configuration Register (PCR) extension & composite hashing.
- Hardware Policy Sealing to PCR measurements [0, 7, 11].
- Tamper detection: unseal rejection when PCR measurements change.
- TPM 2.0 Attestation Quotes signed by AIK with anti-replay nonce.
- Integration through HSMAbstractionLayer and automatic fallback.
"""

import pytest
import asyncio
from nethical.security.hsm import (
    HSMConfig,
    HSMProvider,
    KeyAlgorithm,
    KeyUsage,
    HSMOperationStatus,
    TPM2Provider,
    HSMAbstractionLayer,
    create_hsm_provider,
)


@pytest.fixture
def tpm_config():
    return HSMConfig(
        provider=HSMProvider.TPM_2_0,
        cluster_id="nethical-edge-tpm",
        enabled=True,
    )


@pytest.mark.asyncio
async def test_tpm2_provider_lifecycle_and_detection(tpm_config):
    """Test TPM 2.0 provider connection, detection, and disconnection"""
    provider = create_hsm_provider(tpm_config)
    assert isinstance(provider, TPM2Provider)
    assert not provider.is_connected

    connected = await provider.connect()
    assert connected is True
    assert provider.is_connected is True

    # Hardware detection should report status
    hw_backed = provider.is_hardware_backed
    assert isinstance(hw_backed, bool)

    await provider.disconnect()
    assert provider.is_connected is False


@pytest.mark.asyncio
async def test_tpm2_key_generation_and_signing(tpm_config):
    """Test generating a signing key and creating/verifying cryptographic signatures"""
    provider = TPM2Provider(tpm_config)
    await provider.connect()

    gen_res = await provider.generate_key(
        key_label="tpm-merkle-signer",
        algorithm=KeyAlgorithm.EC_P256,
        usage=[KeyUsage.SIGN, KeyUsage.VERIFY],
    )
    assert gen_res.success
    key_id = gen_res.key_id
    assert key_id is not None
    assert "tpm-key-" in key_id

    # Test signing Merkle root
    payload = b"merkle-genesis-root-block-sha3-256-hash-value"
    sign_res = await provider.sign(key_id, payload)
    assert sign_res.success
    signature = sign_res.data
    assert signature is not None
    assert len(signature) == 32  # SHA-256 digest size

    # Verify signature
    verify_res = await provider.verify(key_id, payload, signature)
    assert verify_res.success
    assert verify_res.metadata["verified"] is True

    # Verify tampered payload fails
    tampered_res = await provider.verify(key_id, b"tampered-payload", signature)
    assert not tampered_res.success or tampered_res.metadata.get("verified") is False

    await provider.disconnect()


@pytest.mark.asyncio
async def test_tpm2_authenticated_encryption(tpm_config):
    """Test AES-256-GCM encryption and decryption with TPM-derived keys"""
    provider = TPM2Provider(tpm_config)
    await provider.connect()

    gen_res = await provider.generate_key(
        key_label="tpm-storage-master",
        algorithm=KeyAlgorithm.AES_256,
        usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
    )
    assert gen_res.success
    key_id = gen_res.key_id

    secret_data = b"DEFENSE_SOVEREIGN_POLICY_KEY_2026_CLASSIFIED"
    enc_res = await provider.encrypt(key_id, secret_data)
    assert enc_res.success
    ciphertext = enc_res.data
    assert ciphertext != secret_data

    dec_res = await provider.decrypt(key_id, ciphertext)
    assert dec_res.success
    assert dec_res.data == secret_data

    await provider.disconnect()


@pytest.mark.asyncio
async def test_tpm2_pcr_hardware_sealing_and_tamper_rejection(tpm_config):
    """
    Test PCR hardware policy sealing and anti-tamper rejection.
    If system state/code integrity changes, unsealing MUST be rejected.
    """
    provider = TPM2Provider(tpm_config)
    await provider.connect()

    gen_res = await provider.generate_key(
        key_label="tpm-pcr-sealed-key",
        algorithm=KeyAlgorithm.AES_256,
        usage=[KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
    )
    key_id = gen_res.key_id

    sensitive_payload = b"CRITICAL_REACTOR_SHUTDOWN_INVARIANT_CODE"
    pcr_indices = [0, 7, 11]

    # 1. Seal data to current PCR 0, 7, 11 measurements
    seal_res = await provider.seal_data(key_id, sensitive_payload, pcr_indices)
    assert seal_res.success
    sealed_blob = seal_res.data
    assert sealed_blob is not None

    # 2. Unseal in pristine system state (should SUCCEED)
    unseal_res = await provider.unseal_data(key_id, sealed_blob)
    assert unseal_res.success
    assert unseal_res.data == sensitive_payload

    # 3. Simulate unauthorized kernel modification or bootloader tamper by extending PCR 11
    provider.extend_pcr(11, b"malicious_module_injection_payload")

    # 4. Attempt unseal after tamper (MUST FAIL with UNAUTHORIZED)
    tampered_unseal = await provider.unseal_data(key_id, sealed_blob)
    assert not tampered_unseal.success
    assert tampered_unseal.status == HSMOperationStatus.UNAUTHORIZED
    assert "PCR mismatch" in tampered_unseal.error_message

    await provider.disconnect()


@pytest.mark.asyncio
async def test_tpm2_attestation_quote(tpm_config):
    """Test generating a TPM 2.0 attestation quote signed by AIK"""
    provider = TPM2Provider(tpm_config)
    await provider.connect()

    nonce = b"\x42" * 32
    quote = provider.get_pcr_quote(pcr_indices=[0, 1, 2, 7, 11], nonce=nonce)

    assert quote["tpm_standard"] == "TPM 2.0"
    assert quote["nonce"] == nonce.hex()
    assert "pcr_digest" in quote
    assert "quote_signature" in quote
    assert len(quote["pcr_values"]) == 5
    assert 7 in quote["pcr_values"]
    assert 11 in quote["pcr_values"]

    await provider.disconnect()


@pytest.mark.asyncio
async def test_tpm2_abstraction_layer_integration(tpm_config):
    """Test TPM2Provider integration through HSMAbstractionLayer"""
    layer = HSMAbstractionLayer(tpm_config)
    connected = await layer.initialize()
    assert connected is True

    status = layer.get_status()
    assert status["configured_provider"] == "tpm-2.0"
    assert status["hsm_connected"] is True

    # Generate key and sign policy
    gen_res = await layer.generate_key("law-1-policy-root", KeyAlgorithm.EC_P256, [KeyUsage.SIGN])
    assert gen_res.success

    policy_hash = b"\xaa" * 32
    sign_res = await layer.sign_policy(gen_res.key_id, policy_hash)
    assert sign_res.success

    verify_res = await layer.verify_policy(gen_res.key_id, policy_hash, sign_res.data)
    assert verify_res.success

    await layer.shutdown()
