# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit tests for content authenticity, C2PA manifest provenance, and deepfake watermarking."""

import numpy as np
import pytest
from datetime import datetime, timezone

from nethical.content_authenticity.c2pa_integration import (
    C2PAAssertion,
    C2PAIngredient,
    C2PAIntegration,
    C2PAManifest,
    C2PAVerificationResult,
    SignedManifest,
)
from nethical.content_authenticity.deepfake_watermark import (
    ContentMetadata,
    ContentProvenance,
    DeepfakeWatermarkingSystem,
    DisclosureLabel,
    ExtractionQuality,
    WatermarkedAudio,
    WatermarkedImage,
    WatermarkedVideo,
    WatermarkStrength,
)


class TestDeepfakeWatermarkingSystem:
    """Tests for DeepfakeWatermarkingSystem across images, videos, audio, and disclosure labels."""

    @pytest.fixture
    def watermarking_system(self) -> DeepfakeWatermarkingSystem:
        return DeepfakeWatermarkingSystem(watermark_strength=WatermarkStrength.MEDIUM)

    @pytest.fixture
    def sample_metadata(self) -> ContentMetadata:
        return ContentMetadata(
            creation_timestamp=datetime.now(timezone.utc),
            creator_id="sentinel_genesis_ai",
            model_name="Nethical-Diffusion",
            model_version="1.0.0",
            generation_params={"guidance_scale": 7.5, "steps": 50},
            synthetic=True,
            content_type="image",
        )

    def test_watermark_image_embedding_and_structure(
        self,
        watermarking_system: DeepfakeWatermarkingSystem,
        sample_metadata: ContentMetadata,
    ) -> None:
        """Verifies watermarking of synthetic image array."""
        dummy_image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        watermarked = watermarking_system.watermark_image(dummy_image, sample_metadata)

        assert isinstance(watermarked, WatermarkedImage)
        assert watermarked.image_data.shape == (64, 64, 3)
        assert watermarked.image_data.dtype == np.uint8
        assert watermarked.watermark_id is not None
        assert watermarked.watermark_strength == 0.3
        assert watermarked.metadata.watermark_id == watermarked.watermark_id

    def test_watermark_video_and_audio_simulation(
        self,
        watermarking_system: DeepfakeWatermarkingSystem,
        sample_metadata: ContentMetadata,
    ) -> None:
        """Verifies simulation of video and audio watermarking."""
        # Video
        video_result = watermarking_system.watermark_video("synthetic_motion.mp4", sample_metadata)
        assert isinstance(video_result, WatermarkedVideo)
        assert video_result.frame_count == 100
        assert video_result.watermark_id in watermarking_system._watermark_registry

        # Audio
        dummy_audio = np.random.uniform(-1.0, 1.0, 44100).astype(np.float32)
        audio_result = watermarking_system.watermark_audio(dummy_audio, sample_metadata, sample_rate=44100)
        assert isinstance(audio_result, WatermarkedAudio)
        assert len(audio_result.audio_data) == 44100

    def test_detection_and_provenance_extraction(
        self,
        watermarking_system: DeepfakeWatermarkingSystem,
        sample_metadata: ContentMetadata,
    ) -> None:
        """Verifies detection of watermark and provenance metadata extraction."""
        dummy_image = np.zeros((32, 32, 3), dtype=np.uint8)
        watermarked = watermarking_system.watermark_image(dummy_image, sample_metadata)

        detection = watermarking_system.detect_watermark(watermarked.image_data)
        assert detection.watermark_detected is True
        assert detection.confidence > 0.7

        provenance = watermarking_system.extract_provenance(watermarked.image_data)
        assert isinstance(provenance, ContentProvenance)
        assert provenance.content_id is not None

    def test_generate_disclosure_label(
        self,
        watermarking_system: DeepfakeWatermarkingSystem,
        sample_metadata: ContentMetadata,
    ) -> None:
        """Verifies regulatory disclosure label generation (Law 10 & Law 12)."""
        label = watermarking_system.generate_disclosure_label("image", sample_metadata)
        assert isinstance(label, DisclosureLabel)
        assert label.is_synthetic is True
        assert label.model_name == "Nethical-Diffusion"
        assert "AI-Generated Image" in label.disclosure_text
        assert "Nethical-Diffusion" in label.disclosure_text


class TestC2PAIntegration:
    """Tests for C2PA manifest provenance tracking, signing, and cryptographic verification."""

    @pytest.fixture
    def c2pa_service(self) -> C2PAIntegration:
        return C2PAIntegration(claim_generator="Nethical Sovereign Engine/2.7.0")

    def test_manifest_creation_and_assertions(self, c2pa_service: C2PAIntegration) -> None:
        """Verifies manifest creation with automatic AI and authorship assertions."""
        raw_bytes = b"PROVENANCE_TEST_IMAGE_BYTES"
        metadata = {
            "title": "Tactical Reconnaissance Orthomosaic",
            "format": "image/png",
            "synthetic": True,
            "model_name": "SAR-GeoSynth",
            "model_version": "2.4",
            "creator_id": "operator_kris",
            "generation_params": {"resolution_m": 0.5},
        }

        manifest = c2pa_service.create_manifest(raw_bytes, metadata)
        assert isinstance(manifest, C2PAManifest)
        assert manifest.claim_generator == "Nethical Sovereign Engine/2.7.0"
        assert manifest.title == "Tactical Reconnaissance Orthomosaic"
        assert manifest.format == "image/png"
        assert len(manifest.assertions) >= 3

        assertion_types = [a.assertion_type for a in manifest.assertions]
        assert "c2pa.ai_generated" in assertion_types
        assert "c2pa.author" in assertion_types
        assert "c2pa.hash.sha256" in assertion_types

    def test_manifest_signing_and_verification_cycle(self, c2pa_service: C2PAIntegration) -> None:
        """Verifies full signing and verification lifecycle."""
        raw_bytes = b"HIGH_INTEGRITY_DECISION_PROOF"
        metadata = {
            "title": "Autonomous Kinetic Boundary Proof",
            "format": "application/json",
            "synthetic": False,
            "creator_id": "sovereign_governor",
        }

        manifest = c2pa_service.create_manifest(raw_bytes, metadata)
        signed = c2pa_service.sign_manifest(manifest, private_key="sovereign-secp256k1-key")

        assert isinstance(signed, SignedManifest)
        assert signed.signature != ""
        assert len(signed.certificate_chain) > 0

        # Verification must succeed for intact manifest
        result = c2pa_service.verify_manifest(signed)
        assert isinstance(result, C2PAVerificationResult)
        assert result.verified is True
        assert result.signature_valid is True
        assert result.manifest_intact is True
        assert len(result.validation_errors) == 0

    def test_tamper_detection(self, c2pa_service: C2PAIntegration) -> None:
        """Verifies that tampering with signed manifest assertions breaks verification."""
        manifest = c2pa_service.create_manifest(b"CONTENT_DATA", {"title": "Original"})
        signed = c2pa_service.sign_manifest(manifest, private_key="key")

        # Tamper with the title after signing
        signed.manifest.title = "Tampered Title By Attacker"

        # Verification must fail
        result = c2pa_service.verify_manifest(signed)
        assert result.verified is False
        assert result.signature_valid is False
        assert len(result.validation_errors) > 0
