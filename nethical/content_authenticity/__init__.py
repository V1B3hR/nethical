"""Content Authenticity Module for Nethical.

This module provides content authenticity and provenance tracking
for AI-generated content, including watermarking and C2PA integration.
"""

from .deepfake_watermark import (
    DeepfakeWatermarkingSystem,
    ContentMetadata,
    WatermarkedImage,
    WatermarkedVideo,
    WatermarkedAudio,
    WatermarkDetectionResult,
    ContentProvenance,
    DisclosureLabel,
    WatermarkStrength,
    ExtractionQuality,
)

from .c2pa_integration import (
    C2PAIntegration,
    C2PAManifest,
    C2PAVerificationResult,
    SignedManifest,
    C2PAAssertion,
    C2PAIngredient,
)

__all__ = [
    "DeepfakeWatermarkingSystem",
    "ContentMetadata",
    "WatermarkedImage",
    "WatermarkedVideo",
    "WatermarkedAudio",
    "WatermarkDetectionResult",
    "ContentProvenance",
    "DisclosureLabel",
    "WatermarkStrength",
    "ExtractionQuality",
    "C2PAIntegration",
    "C2PAManifest",
    "C2PAVerificationResult",
    "SignedManifest",
    "C2PAAssertion",
    "C2PAIngredient",
]
