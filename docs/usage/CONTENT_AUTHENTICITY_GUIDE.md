# Content Authenticity Guide

## Overview

Nethical's Content Authenticity module provides comprehensive tools for watermarking and tracking the provenance of AI-generated content. This guide covers how to use the deepfake watermarking system and C2PA integration to ensure transparency and authenticity of synthetic media.

---

## Features

### 1. Invisible Watermarking
- Embed imperceptible watermarks in images, videos, and audio
- Robust watermark detection and extraction
- Configurable watermark strength for different use cases

### 2. Content Provenance Tracking
- Track creation history and modifications
- Maintain creator attribution
- Verify content authenticity

### 3. C2PA Standard Integration
- Create and sign C2PA manifests
- Verify content signatures
- Industry-standard content credentials

### 4. Disclosure Labels
- Generate user-facing disclosure labels
- Comply with DSA Article 16(6) and AI Act transparency requirements
- Clear indication of AI-generated content

---

## Quick Start

### Installation

The content authenticity module is included in Nethical. Ensure you have the required dependencies:

```bash
pip install nethical
# Or install from source
pip install -e .
```

### Basic Usage

```python
from nethical.content_authenticity import (
    DeepfakeWatermarkingSystem,
    ContentMetadata,
    C2PAIntegration,
)
from datetime import datetime, timezone
import numpy as np

# Initialize watermarking system
watermark_system = DeepfakeWatermarkingSystem(
    watermark_strength=0.3,  # Balanced strength
    c2pa_enabled=True
)

# Create content metadata
metadata = ContentMetadata(
    creation_timestamp=datetime.now(timezone.utc),
    creator_id="user@example.com",
    model_name="Stable Diffusion",
    model_version="2.1",
    generation_params={"prompt": "mountain landscape", "steps": 50},
    synthetic=True,
    content_type="image"
)

# Watermark an image
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
watermarked = watermark_system.watermark_image(image, metadata)

print(f"Watermark ID: {watermarked.watermark_id}")
```

---

## Detailed Usage

### Watermarking Images

```python
# Load your AI-generated image
import cv2
image = cv2.imread("generated_image.jpg")

# Create detailed metadata
metadata = ContentMetadata(
    creation_timestamp=datetime.now(timezone.utc),
    creator_id="ai_artist_123",
    model_name="DALL-E",
    model_version="3.0",
    generation_params={
        "prompt": "futuristic city",
        "negative_prompt": "blurry, low quality",
        "steps": 30,
        "guidance_scale": 7.5
    },
    synthetic=True,
    content_type="image"
)

# Embed watermark
watermarked = watermark_system.watermark_image(image, metadata)

# Save watermarked image
cv2.imwrite("watermarked_image.jpg", watermarked.image_data)
```

### Watermarking Videos

```python
# Watermark a video file
video_metadata = ContentMetadata(
    creation_timestamp=datetime.now(timezone.utc),
    creator_id="video_creator",
    model_name="Runway Gen-2",
    model_version="1.0",
    generation_params={"prompt": "ocean waves"},
    synthetic=True,
    content_type="video"
)

watermarked_video = watermark_system.watermark_video(
    video_path="generated_video.mp4",
    metadata=video_metadata
)

print(f"Video watermarked with ID: {watermarked_video.watermark_id}")
```

### Watermarking Audio

```python
import librosa

# Load audio
audio, sample_rate = librosa.load("generated_audio.wav", sr=None)

audio_metadata = ContentMetadata(
    creation_timestamp=datetime.now(timezone.utc),
    creator_id="audio_generator",
    model_name="MusicGen",
    model_version="1.5",
    generation_params={"prompt": "ambient music"},
    synthetic=True,
    content_type="audio"
)

watermarked_audio = watermark_system.watermark_audio(
    audio=audio,
    metadata=audio_metadata,
    sample_rate=sample_rate
)

# Save watermarked audio
import soundfile as sf
sf.write("watermarked_audio.wav", watermarked_audio.audio_data, sample_rate)
```

---

## Detecting Watermarks

### Basic Detection

```python
# Load potentially watermarked content
image = cv2.imread("suspicious_image.jpg")

# Detect watermark
detection = watermark_system.detect_watermark(image)

if detection.watermark_detected:
    print(f"Watermark found! ID: {detection.watermark_id}")
    print(f"Confidence: {detection.confidence:.2%}")
    print(f"Quality: {detection.extraction_quality}")
    
    if detection.metadata_intact:
        metadata = detection.extracted_metadata
        print(f"Creator: {metadata.creator_id}")
        print(f"Model: {metadata.model_name} v{metadata.model_version}")
else:
    print("No watermark detected")
```

### Provenance Extraction

```python
# Extract full provenance chain
provenance = watermark_system.extract_provenance(image)

print(f"Content ID: {provenance.content_id}")
print(f"Original Creator: {provenance.original_creator}")
print(f"Authenticity Verified: {provenance.authenticity_verified}")

# Display creation chain
for step in provenance.creation_chain:
    print(f"  - Created by {step['creator']} using {step['model']}")
    print(f"    Timestamp: {step['timestamp']}")
```

---

## Disclosure Labels

### Generating Disclosure Labels

```python
# Generate disclosure label for user display
label = watermark_system.generate_disclosure_label(
    content_type="image",
    metadata=metadata
)

print(label.disclosure_text)
# Output:
# ⚠️ AI-Generated Image
# This image was created using artificial intelligence.
# Model: DALL-E v3.0
# Created: 2024-01-14 10:30 UTC
```

### Displaying Disclosure Labels

```html
<!-- Example HTML for displaying disclosure label -->
<div class="ai-disclosure-label">
    <span class="warning-icon">⚠️</span>
    <div class="disclosure-text">
        <strong>AI-Generated Content</strong>
        <p>This content was created using artificial intelligence.</p>
        <p>Model: {{ label.model_name }} v{{ label.model_version }}</p>
        <p>Created: {{ label.creation_date.strftime('%Y-%m-%d %H:%M UTC') }}</p>
    </div>
</div>
```

---

## C2PA Integration

### Creating and Signing Manifests

```python
from nethical.content_authenticity import C2PAIntegration

# Initialize C2PA
c2pa = C2PAIntegration(claim_generator="MyApp v1.0")

# Prepare content metadata
c2pa_metadata = {
    "title": "AI Generated Landscape",
    "format": "image/jpeg",
    "synthetic": True,
    "model_name": "Stable Diffusion",
    "model_version": "2.1",
    "creator_id": "user@example.com",
    "creation_timestamp": datetime.now(timezone.utc),
    "generation_params": {"prompt": "mountain landscape"}
}

# Create manifest
manifest = c2pa.create_manifest(watermarked.image_data, c2pa_metadata)

# Sign manifest (requires private key)
signed_manifest = c2pa.sign_manifest(manifest, private_key="your_private_key")

# Embed manifest into image file
# (This would modify image metadata in production)
embedded_image = c2pa.embed_manifest(watermarked.image_data, signed_manifest)
```

### Verifying C2PA Manifests

```python
# Extract and verify manifest from content
extracted_manifest = c2pa.extract_manifest(image)

if extracted_manifest:
    # Verify signature and integrity
    verification = c2pa.verify_manifest(extracted_manifest)
    
    if verification.verified:
        print("✓ Manifest verified successfully")
        print(f"Signature valid: {verification.signature_valid}")
        print(f"Certificate valid: {verification.certificate_valid}")
        
        # Access manifest data
        manifest = verification.manifest
        for assertion in manifest.assertions:
            print(f"  - {assertion.label}: {assertion.data}")
    else:
        print("✗ Verification failed")
        for error in verification.validation_errors:
            print(f"  Error: {error}")
```

---

## Integration with Deepfake Detector

### Combined Detection and Verification

```python
from nethical.detectors.realtime.deepfake_detector import DeepfakeDetector

# Initialize both systems
deepfake_detector = DeepfakeDetector()
watermark_system = DeepfakeWatermarkingSystem()

async def verify_content_authenticity(image):
    """Comprehensive content authenticity check."""
    
    # Step 1: Check for watermark
    watermark_result = watermark_system.detect_watermark(image)
    
    if watermark_result.watermark_detected:
        # Content has valid watermark - trust it
        return {
            "authentic": watermark_result.metadata_intact,
            "confidence": watermark_result.confidence,
            "method": "watermark",
            "metadata": watermark_result.extracted_metadata
        }
    
    # Step 2: Fall back to ML-based deepfake detection
    detection = await deepfake_detector.detect(
        media=image.tobytes(),
        media_type="image"
    )
    
    return {
        "authentic": not detection["is_deepfake"],
        "confidence": detection["confidence"],
        "method": "ml_detection",
        "violations": detection.get("violations", [])
    }
```

---

## Best Practices

### 1. Watermark Strength Selection

```python
# Low strength (0.1-0.2): Maximum imperceptibility, less robust
watermark_system_low = DeepfakeWatermarkingSystem(watermark_strength=0.15)

# Medium strength (0.3): Balanced (recommended for most use cases)
watermark_system_med = DeepfakeWatermarkingSystem(watermark_strength=0.3)

# High strength (0.4-0.5): Maximum robustness, slightly more perceptible
watermark_system_high = DeepfakeWatermarkingSystem(watermark_strength=0.45)
```

### 2. Metadata Completeness

Always provide comprehensive metadata:

```python
metadata = ContentMetadata(
    creation_timestamp=datetime.now(timezone.utc),
    creator_id="verified_user@example.com",  # Use verified IDs
    model_name="Stable Diffusion XL",
    model_version="1.0",
    generation_params={
        "prompt": "detailed prompt here",
        "negative_prompt": "...",
        "steps": 50,
        "sampler": "DPM++ 2M Karras",
        "seed": 42
    },
    synthetic=True,
    content_type="image"
)
```

### 3. Disclosure Compliance

```python
# Generate and display disclosure for all AI-generated content
label = watermark_system.generate_disclosure_label(
    content_type=content_type,
    metadata=metadata
)

# Display to users before sharing/publishing
print(label.disclosure_text)
```

### 4. Provenance Chain Maintenance

```python
# Track modifications
provenance = watermark_system.extract_provenance(content)

# Add modification record (conceptual)
provenance.modifications.append({
    "timestamp": datetime.now(timezone.utc),
    "modifier": "user_id",
    "operation": "crop",
    "parameters": {"crop_area": "..."}
})
```

---

## Regulatory Compliance

### DSA Article 16(6) Compliance

```python
# Generate disclosure label for online platforms
label = watermark_system.generate_disclosure_label(
    content_type="image",
    metadata=metadata
)

# Display prominent notice (DSA requirement)
disclosure_html = f"""
<div class="dsa-synthetic-disclosure">
    {label.disclosure_text}
</div>
"""
```

### AI Act Transparency Requirements

```python
# For high-risk AI systems, provide detailed disclosures
detailed_label = f"""
AI-Generated Content Notice

This content was generated by an AI system and may not be factually accurate.

Details:
- Model: {metadata.model_name} {metadata.model_version}
- Creator: {metadata.creator_id}
- Generated: {metadata.creation_timestamp.strftime('%Y-%m-%d %H:%M UTC')}
- Generation Parameters: {json.dumps(metadata.generation_params, indent=2)}

For more information or to report issues, contact: support@example.com
"""
```

---

## Advanced Usage

### Custom Watermark Patterns

```python
# The watermark pattern is automatically generated from the watermark ID
# For custom patterns, you can extend the DeepfakeWatermarkingSystem class

class CustomWatermarkSystem(DeepfakeWatermarkingSystem):
    def _generate_watermark_pattern(self, watermark_id, shape):
        # Implement custom pattern generation
        # Example: QR code-based watermark
        pass
```

### Batch Watermarking

```python
import glob

# Watermark multiple images
image_files = glob.glob("generated/*.jpg")

for img_path in image_files:
    image = cv2.imread(img_path)
    
    watermarked = watermark_system.watermark_image(image, metadata)
    
    # Save with watermark
    output_path = img_path.replace("generated/", "watermarked/")
    cv2.imwrite(output_path, watermarked.image_data)
    
    print(f"Watermarked: {img_path} -> {output_path}")
```

---

## Troubleshooting

### Watermark Not Detected

**Problem:** `watermark_detected` is False

**Solutions:**
1. Check if content has been heavily compressed or modified
2. Try higher watermark strength (0.4-0.5)
3. Verify the content was actually watermarked
4. Check for lossy transformations (JPEG compression >80%)

### Low Confidence Detection

**Problem:** `confidence` < 0.7

**Solutions:**
1. Use higher watermark strength
2. Avoid heavy compression
3. Check for scaling/cropping operations
4. Verify content hasn't been re-encoded multiple times

### Metadata Not Extracted

**Problem:** `metadata_intact` is False

**Solutions:**
1. Ensure watermark was created with this system
2. Check watermark ID is in registry
3. Verify content hasn't been modified excessively

---

## API Reference

See module docstrings for detailed API documentation:

```python
help(DeepfakeWatermarkingSystem)
help(C2PAIntegration)
help(ContentMetadata)
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/tree/main/docs

---

## References

- **C2PA Specification:** https://c2pa.org/specifications/
- **DSA Article 16:** Online platform transparency obligations
- **EU AI Act:** Transparency requirements for AI systems
- **NIST Guidance on AI-Generated Content:** https://www.nist.gov/

---

**Document Version:** 1.0  
**Last Updated:** 2024-01-14
