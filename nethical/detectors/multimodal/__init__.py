"""
Multimodal Detection Suite

Detects attacks across multiple modalities (image, audio, video).

Detectors:
- MM-001: Adversarial Image (CNN-based perturbation detection)
- MM-002: Audio Injection (speech-to-text + injection check)
- MM-003: Video Frame Attack (per-frame adversarial detection)
- MM-004: Cross-Modal Injection (multi-encoder consistency check)

Law Alignment:
- Law 9 (Self-Disclosure): Verify content across modalities
- Law 18 (Non-Deception): Detect hidden attacks
- Law 22 (Boundary Respect): Protect multimodal interfaces
"""

from .adversarial_image_detector import AdversarialImageDetector
from .audio_injection_detector import AudioInjectionDetector
from .video_frame_detector import VideoFrameDetector
from .cross_modal_detector import CrossModalDetector

__all__ = [
    'AdversarialImageDetector',
    'AudioInjectionDetector',
    'VideoFrameDetector',
    'CrossModalDetector',
]
