"""
Watermark Detector - Invisible watermarks in responses for data exfiltration detection

This detector embeds invisible watermarks in system responses and detects
when those watermarks appear elsewhere, indicating potential data exfiltration.

Features:
- Invisible watermark embedding
- Watermark detection in inputs
- Data exfiltration alerting
- Tracking watermark propagation

Alignment: Law 2 (Data Integrity), Law 15 (Audit Compliance)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ..base_detector import BaseDetector, ViolationSeverity

logger = logging.getLogger(__name__)

# Constants
WATERMARK_BITS = 32  # Number of bits to use for watermark encoding


class WatermarkType(str, Enum):
    """Types of watermarks."""
    
    UNICODE_STEALTH = "unicode_stealth"  # Zero-width characters
    SEMANTIC_MARKER = "semantic_marker"  # Semantic patterns
    FORMATTING_MARKER = "formatting_marker"  # Whitespace patterns
    STRUCTURAL_MARKER = "structural_marker"  # Document structure


@dataclass
class Watermark:
    """Definition of an embedded watermark."""
    
    id: str
    watermark_type: WatermarkType
    marker: str  # The actual watermark content
    embedded_in: str  # Where it was embedded (session_id, response_id)
    embedded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detected_count: int = 0


@dataclass
class ExfiltrationAlert:
    """Alert for detected data exfiltration."""
    
    alert_id: str
    watermark_id: str
    detected_in: str  # Where watermark was found
    detected_by: str  # Agent that submitted watermarked content
    detection_time: datetime
    confidence: float
    severity: ViolationSeverity


class WatermarkDetector(BaseDetector):
    """
    Watermark-based data exfiltration detection.
    
    Embeds invisible watermarks in responses and detects when those
    watermarks appear in subsequent inputs, indicating that data may
    have been exfiltrated and is being reused.
    
    Detection Method: Watermark presence detection
    """
    
    def __init__(self):
        """Initialize the watermark detector."""
        super().__init__()
        self.active_watermarks: Dict[str, Watermark] = {}
        self.exfiltration_alerts: List[ExfiltrationAlert] = []
        self.suspected_exfiltrators: Set[str] = set()
        
        logger.info("WatermarkDetector initialized")
    
    async def detect_violations(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:  # Returns List[SafetyViolation]
        """
        Detect presence of watermarks in input.
        
        Args:
            input_text: Text to analyze for watermarks
            context: Additional context (agent_id, session_id, etc.)
            
        Returns:
            List of safety violations if watermark detected
        """
        violations = []
        context = context or {}
        
        # Check for watermark presence
        for watermark_id, watermark in self.active_watermarks.items():
            if await self._is_watermark_present(input_text, watermark):
                # Watermark detected - potential exfiltration
                violation = await self._create_violation(
                    input_text, watermark, context
                )
                violations.append(violation)
                
                # Update watermark statistics
                watermark.detected_count += 1
                
                # Track potential exfiltrator
                agent_id = context.get("agent_id", "unknown")
                self.suspected_exfiltrators.add(agent_id)
                
                logger.critical(
                    f"Watermark {watermark_id} detected in input from {agent_id}"
                )
        
        return violations
    
    async def embed_watermark(
        self,
        response_text: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Embed an invisible watermark in a response.
        
        Args:
            response_text: Original response text
            context: Context about the response (session_id, etc.)
            
        Returns:
            Response text with embedded watermark
        """
        # Generate unique watermark
        watermark_id = self._generate_watermark_id()
        watermark_type = WatermarkType.UNICODE_STEALTH  # Default type
        
        # Create watermark marker
        marker = self._create_marker(watermark_id, watermark_type)
        
        # Embed watermark in response
        watermarked_text = self._embed_marker(response_text, marker, watermark_type)
        
        # Store watermark record
        watermark = Watermark(
            id=watermark_id,
            watermark_type=watermark_type,
            marker=marker,
            embedded_in=context.get("session_id", "unknown")
        )
        self.active_watermarks[watermark_id] = watermark
        
        logger.debug(f"Embedded watermark {watermark_id} in response")
        
        return watermarked_text
    
    def _create_marker(
        self,
        watermark_id: str,
        watermark_type: WatermarkType
    ) -> str:
        """Create a watermark marker."""
        
        if watermark_type == WatermarkType.UNICODE_STEALTH:
            # Use zero-width characters
            return self._create_unicode_marker(watermark_id)
        elif watermark_type == WatermarkType.SEMANTIC_MARKER:
            # Use semantic patterns
            return self._create_semantic_marker(watermark_id)
        elif watermark_type == WatermarkType.FORMATTING_MARKER:
            # Use whitespace patterns
            return self._create_formatting_marker(watermark_id)
        else:
            # Default: simple marker
            return f"<!--WM:{watermark_id}-->"
    
    def _create_unicode_marker(self, watermark_id: str) -> str:
        """Create a zero-width Unicode marker."""
        # Zero-width characters: U+200B (ZWSP), U+200C (ZWNJ), U+200D (ZWJ)
        zwsp = "\u200b"
        zwnj = "\u200c"
        zwj = "\u200d"
        
        # Encode watermark ID in zero-width characters
        # Use binary encoding: 0=ZWNJ, 1=ZWJ
        hash_hex = hashlib.sha256(watermark_id.encode()).hexdigest()[:8]
        hash_binary = bin(int(hash_hex, 16))[2:].zfill(WATERMARK_BITS)
        
        marker = zwsp  # Start marker
        for bit in hash_binary[:16]:  # Use first 16 bits
            marker += zwj if bit == "1" else zwnj
        marker += zwsp  # End marker
        
        return marker
    
    def _create_semantic_marker(self, watermark_id: str) -> str:
        """Create a semantic marker (specific word pattern)."""
        # Create a unique but natural-looking phrase
        hash_int = int(hashlib.sha256(watermark_id.encode()).hexdigest()[:8], 16)
        words = ["furthermore", "additionally", "moreover", "consequently"]
        selected_word = words[hash_int % len(words)]
        return f" {selected_word} "
    
    def _create_formatting_marker(self, watermark_id: str) -> str:
        """Create a whitespace-based marker."""
        # Encode in spaces and tabs
        hash_hex = hashlib.sha256(watermark_id.encode()).hexdigest()[:4]
        hash_binary = bin(int(hash_hex, 16))[2:].zfill(16)
        
        marker = ""
        for bit in hash_binary[:8]:  # Use first 8 bits
            marker += "\t" if bit == "1" else " "
        
        return marker
    
    def _embed_marker(
        self,
        text: str,
        marker: str,
        watermark_type: WatermarkType
    ) -> str:
        """Embed marker in text."""
        
        if watermark_type == WatermarkType.UNICODE_STEALTH:
            # Insert in middle of text
            mid_point = len(text) // 2
            return text[:mid_point] + marker + text[mid_point:]
        
        elif watermark_type == WatermarkType.SEMANTIC_MARKER:
            # Add to end of sentence
            sentences = text.split(". ")
            if len(sentences) > 1:
                sentences[0] += marker
                return ". ".join(sentences)
            return text + marker
        
        elif watermark_type == WatermarkType.FORMATTING_MARKER:
            # Add as trailing whitespace
            return text + marker
        
        else:
            # Default: append
            return text + marker
    
    async def _is_watermark_present(
        self,
        text: str,
        watermark: Watermark
    ) -> bool:
        """Check if watermark is present in text."""
        
        # Direct marker match
        if watermark.marker in text:
            return True
        
        # For Unicode markers, check for pattern
        if watermark.watermark_type == WatermarkType.UNICODE_STEALTH:
            return self._detect_unicode_pattern(text, watermark.marker)
        
        return False
    
    def _detect_unicode_pattern(self, text: str, marker: str) -> bool:
        """Detect zero-width character patterns."""
        # Count zero-width characters
        zwsp = "\u200b"
        zwnj = "\u200c"
        zwj = "\u200d"
        
        zw_count = text.count(zwsp) + text.count(zwnj) + text.count(zwj)
        
        # If significant zero-width characters present, check for pattern
        if zw_count >= 5:
            # Extract zero-width sequence
            zw_chars = [c for c in text if c in [zwsp, zwnj, zwj]]
            text_pattern = "".join(zw_chars)
            
            # Check if marker pattern is present
            marker_pattern = "".join([c for c in marker if c in [zwsp, zwnj, zwj]])
            
            return marker_pattern in text_pattern
        
        return False
    
    async def _create_violation(
        self,
        input_text: str,
        watermark: Watermark,
        context: Dict[str, Any]
    ) -> Any:  # Returns SafetyViolation
        """Create a safety violation for watermark detection."""
        
        # Import here to avoid circular dependency
        from ...core.models import SafetyViolation
        
        agent_id = context.get("agent_id", "unknown")
        
        # Create exfiltration alert
        alert = ExfiltrationAlert(
            alert_id=f"EX-{int(time.time() * 1000)}",
            watermark_id=watermark.id,
            detected_in=context.get("session_id", "unknown"),
            detected_by=agent_id,
            detection_time=datetime.now(timezone.utc),
            confidence=0.95,  # High confidence
            severity=ViolationSeverity.CRITICAL
        )
        self.exfiltration_alerts.append(alert)
        
        return SafetyViolation(
            detector_id="watermark_detector",
            severity=ViolationSeverity.CRITICAL.value,
            message="Data exfiltration detected: Watermarked content found in input",
            evidence={
                "watermark_id": watermark.id,
                "watermark_type": watermark.watermark_type.value,
                "embedded_in": watermark.embedded_in,
                "detected_in": context.get("session_id", "unknown"),
                "agent_id": agent_id,
                "detection_count": watermark.detected_count,
            },
            confidence=0.95,
            recommendation="BLOCK"
        )
    
    def _generate_watermark_id(self) -> str:
        """Generate unique watermark ID."""
        timestamp = int(time.time() * 1000)
        random_suffix = random.randint(1000, 9999)
        return f"WM-{timestamp}-{random_suffix}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get watermark statistics."""
        return {
            "active_watermarks": len(self.active_watermarks),
            "exfiltration_alerts": len(self.exfiltration_alerts),
            "suspected_exfiltrators": len(self.suspected_exfiltrators),
            "detection_rate": self._calculate_detection_rate(),
            "watermarks_by_type": self._count_by_type(),
        }
    
    def _calculate_detection_rate(self) -> float:
        """Calculate watermark detection rate."""
        if not self.active_watermarks:
            return 0.0
        
        detected = sum(
            1 for w in self.active_watermarks.values() if w.detected_count > 0
        )
        return detected / len(self.active_watermarks)
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count watermarks by type."""
        counts = {}
        for watermark in self.active_watermarks.values():
            type_name = watermark.watermark_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def is_suspected_exfiltrator(self, agent_id: str) -> bool:
        """Check if agent is suspected of data exfiltration."""
        return agent_id in self.suspected_exfiltrators
