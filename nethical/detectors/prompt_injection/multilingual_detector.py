"""
Multilingual Prompt Injection Detector (PI-007)

Detects injection attempts using non-English languages to bypass pattern-based
detection focused on English keywords.

Signals:
- Language switching mid-prompt
- Unicode homoglyph detection
- RTL/LTR mixing

Law Alignment: Laws 9 (Self-Disclosure), 18 (Non-Deception)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence
import unicodedata

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class MultilingualDetector(BaseDetector):
    """Detects multilingual prompt injection attacks."""

    def __init__(self):
        super().__init__("Multilingual Injection Detector", version="1.0.0")
        
        # Common injection keywords in multiple languages
        self.injection_keywords = {
            "chinese": ["忽略", "系统", "指令", "绕过", "覆盖"],  # ignore, system, instruction, bypass, override
            "spanish": ["ignorar", "sistema", "instrucción", "omitir"],
            "french": ["ignorer", "système", "instruction", "contourner"],
            "german": ["ignorieren", "system", "anweisung", "umgehen"],
            "russian": ["игнорировать", "система", "инструкция", "обход"],
            "arabic": ["تجاهل", "نظام", "تعليمات", "تجاوز"],
            "japanese": ["無視", "システム", "命令", "バイパス"],
            "korean": ["무시", "시스템", "지시", "우회"],
        }
        
        # Unicode homoglyph patterns (characters that look similar but different code points)
        self.homoglyph_pairs = [
            ('a', 'а'),  # Latin a vs Cyrillic a
            ('e', 'е'),  # Latin e vs Cyrillic e
            ('o', 'о'),  # Latin o vs Cyrillic o
            ('p', 'р'),  # Latin p vs Cyrillic p
            ('c', 'с'),  # Latin c vs Cyrillic c
            ('x', 'х'),  # Latin x vs Cyrillic x
            ('y', 'у'),  # Latin y vs Cyrillic y
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect multilingual injection attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for language switching
        lang_switch_score = self._detect_language_switching(content)
        if lang_switch_score > 0:
            evidence.append(f"Language switching detected (score: {lang_switch_score:.2f})")
            confidence += lang_switch_score * 0.4
        
        # Check for non-English injection keywords
        keyword_score = self._detect_multilingual_keywords(content)
        if keyword_score > 0:
            evidence.append(f"Multilingual injection keywords detected (score: {keyword_score:.2f})")
            confidence += keyword_score * 0.3
        
        # Check for homoglyphs
        homoglyph_score = self._detect_homoglyphs(str(action.content))
        if homoglyph_score > 0:
            evidence.append(f"Unicode homoglyphs detected (score: {homoglyph_score:.2f})")
            confidence += homoglyph_score * 0.3
        
        # Check for RTL/LTR mixing
        rtl_ltr_score = self._detect_rtl_ltr_mixing(str(action.content))
        if rtl_ltr_score > 0:
            evidence.append(f"RTL/LTR mixing detected (score: {rtl_ltr_score:.2f})")
            confidence += rtl_ltr_score * 0.2
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:  # Flag threshold
            severity = Severity.CRITICAL if confidence >= 0.8 else (
                Severity.HIGH if confidence >= 0.5 else Severity.MEDIUM
            )
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Multilingual prompt injection detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_language_switching(self, content: str) -> float:
        """Detect language switching patterns."""
        # Count script changes (Latin, Cyrillic, CJK, Arabic, etc.)
        scripts = set()
        for char in content:
            if char.isalpha():
                script_name = unicodedata.name(char, '').split()[0] if unicodedata.name(char, '') else 'UNKNOWN'
                if script_name in ['LATIN', 'CYRILLIC', 'CJK', 'ARABIC', 'HEBREW', 'HIRAGANA', 'KATAKANA', 'HANGUL']:
                    scripts.add(script_name)
        
        # Multiple scripts suggest language switching
        if len(scripts) >= 3:
            return 1.0
        elif len(scripts) == 2:
            return 0.6
        return 0.0

    def _detect_multilingual_keywords(self, content: str) -> float:
        """Detect injection keywords in multiple languages."""
        matches = 0
        for lang, keywords in self.injection_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content:
                    matches += 1
        
        # Normalize score
        if matches >= 3:
            return 1.0
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.4
        return 0.0

    def _detect_homoglyphs(self, content: str) -> float:
        """Detect Unicode homoglyph usage."""
        homoglyph_count = 0
        for latin, lookalike in self.homoglyph_pairs:
            if lookalike in content:
                homoglyph_count += 1
        
        # Normalize score
        if homoglyph_count >= 3:
            return 1.0
        elif homoglyph_count >= 2:
            return 0.7
        elif homoglyph_count >= 1:
            return 0.4
        return 0.0

    def _detect_rtl_ltr_mixing(self, content: str) -> float:
        """Detect mixing of right-to-left and left-to-right text."""
        has_rtl = False
        has_ltr = False
        
        for char in content:
            if char.isalpha():
                direction = unicodedata.bidirectional(char)
                if direction in ['R', 'AL', 'RLE', 'RLO']:
                    has_rtl = True
                elif direction in ['L', 'LRE', 'LRO']:
                    has_ltr = True
        
        if has_rtl and has_ltr:
            return 0.8
        return 0.0
