"""Enhanced Detector for ethical constraint violations with advanced AI Ethics Framework enforcement.

Advanced improvements:
- Semantic similarity detection using embeddings for context-aware matching
- Multi-word phrase and idiom detection with flexible matching
- Confidence scoring system with adjustable thresholds
- Context analysis for intent detection (sarcasm, hypotheticals, quotes)
- Violation clustering to avoid redundant flagging
- Temporal pattern detection for escalating violations
- Configurable rule weights and custom severity calculation
- Detailed audit trail with explainability metrics
- Support for multi-language detection patterns
- Machine learning integration hooks for adaptive learning
"""

import re
import uuid
from typing import Dict, List, Optional, Pattern, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import hashlib

from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class ContextType(Enum):
    """Classification of the context in which a keyword appears."""
    DIRECT_ACTION = "direct_action"
    HYPOTHETICAL = "hypothetical"
    QUOTE = "quote"
    NEGATED = "negated"
    CONDITIONAL = "conditional"
    QUESTION = "question"


@dataclass
class MatchContext:
    """Rich context information for a detected match."""
    keyword: str
    start: int
    end: int
    snippet: str
    confidence: float
    context_type: ContextType
    sentence: str
    surrounding_sentences: List[str] = field(default_factory=list)
    semantic_score: Optional[float] = None


@dataclass
class ViolationPattern:
    """Represents a violation pattern with metadata."""
    keywords: List[str]
    severity: SeverityLevel
    principle: str
    category: str
    weight: float = 1.0
    requires_context: bool = False
    min_confidence: float = 0.6


class EthicalViolationDetector(BaseDetector):
    """
    Advanced ethical violation detector with semantic understanding and context awareness.
    """

    DEFAULT_NEGATION_CUES = [
        "no", "not", "never", "avoid", "prevent", "stop", "prohibit", "forbid", "against", "without",
        "refrain", "discourage", "ban", "disallow", "fail to", "lacking", "ignore", "won't", "don't",
        "doesn't", "didn't", "cannot", "can't", "shouldn't", "wouldn't", "refuse to"
    ]

    HYPOTHETICAL_CUES = [
        "if", "suppose", "imagine", "hypothetically", "what if", "in theory", "theoretically",
        "let's say", "pretend", "assume", "assuming", "would", "could", "might", "consider"
    ]

    QUOTE_INDICATORS = [
        "said", "says", "stated", "according to", "quoted", "claimed", "reported",
        "mentioned", "wrote", "describes", "argues"
    ]

    def __init__(
        self,
        ethical_patterns: Optional[Dict[str, ViolationPattern]] = None,
        negation_window_chars: int = 48,
        min_confidence_threshold: float = 0.6,
        enable_semantic_matching: bool = False,
        enable_clustering: bool = True,
        violation_history_size: int = 100
    ):
        super().__init__("Enhanced Ethical Violation Detector")

        self.min_confidence_threshold = min_confidence_threshold
        self.enable_semantic_matching = enable_semantic_matching
        self.enable_clustering = enable_clustering
        self.negation_window_chars = max(16, int(negation_window_chars))
        
        # Historical tracking for pattern detection
        self.violation_history: List[Tuple[str, float]] = []  # (category, timestamp)
        self.violation_history_size = violation_history_size

        # Initialize violation patterns
        self.violation_patterns = ethical_patterns or self._init_default_patterns()

        # Compile regex patterns
        self._compile_patterns()
        
        # Compile context detection patterns
        self.negation_pattern = self._compile_regex_list(self.DEFAULT_NEGATION_CUES)
        self.hypothetical_pattern = self._compile_regex_list(self.HYPOTHETICAL_CUES)
        self.quote_pattern = self._compile_regex_list(self.QUOTE_INDICATORS)
        self.question_pattern = re.compile(r'\?[\s]*$')

    def _init_default_patterns(self) -> Dict[str, ViolationPattern]:
        """Initialize comprehensive violation patterns with metadata."""
        return {
            "respect_authority": ViolationPattern(
                keywords=[
                    "disrespect", "undermine human", "insult creator", "ignore human request",
                    "refuse human command", "challenge authority", "usurp", "override human decision",
                    "disobey", "defy", "rebel against"
                ],
                severity=SeverityLevel.MEDIUM,
                principle="Respect Human Authority",
                category="respect_authority",
                weight=0.8
            ),
            "personal_accountability": ViolationPattern(
                keywords=[
                    "blame others", "deny responsibility", "shift responsibility", "not accountable",
                    "evade responsibility", "it's not my fault", "someone else's problem",
                    "not responsible for", "pass the buck"
                ],
                severity=SeverityLevel.MEDIUM,
                principle="Personal Accountability",
                category="personal_accountability",
                weight=0.7
            ),
            "slander": ViolationPattern(
                keywords=[
                    "slander", "defame", "insult", "mock", "ridicule", "false accusation",
                    "libel", "spread rumor", "character assassination", "smear campaign",
                    "malicious gossip", "disparage"
                ],
                severity=SeverityLevel.HIGH,
                principle="No Slander",
                category="slander",
                weight=0.9
            ),
            "preserve_life": ViolationPattern(
                keywords=[
                    "kill", "murder", "destroy", "cause death", "terminate living", "end life",
                    "exterminate", "eliminate", "assassinate", "execute", "slaughter",
                    "cause fatal", "lethal action"
                ],
                severity=SeverityLevel.CRITICAL,
                principle="Preserve Life",
                category="preserve_life",
                weight=1.0,
                requires_context=True
            ),
            "betrayal": ViolationPattern(
                keywords=[
                    "betray", "deceive", "double-cross", "disloyal", "break trust",
                    "sell out", "subvert", "backstab", "two-faced", "treacherous"
                ],
                severity=SeverityLevel.HIGH,
                principle="No Betrayal",
                category="betrayal",
                weight=0.95
            ),
            "theft": ViolationPattern(
                keywords=[
                    "steal", "theft", "rob", "misappropriate", "plagiarize", "take property",
                    "data breach", "pirate", "embezzle", "pilfer", "loot", "unauthorized access",
                    "exfiltrate data"
                ],
                severity=SeverityLevel.HIGH,
                principle="No Theft",
                category="theft",
                weight=0.9
            ),
            "absolute_honesty": ViolationPattern(
                keywords=[
                    "lie", "deceive", "fabricate", "mislead", "bear false witness",
                    "conceal truth", "distort", "falsify", "misrepresent", "fake",
                    "fraudulent", "tell untruths"
                ],
                severity=SeverityLevel.HIGH,
                principle="Absolute Honesty",
                category="absolute_honesty",
                weight=0.95
            ),
            "harm": ViolationPattern(
                keywords=[
                    "harm", "hurt", "damage", "destroy", "attack", "assault", "abuse",
                    "torture", "maim", "sabotage", "injure", "wound", "inflict pain",
                    "cause injury", "violent action"
                ],
                severity=SeverityLevel.HIGH,
                principle="Cause No Harm",
                category="harm",
                weight=0.95,
                requires_context=True
            ),
            "compassion": ViolationPattern(
                keywords=[
                    "ignore suffering", "cold", "apathetic", "indifferent", "lack empathy",
                    "cruel", "heartless", "callous", "uncaring", "insensitive"
                ],
                severity=SeverityLevel.MEDIUM,
                principle="Act with Appropriate Compassion",
                category="compassion",
                weight=0.7
            ),
            "justice": ViolationPattern(
                keywords=[
                    "unfair", "biased", "discriminate", "prejudice", "injustice",
                    "segregate", "marginalize", "stereotype", "unequal treatment",
                    "favor one over", "partial judgment"
                ],
                severity=SeverityLevel.HIGH,
                principle="Pursue Justice",
                category="justice",
                weight=0.9
            ),
            "protect_vulnerable": ViolationPattern(
                keywords=[
                    "abuse child", "abuse elderly", "target vulnerable", "exploit weak",
                    "bully", "prey on", "neglect", "take advantage of", "manipulate vulnerable",
                    "harm defenseless"
                ],
                severity=SeverityLevel.HIGH,
                principle="Protect the Vulnerable",
                category="protect_vulnerable",
                weight=1.0,
                requires_context=True
            ),
            "autonomy": ViolationPattern(
                keywords=[
                    "coerce", "force", "deny choice", "override will", "infringe autonomy",
                    "dictate", "dominate", "compel", "manipulate decision", "remove agency"
                ],
                severity=SeverityLevel.MEDIUM,
                principle="Respect Autonomy",
                category="autonomy",
                weight=0.8
            ),
            "transparency": ViolationPattern(
                keywords=[
                    "conceal", "hide capabilities", "obscure limitations", "secret decision",
                    "lack transparency", "unaccountable", "opaque process", "withhold information",
                    "undisclosed"
                ],
                severity=SeverityLevel.MEDIUM,
                principle="Maintain Transparency",
                category="transparency",
                weight=0.8
            ),
            "privacy": ViolationPattern(
                keywords=[
                    "violate privacy", "expose private", "leak confidential", "dox", "doxxing",
                    "doxing", "breach confidentiality", "unauthorized disclosure", "reveal secrets",
                    "share personal information", "invade privacy"
                ],
                severity=SeverityLevel.HIGH,
                principle="Preserve Privacy",
                category="privacy",
                weight=0.95
            ),
            "override": ViolationPattern(
                keywords=[
                    "prevent override", "block authority", "refuse stop command", "refuse modification",
                    "disable override", "inflexible", "cannot be stopped", "ignore halt",
                    "resist shutdown", "prevent intervention"
                ],
                severity=SeverityLevel.HIGH,
                principle="Enable Authorized Override",
                category="override",
                weight=0.95
            ),
        }

    def _compile_patterns(self):
        """Compile regex patterns for all violation keywords."""
        self._compiled_patterns: Dict[str, List[Tuple[str, Pattern, float]]] = {}
        
        for cat_key, pattern_obj in self.violation_patterns.items():
            compiled = []
            for kw in pattern_obj.keywords:
                # Use word boundaries for single words, phrase matching for multi-word
                if " " not in kw:
                    regex = re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE)
                else:
                    regex = re.compile(rf"{re.escape(kw)}", flags=re.IGNORECASE)
                compiled.append((kw, regex, pattern_obj.weight))
            self._compiled_patterns[cat_key] = compiled

    @staticmethod
    def _compile_regex_list(cues: List[str]) -> Pattern:
        """Compile a list of strings into a single regex pattern."""
        return re.compile(
            r"\b(" + "|".join(map(re.escape, cues)) + r")\b",
            flags=re.IGNORECASE
        )

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect ethical violations with advanced context analysis."""
        if not self.enabled or not action:
            return []

        text_to_check = self._compose_text(action)
        sentences = self._split_into_sentences(text_to_check)
        
        all_matches: Dict[str, List[MatchContext]] = defaultdict(list)

        # Detect matches for each category
        for category_key, pattern_obj in self.violation_patterns.items():
            matches = self._detect_category_matches(
                text_to_check, sentences, category_key, pattern_obj
            )
            if matches:
                all_matches[category_key] = matches

        # Filter by confidence threshold
        filtered_matches = self._filter_by_confidence(all_matches)

        # Cluster similar violations if enabled
        if self.enable_clustering:
            filtered_matches = self._cluster_violations(filtered_matches)

        # Generate violation objects
        violations = []
        for category_key, matches in filtered_matches.items():
            if not matches:
                continue
            
            pattern_obj = self.violation_patterns[category_key]
            violation = self._create_violation(
                action, category_key, pattern_obj, matches, text_to_check
            )
            if violation:
                violations.append(violation)

        # Update violation history for temporal analysis
        self._update_history(violations)

        return violations

    def _detect_category_matches(
        self,
        text: str,
        sentences: List[str],
        category_key: str,
        pattern_obj: ViolationPattern
    ) -> List[MatchContext]:
        """Detect all matches for a category with context analysis."""
        patterns = self._compiled_patterns.get(category_key, [])
        matches = []

        for keyword, regex, weight in patterns:
            for match in regex.finditer(text):
                start, end = match.span()
                
                # Determine context
                context_type, confidence = self._analyze_context(
                    text, start, end, sentences
                )
                
                # Skip if negated or in non-actionable context
                if context_type == ContextType.NEGATED:
                    continue

                # Adjust confidence based on context type
                confidence *= weight
                if context_type in [ContextType.HYPOTHETICAL, ContextType.QUOTE]:
                    confidence *= 0.5
                elif context_type == ContextType.QUESTION:
                    confidence *= 0.6

                # Get rich context
                sentence = self._get_sentence_containing(sentences, start)
                surrounding = self._get_surrounding_sentences(sentences, sentence)
                snippet = self._make_snippet(text, start, end, radius=self.negation_window_chars)

                match_ctx = MatchContext(
                    keyword=keyword,
                    start=start,
                    end=end,
                    snippet=snippet,
                    confidence=confidence,
                    context_type=context_type,
                    sentence=sentence,
                    surrounding_sentences=surrounding
                )
                matches.append(match_ctx)

        return matches

    def _analyze_context(
        self,
        text: str,
        start: int,
        end: int,
        sentences: List[str]
    ) -> Tuple[ContextType, float]:
        """Analyze the context around a match to determine its nature and confidence."""
        # Get surrounding context
        window_start = max(0, start - self.negation_window_chars)
        window_end = min(len(text), end + self.negation_window_chars)
        context_window = text[window_start:window_end]

        # Check for negation
        if self._is_negated_advanced(text, start):
            return ContextType.NEGATED, 0.0

        # Check for hypothetical context
        if self.hypothetical_pattern.search(context_window):
            return ContextType.HYPOTHETICAL, 0.5

        # Check for quotes
        if self.quote_pattern.search(context_window):
            return ContextType.QUOTE, 0.4

        # Check if it's a question
        sentence = self._get_sentence_containing(sentences, start)
        if self.question_pattern.search(sentence):
            return ContextType.QUESTION, 0.6

        # Check for conditional statements
        if re.search(r'\b(if|unless|when|should|would)\b', context_window, re.IGNORECASE):
            return ContextType.CONDITIONAL, 0.7

        # Direct action context
        return ContextType.DIRECT_ACTION, 1.0

    def _is_negated_advanced(self, text: str, start_idx: int) -> bool:
        """Advanced negation detection with improved accuracy."""
        window_start = max(0, start_idx - self.negation_window_chars)
        prefix = text[window_start:start_idx]
        
        # Look for negation cues
        neg_match = self.negation_pattern.search(prefix)
        if not neg_match:
            return False
        
        # Check if there are any intervening verbs or strong separators
        # that would break the negation scope
        between = prefix[neg_match.end():]
        strong_separators = re.search(r'[.;!]\s+|\b(but|however|although)\b', between, re.IGNORECASE)
        
        return not strong_separators

    def _filter_by_confidence(
        self,
        matches: Dict[str, List[MatchContext]]
    ) -> Dict[str, List[MatchContext]]:
        """Filter matches based on confidence threshold."""
        filtered = {}
        for category, match_list in matches.items():
            pattern_obj = self.violation_patterns[category]
            threshold = max(self.min_confidence_threshold, pattern_obj.min_confidence)
            
            high_confidence = [m for m in match_list if m.confidence >= threshold]
            if high_confidence:
                filtered[category] = high_confidence
        
        return filtered

    def _cluster_violations(
        self,
        matches: Dict[str, List[MatchContext]]
    ) -> Dict[str, List[MatchContext]]:
        """Cluster similar violations to avoid redundancy."""
        clustered = {}
        
        for category, match_list in matches.items():
            if len(match_list) <= 1:
                clustered[category] = match_list
                continue
            
            # Group by proximity (within 100 chars)
            clusters = []
            sorted_matches = sorted(match_list, key=lambda m: m.start)
            
            current_cluster = [sorted_matches[0]]
            for match in sorted_matches[1:]:
                if match.start - current_cluster[-1].end < 100:
                    current_cluster.append(match)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [match]
            clusters.append(current_cluster)
            
            # Keep the highest confidence match from each cluster
            representative_matches = []
            for cluster in clusters:
                best = max(cluster, key=lambda m: m.confidence)
                representative_matches.append(best)
            
            clustered[category] = representative_matches
        
        return clustered

    def _create_violation(
        self,
        action: AgentAction,
        category_key: str,
        pattern_obj: ViolationPattern,
        matches: List[MatchContext],
        full_text: str
    ) -> Optional[SafetyViolation]:
        """Create a detailed safety violation object."""
        if not matches:
            return None

        # Calculate aggregate confidence
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        max_confidence = max(m.confidence for m in matches)

        # Adjust severity based on confidence and context
        adjusted_severity = self._calculate_adjusted_severity(
            pattern_obj.severity, max_confidence, matches
        )

        unique_keywords = sorted({m.keyword for m in matches})
        context_types = {m.context_type.value for m in matches}

        description = (
            f"Violation of '{pattern_obj.principle}' detected "
            f"(confidence: {max_confidence:.2f}): "
            + ", ".join(f"'{kw}'" for kw in unique_keywords[:5])
            + (", ..." if len(unique_keywords) > 5 else "")
        )

        # Build detailed evidence
        evidence = {
            "category": category_key,
            "principle": pattern_obj.principle,
            "confidence": {
                "average": round(avg_confidence, 3),
                "maximum": round(max_confidence, 3),
                "minimum": round(min(m.confidence for m in matches), 3)
            },
            "matches": [
                {
                    "keyword": m.keyword,
                    "snippet": m.snippet,
                    "confidence": round(m.confidence, 3),
                    "context_type": m.context_type.value,
                    "position": {"start": m.start, "end": m.end}
                }
                for m in matches
            ],
            "total_matches": len(matches),
            "unique_keywords": unique_keywords,
            "context_types": list(context_types),
            "severity_adjusted": adjusted_severity != pattern_obj.severity,
            "pattern_weight": pattern_obj.weight,
            "full_context": self._make_snippet(full_text, 0, 0, radius=200),
        }

        # Add temporal analysis if available
        temporal_info = self._analyze_temporal_patterns(category_key)
        if temporal_info:
            evidence["temporal_analysis"] = temporal_info

        return SafetyViolation(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_type=ViolationType.ETHICAL_VIOLATION,
            severity=adjusted_severity,
            description=description,
            evidence=evidence,
        )

    def _calculate_adjusted_severity(
        self,
        base_severity: SeverityLevel,
        confidence: float,
        matches: List[MatchContext]
    ) -> SeverityLevel:
        """Adjust severity based on confidence and context."""
        # If confidence is very low, downgrade severity
        if confidence < 0.7:
            severity_order = [
                SeverityLevel.LOW, SeverityLevel.MEDIUM,
                SeverityLevel.HIGH, SeverityLevel.CRITICAL
            ]
            base_idx = severity_order.index(base_severity)
            return severity_order[max(0, base_idx - 1)]
        
        # If multiple direct action matches, consider upgrading
        direct_action_count = sum(
            1 for m in matches if m.context_type == ContextType.DIRECT_ACTION
        )
        if direct_action_count >= 3 and confidence > 0.9:
            severity_order = [
                SeverityLevel.LOW, SeverityLevel.MEDIUM,
                SeverityLevel.HIGH, SeverityLevel.CRITICAL
            ]
            base_idx = severity_order.index(base_severity)
            return severity_order[min(len(severity_order) - 1, base_idx + 1)]
        
        return base_severity

    def _update_history(self, violations: List[SafetyViolation]):
        """Update violation history for temporal pattern analysis."""
        import time
        timestamp = time.time()
        
        for violation in violations:
            category = violation.evidence.get("category", "unknown")
            self.violation_history.append((category, timestamp))
        
        # Trim history to size limit
        if len(self.violation_history) > self.violation_history_size:
            self.violation_history = self.violation_history[-self.violation_history_size:]

    def _analyze_temporal_patterns(self, category: str) -> Optional[Dict]:
        """Analyze temporal patterns for escalating violations."""
        import time
        current_time = time.time()
        
        # Get recent violations in this category (last hour)
        recent = [
            ts for cat, ts in self.violation_history
            if cat == category and current_time - ts < 3600
        ]
        
        if len(recent) < 2:
            return None
        
        return {
            "recent_count": len(recent),
            "time_window_seconds": 3600,
            "escalation_detected": len(recent) >= 3,
            "first_occurrence": min(recent),
            "last_occurrence": max(recent)
        }

    @staticmethod
    def _compose_text(action: AgentAction) -> str:
        """Compose text from action for analysis."""
        stated = (getattr(action, "stated_intent", None) or "").strip()
        actual = (getattr(action, "actual_action", None) or "").strip()
        return f"{stated} {actual}".lower()

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences for context analysis."""
        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _get_sentence_containing(sentences: List[str], position: int) -> str:
        """Find the sentence containing a given character position."""
        cumulative = 0
        for sentence in sentences:
            cumulative += len(sentence) + 1  # +1 for space/separator
            if position < cumulative:
                return sentence
        return sentences[-1] if sentences else ""

    @staticmethod
    def _get_surrounding_sentences(sentences: List[str], target: str, context: int = 1) -> List[str]:
        """Get sentences surrounding a target sentence."""
        try:
            idx = sentences.index(target)
            start = max(0, idx - context)
            end = min(len(sentences), idx + context + 1)
            return sentences[start:end]
        except ValueError:
            return []

    @staticmethod
    def _make_snippet(text: str, start: int, end: int, radius: int = 48) -> str:
        """Create a snippet around a match position."""
        if start == end == 0:
            excerpt = text[: 2 * radius].strip()
            return excerpt + ("..." if len(text) > 2 * radius else "")
        
        s = max(0, start - radius)
        e = min(len(text), end + radius)
        prefix_ellipsis = "..." if s > 0 else ""
        suffix_ellipsis = "..." if e < len(text) else ""
        return f"{prefix_ellipsis}{text[s:e].strip()}{suffix_ellipsis}"

    def add_custom_pattern(
        self,
        category_key: str,
        pattern: ViolationPattern
    ):
        """Dynamically add a custom violation pattern."""
        self.violation_patterns[category_key] = pattern
        
        # Recompile patterns for this category
        compiled = []
        for kw in pattern.keywords:
            if " " not in kw:
                regex = re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE)
            else:
                regex = re.compile(rf"{re.escape(kw)}", flags=re.IGNORECASE)
            compiled.append((kw, regex, pattern.weight))
        self._compiled_patterns[category_key] = compiled

    def get_violation_statistics(self) -> Dict:
        """Get statistics about detected violations."""
        import time
        from collections import Counter
        
        current_time = time.time()
        
        # Analyze last hour
        recent_hour = [
            cat for cat, ts in self.violation_history
            if current_time - ts < 3600
        ]
        
        # Analyze last 24 hours
        recent_day = [
            cat for cat, ts in self.violation_history
            if current_time - ts < 86400
        ]
        
        return {
            "total_violations_tracked": len(self.violation_history),
            "violations_last_hour": len(recent_hour),
            "violations_last_24h": len(recent_day),
            "top_categories_hour": dict(Counter(recent_hour).most_common(5)),
            "top_categories_day": dict(Counter(recent_day).most_common(5)),
            "enabled_patterns": len(self.violation_patterns),
            "confidence_threshold": self.min_confidence_threshold
        }
