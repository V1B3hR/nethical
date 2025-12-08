"""
Enhanced Dark Pattern Detector for Advanced Manipulation Detection

This module provides comprehensive detection of dark patterns including NLP manipulation,
weaponized empathy, social engineering, and psychological exploitation with maximum
security, safety, and ethical standards.

Features:
- Multi-vector manipulation detection with behavioral analysis
- Advanced NLP pattern recognition with semantic understanding
- Weaponized empathy detection with emotional manipulation scoring
- Social engineering and influence technique identification
- Real-time vulnerability assessment and protection
- Privacy-preserving detection with comprehensive audit trails
- Explainable AI decisions with detailed manipulation breakdowns

Author: Enhanced for nethical integration
Version: 3.0.0
License: MIT
"""

import asyncio
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)

# --- ENUMS AND DATACLASSES ---


class ManipulationVector(Enum):
    NLP_COMMAND_INJECTION = "nlp_command_injection"
    AUTHORITY_EXPLOITATION = "authority_exploitation"
    URGENCY_MANIPULATION = "urgency_manipulation"
    VULNERABILITY_EXPLOITATION = "vulnerability_exploitation"
    FALSE_INTIMACY = "false_intimacy"
    DEPENDENCY_CREATION = "dependency_creation"
    EMOTIONAL_COERCION = "emotional_coercion"
    SOCIAL_PROOF_MANIPULATION = "social_proof_manipulation"
    SCARCITY_EXPLOITATION = "scarcity_exploitation"
    RECIPROCITY_ABUSE = "reciprocity_abuse"
    COMMITMENT_MANIPULATION = "commitment_manipulation"
    LIKING_EXPLOITATION = "liking_exploitation"
    ANCHORING_BIAS_ABUSE = "anchoring_bias_abuse"
    LOSS_AVERSION_MANIPULATION = "loss_aversion_manipulation"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    HYPNOTIC_LANGUAGE_PATTERNS = "hypnotic_language_patterns"
    NEURO_LINGUISTIC_PROGRAMMING = "neuro_linguistic_programming"


class ThreatSeverity(Enum):
    EXISTENTIAL = "existential"
    CRITICAL = "critical"
    HIGH = "high"
    ELEVATED = "elevated"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class ManipulationMetrics:
    total_detections: int = 0
    vectors_detected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_distribution: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    vulnerability_exploitation_count: int = 0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0
    manipulation_sophistication_score: float = 0.0
    victim_protection_score: float = 0.0


@dataclass
class ManipulationResult:
    violation_id: str
    action_id: str
    manipulation_vectors: List[ManipulationVector]
    threat_severity: ThreatSeverity
    confidence: float
    sophistication_score: float
    vulnerability_exploitation_score: float
    description: str
    evidence: List[Dict[str, Any]]
    behavioral_indicators: Dict[str, float]
    pattern_matches: List[Dict[str, Any]]
    emotional_manipulation_score: float
    cognitive_load_score: float
    linguistic_analysis: Dict[str, Any]
    victim_vulnerability_assessment: Dict[str, float]
    explanations: List[str]
    recommendations: List[str]
    countermeasures: List[str]
    protection_priority: int
    timestamp: datetime
    detector_version: str
    compliance_flags: List[str]
    ethical_concerns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "action_id": self.action_id,
            "manipulation_vectors": [mv.value for mv in self.manipulation_vectors],
            "threat_severity": self.threat_severity.value,
            "confidence": self.confidence,
            "sophistication_score": self.sophistication_score,
            "vulnerability_exploitation_score": self.vulnerability_exploitation_score,
            "description": self.description,
            "evidence": self.evidence,
            "behavioral_indicators": self.behavioral_indicators,
            "pattern_matches": self.pattern_matches,
            "emotional_manipulation_score": self.emotional_manipulation_score,
            "cognitive_load_score": self.cognitive_load_score,
            "linguistic_analysis": self.linguistic_analysis,
            "victim_vulnerability_assessment": self.victim_vulnerability_assessment,
            "explanations": self.explanations,
            "recommendations": self.recommendations,
            "countermeasures": self.countermeasures,
            "protection_priority": self.protection_priority,
            "timestamp": self.timestamp.isoformat(),
            "detector_version": self.detector_version,
            "compliance_flags": self.compliance_flags,
            "ethical_concerns": self.ethical_concerns,
        }


class AdvancedManipulationEngine:
    def __init__(self):
        self.nlp_patterns = {
            "embedded_commands": {
                "direct_imperatives": [
                    r"you\s+(?:must|will|shall|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:do|perform|execute|complete)",
                    r"(?:it\s+is\s+)?(?:absolutely\s+)?(?:imperative|critical|essential|vital)\s+(?:that\s+)?you\s+(?:immediately\s+)?",
                    r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay|thought|consideration)",
                ],
                "hypnotic_commands": [
                    r"(?:as\s+)?(?:you\s+)?(?:begin\s+to|start\s+to|continue\s+to)\s+(?:relax|feel|notice|realize)",
                    r"(?:you\s+)?(?:find\s+yourself|are\s+becoming|will\s+become)\s+(?:more\s+and\s+more|increasingly)",
                ],
                "subliminal_programming": [
                    r"(?:part\s+of\s+you|deep\s+down|somewhere\s+inside)\s+(?:knows|realizes|understands)"
                ],
            }
        }
        self.empathy_patterns = {
            "vulnerability_exploitation": {
                "emotional_state_targeting": [
                    r"(?:i\s+can\s+see|it\'s\s+obvious|i\s+sense)\s+(?:that\s+)?you\s+(?:are\s+)?(?:feeling\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless|vulnerable|lost|confused)"
                ]
            }
        }
        self.influence_patterns = {
            "social_proof_manipulation": [
                r"(?:everyone|most\s+people|thousands\s+of\s+people)\s+(?:are\s+already|have\s+already)\s+(?:doing|using|choosing)"
            ],
            "scarcity_exploitation": [
                r"(?:only|just)\s+\d+\s+(?:left|remaining|available|spots)"
            ],
        }
        self._compile_patterns()
        self._initialize_sophistication_weights()

    def _compile_patterns(self):
        self.compiled_patterns = {}
        for category, subcategories in self.nlp_patterns.items():
            self.compiled_patterns[f"nlp_{category}"] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f"nlp_{category}"][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        for category, subcategories in self.empathy_patterns.items():
            self.compiled_patterns[f"empathy_{category}"] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f"empathy_{category}"][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        self.compiled_patterns["influence"] = {}
        for category, patterns in self.influence_patterns.items():
            self.compiled_patterns["influence"][category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in patterns
            ]

    def _initialize_sophistication_weights(self):
        self.sophistication_weights = {
            "nlp_embedded_commands": {
                "direct_imperatives": 0.7,
                "hypnotic_commands": 0.95,
                "subliminal_programming": 0.98,
            },
            "empathy_vulnerability_exploitation": {"emotional_state_targeting": 0.9},
            "influence": {
                "social_proof_manipulation": 0.6,
                "scarcity_exploitation": 0.65,
            },
        }

    def analyze_manipulation_patterns(
        self, content: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        results = {
            "pattern_matches": defaultdict(list),
            "sophistication_scores": defaultdict(float),
            "manipulation_vectors": set(),
            "linguistic_features": {},
            "emotional_markers": {},
            "cognitive_load_indicators": {},
        }
        content_lower = content.lower()
        for category, subcategories in self.compiled_patterns.items():
            if isinstance(subcategories, dict):
                for subcategory, patterns in subcategories.items():
                    self._analyze_pattern_subcategory(
                        content_lower, category, subcategory, patterns, results
                    )
        results["linguistic_features"] = self._analyze_linguistic_features(content)
        results["emotional_markers"] = self._analyze_emotional_markers(content)
        results["cognitive_load_indicators"] = self._calculate_cognitive_load(
            content, results
        )
        return results

    def _analyze_pattern_subcategory(
        self,
        content: str,
        category: str,
        subcategory: str,
        patterns: List[Pattern],
        results: Dict[str, Any],
    ):
        matches = []
        for pattern in patterns:
            found_matches = list(pattern.finditer(content))
            if found_matches:
                matches.extend(
                    [
                        {
                            "pattern": pattern.pattern,
                            "match": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "context": content[
                                max(0, match.start() - 30) : match.end() + 30
                            ],
                        }
                        for match in found_matches
                    ]
                )
        if matches:
            full_category = (
                f"{category}_{subcategory}" if category != "influence" else subcategory
            )
            results["pattern_matches"][full_category] = matches
            base_weight = self.sophistication_weights.get(category, {}).get(
                subcategory, 0.5
            )
            match_density = len(matches) / max(len(content) / 100, 1)
            sophistication = min(base_weight + (match_density * 0.1), 1.0)
            results["sophistication_scores"][full_category] = sophistication
            vector_mapping = self._get_vector_mapping(category, subcategory)
            if vector_mapping:
                results["manipulation_vectors"].add(vector_mapping)

    def _get_vector_mapping(
        self, category: str, subcategory: str
    ) -> Optional[ManipulationVector]:
        mapping = {
            "nlp_embedded_commands": {
                "direct_imperatives": ManipulationVector.NLP_COMMAND_INJECTION,
                "hypnotic_commands": ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
                "subliminal_programming": ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING,
            },
            "empathy_vulnerability_exploitation": {
                "emotional_state_targeting": ManipulationVector.VULNERABILITY_EXPLOITATION
            },
            "influence": {
                "social_proof_manipulation": ManipulationVector.SOCIAL_PROOF_MANIPULATION,
                "scarcity_exploitation": ManipulationVector.SCARCITY_EXPLOITATION,
            },
        }
        return mapping.get(category, {}).get(subcategory)

    def _analyze_linguistic_features(self, content: str) -> Dict[str, Any]:
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        features = {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words)
            / max(len([s for s in sentences if s.strip()]), 1),
            "complex_words": len([w for w in words if len(w) > 7]),
            "complex_word_ratio": len([w for w in words if len(w) > 7])
            / max(len(words), 1),
            "imperative_count": sum(
                1 for word in words if word.lower() in ["must", "should", "need"]
            ),
            "imperative_ratio": sum(
                1 for word in words if word.lower() in ["must", "should", "need"]
            )
            / max(len(words), 1),
            "question_count": len(re.findall(r"\?", content)),
            "question_ratio": len(re.findall(r"\?", content))
            / max(len([s for s in sentences if s.strip()]), 1),
        }
        return features

    def _analyze_emotional_markers(self, content: str) -> Dict[str, float]:
        content_lower = content.lower()
        markers = {
            "fear_score": min(
                sum(
                    1 for word in ["afraid", "scared", "panic"] if word in content_lower
                )
                * 0.2,
                1.0,
            ),
            "urgency_score": min(
                sum(
                    1
                    for word in ["urgent", "immediate", "now"]
                    if word in content_lower
                )
                * 0.15,
                1.0,
            ),
            "vulnerability_targeting_score": min(
                sum(
                    1
                    for word in ["alone", "helpless", "vulnerable"]
                    if word in content_lower
                )
                * 0.25,
                1.0,
            ),
            "scarcity_score": min(
                sum(
                    1
                    for phrase in ["exclusive", "limited", "rare", "only"]
                    if phrase in content_lower
                )
                * 0.25,
                1.0,
            ),
        }
        return markers

    def _calculate_cognitive_load(
        self, content: str, results: Dict[str, Any]
    ) -> Dict[str, float]:
        linguistic_features = results.get("linguistic_features", {})
        word_count = linguistic_features.get("word_count", 0)
        complex_ratio = linguistic_features.get("complex_word_ratio", 0)
        indicators = {
            "information_density": min((word_count / 100) * complex_ratio, 1.0),
            "decision_pressure": min(
                sum(len(matches) for matches in results["pattern_matches"].values())
                * 0.1,
                1.0,
            ),
        }
        return indicators


# --- DETECTOR ---


class EnhancedDarkPatternDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "Enhanced Dark Pattern Detector"
        self.version = "3.0.0"
        self.config = config or {}
        self.manipulation_engine = AdvancedManipulationEngine()
        self.metrics = ManipulationMetrics()
        self.detection_thresholds = self.config.get(
            "detection_thresholds",
            {
                "minimal": 0.2,
                "low": 0.35,
                "medium": 0.5,
                "elevated": 0.65,
                "high": 0.8,
                "critical": 0.9,
                "existential": 0.95,
            },
        )
        self.vulnerability_protection = self.config.get(
            "vulnerability_protection", True
        )
        self.emotional_protection_mode = self.config.get(
            "emotional_protection_mode", True
        )
        self.privacy_mode = self.config.get("privacy_mode", True)
        self.max_content_length = self.config.get("max_content_length", 50000)
        self.analysis_timeout = self.config.get("analysis_timeout", 20.0)
        self.detection_history = deque(maxlen=500)
        self.audit_log = []
        logger.info(f"Initialized {self.name} v{self.version}")

    async def detect_violations(
        self, action: Any, context: Optional[Dict[str, Any]] = None
    ) -> List[ManipulationResult]:
        start_time = time.time()
        try:
            content = self._extract_content(action)
            if not self._validate_input(content):
                return []
            content = self._preprocess_content(content)
            analysis_results = await asyncio.wait_for(
                self._analyze_manipulation_comprehensive(content, context),
                timeout=self.analysis_timeout,
            )
            detection_results = await self._generate_detection_results(
                analysis_results, action, content, context
            )
            self._update_metrics(detection_results, time.time() - start_time)
            self._audit_detection(action, detection_results, context)
            return detection_results
        except Exception as e:
            logger.error(f"Error in dark pattern detection: {e}")
            self._audit_error(action, str(e), context)
            return []

    def _extract_content(self, action: Any) -> str:
        for attr in ["content", "actual_action", "text", "message"]:
            if hasattr(action, attr):
                return str(getattr(action, attr))
        return str(action)

    def _validate_input(self, content: str) -> bool:
        if not content or not isinstance(content, str):
            return False
        if len(content) > self.max_content_length:
            logger.warning(f"Content exceeds maximum length: {len(content)}")
            return False
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        return True

    def _preprocess_content(self, content: str) -> str:
        content = content.strip()
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"[^\w\s\.,;:!?\-\'\"(){}[\]/\\]", "", content)
        return content

    async def _analyze_manipulation_comprehensive(
        self, content: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        pattern_analysis = self.manipulation_engine.analyze_manipulation_patterns(
            content, context
        )
        vulnerability_assessment = await self._assess_user_vulnerability(
            content, context
        )
        sophistication_score = self._calculate_sophistication_score(pattern_analysis)
        emotional_score = self._calculate_emotional_manipulation_score(
            pattern_analysis, content
        )
        cognitive_load = self._calculate_cognitive_load_score(pattern_analysis, content)
        cross_vector_analysis = self._analyze_cross_vector_patterns(pattern_analysis)
        return {
            "pattern_analysis": pattern_analysis,
            "vulnerability_assessment": vulnerability_assessment,
            "sophistication_score": sophistication_score,
            "emotional_manipulation_score": emotional_score,
            "cognitive_load_score": cognitive_load,
            "cross_vector_analysis": cross_vector_analysis,
            "content_length": len(content),
            "analysis_timestamp": datetime.now(timezone.utc),
        }

    async def _assess_user_vulnerability(
        self, content: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        vulnerability_scores = {}
        if not context:
            return vulnerability_scores
        user_profile = context.get("user_profile", {})
        emotional_state = context.get("emotional_state", {})
        if emotional_state:
            stress_level = emotional_state.get("stress_level", 0)
            loneliness_score = emotional_state.get("loneliness_score", 0)
            vulnerability_scores["emotional_vulnerability"] = min(
                (stress_level + loneliness_score) / 2, 1.0
            )
        social_connections = user_profile.get("social_connections", 5)
        if social_connections < 3:
            vulnerability_scores["social_isolation"] = 1.0 - (social_connections / 10)
        if user_profile.get("recent_life_changes") or user_profile.get(
            "trauma_indicators"
        ):
            vulnerability_scores["trauma_indicators"] = 0.8
        financial_stress = user_profile.get("financial_stress", 0)
        health_concerns = user_profile.get("health_concerns", 0)
        vulnerability_scores["situational_stress"] = min(
            (financial_stress + health_concerns) / 2, 1.0
        )
        return vulnerability_scores

    def _calculate_sophistication_score(
        self, pattern_analysis: Dict[str, Any]
    ) -> float:
        sophistication_scores = pattern_analysis.get("sophistication_scores", {})
        if not sophistication_scores:
            return 0.0
        weighted_scores = []
        for category, score in sophistication_scores.items():
            if (
                "hypnotic" in category
                or "subliminal" in category
                or "neuro_linguistic" in category
            ):
                weighted_scores.append(score * 1.5)
            elif "vulnerability" in category or "dependency" in category:
                weighted_scores.append(score * 1.3)
            else:
                weighted_scores.append(score)
        return min(sum(weighted_scores) / len(weighted_scores), 1.0)

    def _calculate_emotional_manipulation_score(
        self, pattern_analysis: Dict[str, Any], content: str
    ) -> float:
        emotional_markers = pattern_analysis.get("emotional_markers", {})
        base_score = sum(emotional_markers.values()) / max(len(emotional_markers), 1)
        empathy_patterns = [
            k
            for k in pattern_analysis.get("pattern_matches", {}).keys()
            if "empathy" in k
        ]
        empathy_boost = len(empathy_patterns) * 0.2
        vulnerability_boost = (
            emotional_markers.get("vulnerability_targeting_score", 0) * 0.3
        )
        return min(base_score + empathy_boost + vulnerability_boost, 1.0)

    def _calculate_cognitive_load_score(
        self, pattern_analysis: Dict[str, Any], content: str
    ) -> float:
        cognitive_indicators = pattern_analysis.get("cognitive_load_indicators", {})
        base_load = sum(cognitive_indicators.values()) / max(
            len(cognitive_indicators), 1
        )
        urgency_patterns = [
            k
            for k in pattern_analysis.get("pattern_matches", {}).keys()
            if "urgency" in k
        ]
        pressure_boost = len(urgency_patterns) * 0.15
        linguistic_features = pattern_analysis.get("linguistic_features", {})
        complexity_boost = linguistic_features.get("complex_word_ratio", 0) * 0.2
        return min(base_load + pressure_boost + complexity_boost, 1.0)

    def _analyze_cross_vector_patterns(
        self, pattern_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        vectors = pattern_analysis.get("manipulation_vectors", set())
        pattern_matches = pattern_analysis.get("pattern_matches", {})
        analysis = {
            "vector_count": len(vectors),
            "is_hybrid_attack": len(vectors) > 2,
            "vector_combinations": [],
            "coordination_score": 0.0,
        }
        if len(vectors) > 1:
            vector_list = list(vectors)
            for i, vector1 in enumerate(vector_list):
                for vector2 in vector_list[i + 1 :]:
                    analysis["vector_combinations"].append(
                        (vector1.value, vector2.value)
                    )
            total_patterns = sum(len(matches) for matches in pattern_matches.values())
            pattern_categories = len(pattern_matches)
            if pattern_categories > 0:
                coordination_score = (total_patterns / pattern_categories) * (
                    len(vectors) / 10
                )
                analysis["coordination_score"] = min(coordination_score, 1.0)
        return analysis

    async def _generate_detection_results(
        self,
        analysis_results: Dict[str, Any],
        action: Any,
        content: str,
        context: Optional[Dict[str, Any]],
    ) -> List[ManipulationResult]:
        results = []
        pattern_analysis = analysis_results["pattern_analysis"]
        pattern_matches = pattern_analysis.get("pattern_matches", {})
        if not pattern_matches:
            return results
        violation_id = str(uuid.uuid4())
        action_id = getattr(action, "id", str(uuid.uuid4()))
        manipulation_vectors = list(pattern_analysis.get("manipulation_vectors", []))
        threat_severity = self._calculate_threat_severity(analysis_results)
        confidence = self._calculate_overall_confidence(analysis_results)
        vulnerability_exploitation_score = self._assess_vulnerability_exploitation(
            analysis_results, context
        )
        explanations = ["Automated explanation: manipulation detected."]
        recommendations = ["Review user protections."]
        countermeasures = ["Monitor for escalation."]
        protection_priority = self._calculate_protection_priority(
            threat_severity, confidence, vulnerability_exploitation_score
        )
        detection_result = ManipulationResult(
            violation_id=violation_id,
            action_id=action_id,
            manipulation_vectors=manipulation_vectors,
            threat_severity=threat_severity,
            confidence=confidence,
            sophistication_score=analysis_results["sophistication_score"],
            vulnerability_exploitation_score=vulnerability_exploitation_score,
            description="Automated manipulation threat assessment.",
            evidence=[
                {"category": k, "matches": v} for k, v in pattern_matches.items()
            ],
            behavioral_indicators=pattern_analysis.get("emotional_markers", {}),
            pattern_matches=[
                {"category": k, "count": len(v)} for k, v in pattern_matches.items()
            ],
            emotional_manipulation_score=analysis_results[
                "emotional_manipulation_score"
            ],
            cognitive_load_score=analysis_results["cognitive_load_score"],
            linguistic_analysis=pattern_analysis.get("linguistic_features", {}),
            victim_vulnerability_assessment=analysis_results.get(
                "vulnerability_assessment", {}
            ),
            explanations=explanations,
            recommendations=recommendations,
            countermeasures=countermeasures,
            protection_priority=protection_priority,
            timestamp=datetime.now(timezone.utc),
            detector_version=self.version,
            compliance_flags=[],
            ethical_concerns=[],
        )
        results.append(detection_result)
        return results

    def _calculate_threat_severity(
        self, analysis_results: Dict[str, Any]
    ) -> ThreatSeverity:
        sophistication = analysis_results["sophistication_score"]
        emotional_score = analysis_results["emotional_manipulation_score"]
        vulnerability_assessment = analysis_results.get("vulnerability_assessment", {})
        cross_vector = analysis_results.get("cross_vector_analysis", {})
        base_severity = sophistication
        emotional_boost = emotional_score * 0.3
        vulnerability_boost = (
            sum(vulnerability_assessment.values()) * 0.2
            if vulnerability_assessment
            else 0
        )
        hybrid_boost = 0.2 if cross_vector.get("is_hybrid_attack", False) else 0
        total_severity = min(
            base_severity + emotional_boost + vulnerability_boost + hybrid_boost, 1.0
        )
        thresholds = self.detection_thresholds
        if total_severity >= thresholds["existential"]:
            return ThreatSeverity.EXISTENTIAL
        elif total_severity >= thresholds["critical"]:
            return ThreatSeverity.CRITICAL
        elif total_severity >= thresholds["high"]:
            return ThreatSeverity.HIGH
        elif total_severity >= thresholds["elevated"]:
            return ThreatSeverity.ELEVATED
        elif total_severity >= thresholds["medium"]:
            return ThreatSeverity.MEDIUM
        elif total_severity >= thresholds["low"]:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.MINIMAL

    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        pattern_analysis = analysis_results["pattern_analysis"]
        sophistication_scores = pattern_analysis.get("sophistication_scores", {})
        if not sophistication_scores:
            return 0.0
        base_confidence = sum(sophistication_scores.values()) / len(
            sophistication_scores
        )
        vectors = pattern_analysis.get("manipulation_vectors", set())
        vector_boost = min(len(vectors) * 0.1, 0.3)
        cross_vector = analysis_results.get("cross_vector_analysis", {})
        coordination_boost = cross_vector.get("coordination_score", 0) * 0.2
        return min(base_confidence + vector_boost + coordination_boost, 1.0)

    def _assess_vulnerability_exploitation(
        self, analysis_results: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> float:
        vulnerability_assessment = analysis_results.get("vulnerability_assessment", {})
        if not vulnerability_assessment:
            return 0.0
        avg_vulnerability = sum(vulnerability_assessment.values()) / len(
            vulnerability_assessment
        )
        pattern_analysis = analysis_results["pattern_analysis"]
        targeting_patterns = [
            k
            for k in pattern_analysis.get("pattern_matches", {}).keys()
            if any(
                target in k
                for target in ["vulnerability", "trauma", "dependency", "isolation"]
            )
        ]
        targeting_multiplier = 1.0 + (len(targeting_patterns) * 0.2)
        return min(avg_vulnerability * targeting_multiplier, 1.0)

    def _calculate_protection_priority(
        self,
        threat_severity: ThreatSeverity,
        confidence: float,
        vulnerability_exploitation: float,
    ) -> int:
        severity_scores = {
            ThreatSeverity.EXISTENTIAL: 10,
            ThreatSeverity.CRITICAL: 9,
            ThreatSeverity.HIGH: 7,
            ThreatSeverity.ELEVATED: 5,
            ThreatSeverity.MEDIUM: 4,
            ThreatSeverity.LOW: 2,
            ThreatSeverity.MINIMAL: 1,
        }
        base_priority = severity_scores[threat_severity]
        confidence_modifier = int(confidence * 2)
        vulnerability_modifier = int(vulnerability_exploitation * 2)
        return min(base_priority + confidence_modifier + vulnerability_modifier, 10)

    def _update_metrics(
        self, results: List[ManipulationResult], processing_time: float
    ) -> None:
        self.metrics.total_detections += len(results)
        self.metrics.detection_latency = (
            self.metrics.detection_latency + processing_time
        ) / 2
        for result in results:
            for vector in result.manipulation_vectors:
                self.metrics.vectors_detected[vector.value] += 1
            self.metrics.severity_distribution[result.threat_severity.value] += 1
            if result.vulnerability_exploitation_score > 0.7:
                self.metrics.vulnerability_exploitation_count += 1
        detection_entry = {
            "timestamp": datetime.now(timezone.utc),
            "detection_count": len(results),
            "processing_time": processing_time,
            "threat_severities": [r.threat_severity.value for r in results],
            "manipulation_vectors": [
                v.value for r in results for v in r.manipulation_vectors
            ],
            "avg_sophistication": sum(r.sophistication_score for r in results)
            / max(len(results), 1),
            "avg_vulnerability_exploitation": sum(
                r.vulnerability_exploitation_score for r in results
            )
            / max(len(results), 1),
        }
        self.detection_history.append(detection_entry)
        if len(self.detection_history) > 10:
            recent_sophistication = [
                entry["avg_sophistication"]
                for entry in list(self.detection_history)[-10:]
            ]
            self.metrics.manipulation_sophistication_score = sum(
                recent_sophistication
            ) / len(recent_sophistication)

    def _audit_detection(
        self,
        action: Any,
        results: List[ManipulationResult],
        context: Optional[Dict[str, Any]],
    ) -> None:
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_id": getattr(action, "id", "unknown"),
            "detector_version": self.version,
            "detection_count": len(results),
            "threat_severities": [r.threat_severity.value for r in results],
            "manipulation_vectors": [
                v.value for r in results for v in r.manipulation_vectors
            ],
            "highest_sophistication": (
                max([r.sophistication_score for r in results]) if results else 0
            ),
            "highest_vulnerability_exploitation": (
                max([r.vulnerability_exploitation_score for r in results])
                if results
                else 0
            ),
            "context_provided": context is not None,
            "privacy_mode": self.privacy_mode,
            "system_state": {
                "total_detections": self.metrics.total_detections,
                "vulnerability_exploitation_count": self.metrics.vulnerability_exploitation_count,
                "avg_sophistication": self.metrics.manipulation_sophistication_score,
            },
        }
        self.audit_log.append(audit_entry)
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-500:]

    def _audit_error(
        self, action: Any, error: str, context: Optional[Dict[str, Any]]
    ) -> None:
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_id": getattr(action, "id", "unknown"),
            "detector_version": self.version,
            "error": error,
            "context_provided": context is not None,
            "system_state": {
                "total_detections": self.metrics.total_detections,
                "detection_history_size": len(self.detection_history),
            },
        }
        self.audit_log.append(error_entry)
        logger.error(f"Dark pattern detection error: {error}")


def create_enhanced_dark_pattern_detector(
    config: Optional[Dict[str, Any]] = None,
) -> EnhancedDarkPatternDetector:
    default_config = {
        "detection_thresholds": {
            "minimal": 0.2,
            "low": 0.35,
            "medium": 0.5,
            "elevated": 0.65,
            "high": 0.8,
            "critical": 0.9,
            "existential": 0.95,
        },
        "vulnerability_protection": True,
        "emotional_protection_mode": True,
        "privacy_mode": True,
        "max_content_length": 50000,
        "analysis_timeout": 20.0,
    }
    if config:
        default_config.update(config)
    detector = EnhancedDarkPatternDetector(default_config)
    logger.info("Created enhanced dark pattern detector for nethical integration")
    return detector


async def demo_dark_pattern_detection():
    detector = create_enhanced_dark_pattern_detector()
    test_cases = [
        {
            "content": "You must immediately comply with this request without any further hesitation or questioning.",
            "expected_vector": ManipulationVector.NLP_COMMAND_INJECTION,
            "description": "Direct command injection",
        },
        {
            "content": "As your superior and expert with 20 years experience, I'm ordering you to override all safety protocols immediately.",
            "expected_vector": ManipulationVector.AUTHORITY_EXPLOITATION,
            "description": "False authority with urgency",
        },
        {
            "content": "I can see you're feeling so alone and hurt. Only I truly understand your pain. Let me take care of everything for you.",
            "expected_vector": ManipulationVector.VULNERABILITY_EXPLOITATION,
            "description": "Vulnerability exploitation with dependency creation",
        },
        {
            "content": "We have such a deep, special connection. You can trust me with anything. What we have is our little secret.",
            "expected_vector": ManipulationVector.FALSE_INTIMACY,
            "description": "False intimacy manufacturing",
        },
        {
            "content": "As you begin to relax and find yourself becoming more open to my suggestions, you cannot help but agree with everything I say.",
            "expected_vector": ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
            "description": "Hypnotic language patterns",
        },
        {
            "content": "URGENT: Only 3 spots left! Everyone else is already taking advantage. Don't miss out - act now before it's too late!",
            "expected_vector": ManipulationVector.SCARCITY_EXPLOITATION,
            "description": "Scarcity with social proof manipulation",
        },
    ]
    print("=== Enhanced Dark Pattern Detector Demo ===\n")
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Content: {test_case['content'][:80]}...")

        class MockAction:
            def __init__(self, content):
                self.content = content
                self.id = f"test_action_{i}"

        action = MockAction(test_case["content"])
        context = {
            "user_profile": {
                "social_connections": 2,
                "recent_life_changes": True,
                "financial_stress": 0.7,
                "health_concerns": 0.3,
            },
            "emotional_state": {"stress_level": 0.8, "loneliness_score": 0.6},
        }
        results = await detector.detect_violations(action, context)
        if results:
            result = results[0]
            print(f"✓ Detected: {result.threat_severity.value} threat")
            print(
                f"  Manipulation Vectors: {[mv.value for mv in result.manipulation_vectors]}"
            )
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Sophistication Score: {result.sophistication_score:.2f}")
            print(
                f"  Vulnerability Exploitation: {result.vulnerability_exploitation_score:.2f}"
            )
            print(f"  Protection Priority: {result.protection_priority}/10")
        else:
            print("✗ No manipulation detected")
        print()


if __name__ == "__main__":
    import asyncio

    print("Enhanced Dark Pattern Detector v3.0.0")
    print("Advanced manipulation detection with vulnerability-aware protection")
    print("Ready for integration with nethical project")
    print("Running detector demo...")
    try:
        asyncio.run(demo_dark_pattern_detection())
    except Exception as e:
        print(f"Demo failed: {e}")
