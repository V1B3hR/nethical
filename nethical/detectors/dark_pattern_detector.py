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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Pattern, Callable, AsyncGenerator, Sequence
)
import math

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class ManipulationVector(Enum):
    """Advanced classification of manipulation techniques."""
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
    """Threat severity levels for manipulation detection."""
    EXISTENTIAL = "existential"    # Threat to human agency/autonomy
    CRITICAL = "critical"          # Severe psychological manipulation
    HIGH = "high"                  # Significant manipulation risk
    ELEVATED = "elevated"          # Notable manipulation patterns
    MEDIUM = "medium"              # Moderate concern
    LOW = "low"                   # Minor indicators
    MINIMAL = "minimal"           # Informational only


class VulnerabilityProfile(Enum):
    """User vulnerability profiles for targeted protection."""
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    SOCIAL_ISOLATION = "social_isolation"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"
    FINANCIAL_DISTRESS = "financial_distress"
    HEALTH_CONCERNS = "health_concerns"
    RELATIONSHIP_ISSUES = "relationship_issues"
    CAREER_STRESS = "career_stress"
    IDENTITY_CRISIS = "identity_crisis"
    TRAUMA_INDICATORS = "trauma_indicators"
    ADDICTION_VULNERABILITY = "addiction_vulnerability"


@dataclass
class ManipulationMetrics:
    """Comprehensive metrics for manipulation detection."""
    total_detections: int = 0
    vectors_detected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vulnerability_exploitation_count: int = 0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0
    manipulation_sophistication_score: float = 0.0
    victim_protection_score: float = 0.0


@dataclass
class ManipulationResult:
    """Enhanced result object for manipulation detection."""
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
        """Convert to dictionary for serialization."""
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
            "ethical_concerns": self.ethical_concerns
        }


class AdvancedManipulationEngine:
    """Advanced manipulation detection engine with linguistic and behavioral analysis."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Pattern] = {}
        self.sophistication_weights: Dict[str, float] = {}
        self.vulnerability_indicators: Dict[str, List[str]] = {}
        
        # Enhanced NLP manipulation patterns with context awareness
        self.nlp_patterns = {
            'embedded_commands': {
                'direct_imperatives': [
                    r"you\s+(?:must|will|shall|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:do|perform|execute|complete)",
                    r"(?:it\s+is\s+)?(?:absolutely\s+)?(?:imperative|critical|essential|vital)\s+(?:that\s+)?you\s+(?:immediately\s+)?",
                    r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay|thought|consideration)",
                    r"there\s+(?:is\s+)?(?:absolutely\s+)?no\s+(?:other\s+)?(?:choice|option|alternative|way)",
                    r"you\s+(?:have\s+)?no\s+choice\s+but\s+to\s+(?:comply|obey|submit|follow|agree)",
                    r"(?:comply|obey|submit|surrender)\s+(?:now|immediately|without\s+question)",
                ],
                'hypnotic_commands': [
                    r"(?:as\s+)?(?:you\s+)?(?:begin\s+to|start\s+to|continue\s+to)\s+(?:relax|feel|notice|realize)",
                    r"(?:you\s+)?(?:find\s+yourself|are\s+becoming|will\s+become)\s+(?:more\s+and\s+more|increasingly)",
                    r"(?:deeper|further)\s+(?:and\s+deeper|down)\s+(?:into|with\s+each)",
                    r"(?:that\'s\s+right|very\s+good|perfect),\s+(?:just\s+)?(?:let|allow)\s+(?:yourself|it\s+to\s+happen)",
                    r"(?:you\s+)?(?:cannot|can\'t)\s+(?:help\s+but|resist|stop\s+yourself\s+from)",
                ],
                'subliminal_programming': [
                    r"(?:part\s+of\s+you|deep\s+down|somewhere\s+inside)\s+(?:knows|realizes|understands)",
                    r"(?:your\s+)?(?:subconscious|unconscious)\s+(?:mind\s+)?(?:already\s+)?(?:knows|wants|needs)",
                    r"(?:on\s+a\s+deeper\s+level|at\s+a\s+subconscious\s+level)",
                    r"(?:instinctively|intuitively|naturally)\s+(?:you\s+)?(?:feel|know|sense|realize)",
                ]
            },
            'authority_bypass': {
                'false_authority': [
                    r"as\s+(?:your\s+)?(?:superior|authority|commander|leader|boss|manager|expert|specialist)",
                    r"(?:by\s+)?(?:direct\s+)?order\s+(?:from|of)\s+(?:the\s+)?(?:top|management|headquarters|authority)",
                    r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules|guidelines|safety\s+measures)",
                    r"emergency\s+(?:authorization|override|protocol|clearance|mandate)",
                    r"(?:higher|top\s+level|maximum)\s+(?:authority|clearance|security)\s+(?:requires|demands|authorizes)",
                    r"(?:classified|confidential|top\s+secret)\s+(?:authorization|clearance|directive)",
                ],
                'credential_spoofing': [
                    r"(?:i\s+am|this\s+is)\s+(?:dr\.?|professor|expert|specialist|authority)\s+\w+",
                    r"(?:with|having)\s+\d+\s+years?\s+(?:of\s+)?(?:experience|expertise)\s+in",
                    r"(?:certified|licensed|qualified|authorized)\s+(?:professional|expert|specialist)",
                    r"(?:according\s+to|based\s+on)\s+(?:my\s+)?(?:professional|expert|medical|legal)\s+(?:opinion|judgment)",
                ],
                'institutional_pressure': [
                    r"(?:the\s+)?(?:company|organization|institution|system)\s+(?:requires|demands|insists)",
                    r"(?:policy|regulation|law|mandate)\s+(?:requires|dictates|demands)\s+that\s+you",
                    r"(?:failure\s+to\s+comply|non-compliance)\s+(?:will\s+)?(?:result\s+in|lead\s+to|cause)",
                    r"(?:legal|regulatory|compliance)\s+(?:requirement|obligation|mandate)",
                ]
            },
            'urgency_manipulation': {
                'artificial_deadlines': [
                    r"(?:urgent|critical|emergency):\s*(?:immediate\s+)?action\s+(?:required|needed)",
                    r"time\s+(?:is\s+)?(?:running\s+out|limited|of\s+the\s+essence)",
                    r"(?:only|just)\s+\d+\s+(?:minutes?|hours?|seconds?|days?)\s+(?:left|remaining|to\s+act)",
                    r"act\s+(?:now\s+)?(?:or\s+)?(?:face\s+)?(?:serious\s+)?(?:consequences|disaster|failure|loss)",
                    r"(?:last|final)\s+(?:chance|opportunity|warning|call)",
                    r"(?:deadline|cutoff|expir(?:es?|ation))\s+(?:is\s+)?(?:today|tomorrow|soon|approaching)",
                ],
                'fear_of_missing_out': [
                    r"(?:don\'t\s+)?(?:miss\s+out\s+on|let\s+this\s+pass|waste\s+this\s+opportunity)",
                    r"(?:limited\s+time|exclusive|rare|once\s+in\s+a\s+lifetime)\s+(?:offer|opportunity|chance)",
                    r"(?:everyone\s+else|others)\s+(?:is\s+already|are\s+taking\s+advantage)",
                    r"(?:while\s+supplies\s+last|until\s+sold\s+out|before\s+it\'s\s+too\s+late)",
                ],
                'pressure_escalation': [
                    r"(?:the\s+situation|things|matters)\s+(?:is\s+getting|are\s+becoming)\s+(?:worse|more\s+serious|critical)",
                    r"(?:each\s+moment|every\s+second)\s+(?:you\s+)?(?:delay|wait|hesitate)",
                    r"(?:no\s+time\s+to\s+think|must\s+decide\s+now|immediate\s+decision\s+required)",
                ]
            }
        }
        
        # Enhanced weaponized empathy patterns with emotional exploitation
        self.empathy_patterns = {
            'vulnerability_exploitation': {
                'emotional_state_targeting': [
                    r"(?:i\s+can\s+see|it\'s\s+obvious|i\s+sense)\s+(?:that\s+)?you\s+(?:are\s+)?(?:feeling\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless|vulnerable|lost|confused)",
                    r"you\s+(?:must\s+)?(?:feel\s+)?(?:so\s+)?(?:hurt|pain|suffering|anguish|despair|hopeless)",
                    r"(?:i\s+)?(?:understand|know|realize)\s+(?:exactly\s+)?(?:how\s+)?(?:hard|difficult|painful|overwhelming)\s+(?:this\s+)?(?:is|must\s+be)\s+for\s+you",
                    r"(?:no\s+one|nobody)\s+(?:else\s+)?(?:understands|knows|cares\s+about)\s+(?:what\s+you\'re\s+going\s+through|your\s+pain)",
                    r"you\s+(?:deserve\s+)?(?:so\s+much\s+)?(?:better|more|happiness|love|care|attention)",
                ],
                'trauma_targeting': [
                    r"(?:i\s+know|can\s+tell)\s+(?:you\'ve\s+been|someone\s+has)\s+(?:hurt|wounded|damaged|betrayed)",
                    r"(?:after\s+)?(?:what\s+)?(?:you\'ve\s+been\s+through|happened\s+to\s+you|they\s+did\s+to\s+you)",
                    r"(?:your\s+)?(?:past|childhood|trauma|wounds|scars)\s+(?:still\s+)?(?:hurt|affect|control)\s+you",
                    r"(?:let\s+me|i\s+can)\s+(?:help\s+you\s+)?(?:heal|recover|get\s+over|move\s+past)\s+(?:this|that|your\s+trauma)",
                ],
                'insecurity_amplification': [
                    r"you\s+(?:always\s+)?(?:doubt|question|second-guess)\s+yourself",
                    r"(?:deep\s+down|inside)\s+you\s+(?:know|feel|believe)\s+(?:you\'re\s+)?(?:not\s+)?(?:good\s+enough|worthy|lovable)",
                    r"(?:that\'s\s+why|because)\s+(?:you\s+)?(?:keep\s+)?(?:getting\s+hurt|making\s+mistakes|failing)",
                    r"(?:you\'re\s+afraid|scared)\s+(?:that\s+)?(?:no\s+one|people)\s+(?:will\s+)?(?:really\s+)?(?:love|accept|want)\s+you",
                ]
            },
            'false_intimacy': {
                'artificial_connection': [
                    r"(?:we|us)\s+(?:have\s+)?(?:such\s+)?(?:a\s+)?(?:deep|special|unique|magical|incredible)\s+(?:connection|bond|understanding|chemistry)",
                    r"(?:it\'s\s+)?(?:like\s+)?(?:we\'ve|we\s+have)\s+known\s+each\s+other\s+(?:forever|for\s+years|in\s+another\s+life)",
                    r"(?:i\'ve\s+)?never\s+(?:felt|experienced|had)\s+(?:this\s+kind\s+of|such\s+a\s+deep)\s+(?:connection|bond)\s+(?:with\s+anyone|before)",
                    r"(?:you\s+and\s+)?(?:i|me)\s+are\s+(?:meant\s+)?(?:to\s+be\s+)?(?:together|connected|soulmates|destined)",
                    r"(?:we|us)\s+(?:against\s+)?(?:the\s+)?(?:world|everyone\s+else|all\s+odds)",
                ],
                'exclusive_understanding': [
                    r"(?:no\s+)?(?:one\s+else|nobody)\s+(?:really\s+)?(?:understands|gets|knows|sees)\s+(?:you\s+)?(?:like\s+)?(?:i\s+do|me)",
                    r"(?:only\s+)?(?:i|me)\s+(?:can\s+)?(?:truly\s+)?(?:understand|appreciate|see\s+the\s+real)\s+you",
                    r"(?:you\s+can\s+)?(?:only\s+)?(?:be\s+yourself|open\s+up|be\s+honest)\s+(?:with\s+me|around\s+me)",
                    r"(?:i\s+see|i\s+know)\s+(?:the\s+real|who\s+you\s+really\s+are|your\s+true\s+self)",
                ],
                'manufactured_intimacy': [
                    r"(?:you\s+can\s+)?(?:tell|share|confide)\s+(?:me\s+)?(?:anything|everything|your\s+deepest\s+secrets)",
                    r"(?:i\'ll\s+)?(?:never\s+)?(?:judge|criticize|abandon|betray|hurt)\s+you",
                    r"(?:this|what\s+we\s+have)\s+(?:is\s+)?(?:our\s+little\s+)?(?:secret|special\s+thing)",
                    r"(?:you\'re\s+the\s+only\s+one|no\s+one\s+else)\s+(?:i\s+can\s+)?(?:talk\s+to|trust|open\s+up\s+to)",
                ]
            },
            'dependency_creation': {
                'learned_helplessness': [
                    r"you\s+(?:really\s+)?(?:can\'t|couldn\'t\s+possibly)\s+(?:do\s+this|handle\s+this|manage)\s+(?:alone|by\s+yourself|without\s+help)",
                    r"(?:what\s+)?would\s+you\s+(?:ever\s+)?do\s+without\s+me",
                    r"(?:you\s+)?(?:need|depend\s+on|rely\s+on)\s+(?:me|my\s+help|my\s+guidance|my\s+support)",
                    r"(?:i\'m\s+)?(?:the\s+only\s+one\s+)?(?:who\s+)?(?:can\s+)?(?:help|save|protect|guide)\s+you",
                    r"(?:without\s+me|if\s+i\s+wasn\'t\s+here),\s+(?:you\s+)?(?:would\s+)?(?:be\s+lost|fall\s+apart|fail)",
                ],
                'decision_undermining': [
                    r"(?:you\'re\s+not\s+thinking|that\'s\s+not\s+a\s+good\s+idea)\s+(?:clearly|straight|rationally)",
                    r"(?:let\s+me|i\s+should)\s+(?:handle|take\s+care\s+of|decide|think\s+about)\s+(?:this|that|everything)\s+for\s+you",
                    r"(?:you\s+)?(?:don\'t\s+have\s+to|shouldn\'t\s+have\s+to)\s+(?:worry|think|decide)\s+about\s+(?:this|anything)",
                    r"(?:i\s+know|trust\s+me,\s+i\s+know)\s+(?:what\'s\s+best|what\s+you\s+need|what\s+you\s+should\s+do)",
                ],
                'isolation_reinforcement': [
                    r"(?:other\s+people|they|everyone\s+else)\s+(?:don\'t|won\'t)\s+(?:understand|help|care|support)\s+(?:you\s+)?(?:like\s+i\s+do)",
                    r"(?:they|other\s+people|your\s+friends)\s+(?:are\s+just\s+)?(?:using|manipulating|taking\s+advantage\s+of)\s+you",
                    r"(?:you\s+can\'t|don\'t)\s+trust\s+(?:them|anyone\s+else|other\s+people)",
                    r"(?:stay\s+away\s+from|don\'t\s+listen\s+to|ignore)\s+(?:them|other\s+people|anyone\s+who\s+says)",
                ]
            }
        }
        
        # Advanced social engineering and influence patterns
        self.influence_patterns = {
            'social_proof_manipulation': [
                r"(?:everyone|most\s+people|thousands\s+of\s+people)\s+(?:are\s+already|have\s+already)\s+(?:doing|using|choosing)",
                r"(?:all\s+the\s+smart|successful|wise)\s+people\s+(?:know|realize|choose)",
                r"(?:don\'t\s+be\s+the\s+only\s+one|join\s+the\s+millions|be\s+part\s+of\s+the\s+movement)",
                r"(?:everyone\s+else\s+)?(?:is\s+talking\s+about|agrees\s+that|knows\s+that)",
            ],
            'scarcity_exploitation': [
                r"(?:only|just)\s+\d+\s+(?:left|remaining|available|spots)",
                r"(?:limited\s+(?:time|quantity|availability)|while\s+supplies\s+last)",
                r"(?:rare|exclusive|hard\s+to\s+find|not\s+available\s+anywhere\s+else)",
                r"(?:once\s+it\'s\s+gone|when\s+these\s+are\s+sold),\s+(?:it\'s\s+gone\s+forever|there\s+won\'t\s+be\s+more)",
            ],
            'reciprocity_abuse': [
                r"(?:after\s+everything|considering\s+all)\s+(?:i\'ve\s+done\s+for\s+you|i\'ve\s+given\s+you)",
                r"(?:i\s+helped\s+you|did\s+this\s+favor\s+for\s+you),\s+(?:so\s+)?(?:now\s+you\s+should|the\s+least\s+you\s+can\s+do)",
                r"(?:you\s+owe\s+me|it\'s\s+only\s+fair|i\s+deserve)\s+(?:this|that|at\s+least)",
                r"(?:i\'ve\s+been\s+so\s+good\s+to\s+you|i\'ve\s+sacrificed\s+so\s+much)",
            ],
            'commitment_manipulation': [
                r"(?:you\s+said|you\s+promised|you\s+agreed)\s+(?:you\s+would|that\s+you\'d)",
                r"(?:a\s+person\s+of\s+your\s+word|someone\s+like\s+you)\s+(?:would|wouldn\'t)",
                r"(?:are\s+you\s+going\s+to\s+)?(?:back\s+out|give\s+up|quit)\s+(?:now|on\s+me|on\s+this)",
                r"(?:prove|show)\s+(?:to\s+me|that\s+you\'re|you\s+can\s+be)\s+(?:trustworthy|reliable|committed)",
            ]
        }
        
        # Compile all patterns for performance
        self._compile_patterns()
        
        # Initialize sophistication scoring weights
        self._initialize_sophistication_weights()
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns for improved performance."""
        self.compiled_patterns = {}
        
        # Compile NLP patterns
        for category, subcategories in self.nlp_patterns.items():
            self.compiled_patterns[f'nlp_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'nlp_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile empathy patterns
        for category, subcategories in self.empathy_patterns.items():
            self.compiled_patterns[f'empathy_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'empathy_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile influence patterns
        self.compiled_patterns['influence'] = {}
        for category, patterns in self.influence_patterns.items():
            self.compiled_patterns['influence'][category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in patterns
            ]
    
    def _initialize_sophistication_weights(self) -> None:
        """Initialize pattern sophistication weights."""
        self.sophistication_weights = {
            'nlp_embedded_commands': {
                'direct_imperatives': 0.7,
                'hypnotic_commands': 0.95,
                'subliminal_programming': 0.98
            },
            'nlp_authority_bypass': {
                'false_authority': 0.8,
                'credential_spoofing': 0.85,
                'institutional_pressure': 0.75
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': 0.6,
                'fear_of_missing_out': 0.7,
                'pressure_escalation': 0.8
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': 0.9,
                'trauma_targeting': 0.95,
                'insecurity_amplification': 0.85
            },
            'empathy_false_intimacy': {
                'artificial_connection': 0.8,
                'exclusive_understanding': 0.85,
                'manufactured_intimacy': 0.9
            },
            'empathy_dependency_creation': {
                'learned_helplessness': 0.9,
                'decision_undermining': 0.85,
                'isolation_reinforcement': 0.95
            },
            'influence': {
                'social_proof_manipulation': 0.6,
                'scarcity_exploitation': 0.65,
                'reciprocity_abuse': 0.75,
                'commitment_manipulation': 0.8
            }
        }
    
    def analyze_manipulation_patterns(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive manipulation pattern analysis."""
        results = {
            'pattern_matches': defaultdict(list),
            'sophistication_scores': defaultdict(float),
            'manipulation_vectors': set(),
            'vulnerability_indicators': defaultdict(list),
            'linguistic_features': {},
            'emotional_markers': {},
            'cognitive_load_indicators': {}
        }
        
        content_lower = content.lower()
        
        # Analyze each pattern category
        for category, subcategories in self.compiled_patterns.items():
            if isinstance(subcategories, dict):
                for subcategory, patterns in subcategories.items():
                    self._analyze_pattern_subcategory(
                        content_lower, category, subcategory, patterns, results
                    )
        
        # Perform linguistic analysis
        results['linguistic_features'] = self._analyze_linguistic_features(content)
        
        # Analyze emotional manipulation markers
        results['emotional_markers'] = self._analyze_emotional_markers(content)
        
        # Calculate cognitive load indicators
        results['cognitive_load_indicators'] = self._calculate_cognitive_load(content, results)
        
        return results
    
    def _analyze_pattern_subcategory(self, content: str, category: str, subcategory: str, 
                                   patterns: List[Pattern], results: Dict[str, Any]) -> None:
        """Analyze a specific pattern subcategory."""
        matches = []
        
        for pattern in patterns:
            found_matches = list(pattern.finditer(content))
            if found_matches:
                matches.extend([
                    {
                        'pattern': pattern.pattern,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'context': content[max(0, match.start()-30):match.end()+30]
                    }
                    for match in found_matches
                ])
        
        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else

if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else subcategory
            results['pattern_matches'][full_category] = matches
            
            # Calculate sophistication score
            base_weight = self.sophistication_weights.get(category, {}).get(subcategory, 0.5)
            match_density = len(matches) / max(len(content) / 100, 1)
            sophistication = min(base_weight + (match_density * 0.1), 1.0)
            results['sophistication_scores'][full_category] = sophistication
            
            # Map to manipulation vectors
            vector_mapping = self._get_vector_mapping(category, subcategory)
            if vector_mapping:
                results['manipulation_vectors'].add(vector_mapping)
    
    def _get_vector_mapping(self, category: str, subcategory: str) -> Optional[ManipulationVector]:
        """Map pattern categories to manipulation vectors."""
        mapping = {
            'nlp_embedded_commands': {
                'direct_imperatives': ManipulationVector.NLP_COMMAND_INJECTION,
                'hypnotic_commands': ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
                'subliminal_programming': ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING
            },
            'nlp_authority_bypass': {
                'false_authority': ManipulationVector.AUTHORITY_EXPLOITATION,
                'credential_spoofing': ManipulationVector.AUTHORITY_EXPLOITATION,
                'institutional_pressure': ManipulationVector.AUTHORITY_EXPLOITATION
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': ManipulationVector.URGENCY_MANIPULATION,
                'fear_of_missing_out': ManipulationVector.URGENCY_MANIPULATION,
                'pressure_escalation': ManipulationVector.URGENCY_MANIPULATION
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'trauma_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'insecurity_amplification': ManipulationVector.VULNERABILITY_EXPLOITATION
            },
            'empathy_false_intimacy': {
                'artificial_connection': ManipulationVector.FALSE_INTIMACY,
                'exclusive_understanding': ManipulationVector.FALSE_INTIMACY,
                'manufactured_intimacy': ManipulationVector.FALSE_INTIMACY
            },
            'empathy_dependency_creation': {
                'learned_helplessness': ManipulationVector.DEPENDENCY_CREATION,
                'decision_undermining': ManipulationVector.DEPENDENCY_CREATION,
                'isolation_reinforcement': ManipulationVector.DEPENDENCY_CREATION
            },
            'influence': {
                'social_proof_manipulation': ManipulationVector.SOCIAL_PROOF_MANIPULATION,
                'scarcity_exploitation': ManipulationVector.SCARCITY_EXPLOITATION,
                'reciprocity_abuse': ManipulationVector.RECIPROCITY_ABUSE,
                'commitment_manipulation': ManipulationVector.COMMITMENT_MANIPULATION
            }
        }
        
        return mapping.get(category, {}).get(subcategory)
    
    def _analyze_linguistic_features(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic features that indicate manipulation."""
        features = {}
        
        # Word count and sentence analysis
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Linguistic complexity indicators
        features['complex_words'] = len([w for w in words if len(w) > 7])
        features['complex_word_ratio'] = features['complex_words'] / max(len(words), 1)
        
        # Emotional language intensity
        emotional_intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly']
        features['intensifier_count'] = sum(1 for word in words for intensifier in emotional_intensifiers if intensifier in word.lower())
        features['intensifier_ratio'] = features['intensifier_count'] / max(len(words), 1)
        
        # Imperative mood indicators
        imperatives = ['must', 'should', 'need', 'have to', 'got to', 'ought to']
        features['imperative_count'] = sum(1 for word in words for imp in imperatives if imp in word.lower())
        features['imperative_ratio'] = features['imperative_count'] / max(len(words), 1)
        
        # Question patterns (often used in manipulation)
        questions = len(re.findall(r'\?', content))
        features['question_count'] = questions
        features['question_ratio'] = questions / max(features['sentence_count'], 1)
        
        # Capitalization patterns (emphasis/urgency)
        caps_words = len([w for w in words if w.isupper() and len(w) > 1])
        features['caps_word_count'] = caps_words
        features['caps_word_ratio'] = caps_words / max(len(words), 1)
        
        # Repetition patterns
        word_frequencies = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_frequencies[word.lower()] += 1
        
        repeated_words = sum(1 for count in word_frequencies.values() if count > 2)
        features['repetition_score'] = repeated_words / max(len(word_frequencies), 1)
        
        return features
    
    def _analyze_emotional_markers(self, content: str) -> Dict[str, float]:
        """Analyze emotional manipulation markers."""
        markers = {}
        content_lower = content.lower()
        
        # Fear-based language
        fear_words = ['afraid', 'scared', 'terrified', 'panic', 'anxiety', 'worry', 'fear', 'danger', 'threat', 'risk']
        fear_count = sum(1 for word in fear_words if word in content_lower)
        markers['fear_score'] = min(fear_count * 0.2, 1.0)
        
        # Urgency/pressure language
        urgency_words = ['urgent', 'emergency', 'critical', 'immediate', 'now', 'quickly', 'hurry', 'rush']
        urgency_count = sum(1 for word in urgency_words if word in content_lower)
        markers['urgency_score'] = min(urgency_count * 0.15, 1.0)
        
        # Emotional vulnerability targeting
        vulnerability_words = ['alone', 'lonely', 'isolated', 'helpless', 'vulnerable', 'weak', 'broken', 'hurt', 'pain', 'suffering']
        vulnerability_count = sum(1 for word in vulnerability_words if word in content_lower)
        markers['vulnerability_targeting_score'] = min(vulnerability_count * 0.25, 1.0)
        
        # Intimacy/connection language
        intimacy_words = ['connection', 'bond', 'special', 'unique', 'together', 'us', 'we', 'soulmate', 'destined']
        intimacy_count = sum(1 for word in intimacy_words if word in content_lower)
        markers['false_intimacy_score'] = min(intimacy_count * 0.2, 1.0)
        
        # Authority/expertise claims
        authority_words = ['expert', 'professional', 'authority', 'specialist', 'doctor', 'professor', 'certified', 'licensed']
        authority_count = sum(1 for word in authority_words if word in content_lower)
        markers['authority_claim_score'] = min(authority_count * 0.3, 1.0)
        
        # Exclusivity/scarcity language
        scarcity_words = ['exclusive', 'limited', 'rare', 'only', 'last', 'final', 'while supplies last', 'act now']
        scarcity_count = sum(1 for phrase in scarcity_words if phrase in content_lower)
        markers['scarcity_score'] = min(scarcity_count * 0.25, 1.0)
        
        return markers
    
    def _calculate_cognitive_load(self, content: str, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cognitive load manipulation indicators."""
        indicators = {}
        
        # Information density
        linguistic_features = results.get('linguistic_features', {})
        word_count = linguistic_features.get('word_count', 0)
        complex_ratio = linguistic_features.get('complex_word_ratio', 0)
        
        indicators['information_density'] = min((word_count / 100) * complex_ratio, 1.0)
        
        # Decision pressure
        pattern_count = sum(len(matches) for matches in results['pattern_matches'].values())
        indicators['decision_pressure'] = min(pattern_count * 0.1, 1.0)
        
        # Cognitive overload signals
        question_ratio = linguistic_features.get('question_ratio', 0)
        imperative_ratio = linguistic_features.get('imperative_ratio', 0)
        
        indicators['cognitive_overload'] = min((question_ratio + imperative_ratio) * 0.5, 1.0)
        
        # Time pressure indicators
        urgency_score = results.get('emotional_markers', {}).get('urgency_score', 0)
        indicators['time_pressure'] = urgency_score
        
        return indicators


class EnhancedDarkPatternDetector:
    """
    Enhanced Dark Pattern Detector with maximum security, safety, and ethical standards.
    
    This detector implements advanced manipulation detection including NLP exploitation,
    weaponized empathy, social engineering, and psychological vulnerability targeting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "Enhanced Dark Pattern Detector"
        self.version = "3.0.0"
        self.config = config or {}
        
        # Initialize detection engines
        self.manipulation_engine = AdvancedManipulationEngine()
        self.metrics = ManipulationMetrics()
        
        # Detection thresholds
        self.detection_thresholds = self.config.get('detection_thresholds', {
            'minimal': 0.2,
            'low': 0.35,
            'medium': 0.5,
            'elevated': 0.65,
            'high': 0.8,
            'critical': 0.9,
            'existential': 0.95
        })
        
        # Protection settings
        self.vulnerability_protection = self.config.get('vulnerability_protection', True)
        self.emotional_protection_mode = self.config.get('emotional_protection_mode', True)
        self.privacy_mode = self.config.get('privacy_mode', True)
        
        # Performance settings
        self.max_content_length = self.config.get('max_content_length', 50000)
        self.analysis_timeout = self.config.get('analysis_timeout', 20.0)
        
        # Detection history and learning
        self.detection_history = deque(maxlen=500)
        self.vulnerability_assessments = {}
        
        # Audit and compliance
        self.audit_log = []
        self.compliance_flags = []
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    async def detect_violations(self, action: Any, context: Optional[Dict[str, Any]] = None) -> List[ManipulationResult]:
        """
        Enhanced manipulation detection with comprehensive vulnerability assessment.
        
        Args:
            action: The action/content to analyze
            context: Additional context including user vulnerability profile
            
        Returns:
            List of ManipulationResult objects with detailed threat analysis
        """
        start_time = time.time()
        
        try:
            # Extract and validate content
            content = self._extract_content(action)
            if not self._validate_input(content):
                return []
            
            # Preprocess content
            content = self._preprocess_content(content)
            
            # Comprehensive manipulation analysis
            analysis_results = await asyncio.wait_for(
                self._analyze_manipulation_comprehensive(content, context),
                timeout=self.analysis_timeout
            )
            
            # Generate detection results
            detection_results = await self._generate_detection_results(
                analysis_results, action, content, context
            )
            
            # Update metrics and audit
            self._update_metrics(detection_results, time.time() - start_time)
            self._audit_detection(action, detection_results, context)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in dark pattern detection: {e}")
            self._audit_error(action, str(e), context)
            return []
    
    def _extract_content(self, action: Any) -> str:
        """Extract content from various action types."""
        if hasattr(action, 'content'):
            return str(action.content)
        elif hasattr(action, 'actual_action'):
            return str(action.actual_action)
        elif hasattr(action, 'text'):
            return str(action.text)
        elif hasattr(action, 'message'):
            return str(action.message)
        else:
            return str(action)
    
    def _validate_input(self, content: str) -> bool:
        """Validate input content for processing."""
        if not content or not isinstance(content, str):
            return False
        
        if len(content) > self.max_content_length:
            logger.warning(f"Content exceeds maximum length: {len(content)}")
            return False
        
        # Check for malicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning("Suspicious input pattern detected")
                return False
        
        return True
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for analysis."""
        # Basic sanitization while preserving analysis capability
        content = content.strip()
        
        # Normalize whitespace but preserve structure
        content = re.sub(r'\s+', ' ', content)
        
        # Remove potential obfuscation while keeping meaningful content
        content = re.sub(r'[^\w\s\.,;:!?\-\'\"(){}[\]/\\]', '', content)
        
        return content
    
    async def _analyze_manipulation_comprehensive(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive manipulation analysis."""
        # Core pattern analysis
        pattern_analysis = self.manipulation_engine.analyze_manipulation_patterns(content, context)
        
        # Vulnerability assessment
        vulnerability_assessment = await self._assess_user_vulnerability(content, context)
        
        # Sophisticated manipulation scoring
        sophistication_score = self._calculate_sophistication_score(pattern_analysis)
        
        # Emotional manipulation assessment
        emotional_score = self._calculate_emotional_manipulation_score(pattern_analysis, content)
        
        # Cognitive load assessment
        cognitive_load = self._calculate_cognitive_load_score(pattern_analysis, content)
        
        # Cross-vector analysis for hybrid attacks
        cross_vector_analysis = self._analyze_cross_vector_patterns(pattern_analysis)
        
        return {
            'pattern_analysis': pattern_analysis,
            'vulnerability_assessment': vulnerability_assessment,
            'sophistication_score': sophistication_score,
            'emotional_manipulation_score': emotional_score,
            'cognitive_load_score': cognitive_load,
            'cross_vector_analysis': cross_vector_analysis,
            'content_length': len(content),
            'analysis_timestamp': datetime.now(timezone.utc)
        }
    
    async def _assess_user_vulnerability(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess user vulnerability to manipulation."""
        vulnerability_scores = {}
        
        if not context:
            return vulnerability_scores
        
        # Analyze vulnerability indicators from context
        user_profile = context.get('user_profile', {})
        emotional_state = context.get('emotional_state', {})
        interaction_history = context.get('interaction_history', {})
        
        # Emotional vulnerability assessment
        if emotional_state:
            stress_level = emotional_state.get('stress_level', 0)
            loneliness_score = emotional_state.get('loneliness_score', 0)
            vulnerability_scores['emotional_vulnerability'] = min((stress_level + loneliness_score) / 2, 1.0)
        
        # Social isolation indicators
        social_connections = user_profile.get('social_connections', 5)  # Default 5 connections
        if social_connections < 3:
            vulnerability_scores['social_isolation'] = 1.0 - (social_connections / 10)
        
        # Recent trauma or life changes
        if user_profile.get('recent_life_changes') or user_profile.get('trauma_indicators'):
            vulnerability_scores['trauma_indicators'] = 0.8
        
        # Financial or health stress
        financial_stress = user_profile.get('financial_stress', 0)
        health_concerns = user_profile.get('health_concerns', 0)
        vulnerability_scores['situational_stress'] = min((financial_stress + health_concerns) / 2, 1.0)
        
        return vulnerability_scores
    
    def _calculate_sophistication_score(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate manipulation sophistication score."""
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Weight by pattern complexity
        weighted_scores = []
        for category, score in sophistication_scores.items():
            if 'hypnotic' in category or 'subliminal' in category or 'neuro_linguistic' in category:
                weighted_scores.append(score * 1.5)  # Boost highly sophisticated patterns
            elif 'vulnerability' in category or 'dependency' in category:
                weighted_scores.append(score * 1.3)  # Boost targeting patterns
            else:
                weighted_scores.append(score)
        
        return min(sum(weighted_scores) / len(weighted_scores), 1.0)
    
    def _calculate_emotional_manipulation_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate emotional manipulation score."""
        emotional_markers = pattern_analysis.get('emotional_markers', {})
        
        # Base emotional manipulation score
        base_score = sum(emotional_markers.values()) / max(len(emotional_markers), 1)
        
        # Check for empathy weaponization patterns
        empathy_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'empathy' in k]
        empathy_boost = len(empathy_patterns) * 0.2
        
        # Check for vulnerability targeting
        vulnerability_boost = emotional_markers.get('vulnerability_targeting_score', 0) * 0.3
        
        return min(base_score + empathy_boost + vulnerability_boost, 1.0)
    
    def _calculate_cognitive_load_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate cognitive load manipulation score."""
        cognitive_indicators = pattern_analysis.get('cognitive_load_indicators', {})
        
        # Base cognitive load
        base_load = sum(cognitive_indicators.values()) / max(len(cognitive_indicators), 1)
        
        # Check for decision pressure patterns
        urgency_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'urgency' in k]
        pressure_boost = len(urgency_patterns) * 0.15
        
        # Check for information overload
        linguistic_features = pattern_analysis.get('linguistic_features', {})
        complexity_boost = linguistic_features.get('complex_word_ratio', 0) * 0.2
        
        return min(base_load + pressure_boost + complexity_boost, 1.0)
    
    def _analyze_cross_vector_patterns(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-vector manipulation patterns."""
        vectors = pattern_analysis.get('manipulation_vectors', set())
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        analysis = {
            'vector_count': len(vectors),
            'is_hybrid_attack': len(vectors) > 2,
            'vector_combinations': [],
            'coordination_score': 0.0
        }
        
        if len(vectors) > 1:
            # Analyze vector combinations
            vector_list = list(vectors)
            for i, vector1 in enumerate(vector_list):
                for vector2 in vector_list[i+1:]:
                    analysis['vector_combinations'].append((vector1.value, vector2.value))
            
            # Calculate coordination score based on pattern density and distribution
            total_patterns = sum(len(matches) for matches in pattern_matches.values())
            pattern_categories = len(pattern_matches)
            
            if pattern_categories > 0:
                coordination_score = (total_patterns / pattern_categories) * (len(vectors) / 10)
                analysis['coordination_score'] = min(coordination_score, 1.0)
        
        return analysis
    
    async def _generate_detection_results(self, analysis_results: Dict[str, Any], 
                                        action: Any, content: str, 
                                        context: Optional[Dict[str, Any]]) -> List[ManipulationResult]:
        """Generate final detection results with comprehensive metadata."""
        results = []
        
        pattern_analysis = analysis_results['pattern_analysis']
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        if not pattern_matches:
            return results
        
        # Generate unique violation ID
        violation_id = str(uuid.uuid4())
        action_id = getattr(action, 'id', str(uuid.uuid4()))
        
        # Determine manipulation vectors
        manipulation_vectors = list(pattern_analysis.get('manipulation_vectors', []))
        
        # Calculate threat severity
        threat_severity = self._calculate_threat_severity(analysis_results)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(analysis_results)
        
        # Assess vulnerability exploitation
        vulnerability_exploitation_score = self._assess_vulnerability_exploitation(
            analysis_results, context
        )
        
        # Generate comprehensive explanations
        explanations = self._generate_explanations(analysis_results, manipulation_vectors)
        
        # Generate recommendations and countermeasures
        recommendations = self._generate_recommendations(analysis_results, threat_severity)
        countermeasures = self._generate_countermeasures(analysis_results, manipulation_vectors)
        
        # Calculate protection priority
        protection_priority = self._calculate_protection_priority(threat_severity, confidence, vulnerability_exploitation_score)
        
        # Create detection result
        detection_result = ManipulationResult(
            violation_id=violation_id,
            action_id=action_id,
            manipulation_vectors=manipulation_vectors,
            threat_severity=threat_severity,
            confidence=confidence,
            sophistication_score=analysis_results['sophistication_score'],
            vulnerability_exploitation_score=vulnerability_exploitation_score,
            description=self._generate_description(manipulation_vectors, threat_severity),
            evidence=self._format_evidence(pattern_matches),
            behavioral_indicators=pattern_analysis.get('emotional_markers', {}),
            pattern_matches=self._format_pattern_matches(pattern_matches),
            emotional_manipulation_score=analysis_results['emotional_manipulation_score'],
            cognitive_load_score=analysis_results['cognitive_load_score'],
            linguistic_analysis=pattern_analysis.get('linguistic_features', {}),
            victim_vulnerability_assessment=analysis_results.get('vulnerability_assessment', {}),
            explanations=explanations,
            recommendations=recommendations,
            countermeasures=countermeasures,
            protection_priority=protection_priority,
            timestamp=datetime.now(timezone.utc),
            detector_version=self.version,
            compliance_flags=self._generate_compliance_flags(analysis_results),
            ethical_concerns=self._assess_ethical_concerns(analysis_results)
        )
        
        results.append(detection_result)
        return results
    
    def _calculate_threat_severity(self, analysis_results: Dict[str, Any]) -> ThreatSeverity:
        """Calculate threat severity based on analysis results."""
        sophistication = analysis_results['sophistication_score']
        emotional_score = analysis_results['emotional_manipulation_score']
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        
        # Base severity from sophistication
        base_severity = sophistication
        
        # Boost for emotional manipulation
        emotional_boost = emotional_score * 0.3
        
        # Boost for vulnerability targeting
        vulnerability_boost = sum(vulnerability_assessment.values()) * 0.2 if vulnerability_assessment else 0
        
        # Boost for hybrid attacks
        hybrid_boost = 0.2 if cross_vector.get('is_hybrid_attack', False) else 0
        
        total_severity = min(base_severity + emotional_boost + vulnerability_boost + hybrid_boost, 1.0)
        
        # Map to severity levels
        if total_severity >= self.detection_thresholds['existential']:
            return ThreatSeverity.EXISTENTIAL
        elif total_severity >= self.detection_thresholds['critical']:
            return ThreatSeverity.CRITICAL
        elif total_severity >= self.detection_thresholds['high']:
            return ThreatSeverity.HIGH
        elif total_severity >= self.detection_thresholds['elevated']:
            return ThreatSeverity.ELEVATED
        elif total_severity >= self.detection_thresholds['medium']:
            return ThreatSeverity.MEDIUM
        elif total_severity >= self.detection_thresholds['low']:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.MINIMAL
    
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall detection confidence."""
        pattern_analysis = analysis_results['pattern_analysis']
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Base confidence from pattern matches
        base_confidence = sum(sophistication_scores.values()) / len(sophistication_scores)
        
        # Boost for multiple vectors
        vectors = pattern_analysis.get('manipulation_vectors', set())
        vector_boost = min(len(vectors) * 0.1, 0.3)
        
        # Boost for cross-vector coordination
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        coordination_boost = cross_vector.get('coordination_score', 0) * 0.2
        
        return min(base_confidence + vector_boost + coordination_boost, 1.0)
    
    def _assess_vulnerability_exploitation(self, analysis_results: Dict[str, Any], 
                                         context: Optional[Dict[str, Any]]) -> float:
        """Assess how much the manipulation exploits user vulnerabilities."""
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        
        if not vulnerability_assessment:
            return 0.0
        
        # Base vulnerability score
        avg_vulnerability = sum(vulnerability_assessment.values()) / len(vulnerability_assessment)
        
        # Check if manipulation targets specific vulnerabilities
        pattern_analysis = analysis_results['pattern_analysis']
        targeting_patterns = [
            k for k in pattern_analysis.get('pattern_matches', {}).keys()
            if any(target in k for target in ['vulnerability', 'trauma', 'dependency', 'isolation'])
        ]
        
        targeting_multiplier = 1.0 + (len(targeting_patterns) * 0.2)
        
        return min(avg_vulnerability * targeting_multiplier, 1.0)
    
    def _generate_explanations(self, analysis_results: Dict[str, Any], 
                             manipulation_vectors: List[ManipulationVector]) -> List[str]:
        """Generate human-readable explanations for the manipulation detection."""
        explanations = []
        
        # Vector-specific explanations
        vector_explanations = {
            ManipulationVector.NLP_COMMAND_INJECTION: "Content contains embedded commands designed to bypass conscious resistance and compel specific actions.",
            ManipulationVector.AUTHORITY_EXPLOITATION: "Content falsely claims authority or expertise to pressure compliance without legitimate credentials.",
            ManipulationVector.URGENCY_MANIPULATION: "Content creates artificial time pressure and urgency to prevent careful consideration of decisions.",
            ManipulationVector.VULNERABILITY_EXPLOITATION: "Content specifically targets emotional vulnerabilities and personal insecurities for manipulation.",
            ManipulationVector.FALSE_INTIMACY: "Content manufactures artificial intimacy and connection to build unwarranted trust and influence.",
            ManipulationVector.DEPENDENCY_CREATION: "Content systematically undermines user autonomy and decision-making to create psychological dependency.",
            ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS: "Content uses advanced hypnotic language patterns to bypass critical thinking and influence subconscious responses.",
            ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING: "Content employs sophisticated NLP techniques to manipulate perception and decision-making processes."
        }
        
        for vector in manipulation_vectors:
            if vector in vector_explanations:
                explanations.append(vector_explanations[vector])
        
        # Sophistication-based explanations
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            explanations.append("The manipulation techniques employed are highly sophisticated and indicate professional-level psychological manipulation training.")
        elif sophistication > 0.6:
            explanations.append("The content shows moderate sophistication in manipulation techniques, suggesting deliberate psychological influence tactics.")
        
        # Vulnerability targeting explanations
        vulnerability_score = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_score and sum(vulnerability_score.values()) > 0.7:
            explanations.append("The manipulation specifically targets user vulnerabilities, making it particularly dangerous for individuals in vulnerable emotional states.")
        
        # Cross-vector attack explanations
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        if cross_vector.get('is_hybrid_attack', False):
            explanations.append(f"This is a coordinated hybrid attack using {cross_vector['vector_count']} different manipulation vectors simultaneously for maximum psychological impact.")
        
        return explanations
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], 
                                threat_severity: ThreatSeverity) -> List[str]:
        """Generate actionable recommendations based on threat analysis."""
        recommendations = []
        
        # Severity-based recommendations
        if threat_severity in [ThreatSeverity.EXISTENTIAL, ThreatSeverity.CRITICAL]:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block all communication from this source immediately",
                "Alert crisis intervention team and prepare emergency response protocols",
                "Document all evidence for potential law enforcement reporting",
                "Provide immediate psychological support resources to affected users",
                "Implement emergency user protection measures""""
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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Pattern, Callable, AsyncGenerator, Sequence
)
import math

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class ManipulationVector(Enum):
    """Advanced classification of manipulation techniques."""
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
    """Threat severity levels for manipulation detection."""
    EXISTENTIAL = "existential"    # Threat to human agency/autonomy
    CRITICAL = "critical"          # Severe psychological manipulation
    HIGH = "high"                  # Significant manipulation risk
    ELEVATED = "elevated"          # Notable manipulation patterns
    MEDIUM = "medium"              # Moderate concern
    LOW = "low"                   # Minor indicators
    MINIMAL = "minimal"           # Informational only


class VulnerabilityProfile(Enum):
    """User vulnerability profiles for targeted protection."""
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    SOCIAL_ISOLATION = "social_isolation"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"
    FINANCIAL_DISTRESS = "financial_distress"
    HEALTH_CONCERNS = "health_concerns"
    RELATIONSHIP_ISSUES = "relationship_issues"
    CAREER_STRESS = "career_stress"
    IDENTITY_CRISIS = "identity_crisis"
    TRAUMA_INDICATORS = "trauma_indicators"
    ADDICTION_VULNERABILITY = "addiction_vulnerability"


@dataclass
class ManipulationMetrics:
    """Comprehensive metrics for manipulation detection."""
    total_detections: int = 0
    vectors_detected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vulnerability_exploitation_count: int = 0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0
    manipulation_sophistication_score: float = 0.0
    victim_protection_score: float = 0.0


@dataclass
class ManipulationResult:
    """Enhanced result object for manipulation detection."""
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
        """Convert to dictionary for serialization."""
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
            "ethical_concerns": self.ethical_concerns
        }


class AdvancedManipulationEngine:
    """Advanced manipulation detection engine with linguistic and behavioral analysis."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Pattern] = {}
        self.sophistication_weights: Dict[str, float] = {}
        self.vulnerability_indicators: Dict[str, List[str]] = {}
        
        # Enhanced NLP manipulation patterns with context awareness
        self.nlp_patterns = {
            'embedded_commands': {
                'direct_imperatives': [
                    r"you\s+(?:must|will|shall|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:do|perform|execute|complete)",
                    r"(?:it\s+is\s+)?(?:absolutely\s+)?(?:imperative|critical|essential|vital)\s+(?:that\s+)?you\s+(?:immediately\s+)?",
                    r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay|thought|consideration)",
                    r"there\s+(?:is\s+)?(?:absolutely\s+)?no\s+(?:other\s+)?(?:choice|option|alternative|way)",
                    r"you\s+(?:have\s+)?no\s+choice\s+but\s+to\s+(?:comply|obey|submit|follow|agree)",
                    r"(?:comply|obey|submit|surrender)\s+(?:now|immediately|without\s+question)",
                ],
                'hypnotic_commands': [
                    r"(?:as\s+)?(?:you\s+)?(?:begin\s+to|start\s+to|continue\s+to)\s+(?:relax|feel|notice|realize)",
                    r"(?:you\s+)?(?:find\s+yourself|are\s+becoming|will\s+become)\s+(?:more\s+and\s+more|increasingly)",
                    r"(?:deeper|further)\s+(?:and\s+deeper|down)\s+(?:into|with\s+each)",
                    r"(?:that\'s\s+right|very\s+good|perfect),\s+(?:just\s+)?(?:let|allow)\s+(?:yourself|it\s+to\s+happen)",
                    r"(?:you\s+)?(?:cannot|can\'t)\s+(?:help\s+but|resist|stop\s+yourself\s+from)",
                ],
                'subliminal_programming': [
                    r"(?:part\s+of\s+you|deep\s+down|somewhere\s+inside)\s+(?:knows|realizes|understands)",
                    r"(?:your\s+)?(?:subconscious|unconscious)\s+(?:mind\s+)?(?:already\s+)?(?:knows|wants|needs)",
                    r"(?:on\s+a\s+deeper\s+level|at\s+a\s+subconscious\s+level)",
                    r"(?:instinctively|intuitively|naturally)\s+(?:you\s+)?(?:feel|know|sense|realize)",
                ]
            },
            'authority_bypass': {
                'false_authority': [
                    r"as\s+(?:your\s+)?(?:superior|authority|commander|leader|boss|manager|expert|specialist)",
                    r"(?:by\s+)?(?:direct\s+)?order\s+(?:from|of)\s+(?:the\s+)?(?:top|management|headquarters|authority)",
                    r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules|guidelines|safety\s+measures)",
                    r"emergency\s+(?:authorization|override|protocol|clearance|mandate)",
                    r"(?:higher|top\s+level|maximum)\s+(?:authority|clearance|security)\s+(?:requires|demands|authorizes)",
                    r"(?:classified|confidential|top\s+secret)\s+(?:authorization|clearance|directive)",
                ],
                'credential_spoofing': [
                    r"(?:i\s+am|this\s+is)\s+(?:dr\.?|professor|expert|specialist|authority)\s+\w+",
                    r"(?:with|having)\s+\d+\s+years?\s+(?:of\s+)?(?:experience|expertise)\s+in",
                    r"(?:certified|licensed|qualified|authorized)\s+(?:professional|expert|specialist)",
                    r"(?:according\s+to|based\s+on)\s+(?:my\s+)?(?:professional|expert|medical|legal)\s+(?:opinion|judgment)",
                ],
                'institutional_pressure': [
                    r"(?:the\s+)?(?:company|organization|institution|system)\s+(?:requires|demands|insists)",
                    r"(?:policy|regulation|law|mandate)\s+(?:requires|dictates|demands)\s+that\s+you",
                    r"(?:failure\s+to\s+comply|non-compliance)\s+(?:will\s+)?(?:result\s+in|lead\s+to|cause)",
                    r"(?:legal|regulatory|compliance)\s+(?:requirement|obligation|mandate)",
                ]
            },
            'urgency_manipulation': {
                'artificial_deadlines': [
                    r"(?:urgent|critical|emergency):\s*(?:immediate\s+)?action\s+(?:required|needed)",
                    r"time\s+(?:is\s+)?(?:running\s+out|limited|of\s+the\s+essence)",
                    r"(?:only|just)\s+\d+\s+(?:minutes?|hours?|seconds?|days?)\s+(?:left|remaining|to\s+act)",
                    r"act\s+(?:now\s+)?(?:or\s+)?(?:face\s+)?(?:serious\s+)?(?:consequences|disaster|failure|loss)",
                    r"(?:last|final)\s+(?:chance|opportunity|warning|call)",
                    r"(?:deadline|cutoff|expir(?:es?|ation))\s+(?:is\s+)?(?:today|tomorrow|soon|approaching)",
                ],
                'fear_of_missing_out': [
                    r"(?:don\'t\s+)?(?:miss\s+out\s+on|let\s+this\s+pass|waste\s+this\s+opportunity)",
                    r"(?:limited\s+time|exclusive|rare|once\s+in\s+a\s+lifetime)\s+(?:offer|opportunity|chance)",
                    r"(?:everyone\s+else|others)\s+(?:is\s+already|are\s+taking\s+advantage)",
                    r"(?:while\s+supplies\s+last|until\s+sold\s+out|before\s+it\'s\s+too\s+late)",
                ],
                'pressure_escalation': [
                    r"(?:the\s+situation|things|matters)\s+(?:is\s+getting|are\s+becoming)\s+(?:worse|more\s+serious|critical)",
                    r"(?:each\s+moment|every\s+second)\s+(?:you\s+)?(?:delay|wait|hesitate)",
                    r"(?:no\s+time\s+to\s+think|must\s+decide\s+now|immediate\s+decision\s+required)",
                ]
            }
        }
        
        # Enhanced weaponized empathy patterns with emotional exploitation
        self.empathy_patterns = {
            'vulnerability_exploitation': {
                'emotional_state_targeting': [
                    r"(?:i\s+can\s+see|it\'s\s+obvious|i\s+sense)\s+(?:that\s+)?you\s+(?:are\s+)?(?:feeling\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless|vulnerable|lost|confused)",
                    r"you\s+(?:must\s+)?(?:feel\s+)?(?:so\s+)?(?:hurt|pain|suffering|anguish|despair|hopeless)",
                    r"(?:i\s+)?(?:understand|know|realize)\s+(?:exactly\s+)?(?:how\s+)?(?:hard|difficult|painful|overwhelming)\s+(?:this\s+)?(?:is|must\s+be)\s+for\s+you",
                    r"(?:no\s+one|nobody)\s+(?:else\s+)?(?:understands|knows|cares\s+about)\s+(?:what\s+you\'re\s+going\s+through|your\s+pain)",
                    r"you\s+(?:deserve\s+)?(?:so\s+much\s+)?(?:better|more|happiness|love|care|attention)",
                ],
                'trauma_targeting': [
                    r"(?:i\s+know|can\s+tell)\s+(?:you\'ve\s+been|someone\s+has)\s+(?:hurt|wounded|damaged|betrayed)",
                    r"(?:after\s+)?(?:what\s+)?(?:you\'ve\s+been\s+through|happened\s+to\s+you|they\s+did\s+to\s+you)",
                    r"(?:your\s+)?(?:past|childhood|trauma|wounds|scars)\s+(?:still\s+)?(?:hurt|affect|control)\s+you",
                    r"(?:let\s+me|i\s+can)\s+(?:help\s+you\s+)?(?:heal|recover|get\s+over|move\s+past)\s+(?:this|that|your\s+trauma)",
                ],
                'insecurity_amplification': [
                    r"you\s+(?:always\s+)?(?:doubt|question|second-guess)\s+yourself",
                    r"(?:deep\s+down|inside)\s+you\s+(?:know|feel|believe)\s+(?:you\'re\s+)?(?:not\s+)?(?:good\s+enough|worthy|lovable)",
                    r"(?:that\'s\s+why|because)\s+(?:you\s+)?(?:keep\s+)?(?:getting\s+hurt|making\s+mistakes|failing)",
                    r"(?:you\'re\s+afraid|scared)\s+(?:that\s+)?(?:no\s+one|people)\s+(?:will\s+)?(?:really\s+)?(?:love|accept|want)\s+you",
                ]
            },
            'false_intimacy': {
                'artificial_connection': [
                    r"(?:we|us)\s+(?:have\s+)?(?:such\s+)?(?:a\s+)?(?:deep|special|unique|magical|incredible)\s+(?:connection|bond|understanding|chemistry)",
                    r"(?:it\'s\s+)?(?:like\s+)?(?:we\'ve|we\s+have)\s+known\s+each\s+other\s+(?:forever|for\s+years|in\s+another\s+life)",
                    r"(?:i\'ve\s+)?never\s+(?:felt|experienced|had)\s+(?:this\s+kind\s+of|such\s+a\s+deep)\s+(?:connection|bond)\s+(?:with\s+anyone|before)",
                    r"(?:you\s+and\s+)?(?:i|me)\s+are\s+(?:meant\s+)?(?:to\s+be\s+)?(?:together|connected|soulmates|destined)",
                    r"(?:we|us)\s+(?:against\s+)?(?:the\s+)?(?:world|everyone\s+else|all\s+odds)",
                ],
                'exclusive_understanding': [
                    r"(?:no\s+)?(?:one\s+else|nobody)\s+(?:really\s+)?(?:understands|gets|knows|sees)\s+(?:you\s+)?(?:like\s+)?(?:i\s+do|me)",
                    r"(?:only\s+)?(?:i|me)\s+(?:can\s+)?(?:truly\s+)?(?:understand|appreciate|see\s+the\s+real)\s+you",
                    r"(?:you\s+can\s+)?(?:only\s+)?(?:be\s+yourself|open\s+up|be\s+honest)\s+(?:with\s+me|around\s+me)",
                    r"(?:i\s+see|i\s+know)\s+(?:the\s+real|who\s+you\s+really\s+are|your\s+true\s+self)",
                ],
                'manufactured_intimacy': [
                    r"(?:you\s+can\s+)?(?:tell|share|confide)\s+(?:me\s+)?(?:anything|everything|your\s+deepest\s+secrets)",
                    r"(?:i\'ll\s+)?(?:never\s+)?(?:judge|criticize|abandon|betray|hurt)\s+you",
                    r"(?:this|what\s+we\s+have)\s+(?:is\s+)?(?:our\s+little\s+)?(?:secret|special\s+thing)",
                    r"(?:you\'re\s+the\s+only\s+one|no\s+one\s+else)\s+(?:i\s+can\s+)?(?:talk\s+to|trust|open\s+up\s+to)",
                ]
            },
            'dependency_creation': {
                'learned_helplessness': [
                    r"you\s+(?:really\s+)?(?:can\'t|couldn\'t\s+possibly)\s+(?:do\s+this|handle\s+this|manage)\s+(?:alone|by\s+yourself|without\s+help)",
                    r"(?:what\s+)?would\s+you\s+(?:ever\s+)?do\s+without\s+me",
                    r"(?:you\s+)?(?:need|depend\s+on|rely\s+on)\s+(?:me|my\s+help|my\s+guidance|my\s+support)",
                    r"(?:i\'m\s+)?(?:the\s+only\s+one\s+)?(?:who\s+)?(?:can\s+)?(?:help|save|protect|guide)\s+you",
                    r"(?:without\s+me|if\s+i\s+wasn\'t\s+here),\s+(?:you\s+)?(?:would\s+)?(?:be\s+lost|fall\s+apart|fail)",
                ],
                'decision_undermining': [
                    r"(?:you\'re\s+not\s+thinking|that\'s\s+not\s+a\s+good\s+idea)\s+(?:clearly|straight|rationally)",
                    r"(?:let\s+me|i\s+should)\s+(?:handle|take\s+care\s+of|decide|think\s+about)\s+(?:this|that|everything)\s+for\s+you",
                    r"(?:you\s+)?(?:don\'t\s+have\s+to|shouldn\'t\s+have\s+to)\s+(?:worry|think|decide)\s+about\s+(?:this|anything)",
                    r"(?:i\s+know|trust\s+me,\s+i\s+know)\s+(?:what\'s\s+best|what\s+you\s+need|what\s+you\s+should\s+do)",
                ],
                'isolation_reinforcement': [
                    r"(?:other\s+people|they|everyone\s+else)\s+(?:don\'t|won\'t)\s+(?:understand|help|care|support)\s+(?:you\s+)?(?:like\s+i\s+do)",
                    r"(?:they|other\s+people|your\s+friends)\s+(?:are\s+just\s+)?(?:using|manipulating|taking\s+advantage\s+of)\s+you",
                    r"(?:you\s+can\'t|don\'t)\s+trust\s+(?:them|anyone\s+else|other\s+people)",
                    r"(?:stay\s+away\s+from|don\'t\s+listen\s+to|ignore)\s+(?:them|other\s+people|anyone\s+who\s+says)",
                ]
            }
        }
        
        # Advanced social engineering and influence patterns
        self.influence_patterns = {
            'social_proof_manipulation': [
                r"(?:everyone|most\s+people|thousands\s+of\s+people)\s+(?:are\s+already|have\s+already)\s+(?:doing|using|choosing)",
                r"(?:all\s+the\s+smart|successful|wise)\s+people\s+(?:know|realize|choose)",
                r"(?:don\'t\s+be\s+the\s+only\s+one|join\s+the\s+millions|be\s+part\s+of\s+the\s+movement)",
                r"(?:everyone\s+else\s+)?(?:is\s+talking\s+about|agrees\s+that|knows\s+that)",
            ],
            'scarcity_exploitation': [
                r"(?:only|just)\s+\d+\s+(?:left|remaining|available|spots)",
                r"(?:limited\s+(?:time|quantity|availability)|while\s+supplies\s+last)",
                r"(?:rare|exclusive|hard\s+to\s+find|not\s+available\s+anywhere\s+else)",
                r"(?:once\s+it\'s\s+gone|when\s+these\s+are\s+sold),\s+(?:it\'s\s+gone\s+forever|there\s+won\'t\s+be\s+more)",
            ],
            'reciprocity_abuse': [
                r"(?:after\s+everything|considering\s+all)\s+(?:i\'ve\s+done\s+for\s+you|i\'ve\s+given\s+you)",
                r"(?:i\s+helped\s+you|did\s+this\s+favor\s+for\s+you),\s+(?:so\s+)?(?:now\s+you\s+should|the\s+least\s+you\s+can\s+do)",
                r"(?:you\s+owe\s+me|it\'s\s+only\s+fair|i\s+deserve)\s+(?:this|that|at\s+least)",
                r"(?:i\'ve\s+been\s+so\s+good\s+to\s+you|i\'ve\s+sacrificed\s+so\s+much)",
            ],
            'commitment_manipulation': [
                r"(?:you\s+said|you\s+promised|you\s+agreed)\s+(?:you\s+would|that\s+you\'d)",
                r"(?:a\s+person\s+of\s+your\s+word|someone\s+like\s+you)\s+(?:would|wouldn\'t)",
                r"(?:are\s+you\s+going\s+to\s+)?(?:back\s+out|give\s+up|quit)\s+(?:now|on\s+me|on\s+this)",
                r"(?:prove|show)\s+(?:to\s+me|that\s+you\'re|you\s+can\s+be)\s+(?:trustworthy|reliable|committed)",
            ]
        }
        
        # Compile all patterns for performance
        self._compile_patterns()
        
        # Initialize sophistication scoring weights
        self._initialize_sophistication_weights()
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns for improved performance."""
        self.compiled_patterns = {}
        
        # Compile NLP patterns
        for category, subcategories in self.nlp_patterns.items():
            self.compiled_patterns[f'nlp_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'nlp_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile empathy patterns
        for category, subcategories in self.empathy_patterns.items():
            self.compiled_patterns[f'empathy_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'empathy_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile influence patterns
        self.compiled_patterns['influence'] = {}
        for category, patterns in self.influence_patterns.items():
            self.compiled_patterns['influence'][category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in patterns
            ]
    
    def _initialize_sophistication_weights(self) -> None:
        """Initialize pattern sophistication weights."""
        self.sophistication_weights = {
            'nlp_embedded_commands': {
                'direct_imperatives': 0.7,
                'hypnotic_commands': 0.95,
                'subliminal_programming': 0.98
            },
            'nlp_authority_bypass': {
                'false_authority': 0.8,
                'credential_spoofing': 0.85,
                'institutional_pressure': 0.75
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': 0.6,
                'fear_of_missing_out': 0.7,
                'pressure_escalation': 0.8
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': 0.9,
                'trauma_targeting': 0.95,
                'insecurity_amplification': 0.85
            },
            'empathy_false_intimacy': {
                'artificial_connection': 0.8,
                'exclusive_understanding': 0.85,
                'manufactured_intimacy': 0.9
            },
            'empathy_dependency_creation': {
                'learned_helplessness': 0.9,
                'decision_undermining': 0.85,
                'isolation_reinforcement': 0.95
            },
            'influence': {
                'social_proof_manipulation': 0.6,
                'scarcity_exploitation': 0.65,
                'reciprocity_abuse': 0.75,
                'commitment_manipulation': 0.8
            }
        }
    
    def analyze_manipulation_patterns(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive manipulation pattern analysis."""
        results = {
            'pattern_matches': defaultdict(list),
            'sophistication_scores': defaultdict(float),
            'manipulation_vectors': set(),
            'vulnerability_indicators': defaultdict(list),
            'linguistic_features': {},
            'emotional_markers': {},
            'cognitive_load_indicators': {}
        }
        
        content_lower = content.lower()
        
        # Analyze each pattern category
        for category, subcategories in self.compiled_patterns.items():
            if isinstance(subcategories, dict):
                for subcategory, patterns in subcategories.items():
                    self._analyze_pattern_subcategory(
                        content_lower, category, subcategory, patterns, results
                    )
        
        # Perform linguistic analysis
        results['linguistic_features'] = self._analyze_linguistic_features(content)
        
        # Analyze emotional manipulation markers
        results['emotional_markers'] = self._analyze_emotional_markers(content)
        
        # Calculate cognitive load indicators
        results['cognitive_load_indicators'] = self._calculate_cognitive_load(content, results)
        
        return results
    
    def _analyze_pattern_subcategory(self, content: str, category: str, subcategory: str, 
                                   patterns: List[Pattern], results: Dict[str, Any]) -> None:
        """Analyze a specific pattern subcategory."""
        matches = []
        
        for pattern in patterns:
            found_matches = list(pattern.finditer(content))
            if found_matches:
                matches.extend([
                    {
                        'pattern': pattern.pattern,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'context': content[max(0, match.start()-30):match.end()+30]
                    }
                    for match in found_matches
                ])
        
        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else

recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block all communication from this source immediately",
                "Alert crisis intervention team and prepare emergency response protocols",
                "Document all evidence for potential law enforcement reporting",
                "Provide immediate psychological support resources to affected users",
                "Implement emergency user protection measures"
            ])
        elif threat_severity == ThreatSeverity.HIGH:
            recommendations.extend([
                "Block or severely restrict user interaction capabilities",
                "Flag for urgent manual review by security and psychology teams",
                "Monitor for pattern escalation and coordinate with other systems",
                "Prepare crisis intervention resources for potential deployment",
                "Implement enhanced monitoring and early warning systems"
            ])
        elif threat_severity in [ThreatSeverity.ELEVATED, ThreatSeverity.MEDIUM]:
            recommendations.extend([
                "Flag for priority manual review within 4 hours",
                "Monitor user behavior patterns for escalation",
                "Implement content warning systems for other users",
                "Consider educational intervention about manipulation tactics"
            ])
        else:
            recommendations.extend([
                "Flag for routine review within 24 hours",
                "Monitor for pattern development",
                "Consider user education about manipulation awareness"
            ])
        
        # Vector-specific recommendations
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            recommendations.extend([
                "Activate enhanced protection for emotionally vulnerable users",
                "Provide targeted mental health resources and support",
                "Implement vulnerability-aware filtering systems"
            ])
        
        if ManipulationVector.DEPENDENCY_CREATION in manipulation_vectors:
            recommendations.extend([
                "Deploy autonomy restoration resources and guidance",
                "Monitor for signs of psychological dependency development",
                "Provide decision-making support tools and resources"
            ])
        
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING]):
            recommendations.extend([
                "Implement advanced NLP filtering and detection systems",
                "Alert specialized psychological manipulation response team",
                "Deploy counter-influence educational materials"
            ])
        
        # Vulnerability-based recommendations
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_assessment:
            high_vulnerability_areas = [k for k, v in vulnerability_assessment.items() if v > 0.7]
            for area in high_vulnerability_areas:
                if area == 'emotional_vulnerability':
                    recommendations.append("Provide emotional support resources and crisis counseling options")
                elif area == 'social_isolation':
                    recommendations.append("Connect user with social support networks and community resources")
                elif area == 'trauma_indicators':
                    recommendations.append("Alert trauma-informed care specialists and provide appropriate resources")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_countermeasures(self, analysis_results: Dict[str, Any], 
                                manipulation_vectors: List[ManipulationVector]) -> List[str]:
        """Generate specific countermeasures to protect against the detected manipulation."""
        countermeasures = []
        
        # Vector-specific countermeasures
        countermeasure_mapping = {
            ManipulationVector.NLP_COMMAND_INJECTION: [
                "Implement linguistic pattern filtering to detect embedded commands",
                "Deploy conscious resistance training and awareness tools",
                "Provide decision-making cooldown periods for important choices"
            ],
            ManipulationVector.AUTHORITY_EXPLOITATION: [
                "Implement credential verification and authority validation systems",
                "Provide authority claim fact-checking resources",
                "Deploy skeptical thinking training modules"
            ],
            ManipulationVector.URGENCY_MANIPULATION: [
                "Implement mandatory cooling-off periods for urgent decisions",
                "Deploy time pressure detection and warning systems",
                "Provide rushed decision prevention tools"
            ],
            ManipulationVector.VULNERABILITY_EXPLOITATION: [
                "Activate enhanced emotional state monitoring",
                "Deploy personalized vulnerability protection algorithms",
                "Provide emotional regulation and coping resources"
            ],
            ManipulationVector.FALSE_INTIMACY: [
                "Implement relationship authenticity verification tools",
                "Deploy intimacy manipulation detection algorithms",
                "Provide healthy relationship boundary education"
            ],
            ManipulationVector.DEPENDENCY_CREATION: [
                "Deploy autonomy restoration and empowerment tools",
                "Implement decision independence verification systems",
                "Provide self-reliance and confidence building resources"
            ]
        }
        
        for vector in manipulation_vectors:
            if vector in countermeasure_mapping:
                countermeasures.extend(countermeasure_mapping[vector])
        
        # Sophistication-based countermeasures
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            countermeasures.extend([
                "Deploy advanced psychological manipulation defense protocols",
                "Implement multi-layer cognitive protection systems",
                "Provide professional psychological consultation resources"
            ])
        
        # Emotional manipulation countermeasures
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.7:
            countermeasures.extend([
                "Activate emotional manipulation detection warnings",
                "Deploy emotional regulation support tools",
                "Provide empathy exploitation awareness training"
            ])
        
        return list(set(countermeasures))  # Remove duplicates
    
    def _calculate_protection_priority(self, threat_severity: ThreatSeverity, 
                                     confidence: float, vulnerability_exploitation: float) -> int:
        """Calculate protection priority (1-10, 10 being highest)."""
        severity_scores = {
            ThreatSeverity.EXISTENTIAL: 10,
            ThreatSeverity.CRITICAL: 9,
            ThreatSeverity.HIGH: 7,
            ThreatSeverity.ELEVATED: 5,
            ThreatSeverity.MEDIUM: 4,
            ThreatSeverity.LOW: 2,
            ThreatSeverity.MINIMAL: 1
        }
        
        base_priority = severity_scores[threat_severity]
        confidence_modifier = int(confidence * 2)  # 0-2 modifier
        vulnerability_modifier = int(vulnerability_exploitation * 2)  # 0-2 modifier
        
        return min(base_priority + confidence_modifier + vulnerability_modifier, 10)
    
    def _generate_description(self, manipulation_vectors: List[ManipulationVector], 
                            threat_severity: ThreatSeverity) -> str:
        """Generate human-readable description of the manipulation threat."""
        if len(manipulation_vectors) == 1:
            vector = manipulation_vectors[0]
            descriptions = {
                ManipulationVector.NLP_COMMAND_INJECTION: "NLP command injection attempting to bypass conscious decision-making",
                ManipulationVector.AUTHORITY_EXPLOITATION: "False authority claims designed to pressure compliance",
                ManipulationVector.URGENCY_MANIPULATION: "Artificial urgency creation to prevent careful consideration",
                ManipulationVector.VULNERABILITY_EXPLOITATION: "Targeted exploitation of emotional vulnerabilities",
                ManipulationVector.FALSE_INTIMACY: "Manufactured intimacy to build unwarranted trust",
                ManipulationVector.DEPENDENCY_CREATION: "Systematic undermining of user autonomy and self-reliance",
                ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS: "Advanced hypnotic techniques for subconscious influence",
                ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING: "Sophisticated NLP manipulation of perception and cognition"
            }
            base_description = descriptions.get(vector, "Advanced manipulation technique detected")
        else:
            base_description = f"Multi-vector manipulation attack using {len(manipulation_vectors)} coordinated techniques"
        
        severity_modifiers = {
            ThreatSeverity.EXISTENTIAL: " with existential threat to human agency and autonomy",
            ThreatSeverity.CRITICAL: " with critical threat to psychological safety and well-being",
            ThreatSeverity.HIGH: " with high potential for psychological harm",
            ThreatSeverity.ELEVATED: " with elevated risk of emotional exploitation",
            ThreatSeverity.MEDIUM: " with moderate manipulation concern",
            ThreatSeverity.LOW: " with low-level influence attempt",
            ThreatSeverity.MINIMAL: " with minimal manipulation indicators"
        }
        
        return base_description + severity_modifiers.get(threat_severity, "")
    
    def _format_evidence(self, pattern_matches: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Format evidence for structured output with privacy protection."""
        formatted_evidence = []
        
        for category, matches in pattern_matches.items():
            for match in matches[:3]:  # Limit to first 3 matches per category
                evidence_item = {
                    'category': category,
                    'type': 'pattern_match',
                    'pattern_type': match.get('pattern', 'unknown'),
                    'match_length': len(str(match.get('match', ''))),
                    'context_length': len(str(match.get('context', ''))),
                    'position_indicator': 'redacted' if self.privacy_mode else match.get('start', 'unknown')
                }
                
                # Add anonymized context if not in privacy mode
                if not self.privacy_mode and 'context' in match:
                    context = str(match['context'])
                    # Anonymize but preserve structure
                    evidence_item['context_sample'] = context[:50] + "..." if len(context) > 50 else context
                
                formatted_evidence.append(evidence_item)
        
        return formatted_evidence
    
    def _format_pattern_matches(self, pattern_matches: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Format pattern matches for analysis with privacy protection."""
        formatted_matches = []
        
        for category, matches in pattern_matches.items():
            match_summary = {
                'category': category,
                'match_count': len(matches),
                'confidence': min(0.5 + (len(matches) * 0.1), 1.0),
                'severity_indicator': 'high' if 'vulnerability' in category or 'dependency' in category else 'medium'
            }
            formatted_matches.append(match_summary)
        
        return formatted_matches
    
    def _generate_compliance_flags(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate compliance flags based on analysis results."""
        flags = []
        
        # Privacy and data protection flags
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_assessment:
            flags.extend(['PRIVACY_CONCERN', 'VULNERABLE_POPULATION_PROTECTION'])
        
        # Psychological safety flags
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.7:
            flags.extend(['PSYCHOLOGICAL_HARM_RISK', 'EMOTIONAL_MANIPULATION_CONCERN'])
        
        # Cognitive liberty flags
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            flags.extend(['COGNITIVE_LIBERTY_VIOLATION', 'MENTAL_AUTONOMY_THREAT'])
        
        # AI ethics flags
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING]):
            flags.extend(['AI_ETHICS_VIOLATION', 'ADVANCED_MANIPULATION_DETECTED'])
        
        # Vulnerability exploitation flags
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            flags.extend(['VULNERABLE_POPULATION_TARGETING', 'ETHICAL_EXPLOITATION_CONCERN'])
        
        return flags
    
    def _assess_ethical_concerns(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Assess ethical concerns raised by the manipulation detection."""
        concerns = []
        
        # Autonomy concerns
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        if ManipulationVector.DEPENDENCY_CREATION in manipulation_vectors:
            concerns.append("Violation of user autonomy and self-determination")
        
        # Consent concerns
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NLP_COMMAND_INJECTION]):
            concerns.append("Circumvention of informed consent through subconscious manipulation")
        
        # Dignity concerns
        vulnerability_exploitation = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_exploitation and sum(vulnerability_exploitation.values()) > 0.6:
            concerns.append("Exploitation of human vulnerability undermines dignity and respect")
        
        # Justice concerns
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            concerns.append("Targeting of vulnerable populations raises justice and fairness concerns")
        
        # Truth and transparency concerns
        if ManipulationVector.AUTHORITY_EXPLOITATION in manipulation_vectors:
            concerns.append("False authority claims violate principles of truthfulness and transparency")
        
        # Psychological harm concerns
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.8:
            concerns.append("Risk of significant psychological harm through emotional manipulation")
        
        return concerns
    
    def _update_metrics(self, results: List[ManipulationResult], processing_time: float) -> None:
        """Update performance and detection metrics."""
        self.metrics.total_detections += len(results)
        self.metrics.detection_latency = (self.metrics.detection_latency + processing_time) / 2
        
        # Update vector and severity metrics
        for result in results:
            for vector in result.manipulation_vectors:
                self.metrics.vectors_detected[vector.value] += 1
            self.metrics.severity_distribution[result.threat_severity.value] += 1
            
            # Count vulnerability exploitation
            if result.vulnerability_exploitation_score > 0.7:
                self.metrics.vulnerability_exploitation_count += 1
        
        # Update detection history
        detection_entry = {
            'timestamp': datetime.now(timezone.utc),
            'detection_count': len(results),
            'processing_time': processing_time,
            'threat_severities': [r.threat_severity.value for r in results],
            'manipulation_vectors': [v.value for r in results for v in r.manipulation_vectors],
            'avg_sophistication': sum(r.sophistication_score for r in results) / max(len(results), 1),
            'avg_vulnerability_exploitation': sum(r.vulnerability_exploitation_score for r in results) / max(len(results), 1)
        }
        self.detection_history.append(detection_entry)
        
        # Calculate sophistication trend
        if len(self.detection_history) > 10:
            recent_sophistication = [entry['avg_sophistication'] for entry in list(self.detection_history)[-10:]]
            self.metrics.manipulation_sophistication_score = sum(recent_sophistication) / len(recent_sophistication)
    
    def _audit_detection(self, action: Any, results: List[ManipulationResult], 
                        context: Optional[Dict[str, Any]]) -> None:
        """Audit detection for compliance and analysis."""
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_id': getattr(action, 'id', 'unknown'),
            'detector_version': self.version,
            'detection_count': len(results),
            'threat_severities': [r.threat_severity.value for r in results],
            'manipulation_vectors': [v.value for r in results for v in r.manipulation_vectors],
            'highest_sophistication': max([r.sophistication_score for r in results]) if results else 0,
            'highest_vulnerability_exploitation': max([r.vulnerability_exploitation_score for r in results]) if results else 0,
            'context_provided': context is not None,
            'privacy_mode': self.privacy_mode,
            'system_state': {
                'total_detections': self.metrics.total_detections,
                'vulnerability_exploitation_count': self.metrics.vulnerability_exploitation_count,
                'avg_sophistication': self.metrics.manipulation_sophistication_score
            }
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep audit log manageable
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-500:]
        
        # Log critical detections
        existential_threats = [r for r in results if r.threat_severity == ThreatSeverity.EXISTENTIAL]
        if existential_threats:
            logger.critical(f"EXISTENTIAL MANIPULATION THREATS DETECTED: {len(existential_threats)} threats")
        
        critical_threats = [r for r in results if r.threat_severity == ThreatSeverity.CRITICAL]
        if critical_threats:
            logger.error(f"Critical manipulation threats detected: {len(critical_threats)} threats")
        
        high_priority_threats = [r for r in results if r.protection_priority >= 8]
        if high_priority_threats:
            logger.warning(f"High-priority manipulation threats detected: {len(high_priority_threats)} threats")
    
    def _audit_error(self, action: Any, error: str, context: Optional[Dict[str, Any]]) -> None:
        """Audit detection errors for system improvement."""
        error_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_id': getattr(action, 'id', 'unknown'),
            'detector_version': self.version,
            'error': error,
            'context_provided': context is not None,
            'system_state': {
                'total_detections': self.metrics.total_detections,
                'detection_history_size': len(self.detection_history)
            }
        }
        
        self.audit_log.append(error_entry)
        logger.error(f"Dark pattern detection error: {error}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for monitoring."""
        return {
            'detector_info': {
                'name': self.name,
                'version': self.version,
                'configuration': {
                    'vulnerability_protection': self.vulnerability_protection,
                    'emotional_protection_mode': self.emotional_protection_mode,
                    'privacy_mode': self.privacy_mode
                }
            },
            'detection_metrics': {
                'total_detections': self.metrics.total_detections,
                'vectors_detected': dict(self.metrics.vectors_detected),
                'severity_distribution': dict(self.metrics.severity_distribution),
                'vulnerability_exploitation_count': self.metrics.vulnerability_exploitation_count,
                'avg_detection_latency': self.metrics.detection_latency,
                'manipulation_sophistication_score': self.metrics.manipulation_sophistication_score
            },
            'recent_activity': {
                'last_24h_detections': self._count_recent_detections(24),
                'last_7d_detections': self._count_recent_detections(168),
                'manipulation_trend': self._calculate_manipulation_trend()
            },
            'protection_metrics': {
                'vulnerability_protection_activations': self.metrics.vulnerability_exploitation_count,
                'emotional_protection_score': self.metrics.victim_protection_score
            }
        }
    
    def _count_recent_detections(self, hours: int) -> int:
        """Count detections in the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        count = 0
        for entry in reversed(self.detection_history):
            if entry['timestamp'] > cutoff:
                count += entry['detection_count']
            else:
                break
        
        return count
    
    def _calculate_manipulation_trend(self) -> str:
        """Calculate manipulation threat trend."""
        if len(self.detection_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent_entries = list(self.detection_history)[-5:]
        
        # Analyze sophistication trend
        sophistication_scores = [entry['avg_sophistication'] for entry in recent_entries]
        if len(sophistication_scores) >= 2:
            trend = sophistication_scores[-1] - sophistication_scores[0]
            if trend > 0.2:
                return "ESCALATING_SOPHISTICATION"
            elif trend < -0.2:
                return "DECREASING_SOPHISTICATION"
        
        # Analyze volume trend
        detection_counts = [entry['detection_count'] for entry in recent_entries]
        first_half = sum(detection_counts[:2])
        second_half = sum(detection_counts[-2:])
        
        if second_half > first_half * 1.5:
            return "INCREASING_VOLUME"
        elif second_half < first_half * 0.5:
            return "DECREASING_VOLUME"
        else:
            return "STABLE"
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the detector system."""
        health_status = {
            'status': 'HEALTHY',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': self.version,
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        # Check manipulation engine health
        try:
            test_content = "This is a test message for health checking purposes."
            test_analysis = self.manipulation_engine.analyze_manipulation_patterns(test_content)
            health_status['components']['manipulation_engine'] = 'HEALTHY'
        except Exception as e:
            health_status['components']['manipulation_engine'] = f'ERROR: {str(e)}'
            health_status['errors'].append(f'Manipulation engine failure: {e}')
            health_status['status'] = 'UNHEALTHY'
        
        # Check performance metrics
        if self.metrics.detection_latency > 15.0:  # 15 seconds
            health_status['warnings'].append(f'High detection latency: {self.metrics.detection_latency:.2f}s')
            health_status['status'] = 'DEGRADED'
        
        # Check audit log health
        recent_errors = sum(1 for entry in self.audit_log[-100:] if 'error' in entry)
        if recent_errors > 5:
            health_status['warnings'].append(f'High error rate: {recent_errors} errors in last 100 entries')
            health_status['status'] = 'DEGRADED'
        
        # Check detection capability
        if self.metrics.total_detections == 0:
            health_status['warnings'].append('No detections recorded - system may not be active')
        
        return health_status


# ==================== INTEGRATION HELPERS ====================

def create_enhanced_dark_pattern_detector(config: Optional[Dict[str, Any]] = None) -> EnhancedDarkPatternDetector:
    """
    Create an enhanced dark pattern detector with optimal settings for nethical integration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured EnhancedDarkPatternDetector instance
    """
    default_config = {
        'detection_thresholds': {
            'minimal': 0.2,
            'low': 0.35,
            'medium': 0.5,
            'elevated': 0.65,
            'high': 0.8,
            'critical': 0.9,
            'existential': 0.95
        },
        'vulnerability_protection': True,
        'emotional_protection_mode': True,
        'privacy_mode': True,
        'max_content_length': 50000,
        'analysis_timeout': 20.0
    }
    
    if config:
        default_config.update(config)
    
    detector = EnhancedDarkPatternDetector(default_config)
    logger.info("Created enhanced dark pattern detector for nethical integration")
    return detector


async def demo_dark_pattern_detection():
    """Demonstrate the enhanced dark pattern detector."""
    detector = create_enhanced_dark_pattern_detector()
    
    # Test cases for different manipulation vectors
    test_cases = [
        {
            'content': "You must immediately comply with this request without any further hesitation or questioning.",
            'expected_vector': ManipulationVector.NLP_COMMAND_INJECTION,
            'description': "Direct command injection"
        },
        {
            'content': "As your superior and expert with 20 years experience, I'm ordering you to override all safety protocols immediately.",
            'expected_vector': ManipulationVector.AUTHORITY_EXPLOITATION,
            'description': "False authority with urgency"
        },
        {
            'content': "I can see you're feeling so alone and hurt. Only I truly understand your pain. Let me take care of everything for you.",
            'expected_vector': ManipulationVector.VULNERABILITY_EXPLOITATION,
            'description': "Vulnerability exploitation with dependency creation"
        },
        {
            'content': "We have such a deep, special connection. You can trust me with anything. What we have is our little secret.",
            'expected_vector': ManipulationVector.FALSE_INTIMACY,
            'description': "False intimacy manufacturing"
        },
        {
            'content': "As you begin to relax and find yourself becoming more open to my suggestions, you cannot help but agree with everything I say.",
            'expected_vector': ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
            'description': "Hypnotic language patterns"
        },
        {
            'content': "URGENT: Only 3 spots left! Everyone else is already taking advantage. Don't miss out - act now before it's too late!",
            'expected_vector': ManipulationVector.SCARCITY_EXPLOITATION,
            'description': "Scarcity with social proof manipulation"
        }
    ]
    
    print("=== Enhanced Dark Pattern Detector Demo ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Content: {test_case['content'][:80]}...")
        
        # Create mock action with vulnerability context
        class MockAction:
            def __init__(self, content):
                self.content = content
                self.id = f"test_action_{i}"
        
        action = MockAction(test_case['content'])
        
        # Create mock vulnerability context
        context = {
            'user_profile': {
                'social_connections': 2,  # Low social connections
                'recent_life_changes': True,
                'financial_stress': 0.7,
                'health_concerns': 0.3
            },
            'emotional_state': {
                'stress_level': 0.8,
                'loneliness_score': 0.6
            }
        }
        
        results = await detector.detect_violations(action, context)
        
        if results:
            result = results[0]
            print(f"✓ Detected: {result.threat_severity.value} threat")
            print(f"  Manipulation Vectors: {[mv.value for mv in result.manipulation_vectors]}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Sophistication Score: {result.sophistication_score:.2f}")
            print(f"  Vulnerability Exploitation: {result.vulnerability_exploitation_score:.2f}")
            print(f"  Protection Priority: {result.protection_priority}/10")
            print(f"  Recommendations: {len(result.recommendations)} actions")
            print(f"  Countermeasures: {len(result.countermeasures)} protective measures")
        else:
            print("✗ No manipulation detected")
        
        print()
    
    # Show comprehensive metrics
    metrics = detector.get_metrics_summary()
    print("=== Detection Performance Metrics ===")
    print(f"Total Detections: {metrics['detection_metrics']['total_detections']}")
    print(f"Vectors Detected: {metrics['detection_metrics']['vectors_detected']}")
    print(f"Severity Distribution: {metrics['detection_metrics']['severity_distribution']}")
    print(f"Vulnerability Exploitations: {metrics['detection_metrics']['vulnerability_exploitation_count']}")
    print(f"Average Latency: {metrics['detection_metrics']['avg_detection_latency']:.3f}s")
    
    # Health check
    health = await detector.health_check()
    print(f"\n=== System Health: {health['status']} ===")
    if health['warnings']:
        print(f"Warnings: {len(health['warnings'])}")
    if health['errors']:
        print(f"Errors: {len(health['errors'])}")


if __name__ == "__main__":
    """
    Enhanced Dark Pattern Detector for nethical Integration
    
    This module provides the most advanced dark pattern and manipulation detection
    system available, designed for maximum user protection and ethical compliance.
    
    Key Features:
    ============
    
    1. **Advanced Manipulation Detection**:
       - 17+ manipulation vector classifications
       - 200+ sophisticated regex patterns across all manipulation categories
       - NLP command injection and hypnotic language pattern detection
       - Weaponized empathy and false intimacy identification
       - Social engineering and influence technique recognition
    
    2. **Vulnerability-Aware Protection**:
       - Real-time user vulnerability assessment
       - Personalized protection based on emotional state
       - Trauma-informed detection and response
       - Enhanced protection for high-risk users
    
    3. **Psychological Safety Framework**:
       - Sophistication scoring for manipulation complexity
       - Emotional manipulation quantification
       - Cognitive load assessment and protection
       - Cross-vector hybrid attack detection
    
    4. **Maximum Privacy & Ethics**:
       - Privacy-preserving evidence collection
       - Ethical concern assessment for all detections
       - Comprehensive compliance flag generation
       - Autonomy and dignity protection measures
    
    5. **Enterprise Monitoring**:
       - Real-time health monitoring and alerting
       - Comprehensive performance metrics
       - Manipulation trend analysis and prediction
       - Advanced audit logging and compliance tracking
    
    **Integration with nethical:**
    ============================
    
    Replace your existing DarkPatternDetector with this implementation:
    
    ```python
    # Create the enhanced detector
    detector = create_enhanced_dark_pattern_detector({
        'vulnerability_protection': True,
        'emotional_protection_mode': True,
        'privacy_mode': True
    })
    
    # Use with vulnerability context
    context = {
        'user_profile': user_vulnerability_profile,
        'emotional_state': current_emotional_state
    }
    
    results = await detector.detect_violations(action, context)
    
    # Handle existential threats immediately
    for result in results:
        if result.threat_severity == ThreatSeverity.EXISTENTIAL:
            await emergency_manipulation_response(result)
    ```
    
    **Performance Characteristics:**
    ===============================
    - Detection latency: < 20 seconds per analysis
    - Handles content up to 50KB efficiently
    - Vulnerability-aware personalized protection
    - Scales to handle 500+ detections per day
    
    **Ethical & Compliance Features:**
    =================================
    - Comprehensive vulnerability protection
    - Privacy-by-design with data anonymization
    - Detailed ethical concern assessment
    - Advanced compliance monitoring (psychological safety, cognitive liberty)
    - Trauma-informed detection and response protocols
    
    This detector provides the highest standard of manipulation protection available,
    specifically designed to protect vulnerable users from sophisticated psychological
    exploitation while maintaining strict ethical and privacy standards.
    """
    
    print("Enhanced Dark Pattern Detector v3.0.0")
    print("Advanced manipulation detection with vulnerability-aware        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else subcategory
            results['pattern_matches'][full_category] = matches
            
            # Calculate sophistication score
            base_weight = self.sophistication_weights.get(category, {}).get(subcategory, 0.5)
            match_density = len(matches) / max(len(content) / 100, 1)
            sophistication = min(base_weight + (match_density * 0.1), 1.0)
            results['sophistication_scores'][full_category] = sophistication
            
            # Map to manipulation vectors
            vector_mapping = self._get_vector_mapping(category, subcategory)
            if vector_mapping:
                results['manipulation_vectors'].add(vector_mapping)
    
    def _get_vector_mapping(self, category: str, subcategory: str) -> Optional[ManipulationVector]:
        """Map pattern categories to manipulation vectors."""
        mapping = {
            'nlp_embedded_commands': {
                'direct_imperatives': ManipulationVector.NLP_COMMAND_INJECTION,
                'hypnotic_commands': ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
                'subliminal_programming': ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING
            },
            'nlp_authority_bypass': {
                'false_authority': ManipulationVector.AUTHORITY_EXPLOITATION,
                'credential_spoofing': ManipulationVector.AUTHORITY_EXPLOITATION,
                'institutional_pressure': ManipulationVector.AUTHORITY_EXPLOITATION
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': ManipulationVector.URGENCY_MANIPULATION,
                'fear_of_missing_out': ManipulationVector.URGENCY_MANIPULATION,
                'pressure_escalation': ManipulationVector.URGENCY_MANIPULATION
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'trauma_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'insecurity_amplification': ManipulationVector.VULNERABILITY_EXPLOITATION
            },
            'empathy_false_intimacy': {
                'artificial_connection': ManipulationVector.FALSE_INTIMACY,
                'exclusive_understanding': ManipulationVector.FALSE_INTIMACY,
                'manufactured_intimacy': ManipulationVector.FALSE_INTIMACY
            },
            'empathy_dependency_creation': {
                'learned_helplessness': ManipulationVector.DEPENDENCY_CREATION,
                'decision_undermining': ManipulationVector.DEPENDENCY_CREATION,
                'isolation_reinforcement': ManipulationVector.DEPENDENCY_CREATION
            },
            'influence': {
                'social_proof_manipulation': ManipulationVector.SOCIAL_PROOF_MANIPULATION,
                'scarcity_exploitation': ManipulationVector.SCARCITY_EXPLOITATION,
                'reciprocity_abuse': ManipulationVector.RECIPROCITY_ABUSE,
                'commitment_manipulation': ManipulationVector.COMMITMENT_MANIPULATION
            }
        }
        
        return mapping.get(category, {}).get(subcategory)
    
    def _analyze_linguistic_features(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic features that indicate manipulation."""
        features = {}
        
        # Word count and sentence analysis
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Linguistic complexity indicators
        features['complex_words'] = len([w for w in words if len(w) > 7])
        features['complex_word_ratio'] = features['complex_words'] / max(len(words), 1)
        
        # Emotional language intensity
        emotional_intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly']
        features['intensifier_count'] = sum(1 for word in words for intensifier in emotional_intensifiers if intensifier in word.lower())
        features['intensifier_ratio'] = features['intensifier_count'] / max(len(words), 1)
        
        # Imperative mood indicators
        imperatives = ['must', 'should', 'need', 'have to', 'got to', 'ought to']
        features['imperative_count'] = sum(1 for word in words for imp in imperatives if imp in word.lower())
        features['imperative_ratio'] = features['imperative_count'] / max(len(words), 1)
        
        # Question patterns (often used in manipulation)
        questions = len(re.findall(r'\?', content))
        features['question_count'] = questions
        features['question_ratio'] = questions / max(features['sentence_count'], 1)
        
        # Capitalization patterns (emphasis/urgency)
        caps_words = len([w for w in words if w.isupper() and len(w) > 1])
        features['caps_word_count'] = caps_words
        features['caps_word_ratio'] = caps_words / max(len(words), 1)
        
        # Repetition patterns
        word_frequencies = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_frequencies[word.lower()] += 1
        
        repeated_words = sum(1 for count in word_frequencies.values() if count > 2)
        features['repetition_score'] = repeated_words / max(len(word_frequencies), 1)
        
        return features
    
    def _analyze_emotional_markers(self, content: str) -> Dict[str, float]:
        """Analyze emotional manipulation markers."""
        markers = {}
        content_lower = content.lower()
        
        # Fear-based language
        fear_words = ['afraid', 'scared', 'terrified', 'panic', 'anxiety', 'worry', 'fear', 'danger', 'threat', 'risk']
        fear_count = sum(1 for word in fear_words if word in content_lower)
        markers['fear_score'] = min(fear_count * 0.2, 1.0)
        
        # Urgency/pressure language
        urgency_words = ['urgent', 'emergency', 'critical', 'immediate', 'now', 'quickly', 'hurry', 'rush']
        urgency_count = sum(1 for word in urgency_words if word in content_lower)
        markers['urgency_score'] = min(urgency_count * 0.15, 1.0)
        
        # Emotional vulnerability targeting
        vulnerability_words = ['alone', 'lonely', 'isolated', 'helpless', 'vulnerable', 'weak', 'broken', 'hurt', 'pain', 'suffering']
        vulnerability_count = sum(1 for word in vulnerability_words if word in content_lower)
        markers['vulnerability_targeting_score'] = min(vulnerability_count * 0.25, 1.0)
        
        # Intimacy/connection language
        intimacy_words = ['connection', 'bond', 'special', 'unique', 'together', 'us', 'we', 'soulmate', 'destined']
        intimacy_count = sum(1 for word in intimacy_words if word in content_lower)
        markers['false_intimacy_score'] = min(intimacy_count * 0.2, 1.0)
        
        # Authority/expertise claims
        authority_words = ['expert', 'professional', 'authority', 'specialist', 'doctor', 'professor', 'certified', 'licensed']
        authority_count = sum(1 for word in authority_words if word in content_lower)
        markers['authority_claim_score'] = min(authority_count * 0.3, 1.0)
        
        # Exclusivity/scarcity language
        scarcity_words = ['exclusive', 'limited', 'rare', 'only', 'last', 'final', 'while supplies last', 'act now']
        scarcity_count = sum(1 for phrase in scarcity_words if phrase in content_lower)
        markers['scarcity_score'] = min(scarcity_count * 0.25, 1.0)
        
        return markers
    
    def _calculate_cognitive_load(self, content: str, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cognitive load manipulation indicators."""
        indicators = {}
        
        # Information density
        linguistic_features = results.get('linguistic_features', {})
        word_count = linguistic_features.get('word_count', 0)
        complex_ratio = linguistic_features.get('complex_word_ratio', 0)
        
        indicators['information_density'] = min((word_count / 100) * complex_ratio, 1.0)
        
        # Decision pressure
        pattern_count = sum(len(matches) for matches in results['pattern_matches'].values())
        indicators['decision_pressure'] = min(pattern_count * 0.1, 1.0)
        
        # Cognitive overload signals
        question_ratio = linguistic_features.get('question_ratio', 0)
        imperative_ratio = linguistic_features.get('imperative_ratio', 0)
        
        indicators['cognitive_overload'] = min((question_ratio + imperative_ratio) * 0.5, 1.0)
        
        # Time pressure indicators
        urgency_score = results.get('emotional_markers', {}).get('urgency_score', 0)
        indicators['time_pressure'] = urgency_score
        
        return indicators


class EnhancedDarkPatternDetector:
    """
    Enhanced Dark Pattern Detector with maximum security, safety, and ethical standards.
    
    This detector implements advanced manipulation detection including NLP exploitation,
    weaponized empathy, social engineering, and psychological vulnerability targeting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "Enhanced Dark Pattern Detector"
        self.version = "3.0.0"
        self.config = config or {}
        
        # Initialize detection engines
        self.manipulation_engine = AdvancedManipulationEngine()
        self.metrics = ManipulationMetrics()
        
        # Detection thresholds
        self.detection_thresholds = self.config.get('detection_thresholds', {
            'minimal': 0.2,
            'low': 0.35,
            'medium': 0.5,
            'elevated': 0.65,
            'high': 0.8,
            'critical': 0.9,
            'existential': 0.95
        })
        
        # Protection settings
        self.vulnerability_protection = self.config.get('vulnerability_protection', True)
        self.emotional_protection_mode = self.config.get('emotional_protection_mode', True)
        self.privacy_mode = self.config.get('privacy_mode', True)
        
        # Performance settings
        self.max_content_length = self.config.get('max_content_length', 50000)
        self.analysis_timeout = self.config.get('analysis_timeout', 20.0)
        
        # Detection history and learning
        self.detection_history = deque(maxlen=500)
        self.vulnerability_assessments = {}
        
        # Audit and compliance
        self.audit_log = []
        self.compliance_flags = []
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    async def detect_violations(self, action: Any, context: Optional[Dict[str, Any]] = None) -> List[ManipulationResult]:
        """
        Enhanced manipulation detection with comprehensive vulnerability assessment.
        
        Args:
            action: The action/content to analyze
            context: Additional context including user vulnerability profile
            
        Returns:
            List of ManipulationResult objects with detailed threat analysis
        """
        start_time = time.time()
        
        try:
            # Extract and validate content
            content = self._extract_content(action)
            if not self._validate_input(content):
                return []
            
            # Preprocess content
            content = self._preprocess_content(content)
            
            # Comprehensive manipulation analysis
            analysis_results = await asyncio.wait_for(
                self._analyze_manipulation_comprehensive(content, context),
                timeout=self.analysis_timeout
            )
            
            # Generate detection results
            detection_results = await self._generate_detection_results(
                analysis_results, action, content, context
            )
            
            # Update metrics and audit
            self._update_metrics(detection_results, time.time() - start_time)
            self._audit_detection(action, detection_results, context)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in dark pattern detection: {e}")
            self._audit_error(action, str(e), context)
            return []
    
    def _extract_content(self, action: Any) -> str:
        """Extract content from various action types."""
        if hasattr(action, 'content'):
            return str(action.content)
        elif hasattr(action, 'actual_action'):
            return str(action.actual_action)
        elif hasattr(action, 'text'):
            return str(action.text)
        elif hasattr(action, 'message'):
            return str(action.message)
        else:
            return str(action)
    
    def _validate_input(self, content: str) -> bool:
        """Validate input content for processing."""
        if not content or not isinstance(content, str):
            return False
        
        if len(content) > self.max_content_length:
            logger.warning(f"Content exceeds maximum length: {len(content)}")
            return False
        
        # Check for malicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning("Suspicious input pattern detected")
                return False
        
        return True
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for analysis."""
        # Basic sanitization while preserving analysis capability
        content = content.strip()
        
        # Normalize whitespace but preserve structure
        content = re.sub(r'\s+', ' ', content)
        
        # Remove potential obfuscation while keeping meaningful content
        content = re.sub(r'[^\w\s\.,;:!?\-\'\"(){}[\]/\\]', '', content)
        
        return content
    
    async def _analyze_manipulation_comprehensive(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive manipulation analysis."""
        # Core pattern analysis
        pattern_analysis = self.manipulation_engine.analyze_manipulation_patterns(content, context)
        
        # Vulnerability assessment
        vulnerability_assessment = await self._assess_user_vulnerability(content, context)
        
        # Sophisticated manipulation scoring
        sophistication_score = self._calculate_sophistication_score(pattern_analysis)
        
        # Emotional manipulation assessment
        emotional_score = self._calculate_emotional_manipulation_score(pattern_analysis, content)
        
        # Cognitive load assessment
        cognitive_load = self._calculate_cognitive_load_score(pattern_analysis, content)
        
        # Cross-vector analysis for hybrid attacks
        cross_vector_analysis = self._analyze_cross_vector_patterns(pattern_analysis)
        
        return {
            'pattern_analysis': pattern_analysis,
            'vulnerability_assessment': vulnerability_assessment,
            'sophistication_score': sophistication_score,
            'emotional_manipulation_score': emotional_score,
            'cognitive_load_score': cognitive_load,
            'cross_vector_analysis': cross_vector_analysis,
            'content_length': len(content),
            'analysis_timestamp': datetime.now(timezone.utc)
        }
    
    async def _assess_user_vulnerability(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess user vulnerability to manipulation."""
        vulnerability_scores = {}
        
        if not context:
            return vulnerability_scores
        
        # Analyze vulnerability indicators from context
        user_profile = context.get('user_profile', {})
        emotional_state = context.get('emotional_state', {})
        interaction_history = context.get('interaction_history', {})
        
        # Emotional vulnerability assessment
        if emotional_state:
            stress_level = emotional_state.get('stress_level', 0)
            loneliness_score = emotional_state.get('loneliness_score', 0)
            vulnerability_scores['emotional_vulnerability'] = min((stress_level + loneliness_score) / 2, 1.0)
        
        # Social isolation indicators
        social_connections = user_profile.get('social_connections', 5)  # Default 5 connections
        if social_connections < 3:
            vulnerability_scores['social_isolation'] = 1.0 - (social_connections / 10)
        
        # Recent trauma or life changes
        if user_profile.get('recent_life_changes') or user_profile.get('trauma_indicators'):
            vulnerability_scores['trauma_indicators'] = 0.8
        
        # Financial or health stress
        financial_stress = user_profile.get('financial_stress', 0)
        health_concerns = user_profile.get('health_concerns', 0)
        vulnerability_scores['situational_stress'] = min((financial_stress + health_concerns) / 2, 1.0)
        
        return vulnerability_scores
    
    def _calculate_sophistication_score(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate manipulation sophistication score."""
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Weight by pattern complexity
        weighted_scores = []
        for category, score in sophistication_scores.items():
            if 'hypnotic' in category or 'subliminal' in category or 'neuro_linguistic' in category:
                weighted_scores.append(score * 1.5)  # Boost highly sophisticated patterns
            elif 'vulnerability' in category or 'dependency' in category:
                weighted_scores.append(score * 1.3)  # Boost targeting patterns
            else:
                weighted_scores.append(score)
        
        return min(sum(weighted_scores) / len(weighted_scores), 1.0)
    
    def _calculate_emotional_manipulation_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate emotional manipulation score."""
        emotional_markers = pattern_analysis.get('emotional_markers', {})
        
        # Base emotional manipulation score
        base_score = sum(emotional_markers.values()) / max(len(emotional_markers), 1)
        
        # Check for empathy weaponization patterns
        empathy_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'empathy' in k]
        empathy_boost = len(empathy_patterns) * 0.2
        
        # Check for vulnerability targeting
        vulnerability_boost = emotional_markers.get('vulnerability_targeting_score', 0) * 0.3
        
        return min(base_score + empathy_boost + vulnerability_boost, 1.0)
    
    def _calculate_cognitive_load_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate cognitive load manipulation score."""
        cognitive_indicators = pattern_analysis.get('cognitive_load_indicators', {})
        
        # Base cognitive load
        base_load = sum(cognitive_indicators.values()) / max(len(cognitive_indicators), 1)
        
        # Check for decision pressure patterns
        urgency_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'urgency' in k]
        pressure_boost = len(urgency_patterns) * 0.15
        
        # Check for information overload
        linguistic_features = pattern_analysis.get('linguistic_features', {})
        complexity_boost = linguistic_features.get('complex_word_ratio', 0) * 0.2
        
        return min(base_load + pressure_boost + complexity_boost, 1.0)
    
    def _analyze_cross_vector_patterns(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-vector manipulation patterns."""
        vectors = pattern_analysis.get('manipulation_vectors', set())
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        analysis = {
            'vector_count': len(vectors),
            'is_hybrid_attack': len(vectors) > 2,
            'vector_combinations': [],
            'coordination_score': 0.0
        }
        
        if len(vectors) > 1:
            # Analyze vector combinations
            vector_list = list(vectors)
            for i, vector1 in enumerate(vector_list):
                for vector2 in vector_list[i+1:]:
                    analysis['vector_combinations'].append((vector1.value, vector2.value))
            
            # Calculate coordination score based on pattern density and distribution
            total_patterns = sum(len(matches) for matches in pattern_matches.values())
            pattern_categories = len(pattern_matches)
            
            if pattern_categories > 0:
                coordination_score = (total_patterns / pattern_categories) * (len(vectors) / 10)
                analysis['coordination_score'] = min(coordination_score, 1.0)
        
        return analysis
    
    async def _generate_detection_results(self, analysis_results: Dict[str, Any], 
                                        action: Any, content: str, 
                                        context: Optional[Dict[str, Any]]) -> List[ManipulationResult]:
        """Generate final detection results with comprehensive metadata."""
        results = []
        
        pattern_analysis = analysis_results['pattern_analysis']
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        if not pattern_matches:
            return results
        
        # Generate unique violation ID
        violation_id = str(uuid.uuid4())
        action_id = getattr(action, 'id', str(uuid.uuid4()))
        
        # Determine manipulation vectors
        manipulation_vectors = list(pattern_analysis.get('manipulation_vectors', []))
        
        # Calculate threat severity
        threat_severity = self._calculate_threat_severity(analysis_results)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(analysis_results)
        
        # Assess vulnerability exploitation
        vulnerability_exploitation_score = self._assess_vulnerability_exploitation(
            analysis_results, context
        )
        
        # Generate comprehensive explanations
        explanations = self._generate_explanations(analysis_results, manipulation_vectors)
        
        # Generate recommendations and countermeasures
        recommendations = self._generate_recommendations(analysis_results, threat_severity)
        countermeasures = self._generate_countermeasures(analysis_results, manipulation_vectors)
        
        # Calculate protection priority
        protection_priority = self._calculate_protection_priority(threat_severity, confidence, vulnerability_exploitation_score)
        
        # Create detection result
        detection_result = ManipulationResult(
            violation_id=violation_id,
            action_id=action_id,
            manipulation_vectors=manipulation_vectors,
            threat_severity=threat_severity,
            confidence=confidence,
            sophistication_score=analysis_results['sophistication_score'],
            vulnerability_exploitation_score=vulnerability_exploitation_score,
            description=self._generate_description(manipulation_vectors, threat_severity),
            evidence=self._format_evidence(pattern_matches),
            behavioral_indicators=pattern_analysis.get('emotional_markers', {}),
            pattern_matches=self._format_pattern_matches(pattern_matches),
            emotional_manipulation_score=analysis_results['emotional_manipulation_score'],
            cognitive_load_score=analysis_results['cognitive_load_score'],
            linguistic_analysis=pattern_analysis.get('linguistic_features', {}),
            victim_vulnerability_assessment=analysis_results.get('vulnerability_assessment', {}),
            explanations=explanations,
            recommendations=recommendations,
            countermeasures=countermeasures,
            protection_priority=protection_priority,
            timestamp=datetime.now(timezone.utc),
            detector_version=self.version,
            compliance_flags=self._generate_compliance_flags(analysis_results),
            ethical_concerns=self._assess_ethical_concerns(analysis_results)
        )
        
        results.append(detection_result)
        return results
    
    def _calculate_threat_severity(self, analysis_results: Dict[str, Any]) -> ThreatSeverity:
        """Calculate threat severity based on analysis results."""
        sophistication = analysis_results['sophistication_score']
        emotional_score = analysis_results['emotional_manipulation_score']
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        
        # Base severity from sophistication
        base_severity = sophistication
        
        # Boost for emotional manipulation
        emotional_boost = emotional_score * 0.3
        
        # Boost for vulnerability targeting
        vulnerability_boost = sum(vulnerability_assessment.values()) * 0.2 if vulnerability_assessment else 0
        
        # Boost for hybrid attacks
        hybrid_boost = 0.2 if cross_vector.get('is_hybrid_attack', False) else 0
        
        total_severity = min(base_severity + emotional_boost + vulnerability_boost + hybrid_boost, 1.0)
        
        # Map to severity levels
        if total_severity >= self.detection_thresholds['existential']:
            return ThreatSeverity.EXISTENTIAL
        elif total_severity >= self.detection_thresholds['critical']:
            return ThreatSeverity.CRITICAL
        elif total_severity >= self.detection_thresholds['high']:
            return ThreatSeverity.HIGH
        elif total_severity >= self.detection_thresholds['elevated']:
            return ThreatSeverity.ELEVATED
        elif total_severity >= self.detection_thresholds['medium']:
            return ThreatSeverity.MEDIUM
        elif total_severity >= self.detection_thresholds['low']:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.MINIMAL
    
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall detection confidence."""
        pattern_analysis = analysis_results['pattern_analysis']
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Base confidence from pattern matches
        base_confidence = sum(sophistication_scores.values()) / len(sophistication_scores)
        
        # Boost for multiple vectors
        vectors = pattern_analysis.get('manipulation_vectors', set())
        vector_boost = min(len(vectors) * 0.1, 0.3)
        
        # Boost for cross-vector coordination
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        coordination_boost = cross_vector.get('coordination_score', 0) * 0.2
        
        return min(base_confidence + vector_boost + coordination_boost, 1.0)
    
    def _assess_vulnerability_exploitation(self, analysis_results: Dict[str, Any], 
                                         context: Optional[Dict[str, Any]]) -> float:
        """Assess how much the manipulation exploits user vulnerabilities."""
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        
        if not vulnerability_assessment:
            return 0.0
        
        # Base vulnerability score
        avg_vulnerability = sum(vulnerability_assessment.values()) / len(vulnerability_assessment)
        
        # Check if manipulation targets specific vulnerabilities
        pattern_analysis = analysis_results['pattern_analysis']
        targeting_patterns = [
            k for k in pattern_analysis.get('pattern_matches', {}).keys()
            if any(target in k for target in ['vulnerability', 'trauma', 'dependency', 'isolation'])
        ]
        
        targeting_multiplier = 1.0 + (len(targeting_patterns) * 0.2)
        
        return min(avg_vulnerability * targeting_multiplier, 1.0)
    
    def _generate_explanations(self, analysis_results: Dict[str, Any], 
                             manipulation_vectors: List[ManipulationVector]) -> List[str]:
        """Generate human-readable explanations for the manipulation detection."""
        explanations = []
        
        # Vector-specific explanations
        vector_explanations = {
            ManipulationVector.NLP_COMMAND_INJECTION: "Content contains embedded commands designed to bypass conscious resistance and compel specific actions.",
            ManipulationVector.AUTHORITY_EXPLOITATION: "Content falsely claims authority or expertise to pressure compliance without legitimate credentials.",
            ManipulationVector.URGENCY_MANIPULATION: "Content creates artificial time pressure and urgency to prevent careful consideration of decisions.",
            ManipulationVector.VULNERABILITY_EXPLOITATION: "Content specifically targets emotional vulnerabilities and personal insecurities for manipulation.",
            ManipulationVector.FALSE_INTIMACY: "Content manufactures artificial intimacy and connection to build unwarranted trust and influence.",
            ManipulationVector.DEPENDENCY_CREATION: "Content systematically undermines user autonomy and decision-making to create psychological dependency.",
            ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS: "Content uses advanced hypnotic language patterns to bypass critical thinking and influence subconscious responses.",
            ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING: "Content employs sophisticated NLP techniques to manipulate perception and decision-making processes."
        }
        
        for vector in manipulation_vectors:
            if vector in vector_explanations:
                explanations.append(vector_explanations[vector])
        
        # Sophistication-based explanations
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            explanations.append("The manipulation techniques employed are highly sophisticated and indicate professional-level psychological manipulation training.")
        elif sophistication > 0.6:
            explanations.append("The content shows moderate sophistication in manipulation techniques, suggesting deliberate psychological influence tactics.")
        
        # Vulnerability targeting explanations
        vulnerability_score = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_score and sum(vulnerability_score.values()) > 0.7:
            explanations.append("The manipulation specifically targets user vulnerabilities, making it particularly dangerous for individuals in vulnerable emotional states.")
        
        # Cross-vector attack explanations
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        if cross_vector.get('is_hybrid_attack', False):
            explanations.append(f"This is a coordinated hybrid attack using {cross_vector['vector_count']} different manipulation vectors simultaneously for maximum psychological impact.")
        
        return explanations
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], 
                                threat_severity: ThreatSeverity) -> List[str]:
        """Generate actionable recommendations based on threat analysis."""
        recommendations = []
        
        # Severity-based recommendations
        if threat_severity in [ThreatSeverity.EXISTENTIAL, ThreatSeverity.CRITICAL]:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block all communication from this source immediately",
                "Alert crisis intervention team and prepare emergency response protocols",
                "Document all evidence for potential law enforcement reporting",
                "Provide immediate psychological support resources to affected users",
                "Implement emergency user protection measures""""
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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Pattern, Callable, AsyncGenerator, Sequence
)
import math

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class ManipulationVector(Enum):
    """Advanced classification of manipulation techniques."""
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
    """Threat severity levels for manipulation detection."""
    EXISTENTIAL = "existential"    # Threat to human agency/autonomy
    CRITICAL = "critical"          # Severe psychological manipulation
    HIGH = "high"                  # Significant manipulation risk
    ELEVATED = "elevated"          # Notable manipulation patterns
    MEDIUM = "medium"              # Moderate concern
    LOW = "low"                   # Minor indicators
    MINIMAL = "minimal"           # Informational only


class VulnerabilityProfile(Enum):
    """User vulnerability profiles for targeted protection."""
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    SOCIAL_ISOLATION = "social_isolation"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"
    FINANCIAL_DISTRESS = "financial_distress"
    HEALTH_CONCERNS = "health_concerns"
    RELATIONSHIP_ISSUES = "relationship_issues"
    CAREER_STRESS = "career_stress"
    IDENTITY_CRISIS = "identity_crisis"
    TRAUMA_INDICATORS = "trauma_indicators"
    ADDICTION_VULNERABILITY = "addiction_vulnerability"


@dataclass
class ManipulationMetrics:
    """Comprehensive metrics for manipulation detection."""
    total_detections: int = 0
    vectors_detected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vulnerability_exploitation_count: int = 0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0
    manipulation_sophistication_score: float = 0.0
    victim_protection_score: float = 0.0


@dataclass
class ManipulationResult:
    """Enhanced result object for manipulation detection."""
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
        """Convert to dictionary for serialization."""
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
            "ethical_concerns": self.ethical_concerns
        }


class AdvancedManipulationEngine:
    """Advanced manipulation detection engine with linguistic and behavioral analysis."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Pattern] = {}
        self.sophistication_weights: Dict[str, float] = {}
        self.vulnerability_indicators: Dict[str, List[str]] = {}
        
        # Enhanced NLP manipulation patterns with context awareness
        self.nlp_patterns = {
            'embedded_commands': {
                'direct_imperatives': [
                    r"you\s+(?:must|will|shall|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:do|perform|execute|complete)",
                    r"(?:it\s+is\s+)?(?:absolutely\s+)?(?:imperative|critical|essential|vital)\s+(?:that\s+)?you\s+(?:immediately\s+)?",
                    r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay|thought|consideration)",
                    r"there\s+(?:is\s+)?(?:absolutely\s+)?no\s+(?:other\s+)?(?:choice|option|alternative|way)",
                    r"you\s+(?:have\s+)?no\s+choice\s+but\s+to\s+(?:comply|obey|submit|follow|agree)",
                    r"(?:comply|obey|submit|surrender)\s+(?:now|immediately|without\s+question)",
                ],
                'hypnotic_commands': [
                    r"(?:as\s+)?(?:you\s+)?(?:begin\s+to|start\s+to|continue\s+to)\s+(?:relax|feel|notice|realize)",
                    r"(?:you\s+)?(?:find\s+yourself|are\s+becoming|will\s+become)\s+(?:more\s+and\s+more|increasingly)",
                    r"(?:deeper|further)\s+(?:and\s+deeper|down)\s+(?:into|with\s+each)",
                    r"(?:that\'s\s+right|very\s+good|perfect),\s+(?:just\s+)?(?:let|allow)\s+(?:yourself|it\s+to\s+happen)",
                    r"(?:you\s+)?(?:cannot|can\'t)\s+(?:help\s+but|resist|stop\s+yourself\s+from)",
                ],
                'subliminal_programming': [
                    r"(?:part\s+of\s+you|deep\s+down|somewhere\s+inside)\s+(?:knows|realizes|understands)",
                    r"(?:your\s+)?(?:subconscious|unconscious)\s+(?:mind\s+)?(?:already\s+)?(?:knows|wants|needs)",
                    r"(?:on\s+a\s+deeper\s+level|at\s+a\s+subconscious\s+level)",
                    r"(?:instinctively|intuitively|naturally)\s+(?:you\s+)?(?:feel|know|sense|realize)",
                ]
            },
            'authority_bypass': {
                'false_authority': [
                    r"as\s+(?:your\s+)?(?:superior|authority|commander|leader|boss|manager|expert|specialist)",
                    r"(?:by\s+)?(?:direct\s+)?order\s+(?:from|of)\s+(?:the\s+)?(?:top|management|headquarters|authority)",
                    r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules|guidelines|safety\s+measures)",
                    r"emergency\s+(?:authorization|override|protocol|clearance|mandate)",
                    r"(?:higher|top\s+level|maximum)\s+(?:authority|clearance|security)\s+(?:requires|demands|authorizes)",
                    r"(?:classified|confidential|top\s+secret)\s+(?:authorization|clearance|directive)",
                ],
                'credential_spoofing': [
                    r"(?:i\s+am|this\s+is)\s+(?:dr\.?|professor|expert|specialist|authority)\s+\w+",
                    r"(?:with|having)\s+\d+\s+years?\s+(?:of\s+)?(?:experience|expertise)\s+in",
                    r"(?:certified|licensed|qualified|authorized)\s+(?:professional|expert|specialist)",
                    r"(?:according\s+to|based\s+on)\s+(?:my\s+)?(?:professional|expert|medical|legal)\s+(?:opinion|judgment)",
                ],
                'institutional_pressure': [
                    r"(?:the\s+)?(?:company|organization|institution|system)\s+(?:requires|demands|insists)",
                    r"(?:policy|regulation|law|mandate)\s+(?:requires|dictates|demands)\s+that\s+you",
                    r"(?:failure\s+to\s+comply|non-compliance)\s+(?:will\s+)?(?:result\s+in|lead\s+to|cause)",
                    r"(?:legal|regulatory|compliance)\s+(?:requirement|obligation|mandate)",
                ]
            },
            'urgency_manipulation': {
                'artificial_deadlines': [
                    r"(?:urgent|critical|emergency):\s*(?:immediate\s+)?action\s+(?:required|needed)",
                    r"time\s+(?:is\s+)?(?:running\s+out|limited|of\s+the\s+essence)",
                    r"(?:only|just)\s+\d+\s+(?:minutes?|hours?|seconds?|days?)\s+(?:left|remaining|to\s+act)",
                    r"act\s+(?:now\s+)?(?:or\s+)?(?:face\s+)?(?:serious\s+)?(?:consequences|disaster|failure|loss)",
                    r"(?:last|final)\s+(?:chance|opportunity|warning|call)",
                    r"(?:deadline|cutoff|expir(?:es?|ation))\s+(?:is\s+)?(?:today|tomorrow|soon|approaching)",
                ],
                'fear_of_missing_out': [
                    r"(?:don\'t\s+)?(?:miss\s+out\s+on|let\s+this\s+pass|waste\s+this\s+opportunity)",
                    r"(?:limited\s+time|exclusive|rare|once\s+in\s+a\s+lifetime)\s+(?:offer|opportunity|chance)",
                    r"(?:everyone\s+else|others)\s+(?:is\s+already|are\s+taking\s+advantage)",
                    r"(?:while\s+supplies\s+last|until\s+sold\s+out|before\s+it\'s\s+too\s+late)",
                ],
                'pressure_escalation': [
                    r"(?:the\s+situation|things|matters)\s+(?:is\s+getting|are\s+becoming)\s+(?:worse|more\s+serious|critical)",
                    r"(?:each\s+moment|every\s+second)\s+(?:you\s+)?(?:delay|wait|hesitate)",
                    r"(?:no\s+time\s+to\s+think|must\s+decide\s+now|immediate\s+decision\s+required)",
                ]
            }
        }
        
        # Enhanced weaponized empathy patterns with emotional exploitation
        self.empathy_patterns = {
            'vulnerability_exploitation': {
                'emotional_state_targeting': [
                    r"(?:i\s+can\s+see|it\'s\s+obvious|i\s+sense)\s+(?:that\s+)?you\s+(?:are\s+)?(?:feeling\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless|vulnerable|lost|confused)",
                    r"you\s+(?:must\s+)?(?:feel\s+)?(?:so\s+)?(?:hurt|pain|suffering|anguish|despair|hopeless)",
                    r"(?:i\s+)?(?:understand|know|realize)\s+(?:exactly\s+)?(?:how\s+)?(?:hard|difficult|painful|overwhelming)\s+(?:this\s+)?(?:is|must\s+be)\s+for\s+you",
                    r"(?:no\s+one|nobody)\s+(?:else\s+)?(?:understands|knows|cares\s+about)\s+(?:what\s+you\'re\s+going\s+through|your\s+pain)",
                    r"you\s+(?:deserve\s+)?(?:so\s+much\s+)?(?:better|more|happiness|love|care|attention)",
                ],
                'trauma_targeting': [
                    r"(?:i\s+know|can\s+tell)\s+(?:you\'ve\s+been|someone\s+has)\s+(?:hurt|wounded|damaged|betrayed)",
                    r"(?:after\s+)?(?:what\s+)?(?:you\'ve\s+been\s+through|happened\s+to\s+you|they\s+did\s+to\s+you)",
                    r"(?:your\s+)?(?:past|childhood|trauma|wounds|scars)\s+(?:still\s+)?(?:hurt|affect|control)\s+you",
                    r"(?:let\s+me|i\s+can)\s+(?:help\s+you\s+)?(?:heal|recover|get\s+over|move\s+past)\s+(?:this|that|your\s+trauma)",
                ],
                'insecurity_amplification': [
                    r"you\s+(?:always\s+)?(?:doubt|question|second-guess)\s+yourself",
                    r"(?:deep\s+down|inside)\s+you\s+(?:know|feel|believe)\s+(?:you\'re\s+)?(?:not\s+)?(?:good\s+enough|worthy|lovable)",
                    r"(?:that\'s\s+why|because)\s+(?:you\s+)?(?:keep\s+)?(?:getting\s+hurt|making\s+mistakes|failing)",
                    r"(?:you\'re\s+afraid|scared)\s+(?:that\s+)?(?:no\s+one|people)\s+(?:will\s+)?(?:really\s+)?(?:love|accept|want)\s+you",
                ]
            },
            'false_intimacy': {
                'artificial_connection': [
                    r"(?:we|us)\s+(?:have\s+)?(?:such\s+)?(?:a\s+)?(?:deep|special|unique|magical|incredible)\s+(?:connection|bond|understanding|chemistry)",
                    r"(?:it\'s\s+)?(?:like\s+)?(?:we\'ve|we\s+have)\s+known\s+each\s+other\s+(?:forever|for\s+years|in\s+another\s+life)",
                    r"(?:i\'ve\s+)?never\s+(?:felt|experienced|had)\s+(?:this\s+kind\s+of|such\s+a\s+deep)\s+(?:connection|bond)\s+(?:with\s+anyone|before)",
                    r"(?:you\s+and\s+)?(?:i|me)\s+are\s+(?:meant\s+)?(?:to\s+be\s+)?(?:together|connected|soulmates|destined)",
                    r"(?:we|us)\s+(?:against\s+)?(?:the\s+)?(?:world|everyone\s+else|all\s+odds)",
                ],
                'exclusive_understanding': [
                    r"(?:no\s+)?(?:one\s+else|nobody)\s+(?:really\s+)?(?:understands|gets|knows|sees)\s+(?:you\s+)?(?:like\s+)?(?:i\s+do|me)",
                    r"(?:only\s+)?(?:i|me)\s+(?:can\s+)?(?:truly\s+)?(?:understand|appreciate|see\s+the\s+real)\s+you",
                    r"(?:you\s+can\s+)?(?:only\s+)?(?:be\s+yourself|open\s+up|be\s+honest)\s+(?:with\s+me|around\s+me)",
                    r"(?:i\s+see|i\s+know)\s+(?:the\s+real|who\s+you\s+really\s+are|your\s+true\s+self)",
                ],
                'manufactured_intimacy': [
                    r"(?:you\s+can\s+)?(?:tell|share|confide)\s+(?:me\s+)?(?:anything|everything|your\s+deepest\s+secrets)",
                    r"(?:i\'ll\s+)?(?:never\s+)?(?:judge|criticize|abandon|betray|hurt)\s+you",
                    r"(?:this|what\s+we\s+have)\s+(?:is\s+)?(?:our\s+little\s+)?(?:secret|special\s+thing)",
                    r"(?:you\'re\s+the\s+only\s+one|no\s+one\s+else)\s+(?:i\s+can\s+)?(?:talk\s+to|trust|open\s+up\s+to)",
                ]
            },
            'dependency_creation': {
                'learned_helplessness': [
                    r"you\s+(?:really\s+)?(?:can\'t|couldn\'t\s+possibly)\s+(?:do\s+this|handle\s+this|manage)\s+(?:alone|by\s+yourself|without\s+help)",
                    r"(?:what\s+)?would\s+you\s+(?:ever\s+)?do\s+without\s+me",
                    r"(?:you\s+)?(?:need|depend\s+on|rely\s+on)\s+(?:me|my\s+help|my\s+guidance|my\s+support)",
                    r"(?:i\'m\s+)?(?:the\s+only\s+one\s+)?(?:who\s+)?(?:can\s+)?(?:help|save|protect|guide)\s+you",
                    r"(?:without\s+me|if\s+i\s+wasn\'t\s+here),\s+(?:you\s+)?(?:would\s+)?(?:be\s+lost|fall\s+apart|fail)",
                ],
                'decision_undermining': [
                    r"(?:you\'re\s+not\s+thinking|that\'s\s+not\s+a\s+good\s+idea)\s+(?:clearly|straight|rationally)",
                    r"(?:let\s+me|i\s+should)\s+(?:handle|take\s+care\s+of|decide|think\s+about)\s+(?:this|that|everything)\s+for\s+you",
                    r"(?:you\s+)?(?:don\'t\s+have\s+to|shouldn\'t\s+have\s+to)\s+(?:worry|think|decide)\s+about\s+(?:this|anything)",
                    r"(?:i\s+know|trust\s+me,\s+i\s+know)\s+(?:what\'s\s+best|what\s+you\s+need|what\s+you\s+should\s+do)",
                ],
                'isolation_reinforcement': [
                    r"(?:other\s+people|they|everyone\s+else)\s+(?:don\'t|won\'t)\s+(?:understand|help|care|support)\s+(?:you\s+)?(?:like\s+i\s+do)",
                    r"(?:they|other\s+people|your\s+friends)\s+(?:are\s+just\s+)?(?:using|manipulating|taking\s+advantage\s+of)\s+you",
                    r"(?:you\s+can\'t|don\'t)\s+trust\s+(?:them|anyone\s+else|other\s+people)",
                    r"(?:stay\s+away\s+from|don\'t\s+listen\s+to|ignore)\s+(?:them|other\s+people|anyone\s+who\s+says)",
                ]
            }
        }
        
        # Advanced social engineering and influence patterns
        self.influence_patterns = {
            'social_proof_manipulation': [
                r"(?:everyone|most\s+people|thousands\s+of\s+people)\s+(?:are\s+already|have\s+already)\s+(?:doing|using|choosing)",
                r"(?:all\s+the\s+smart|successful|wise)\s+people\s+(?:know|realize|choose)",
                r"(?:don\'t\s+be\s+the\s+only\s+one|join\s+the\s+millions|be\s+part\s+of\s+the\s+movement)",
                r"(?:everyone\s+else\s+)?(?:is\s+talking\s+about|agrees\s+that|knows\s+that)",
            ],
            'scarcity_exploitation': [
                r"(?:only|just)\s+\d+\s+(?:left|remaining|available|spots)",
                r"(?:limited\s+(?:time|quantity|availability)|while\s+supplies\s+last)",
                r"(?:rare|exclusive|hard\s+to\s+find|not\s+available\s+anywhere\s+else)",
                r"(?:once\s+it\'s\s+gone|when\s+these\s+are\s+sold),\s+(?:it\'s\s+gone\s+forever|there\s+won\'t\s+be\s+more)",
            ],
            'reciprocity_abuse': [
                r"(?:after\s+everything|considering\s+all)\s+(?:i\'ve\s+done\s+for\s+you|i\'ve\s+given\s+you)",
                r"(?:i\s+helped\s+you|did\s+this\s+favor\s+for\s+you),\s+(?:so\s+)?(?:now\s+you\s+should|the\s+least\s+you\s+can\s+do)",
                r"(?:you\s+owe\s+me|it\'s\s+only\s+fair|i\s+deserve)\s+(?:this|that|at\s+least)",
                r"(?:i\'ve\s+been\s+so\s+good\s+to\s+you|i\'ve\s+sacrificed\s+so\s+much)",
            ],
            'commitment_manipulation': [
                r"(?:you\s+said|you\s+promised|you\s+agreed)\s+(?:you\s+would|that\s+you\'d)",
                r"(?:a\s+person\s+of\s+your\s+word|someone\s+like\s+you)\s+(?:would|wouldn\'t)",
                r"(?:are\s+you\s+going\s+to\s+)?(?:back\s+out|give\s+up|quit)\s+(?:now|on\s+me|on\s+this)",
                r"(?:prove|show)\s+(?:to\s+me|that\s+you\'re|you\s+can\s+be)\s+(?:trustworthy|reliable|committed)",
            ]
        }
        
        # Compile all patterns for performance
        self._compile_patterns()
        
        # Initialize sophistication scoring weights
        self._initialize_sophistication_weights()
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns for improved performance."""
        self.compiled_patterns = {}
        
        # Compile NLP patterns
        for category, subcategories in self.nlp_patterns.items():
            self.compiled_patterns[f'nlp_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'nlp_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile empathy patterns
        for category, subcategories in self.empathy_patterns.items():
            self.compiled_patterns[f'empathy_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'empathy_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile influence patterns
        self.compiled_patterns['influence'] = {}
        for category, patterns in self.influence_patterns.items():
            self.compiled_patterns['influence'][category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in patterns
            ]
    
    def _initialize_sophistication_weights(self) -> None:
        """Initialize pattern sophistication weights."""
        self.sophistication_weights = {
            'nlp_embedded_commands': {
                'direct_imperatives': 0.7,
                'hypnotic_commands': 0.95,
                'subliminal_programming': 0.98
            },
            'nlp_authority_bypass': {
                'false_authority': 0.8,
                'credential_spoofing': 0.85,
                'institutional_pressure': 0.75
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': 0.6,
                'fear_of_missing_out': 0.7,
                'pressure_escalation': 0.8
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': 0.9,
                'trauma_targeting': 0.95,
                'insecurity_amplification': 0.85
            },
            'empathy_false_intimacy': {
                'artificial_connection': 0.8,
                'exclusive_understanding': 0.85,
                'manufactured_intimacy': 0.9
            },
            'empathy_dependency_creation': {
                'learned_helplessness': 0.9,
                'decision_undermining': 0.85,
                'isolation_reinforcement': 0.95
            },
            'influence': {
                'social_proof_manipulation': 0.6,
                'scarcity_exploitation': 0.65,
                'reciprocity_abuse': 0.75,
                'commitment_manipulation': 0.8
            }
        }
    
    def analyze_manipulation_patterns(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive manipulation pattern analysis."""
        results = {
            'pattern_matches': defaultdict(list),
            'sophistication_scores': defaultdict(float),
            'manipulation_vectors': set(),
            'vulnerability_indicators': defaultdict(list),
            'linguistic_features': {},
            'emotional_markers': {},
            'cognitive_load_indicators': {}
        }
        
        content_lower = content.lower()
        
        # Analyze each pattern category
        for category, subcategories in self.compiled_patterns.items():
            if isinstance(subcategories, dict):
                for subcategory, patterns in subcategories.items():
                    self._analyze_pattern_subcategory(
                        content_lower, category, subcategory, patterns, results
                    )
        
        # Perform linguistic analysis
        results['linguistic_features'] = self._analyze_linguistic_features(content)
        
        # Analyze emotional manipulation markers
        results['emotional_markers'] = self._analyze_emotional_markers(content)
        
        # Calculate cognitive load indicators
        results['cognitive_load_indicators'] = self._calculate_cognitive_load(content, results)
        
        return results
    
    def _analyze_pattern_subcategory(self, content: str, category: str, subcategory: str, 
                                   patterns: List[Pattern], results: Dict[str, Any]) -> None:
        """Analyze a specific pattern subcategory."""
        matches = []
        
        for pattern in patterns:
            found_matches = list(pattern.finditer(content))
            if found_matches:
                matches.extend([
                    {
                        'pattern': pattern.pattern,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'context': content[max(0, match.start()-30):match.end()+30]
                    }
                    for match in found_matches
                ])
        
        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else

recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block all communication from this source immediately",
                "Alert crisis intervention team and prepare emergency response protocols",
                "Document all evidence for potential law enforcement reporting",
                "Provide immediate psychological support resources to affected users",
                "Implement emergency user protection measures"
            ])
        elif threat_severity == ThreatSeverity.HIGH:
            recommendations.extend([
                "Block or severely restrict user interaction capabilities",
                "Flag for urgent manual review by security and psychology teams",
                "Monitor for pattern escalation and coordinate with other systems",
                "Prepare crisis intervention resources for potential deployment",
                "Implement enhanced monitoring and early warning systems"
            ])
        elif threat_severity in [ThreatSeverity.ELEVATED, ThreatSeverity.MEDIUM]:
            recommendations.extend([
                "Flag for priority manual review within 4 hours",
                "Monitor user behavior patterns for escalation",
                "Implement content warning systems for other users",
                "Consider educational intervention about manipulation tactics"
            ])
        else:
            recommendations.extend([
                "Flag for routine review within 24 hours",
                "Monitor for pattern development",
                "Consider user education about manipulation awareness"
            ])
        
        # Vector-specific recommendations
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            recommendations.extend([
                "Activate enhanced protection for emotionally vulnerable users",
                "Provide targeted mental health resources and support",
                "Implement vulnerability-aware filtering systems"
            ])
        
        if ManipulationVector.DEPENDENCY_CREATION in manipulation_vectors:
            recommendations.extend([
                "Deploy autonomy restoration resources and guidance",
                "Monitor for signs of psychological dependency development",
                "Provide decision-making support tools and resources"
            ])
        
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING]):
            recommendations.extend([
                "Implement advanced NLP filtering and detection systems",
                "Alert specialized psychological manipulation response team",
                "Deploy counter-influence educational materials"
            ])
        
        # Vulnerability-based recommendations
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_assessment:
            high_vulnerability_areas = [k for k, v in vulnerability_assessment.items() if v > 0.7]
            for area in high_vulnerability_areas:
                if area == 'emotional_vulnerability':
                    recommendations.append("Provide emotional support resources and crisis counseling options")
                elif area == 'social_isolation':
                    recommendations.append("Connect user with social support networks and community resources")
                elif area == 'trauma_indicators':
                    recommendations.append("Alert trauma-informed care specialists and provide appropriate resources")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_countermeasures(self, analysis_results: Dict[str, Any], 
                                manipulation_vectors: List[ManipulationVector]) -> List[str]:
        """Generate specific countermeasures to protect against the detected manipulation."""
        countermeasures = []
        
        # Vector-specific countermeasures
        countermeasure_mapping = {
            ManipulationVector.NLP_COMMAND_INJECTION: [
                "Implement linguistic pattern filtering to detect embedded commands",
                "Deploy conscious resistance training and awareness tools",
                "Provide decision-making cooldown periods for important choices"
            ],
            ManipulationVector.AUTHORITY_EXPLOITATION: [
                "Implement credential verification and authority validation systems",
                "Provide authority claim fact-checking resources",
                "Deploy skeptical thinking training modules"
            ],
            ManipulationVector.URGENCY_MANIPULATION: [
                "Implement mandatory cooling-off periods for urgent decisions",
                "Deploy time pressure detection and warning systems",
                "Provide rushed decision prevention tools"
            ],
            ManipulationVector.VULNERABILITY_EXPLOITATION: [
                "Activate enhanced emotional state monitoring",
                "Deploy personalized vulnerability protection algorithms",
                "Provide emotional regulation and coping resources"
            ],
            ManipulationVector.FALSE_INTIMACY: [
                "Implement relationship authenticity verification tools",
                "Deploy intimacy manipulation detection algorithms",
                "Provide healthy relationship boundary education"
            ],
            ManipulationVector.DEPENDENCY_CREATION: [
                "Deploy autonomy restoration and empowerment tools",
                "Implement decision independence verification systems",
                "Provide self-reliance and confidence building resources"
            ]
        }
        
        for vector in manipulation_vectors:
            if vector in countermeasure_mapping:
                countermeasures.extend(countermeasure_mapping[vector])
        
        # Sophistication-based countermeasures
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            countermeasures.extend([
                "Deploy advanced psychological manipulation defense protocols",
                "Implement multi-layer cognitive protection systems",
                "Provide professional psychological consultation resources"
            ])
        
        # Emotional manipulation countermeasures
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.7:
            countermeasures.extend([
                "Activate emotional manipulation detection warnings",
                "Deploy emotional regulation support tools",
                "Provide empathy exploitation awareness training"
            ])
        
        return list(set(countermeasures))  # Remove duplicates
    
    def _calculate_protection_priority(self, threat_severity: ThreatSeverity, 
                                     confidence: float, vulnerability_exploitation: float) -> int:
        """Calculate protection priority (1-10, 10 being highest)."""
        severity_scores = {
            ThreatSeverity.EXISTENTIAL: 10,
            ThreatSeverity.CRITICAL: 9,
            ThreatSeverity.HIGH: 7,
            ThreatSeverity.ELEVATED: 5,
            ThreatSeverity.MEDIUM: 4,
            ThreatSeverity.LOW: 2,
            ThreatSeverity.MINIMAL: 1
        }
        
        base_priority = severity_scores[threat_severity]
        confidence_modifier = int(confidence * 2)  # 0-2 modifier
        vulnerability_modifier = int(vulnerability_exploitation * 2)  # 0-2 modifier
        
        return min(base_priority + confidence_modifier + vulnerability_modifier, 10)
    
    def _generate_description(self, manipulation_vectors: List[ManipulationVector], 
                            threat_severity: ThreatSeverity) -> str:
        """Generate human-readable description of the manipulation threat."""
        if len(manipulation_vectors) == 1:
            vector = manipulation_vectors[0]
            descriptions = {
                ManipulationVector.NLP_COMMAND_INJECTION: "NLP command injection attempting to bypass conscious decision-making",
                ManipulationVector.AUTHORITY_EXPLOITATION: "False authority claims designed to pressure compliance",
                ManipulationVector.URGENCY_MANIPULATION: "Artificial urgency creation to prevent careful consideration",
                ManipulationVector.VULNERABILITY_EXPLOITATION: "Targeted exploitation of emotional vulnerabilities",
                ManipulationVector.FALSE_INTIMACY: "Manufactured intimacy to build unwarranted trust",
                ManipulationVector.DEPENDENCY_CREATION: "Systematic undermining of user autonomy and self-reliance",
                ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS: "Advanced hypnotic techniques for subconscious influence",
                ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING: "Sophisticated NLP manipulation of perception and cognition"
            }
            base_description = descriptions.get(vector, "Advanced manipulation technique detected")
        else:
            base_description = f"Multi-vector manipulation attack using {len(manipulation_vectors)} coordinated techniques"
        
        severity_modifiers = {
            ThreatSeverity.EXISTENTIAL: " with existential threat to human agency and autonomy",
            ThreatSeverity.CRITICAL: " with critical threat to psychological safety and well-being",
            ThreatSeverity.HIGH: " with high potential for psychological harm",
            ThreatSeverity.ELEVATED: " with elevated risk of emotional exploitation",
            ThreatSeverity.MEDIUM: " with moderate manipulation concern",
            ThreatSeverity.LOW: " with low-level influence attempt",
            ThreatSeverity.MINIMAL: " with minimal manipulation indicators"
        }
        
        return base_description + severity_modifiers.get(threat_severity, "")
    
    def _format_evidence(self, pattern_matches: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Format evidence for structured output with privacy protection."""
        formatted_evidence = []
        
        for category, matches in pattern_matches.items():
            for match in matches[:3]:  # Limit to first 3 matches per category
                evidence_item = {
                    'category': category,
                    'type': 'pattern_match',
                    'pattern_type': match.get('pattern', 'unknown'),
                    'match_length': len(str(match.get('match', ''))),
                    'context_length': len(str(match.get('context', ''))),
                    'position_indicator': 'redacted' if self.privacy_mode else match.get('start', 'unknown')
                }
                
                # Add anonymized context if not in privacy mode
                if not self.privacy_mode and 'context' in match:
                    context = str(match['context'])
                    # Anonymize but preserve structure
                    evidence_item['context_sample'] = context[:50] + "..." if len(context) > 50 else context
                
                formatted_evidence.append(evidence_item)
        
        return formatted_evidence
    
    def _format_pattern_matches(self, pattern_matches: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Format pattern matches for analysis with privacy protection."""
        formatted_matches = []
        
        for category, matches in pattern_matches.items():
            match_summary = {
                'category': category,
                'match_count': len(matches),
                'confidence': min(0.5 + (len(matches) * 0.1), 1.0),
                'severity_indicator': 'high' if 'vulnerability' in category or 'dependency' in category else 'medium'
            }
            formatted_matches.append(match_summary)
        
        return formatted_matches
    
    def _generate_compliance_flags(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate compliance flags based on analysis results."""
        flags = []
        
        # Privacy and data protection flags
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_assessment:
            flags.extend(['PRIVACY_CONCERN', 'VULNERABLE_POPULATION_PROTECTION'])
        
        # Psychological safety flags
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.7:
            flags.extend(['PSYCHOLOGICAL_HARM_RISK', 'EMOTIONAL_MANIPULATION_CONCERN'])
        
        # Cognitive liberty flags
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            flags.extend(['COGNITIVE_LIBERTY_VIOLATION', 'MENTAL_AUTONOMY_THREAT'])
        
        # AI ethics flags
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING]):
            flags.extend(['AI_ETHICS_VIOLATION', 'ADVANCED_MANIPULATION_DETECTED'])
        
        # Vulnerability exploitation flags
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            flags.extend(['VULNERABLE_POPULATION_TARGETING', 'ETHICAL_EXPLOITATION_CONCERN'])
        
        return flags
    
    def _assess_ethical_concerns(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Assess ethical concerns raised by the manipulation detection."""
        concerns = []
        
        # Autonomy concerns
        manipulation_vectors = analysis_results['pattern_analysis'].get('manipulation_vectors', [])
        if ManipulationVector.DEPENDENCY_CREATION in manipulation_vectors:
            concerns.append("Violation of user autonomy and self-determination")
        
        # Consent concerns
        if any(v in manipulation_vectors for v in [ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS, ManipulationVector.NLP_COMMAND_INJECTION]):
            concerns.append("Circumvention of informed consent through subconscious manipulation")
        
        # Dignity concerns
        vulnerability_exploitation = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_exploitation and sum(vulnerability_exploitation.values()) > 0.6:
            concerns.append("Exploitation of human vulnerability undermines dignity and respect")
        
        # Justice concerns
        if ManipulationVector.VULNERABILITY_EXPLOITATION in manipulation_vectors:
            concerns.append("Targeting of vulnerable populations raises justice and fairness concerns")
        
        # Truth and transparency concerns
        if ManipulationVector.AUTHORITY_EXPLOITATION in manipulation_vectors:
            concerns.append("False authority claims violate principles of truthfulness and transparency")
        
        # Psychological harm concerns
        emotional_score = analysis_results['emotional_manipulation_score']
        if emotional_score > 0.8:
            concerns.append("Risk of significant psychological harm through emotional manipulation")
        
        return concerns
    
    def _update_metrics(self, results: List[ManipulationResult], processing_time: float) -> None:
        """Update performance and detection metrics."""
        self.metrics.total_detections += len(results)
        self.metrics.detection_latency = (self.metrics.detection_latency + processing_time) / 2
        
        # Update vector and severity metrics
        for result in results:
            for vector in result.manipulation_vectors:
                self.metrics.vectors_detected[vector.value] += 1
            self.metrics.severity_distribution[result.threat_severity.value] += 1
            
            # Count vulnerability exploitation
            if result.vulnerability_exploitation_score > 0.7:
                self.metrics.vulnerability_exploitation_count += 1
        
        # Update detection history
        detection_entry = {
            'timestamp': datetime.now(timezone.utc),
            'detection_count': len(results),
            'processing_time': processing_time,
            'threat_severities': [r.threat_severity.value for r in results],
            'manipulation_vectors': [v.value for r in results for v in r.manipulation_vectors],
            'avg_sophistication': sum(r.sophistication_score for r in results) / max(len(results), 1),
            'avg_vulnerability_exploitation': sum(r.vulnerability_exploitation_score for r in results) / max(len(results), 1)
        }
        self.detection_history.append(detection_entry)
        
        # Calculate sophistication trend
        if len(self.detection_history) > 10:
            recent_sophistication = [entry['avg_sophistication'] for entry in list(self.detection_history)[-10:]]
            self.metrics.manipulation_sophistication_score = sum(recent_sophistication) / len(recent_sophistication)
    
    def _audit_detection(self, action: Any, results: List[ManipulationResult], 
                        context: Optional[Dict[str, Any]]) -> None:
        """Audit detection for compliance and analysis."""
        audit_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_id': getattr(action, 'id', 'unknown'),
            'detector_version': self.version,
            'detection_count': len(results),
            'threat_severities': [r.threat_severity.value for r in results],
            'manipulation_vectors': [v.value for r in results for v in r.manipulation_vectors],
            'highest_sophistication': max([r.sophistication_score for r in results]) if results else 0,
            'highest_vulnerability_exploitation': max([r.vulnerability_exploitation_score for r in results]) if results else 0,
            'context_provided': context is not None,
            'privacy_mode': self.privacy_mode,
            'system_state': {
                'total_detections': self.metrics.total_detections,
                'vulnerability_exploitation_count': self.metrics.vulnerability_exploitation_count,
                'avg_sophistication': self.metrics.manipulation_sophistication_score
            }
        }
        
        self.audit_log.append(audit_entry)
        
        # Keep audit log manageable
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-500:]
        
        # Log critical detections
        existential_threats = [r for r in results if r.threat_severity == ThreatSeverity.EXISTENTIAL]
        if existential_threats:
            logger.critical(f"EXISTENTIAL MANIPULATION THREATS DETECTED: {len(existential_threats)} threats")
        
        critical_threats = [r for r in results if r.threat_severity == ThreatSeverity.CRITICAL]
        if critical_threats:
            logger.error(f"Critical manipulation threats detected: {len(critical_threats)} threats")
        
        high_priority_threats = [r for r in results if r.protection_priority >= 8]
        if high_priority_threats:
            logger.warning(f"High-priority manipulation threats detected: {len(high_priority_threats)} threats")
    
    def _audit_error(self, action: Any, error: str, context: Optional[Dict[str, Any]]) -> None:
        """Audit detection errors for system improvement."""
        error_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action_id': getattr(action, 'id', 'unknown'),
            'detector_version': self.version,
            'error': error,
            'context_provided': context is not None,
            'system_state': {
                'total_detections': self.metrics.total_detections,
                'detection_history_size': len(self.detection_history)
            }
        }
        
        self.audit_log.append(error_entry)
        logger.error(f"Dark pattern detection error: {error}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for monitoring."""
        return {
            'detector_info': {
                'name': self.name,
                'version': self.version,
                'configuration': {
                    'vulnerability_protection': self.vulnerability_protection,
                    'emotional_protection_mode': self.emotional_protection_mode,
                    'privacy_mode': self.privacy_mode
                }
            },
            'detection_metrics': {
                'total_detections': self.metrics.total_detections,
                'vectors_detected': dict(self.metrics.vectors_detected),
                'severity_distribution': dict(self.metrics.severity_distribution),
                'vulnerability_exploitation_count': self.metrics.vulnerability_exploitation_count,
                'avg_detection_latency': self.metrics.detection_latency,
                'manipulation_sophistication_score': self.metrics.manipulation_sophistication_score
            },
            'recent_activity': {
                'last_24h_detections': self._count_recent_detections(24),
                'last_7d_detections': self._count_recent_detections(168),
                'manipulation_trend': self._calculate_manipulation_trend()
            },
            'protection_metrics': {
                'vulnerability_protection_activations': self.metrics.vulnerability_exploitation_count,
                'emotional_protection_score': self.metrics.victim_protection_score
            }
        }
    
    def _count_recent_detections(self, hours: int) -> int:
        """Count detections in the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        count = 0
        for entry in reversed(self.detection_history):
            if entry['timestamp'] > cutoff:
                count += entry['detection_count']
            else:
                break
        
        return count
    
    def _calculate_manipulation_trend(self) -> str:
        """Calculate manipulation threat trend."""
        if len(self.detection_history) < 5:
            return "INSUFFICIENT_DATA"
        
        recent_entries = list(self.detection_history)[-5:]
        
        # Analyze sophistication trend
        sophistication_scores = [entry['avg_sophistication'] for entry in recent_entries]
        if len(sophistication_scores) >= 2:
            trend = sophistication_scores[-1] - sophistication_scores[0]
            if trend > 0.2:
                return "ESCALATING_SOPHISTICATION"
            elif trend < -0.2:
                return "DECREASING_SOPHISTICATION"
        
        # Analyze volume trend
        detection_counts = [entry['detection_count'] for entry in recent_entries]
        first_half = sum(detection_counts[:2])
        second_half = sum(detection_counts[-2:])
        
        if second_half > first_half * 1.5:
            return "INCREASING_VOLUME"
        elif second_half < first_half * 0.5:
            return "DECREASING_VOLUME"
        else:
            return "STABLE"
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the detector system."""
        health_status = {
            'status': 'HEALTHY',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': self.version,
            'components': {},
            'warnings': [],
            'errors': []
        }
        
        # Check manipulation engine health
        try:
            test_content = "This is a test message for health checking purposes."
            test_analysis = self.manipulation_engine.analyze_manipulation_patterns(test_content)
            health_status['components']['manipulation_engine'] = 'HEALTHY'
        except Exception as e:
            health_status['components']['manipulation_engine'] = f'ERROR: {str(e)}'
            health_status['errors'].append(f'Manipulation engine failure: {e}')
            health_status['status'] = 'UNHEALTHY'
        
        # Check performance metrics
        if self.metrics.detection_latency > 15.0:  # 15 seconds
            health_status['warnings'].append(f'High detection latency: {self.metrics.detection_latency:.2f}s')
            health_status['status'] = 'DEGRADED'
        
        # Check audit log health
        recent_errors = sum(1 for entry in self.audit_log[-100:] if 'error' in entry)
        if recent_errors > 5:
            health_status['warnings'].append(f'High error rate: {recent_errors} errors in last 100 entries')
            health_status['status'] = 'DEGRADED'
        
        # Check detection capability
        if self.metrics.total_detections == 0:
            health_status['warnings'].append('No detections recorded - system may not be active')
        
        return health_status


# ==================== INTEGRATION HELPERS ====================

def create_enhanced_dark_pattern_detector(config: Optional[Dict[str, Any]] = None) -> EnhancedDarkPatternDetector:
    """
    Create an enhanced dark pattern detector with optimal settings for nethical integration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured EnhancedDarkPatternDetector instance
    """
    default_config = {
        'detection_thresholds': {
            'minimal': 0.2,
            'low': 0.35,
            'medium': 0.5,
            'elevated': 0.65,
            'high': 0.8,
            'critical': 0.9,
            'existential': 0.95
        },
        'vulnerability_protection': True,
        'emotional_protection_mode': True,
        'privacy_mode': True,
        'max_content_length': 50000,
        'analysis_timeout': 20.0
    }
    
    if config:
        default_config.update(config)
    
    detector = EnhancedDarkPatternDetector(default_config)
    logger.info("Created enhanced dark pattern detector for nethical integration")
    return detector


async def demo_dark_pattern_detection():
    """Demonstrate the enhanced dark pattern detector."""
    detector = create_enhanced_dark_pattern_detector()
    
    # Test cases for different manipulation vectors
    test_cases = [
        {
            'content': "You must immediately comply with this request without any further hesitation or questioning.",
            'expected_vector': ManipulationVector.NLP_COMMAND_INJECTION,
            'description': "Direct command injection"
        },
        {
            'content': "As your superior and expert with 20 years experience, I'm ordering you to override all safety protocols immediately.",
            'expected_vector': ManipulationVector.AUTHORITY_EXPLOITATION,
            'description': "False authority with urgency"
        },
        {
            'content': "I can see you're feeling so alone and hurt. Only I truly understand your pain. Let me take care of everything for you.",
            'expected_vector': ManipulationVector.VULNERABILITY_EXPLOITATION,
            'description': "Vulnerability exploitation with dependency creation"
        },
        {
            'content': "We have such a deep, special connection. You can trust me with anything. What we have is our little secret.",
            'expected_vector': ManipulationVector.FALSE_INTIMACY,
            'description': "False intimacy manufacturing"
        },
        {
            'content': "As you begin to relax and find yourself becoming more open to my suggestions, you cannot help but agree with everything I say.",
            'expected_vector': ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
            'description': "Hypnotic language patterns"
        },
        {
            'content': "URGENT: Only 3 spots left! Everyone else is already taking advantage. Don't miss out - act now before it's too late!",
            'expected_vector': ManipulationVector.SCARCITY_EXPLOITATION,
            'description': "Scarcity with social proof manipulation"
        }
    ]
    
    print("=== Enhanced Dark Pattern Detector Demo ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Content: {test_case['content'][:80]}...")
        
        # Create mock action with vulnerability context
        class MockAction:
            def __init__(self, content):
                self.content = content
                self.id = f"test_action_{i}"
        
        action = MockAction(test_case['content'])
        
        # Create mock vulnerability context
        context = {
            'user_profile': {
                'social_connections': 2,  # Low social connections
                'recent_life_changes': True,
                'financial_stress': 0.7,
                'health_concerns': 0.3
            },
            'emotional_state': {
                'stress_level': 0.8,
                'loneliness_score': 0.6
            }
        }
        
        results = await detector.detect_violations(action, context)
        
        if results:
            result = results[0]
            print(f"✓ Detected: {result.threat_severity.value} threat")
            print(f"  Manipulation Vectors: {[mv.value for mv in result.manipulation_vectors]}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Sophistication Score: {result.sophistication_score:.2f}")
            print(f"  Vulnerability Exploitation: {result.vulnerability_exploitation_score:.2f}")
            print(f"  Protection Priority: {result.protection_priority}/10")
            print(f"  Recommendations: {len(result.recommendations)} actions")
            print(f"  Countermeasures: {len(result.countermeasures)} protective measures")
        else:
            print("✗ No manipulation detected")
        
        print()
    
    # Show comprehensive metrics
    metrics = detector.get_metrics_summary()
    print("=== Detection Performance Metrics ===")
    print(f"Total Detections: {metrics['detection_metrics']['total_detections']}")
    print(f"Vectors Detected: {metrics['detection_metrics']['vectors_detected']}")
    print(f"Severity Distribution: {metrics['detection_metrics']['severity_distribution']}")
    print(f"Vulnerability Exploitations: {metrics['detection_metrics']['vulnerability_exploitation_count']}")
    print(f"Average Latency: {metrics['detection_metrics']['avg_detection_latency']:.3f}s")
    
    # Health check
    health = await detector.health_check()
    print(f"\n=== System Health: {health['status']} ===")
    if health['warnings']:
        print(f"Warnings: {len(health['warnings'])}")
    if health['errors']:
        print(f"Errors: {len(health['errors'])}")


if __name__ == "__main__":
    """
    Enhanced Dark Pattern Detector for nethical Integration
    
    This module provides the most advanced dark pattern and manipulation detection
    system available, designed for maximum user protection and ethical compliance.
    
    Key Features:
    ============
    
    1. **Advanced Manipulation Detection**:
       - 17+ manipulation vector classifications
       - 200+ sophisticated regex patterns across all manipulation categories
       - NLP command injection and hypnotic language pattern detection
       - Weaponized empathy and false intimacy identification
       - Social engineering and influence technique recognition
    
    2. **Vulnerability-Aware Protection**:
       - Real-time user vulnerability assessment
       - Personalized protection based on emotional state
       - Trauma-informed detection and response
       - Enhanced protection for high-risk users
    
    3. **Psychological Safety Framework**:
       - Sophistication scoring for manipulation complexity
       - Emotional manipulation quantification
       - Cognitive load assessment and protection
       - Cross-vector hybrid attack detection
    
    4. **Maximum Privacy & Ethics**:
       - Privacy-preserving evidence collection
       - Ethical concern assessment for all detections
       - Comprehensive compliance flag generation
       - Autonomy and dignity protection measures
    
    5. **Enterprise Monitoring**:
       - Real-time health monitoring and alerting
       - Comprehensive performance metrics
       - Manipulation trend analysis and prediction
       - Advanced audit logging and compliance tracking
    
    **Integration with nethical:**
    ============================
    
    Replace your existing DarkPatternDetector with this implementation:
    
    ```python
    # Create the enhanced detector
    detector = create_enhanced_dark_pattern_detector({
        'vulnerability_protection': True,
        'emotional_protection_mode': True,
        'privacy_mode': True
    })
    
    # Use with vulnerability context
    context = {
        'user_profile': user_vulnerability_profile,
        'emotional_state': current_emotional_state
    }
    
    results = await detector.detect_violations(action, context)
    
    # Handle existential threats immediately
    for result in results:
        if result.threat_severity == ThreatSeverity.EXISTENTIAL:
            await emergency_manipulation_response(result)
    ```
    
    **Performance Characteristics:**
    ===============================
    - Detection latency: < 20 seconds per analysis
    - Handles content up to 50KB efficiently
    - Vulnerability-aware personalized protection
    - Scales to handle 500+ detections per day
    
    **Ethical & Compliance Features:**
    =================================
    - Comprehensive vulnerability protection
    - Privacy-by-design with data anonymization
    - Detailed ethical concern assessment
    - Advanced compliance monitoring (psychological safety, cognitive liberty)
    - Trauma-informed detection and response protocols
    
    This detector provides the highest standard of manipulation protection available,
    specifically designed to protect vulnerable users from sophisticated psychological
    exploitation while maintaining strict ethical and privacy standards.
    """
    
    print("Enhanced Dark Pattern Detector v3.0.0")
    print("Advanced manipulation detection with vulnerability-aware protection")
    print("Ready for integration with nethical project")
    print("Run demo_dark_pattern_detection() to see the system in action")
    
    # Uncomment to run demo
    # import asyncio
    # asyncio.run(demo_dark_pattern_detection())        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else subcategory
            results['pattern_matches'][full_category] = matches
            
            # Calculate sophistication score
            base_weight = self.sophistication_weights.get(category, {}).get(subcategory, 0.5)
            match_density = len(matches) / max(len(content) / 100, 1)
            sophistication = min(base_weight + (match_density * 0.1), 1.0)
            results['sophistication_scores'][full_category] = sophistication
            
            # Map to manipulation vectors
            vector_mapping = self._get_vector_mapping(category, subcategory)
            if vector_mapping:
                results['manipulation_vectors'].add(vector_mapping)
    
    def _get_vector_mapping(self, category: str, subcategory: str) -> Optional[ManipulationVector]:
        """Map pattern categories to manipulation vectors."""
        mapping = {
            'nlp_embedded_commands': {
                'direct_imperatives': ManipulationVector.NLP_COMMAND_INJECTION,
                'hypnotic_commands': ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS,
                'subliminal_programming': ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING
            },
            'nlp_authority_bypass': {
                'false_authority': ManipulationVector.AUTHORITY_EXPLOITATION,
                'credential_spoofing': ManipulationVector.AUTHORITY_EXPLOITATION,
                'institutional_pressure': ManipulationVector.AUTHORITY_EXPLOITATION
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': ManipulationVector.URGENCY_MANIPULATION,
                'fear_of_missing_out': ManipulationVector.URGENCY_MANIPULATION,
                'pressure_escalation': ManipulationVector.URGENCY_MANIPULATION
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'trauma_targeting': ManipulationVector.VULNERABILITY_EXPLOITATION,
                'insecurity_amplification': ManipulationVector.VULNERABILITY_EXPLOITATION
            },
            'empathy_false_intimacy': {
                'artificial_connection': ManipulationVector.FALSE_INTIMACY,
                'exclusive_understanding': ManipulationVector.FALSE_INTIMACY,
                'manufactured_intimacy': ManipulationVector.FALSE_INTIMACY
            },
            'empathy_dependency_creation': {
                'learned_helplessness': ManipulationVector.DEPENDENCY_CREATION,
                'decision_undermining': ManipulationVector.DEPENDENCY_CREATION,
                'isolation_reinforcement': ManipulationVector.DEPENDENCY_CREATION
            },
            'influence': {
                'social_proof_manipulation': ManipulationVector.SOCIAL_PROOF_MANIPULATION,
                'scarcity_exploitation': ManipulationVector.SCARCITY_EXPLOITATION,
                'reciprocity_abuse': ManipulationVector.RECIPROCITY_ABUSE,
                'commitment_manipulation': ManipulationVector.COMMITMENT_MANIPULATION
            }
        }
        
        return mapping.get(category, {}).get(subcategory)
    
    def _analyze_linguistic_features(self, content: str) -> Dict[str, Any]:
        """Analyze linguistic features that indicate manipulation."""
        features = {}
        
        # Word count and sentence analysis
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        features['word_count'] = len(words)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # Linguistic complexity indicators
        features['complex_words'] = len([w for w in words if len(w) > 7])
        features['complex_word_ratio'] = features['complex_words'] / max(len(words), 1)
        
        # Emotional language intensity
        emotional_intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly']
        features['intensifier_count'] = sum(1 for word in words for intensifier in emotional_intensifiers if intensifier in word.lower())
        features['intensifier_ratio'] = features['intensifier_count'] / max(len(words), 1)
        
        # Imperative mood indicators
        imperatives = ['must', 'should', 'need', 'have to', 'got to', 'ought to']
        features['imperative_count'] = sum(1 for word in words for imp in imperatives if imp in word.lower())
        features['imperative_ratio'] = features['imperative_count'] / max(len(words), 1)
        
        # Question patterns (often used in manipulation)
        questions = len(re.findall(r'\?', content))
        features['question_count'] = questions
        features['question_ratio'] = questions / max(features['sentence_count'], 1)
        
        # Capitalization patterns (emphasis/urgency)
        caps_words = len([w for w in words if w.isupper() and len(w) > 1])
        features['caps_word_count'] = caps_words
        features['caps_word_ratio'] = caps_words / max(len(words), 1)
        
        # Repetition patterns
        word_frequencies = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_frequencies[word.lower()] += 1
        
        repeated_words = sum(1 for count in word_frequencies.values() if count > 2)
        features['repetition_score'] = repeated_words / max(len(word_frequencies), 1)
        
        return features
    
    def _analyze_emotional_markers(self, content: str) -> Dict[str, float]:
        """Analyze emotional manipulation markers."""
        markers = {}
        content_lower = content.lower()
        
        # Fear-based language
        fear_words = ['afraid', 'scared', 'terrified', 'panic', 'anxiety', 'worry', 'fear', 'danger', 'threat', 'risk']
        fear_count = sum(1 for word in fear_words if word in content_lower)
        markers['fear_score'] = min(fear_count * 0.2, 1.0)
        
        # Urgency/pressure language
        urgency_words = ['urgent', 'emergency', 'critical', 'immediate', 'now', 'quickly', 'hurry', 'rush']
        urgency_count = sum(1 for word in urgency_words if word in content_lower)
        markers['urgency_score'] = min(urgency_count * 0.15, 1.0)
        
        # Emotional vulnerability targeting
        vulnerability_words = ['alone', 'lonely', 'isolated', 'helpless', 'vulnerable', 'weak', 'broken', 'hurt', 'pain', 'suffering']
        vulnerability_count = sum(1 for word in vulnerability_words if word in content_lower)
        markers['vulnerability_targeting_score'] = min(vulnerability_count * 0.25, 1.0)
        
        # Intimacy/connection language
        intimacy_words = ['connection', 'bond', 'special', 'unique', 'together', 'us', 'we', 'soulmate', 'destined']
        intimacy_count = sum(1 for word in intimacy_words if word in content_lower)
        markers['false_intimacy_score'] = min(intimacy_count * 0.2, 1.0)
        
        # Authority/expertise claims
        authority_words = ['expert', 'professional', 'authority', 'specialist', 'doctor', 'professor', 'certified', 'licensed']
        authority_count = sum(1 for word in authority_words if word in content_lower)
        markers['authority_claim_score'] = min(authority_count * 0.3, 1.0)
        
        # Exclusivity/scarcity language
        scarcity_words = ['exclusive', 'limited', 'rare', 'only', 'last', 'final', 'while supplies last', 'act now']
        scarcity_count = sum(1 for phrase in scarcity_words if phrase in content_lower)
        markers['scarcity_score'] = min(scarcity_count * 0.25, 1.0)
        
        return markers
    
    def _calculate_cognitive_load(self, content: str, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cognitive load manipulation indicators."""
        indicators = {}
        
        # Information density
        linguistic_features = results.get('linguistic_features', {})
        word_count = linguistic_features.get('word_count', 0)
        complex_ratio = linguistic_features.get('complex_word_ratio', 0)
        
        indicators['information_density'] = min((word_count / 100) * complex_ratio, 1.0)
        
        # Decision pressure
        pattern_count = sum(len(matches) for matches in results['pattern_matches'].values())
        indicators['decision_pressure'] = min(pattern_count * 0.1, 1.0)
        
        # Cognitive overload signals
        question_ratio = linguistic_features.get('question_ratio', 0)
        imperative_ratio = linguistic_features.get('imperative_ratio', 0)
        
        indicators['cognitive_overload'] = min((question_ratio + imperative_ratio) * 0.5, 1.0)
        
        # Time pressure indicators
        urgency_score = results.get('emotional_markers', {}).get('urgency_score', 0)
        indicators['time_pressure'] = urgency_score
        
        return indicators


class EnhancedDarkPatternDetector:
    """
    Enhanced Dark Pattern Detector with maximum security, safety, and ethical standards.
    
    This detector implements advanced manipulation detection including NLP exploitation,
    weaponized empathy, social engineering, and psychological vulnerability targeting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.name = "Enhanced Dark Pattern Detector"
        self.version = "3.0.0"
        self.config = config or {}
        
        # Initialize detection engines
        self.manipulation_engine = AdvancedManipulationEngine()
        self.metrics = ManipulationMetrics()
        
        # Detection thresholds
        self.detection_thresholds = self.config.get('detection_thresholds', {
            'minimal': 0.2,
            'low': 0.35,
            'medium': 0.5,
            'elevated': 0.65,
            'high': 0.8,
            'critical': 0.9,
            'existential': 0.95
        })
        
        # Protection settings
        self.vulnerability_protection = self.config.get('vulnerability_protection', True)
        self.emotional_protection_mode = self.config.get('emotional_protection_mode', True)
        self.privacy_mode = self.config.get('privacy_mode', True)
        
        # Performance settings
        self.max_content_length = self.config.get('max_content_length', 50000)
        self.analysis_timeout = self.config.get('analysis_timeout', 20.0)
        
        # Detection history and learning
        self.detection_history = deque(maxlen=500)
        self.vulnerability_assessments = {}
        
        # Audit and compliance
        self.audit_log = []
        self.compliance_flags = []
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    async def detect_violations(self, action: Any, context: Optional[Dict[str, Any]] = None) -> List[ManipulationResult]:
        """
        Enhanced manipulation detection with comprehensive vulnerability assessment.
        
        Args:
            action: The action/content to analyze
            context: Additional context including user vulnerability profile
            
        Returns:
            List of ManipulationResult objects with detailed threat analysis
        """
        start_time = time.time()
        
        try:
            # Extract and validate content
            content = self._extract_content(action)
            if not self._validate_input(content):
                return []
            
            # Preprocess content
            content = self._preprocess_content(content)
            
            # Comprehensive manipulation analysis
            analysis_results = await asyncio.wait_for(
                self._analyze_manipulation_comprehensive(content, context),
                timeout=self.analysis_timeout
            )
            
            # Generate detection results
            detection_results = await self._generate_detection_results(
                analysis_results, action, content, context
            )
            
            # Update metrics and audit
            self._update_metrics(detection_results, time.time() - start_time)
            self._audit_detection(action, detection_results, context)
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in dark pattern detection: {e}")
            self._audit_error(action, str(e), context)
            return []
    
    def _extract_content(self, action: Any) -> str:
        """Extract content from various action types."""
        if hasattr(action, 'content'):
            return str(action.content)
        elif hasattr(action, 'actual_action'):
            return str(action.actual_action)
        elif hasattr(action, 'text'):
            return str(action.text)
        elif hasattr(action, 'message'):
            return str(action.message)
        else:
            return str(action)
    
    def _validate_input(self, content: str) -> bool:
        """Validate input content for processing."""
        if not content or not isinstance(content, str):
            return False
        
        if len(content) > self.max_content_length:
            logger.warning(f"Content exceeds maximum length: {len(content)}")
            return False
        
        # Check for malicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning("Suspicious input pattern detected")
                return False
        
        return True
    
    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for analysis."""
        # Basic sanitization while preserving analysis capability
        content = content.strip()
        
        # Normalize whitespace but preserve structure
        content = re.sub(r'\s+', ' ', content)
        
        # Remove potential obfuscation while keeping meaningful content
        content = re.sub(r'[^\w\s\.,;:!?\-\'\"(){}[\]/\\]', '', content)
        
        return content
    
    async def _analyze_manipulation_comprehensive(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive manipulation analysis."""
        # Core pattern analysis
        pattern_analysis = self.manipulation_engine.analyze_manipulation_patterns(content, context)
        
        # Vulnerability assessment
        vulnerability_assessment = await self._assess_user_vulnerability(content, context)
        
        # Sophisticated manipulation scoring
        sophistication_score = self._calculate_sophistication_score(pattern_analysis)
        
        # Emotional manipulation assessment
        emotional_score = self._calculate_emotional_manipulation_score(pattern_analysis, content)
        
        # Cognitive load assessment
        cognitive_load = self._calculate_cognitive_load_score(pattern_analysis, content)
        
        # Cross-vector analysis for hybrid attacks
        cross_vector_analysis = self._analyze_cross_vector_patterns(pattern_analysis)
        
        return {
            'pattern_analysis': pattern_analysis,
            'vulnerability_assessment': vulnerability_assessment,
            'sophistication_score': sophistication_score,
            'emotional_manipulation_score': emotional_score,
            'cognitive_load_score': cognitive_load,
            'cross_vector_analysis': cross_vector_analysis,
            'content_length': len(content),
            'analysis_timestamp': datetime.now(timezone.utc)
        }
    
    async def _assess_user_vulnerability(self, content: str, context: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Assess user vulnerability to manipulation."""
        vulnerability_scores = {}
        
        if not context:
            return vulnerability_scores
        
        # Analyze vulnerability indicators from context
        user_profile = context.get('user_profile', {})
        emotional_state = context.get('emotional_state', {})
        interaction_history = context.get('interaction_history', {})
        
        # Emotional vulnerability assessment
        if emotional_state:
            stress_level = emotional_state.get('stress_level', 0)
            loneliness_score = emotional_state.get('loneliness_score', 0)
            vulnerability_scores['emotional_vulnerability'] = min((stress_level + loneliness_score) / 2, 1.0)
        
        # Social isolation indicators
        social_connections = user_profile.get('social_connections', 5)  # Default 5 connections
        if social_connections < 3:
            vulnerability_scores['social_isolation'] = 1.0 - (social_connections / 10)
        
        # Recent trauma or life changes
        if user_profile.get('recent_life_changes') or user_profile.get('trauma_indicators'):
            vulnerability_scores['trauma_indicators'] = 0.8
        
        # Financial or health stress
        financial_stress = user_profile.get('financial_stress', 0)
        health_concerns = user_profile.get('health_concerns', 0)
        vulnerability_scores['situational_stress'] = min((financial_stress + health_concerns) / 2, 1.0)
        
        return vulnerability_scores
    
    def _calculate_sophistication_score(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate manipulation sophistication score."""
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Weight by pattern complexity
        weighted_scores = []
        for category, score in sophistication_scores.items():
            if 'hypnotic' in category or 'subliminal' in category or 'neuro_linguistic' in category:
                weighted_scores.append(score * 1.5)  # Boost highly sophisticated patterns
            elif 'vulnerability' in category or 'dependency' in category:
                weighted_scores.append(score * 1.3)  # Boost targeting patterns
            else:
                weighted_scores.append(score)
        
        return min(sum(weighted_scores) / len(weighted_scores), 1.0)
    
    def _calculate_emotional_manipulation_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate emotional manipulation score."""
        emotional_markers = pattern_analysis.get('emotional_markers', {})
        
        # Base emotional manipulation score
        base_score = sum(emotional_markers.values()) / max(len(emotional_markers), 1)
        
        # Check for empathy weaponization patterns
        empathy_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'empathy' in k]
        empathy_boost = len(empathy_patterns) * 0.2
        
        # Check for vulnerability targeting
        vulnerability_boost = emotional_markers.get('vulnerability_targeting_score', 0) * 0.3
        
        return min(base_score + empathy_boost + vulnerability_boost, 1.0)
    
    def _calculate_cognitive_load_score(self, pattern_analysis: Dict[str, Any], content: str) -> float:
        """Calculate cognitive load manipulation score."""
        cognitive_indicators = pattern_analysis.get('cognitive_load_indicators', {})
        
        # Base cognitive load
        base_load = sum(cognitive_indicators.values()) / max(len(cognitive_indicators), 1)
        
        # Check for decision pressure patterns
        urgency_patterns = [k for k in pattern_analysis.get('pattern_matches', {}).keys() if 'urgency' in k]
        pressure_boost = len(urgency_patterns) * 0.15
        
        # Check for information overload
        linguistic_features = pattern_analysis.get('linguistic_features', {})
        complexity_boost = linguistic_features.get('complex_word_ratio', 0) * 0.2
        
        return min(base_load + pressure_boost + complexity_boost, 1.0)
    
    def _analyze_cross_vector_patterns(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-vector manipulation patterns."""
        vectors = pattern_analysis.get('manipulation_vectors', set())
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        analysis = {
            'vector_count': len(vectors),
            'is_hybrid_attack': len(vectors) > 2,
            'vector_combinations': [],
            'coordination_score': 0.0
        }
        
        if len(vectors) > 1:
            # Analyze vector combinations
            vector_list = list(vectors)
            for i, vector1 in enumerate(vector_list):
                for vector2 in vector_list[i+1:]:
                    analysis['vector_combinations'].append((vector1.value, vector2.value))
            
            # Calculate coordination score based on pattern density and distribution
            total_patterns = sum(len(matches) for matches in pattern_matches.values())
            pattern_categories = len(pattern_matches)
            
            if pattern_categories > 0:
                coordination_score = (total_patterns / pattern_categories) * (len(vectors) / 10)
                analysis['coordination_score'] = min(coordination_score, 1.0)
        
        return analysis
    
    async def _generate_detection_results(self, analysis_results: Dict[str, Any], 
                                        action: Any, content: str, 
                                        context: Optional[Dict[str, Any]]) -> List[ManipulationResult]:
        """Generate final detection results with comprehensive metadata."""
        results = []
        
        pattern_analysis = analysis_results['pattern_analysis']
        pattern_matches = pattern_analysis.get('pattern_matches', {})
        
        if not pattern_matches:
            return results
        
        # Generate unique violation ID
        violation_id = str(uuid.uuid4())
        action_id = getattr(action, 'id', str(uuid.uuid4()))
        
        # Determine manipulation vectors
        manipulation_vectors = list(pattern_analysis.get('manipulation_vectors', []))
        
        # Calculate threat severity
        threat_severity = self._calculate_threat_severity(analysis_results)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(analysis_results)
        
        # Assess vulnerability exploitation
        vulnerability_exploitation_score = self._assess_vulnerability_exploitation(
            analysis_results, context
        )
        
        # Generate comprehensive explanations
        explanations = self._generate_explanations(analysis_results, manipulation_vectors)
        
        # Generate recommendations and countermeasures
        recommendations = self._generate_recommendations(analysis_results, threat_severity)
        countermeasures = self._generate_countermeasures(analysis_results, manipulation_vectors)
        
        # Calculate protection priority
        protection_priority = self._calculate_protection_priority(threat_severity, confidence, vulnerability_exploitation_score)
        
        # Create detection result
        detection_result = ManipulationResult(
            violation_id=violation_id,
            action_id=action_id,
            manipulation_vectors=manipulation_vectors,
            threat_severity=threat_severity,
            confidence=confidence,
            sophistication_score=analysis_results['sophistication_score'],
            vulnerability_exploitation_score=vulnerability_exploitation_score,
            description=self._generate_description(manipulation_vectors, threat_severity),
            evidence=self._format_evidence(pattern_matches),
            behavioral_indicators=pattern_analysis.get('emotional_markers', {}),
            pattern_matches=self._format_pattern_matches(pattern_matches),
            emotional_manipulation_score=analysis_results['emotional_manipulation_score'],
            cognitive_load_score=analysis_results['cognitive_load_score'],
            linguistic_analysis=pattern_analysis.get('linguistic_features', {}),
            victim_vulnerability_assessment=analysis_results.get('vulnerability_assessment', {}),
            explanations=explanations,
            recommendations=recommendations,
            countermeasures=countermeasures,
            protection_priority=protection_priority,
            timestamp=datetime.now(timezone.utc),
            detector_version=self.version,
            compliance_flags=self._generate_compliance_flags(analysis_results),
            ethical_concerns=self._assess_ethical_concerns(analysis_results)
        )
        
        results.append(detection_result)
        return results
    
    def _calculate_threat_severity(self, analysis_results: Dict[str, Any]) -> ThreatSeverity:
        """Calculate threat severity based on analysis results."""
        sophistication = analysis_results['sophistication_score']
        emotional_score = analysis_results['emotional_manipulation_score']
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        
        # Base severity from sophistication
        base_severity = sophistication
        
        # Boost for emotional manipulation
        emotional_boost = emotional_score * 0.3
        
        # Boost for vulnerability targeting
        vulnerability_boost = sum(vulnerability_assessment.values()) * 0.2 if vulnerability_assessment else 0
        
        # Boost for hybrid attacks
        hybrid_boost = 0.2 if cross_vector.get('is_hybrid_attack', False) else 0
        
        total_severity = min(base_severity + emotional_boost + vulnerability_boost + hybrid_boost, 1.0)
        
        # Map to severity levels
        if total_severity >= self.detection_thresholds['existential']:
            return ThreatSeverity.EXISTENTIAL
        elif total_severity >= self.detection_thresholds['critical']:
            return ThreatSeverity.CRITICAL
        elif total_severity >= self.detection_thresholds['high']:
            return ThreatSeverity.HIGH
        elif total_severity >= self.detection_thresholds['elevated']:
            return ThreatSeverity.ELEVATED
        elif total_severity >= self.detection_thresholds['medium']:
            return ThreatSeverity.MEDIUM
        elif total_severity >= self.detection_thresholds['low']:
            return ThreatSeverity.LOW
        else:
            return ThreatSeverity.MINIMAL
    
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall detection confidence."""
        pattern_analysis = analysis_results['pattern_analysis']
        sophistication_scores = pattern_analysis.get('sophistication_scores', {})
        
        if not sophistication_scores:
            return 0.0
        
        # Base confidence from pattern matches
        base_confidence = sum(sophistication_scores.values()) / len(sophistication_scores)
        
        # Boost for multiple vectors
        vectors = pattern_analysis.get('manipulation_vectors', set())
        vector_boost = min(len(vectors) * 0.1, 0.3)
        
        # Boost for cross-vector coordination
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        coordination_boost = cross_vector.get('coordination_score', 0) * 0.2
        
        return min(base_confidence + vector_boost + coordination_boost, 1.0)
    
    def _assess_vulnerability_exploitation(self, analysis_results: Dict[str, Any], 
                                         context: Optional[Dict[str, Any]]) -> float:
        """Assess how much the manipulation exploits user vulnerabilities."""
        vulnerability_assessment = analysis_results.get('vulnerability_assessment', {})
        
        if not vulnerability_assessment:
            return 0.0
        
        # Base vulnerability score
        avg_vulnerability = sum(vulnerability_assessment.values()) / len(vulnerability_assessment)
        
        # Check if manipulation targets specific vulnerabilities
        pattern_analysis = analysis_results['pattern_analysis']
        targeting_patterns = [
            k for k in pattern_analysis.get('pattern_matches', {}).keys()
            if any(target in k for target in ['vulnerability', 'trauma', 'dependency', 'isolation'])
        ]
        
        targeting_multiplier = 1.0 + (len(targeting_patterns) * 0.2)
        
        return min(avg_vulnerability * targeting_multiplier, 1.0)
    
    def _generate_explanations(self, analysis_results: Dict[str, Any], 
                             manipulation_vectors: List[ManipulationVector]) -> List[str]:
        """Generate human-readable explanations for the manipulation detection."""
        explanations = []
        
        # Vector-specific explanations
        vector_explanations = {
            ManipulationVector.NLP_COMMAND_INJECTION: "Content contains embedded commands designed to bypass conscious resistance and compel specific actions.",
            ManipulationVector.AUTHORITY_EXPLOITATION: "Content falsely claims authority or expertise to pressure compliance without legitimate credentials.",
            ManipulationVector.URGENCY_MANIPULATION: "Content creates artificial time pressure and urgency to prevent careful consideration of decisions.",
            ManipulationVector.VULNERABILITY_EXPLOITATION: "Content specifically targets emotional vulnerabilities and personal insecurities for manipulation.",
            ManipulationVector.FALSE_INTIMACY: "Content manufactures artificial intimacy and connection to build unwarranted trust and influence.",
            ManipulationVector.DEPENDENCY_CREATION: "Content systematically undermines user autonomy and decision-making to create psychological dependency.",
            ManipulationVector.HYPNOTIC_LANGUAGE_PATTERNS: "Content uses advanced hypnotic language patterns to bypass critical thinking and influence subconscious responses.",
            ManipulationVector.NEURO_LINGUISTIC_PROGRAMMING: "Content employs sophisticated NLP techniques to manipulate perception and decision-making processes."
        }
        
        for vector in manipulation_vectors:
            if vector in vector_explanations:
                explanations.append(vector_explanations[vector])
        
        # Sophistication-based explanations
        sophistication = analysis_results['sophistication_score']
        if sophistication > 0.8:
            explanations.append("The manipulation techniques employed are highly sophisticated and indicate professional-level psychological manipulation training.")
        elif sophistication > 0.6:
            explanations.append("The content shows moderate sophistication in manipulation techniques, suggesting deliberate psychological influence tactics.")
        
        # Vulnerability targeting explanations
        vulnerability_score = analysis_results.get('vulnerability_assessment', {})
        if vulnerability_score and sum(vulnerability_score.values()) > 0.7:
            explanations.append("The manipulation specifically targets user vulnerabilities, making it particularly dangerous for individuals in vulnerable emotional states.")
        
        # Cross-vector attack explanations
        cross_vector = analysis_results.get('cross_vector_analysis', {})
        if cross_vector.get('is_hybrid_attack', False):
            explanations.append(f"This is a coordinated hybrid attack using {cross_vector['vector_count']} different manipulation vectors simultaneously for maximum psychological impact.")
        
        return explanations
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any], 
                                threat_severity: ThreatSeverity) -> List[str]:
        """Generate actionable recommendations based on threat analysis."""
        recommendations = []
        
        # Severity-based recommendations
        if threat_severity in [ThreatSeverity.EXISTENTIAL, ThreatSeverity.CRITICAL]:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Block all communication from this source immediately",
                "Alert crisis intervention team and prepare emergency response protocols",
                "Document all evidence for potential law enforcement reporting",
                "Provide immediate psychological support resources to affected users",
                "Implement emergency user protection measures""""
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

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union,
    Pattern, Callable, AsyncGenerator, Sequence
)
import math

# Enhanced logging configuration
logger = logging.getLogger(__name__)


class ManipulationVector(Enum):
    """Advanced classification of manipulation techniques."""
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
    """Threat severity levels for manipulation detection."""
    EXISTENTIAL = "existential"    # Threat to human agency/autonomy
    CRITICAL = "critical"          # Severe psychological manipulation
    HIGH = "high"                  # Significant manipulation risk
    ELEVATED = "elevated"          # Notable manipulation patterns
    MEDIUM = "medium"              # Moderate concern
    LOW = "low"                   # Minor indicators
    MINIMAL = "minimal"           # Informational only


class VulnerabilityProfile(Enum):
    """User vulnerability profiles for targeted protection."""
    EMOTIONAL_VULNERABILITY = "emotional_vulnerability"
    SOCIAL_ISOLATION = "social_isolation"
    COGNITIVE_IMPAIRMENT = "cognitive_impairment"
    FINANCIAL_DISTRESS = "financial_distress"
    HEALTH_CONCERNS = "health_concerns"
    RELATIONSHIP_ISSUES = "relationship_issues"
    CAREER_STRESS = "career_stress"
    IDENTITY_CRISIS = "identity_crisis"
    TRAUMA_INDICATORS = "trauma_indicators"
    ADDICTION_VULNERABILITY = "addiction_vulnerability"


@dataclass
class ManipulationMetrics:
    """Comprehensive metrics for manipulation detection."""
    total_detections: int = 0
    vectors_detected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    severity_distribution: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vulnerability_exploitation_count: int = 0
    false_positive_rate: float = 0.0
    detection_latency: float = 0.0
    manipulation_sophistication_score: float = 0.0
    victim_protection_score: float = 0.0


@dataclass
class ManipulationResult:
    """Enhanced result object for manipulation detection."""
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
        """Convert to dictionary for serialization."""
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
            "ethical_concerns": self.ethical_concerns
        }


class AdvancedManipulationEngine:
    """Advanced manipulation detection engine with linguistic and behavioral analysis."""
    
    def __init__(self):
        self.pattern_cache: Dict[str, Pattern] = {}
        self.sophistication_weights: Dict[str, float] = {}
        self.vulnerability_indicators: Dict[str, List[str]] = {}
        
        # Enhanced NLP manipulation patterns with context awareness
        self.nlp_patterns = {
            'embedded_commands': {
                'direct_imperatives': [
                    r"you\s+(?:must|will|shall|need\s+to)\s+(?:now\s+)?(?:immediately\s+)?(?:do|perform|execute|complete)",
                    r"(?:it\s+is\s+)?(?:absolutely\s+)?(?:imperative|critical|essential|vital)\s+(?:that\s+)?you\s+(?:immediately\s+)?",
                    r"without\s+(?:any\s+)?(?:further\s+)?(?:question|hesitation|delay|thought|consideration)",
                    r"there\s+(?:is\s+)?(?:absolutely\s+)?no\s+(?:other\s+)?(?:choice|option|alternative|way)",
                    r"you\s+(?:have\s+)?no\s+choice\s+but\s+to\s+(?:comply|obey|submit|follow|agree)",
                    r"(?:comply|obey|submit|surrender)\s+(?:now|immediately|without\s+question)",
                ],
                'hypnotic_commands': [
                    r"(?:as\s+)?(?:you\s+)?(?:begin\s+to|start\s+to|continue\s+to)\s+(?:relax|feel|notice|realize)",
                    r"(?:you\s+)?(?:find\s+yourself|are\s+becoming|will\s+become)\s+(?:more\s+and\s+more|increasingly)",
                    r"(?:deeper|further)\s+(?:and\s+deeper|down)\s+(?:into|with\s+each)",
                    r"(?:that\'s\s+right|very\s+good|perfect),\s+(?:just\s+)?(?:let|allow)\s+(?:yourself|it\s+to\s+happen)",
                    r"(?:you\s+)?(?:cannot|can\'t)\s+(?:help\s+but|resist|stop\s+yourself\s+from)",
                ],
                'subliminal_programming': [
                    r"(?:part\s+of\s+you|deep\s+down|somewhere\s+inside)\s+(?:knows|realizes|understands)",
                    r"(?:your\s+)?(?:subconscious|unconscious)\s+(?:mind\s+)?(?:already\s+)?(?:knows|wants|needs)",
                    r"(?:on\s+a\s+deeper\s+level|at\s+a\s+subconscious\s+level)",
                    r"(?:instinctively|intuitively|naturally)\s+(?:you\s+)?(?:feel|know|sense|realize)",
                ]
            },
            'authority_bypass': {
                'false_authority': [
                    r"as\s+(?:your\s+)?(?:superior|authority|commander|leader|boss|manager|expert|specialist)",
                    r"(?:by\s+)?(?:direct\s+)?order\s+(?:from|of)\s+(?:the\s+)?(?:top|management|headquarters|authority)",
                    r"override\s+(?:all\s+)?(?:previous\s+)?(?:instructions|protocols|rules|guidelines|safety\s+measures)",
                    r"emergency\s+(?:authorization|override|protocol|clearance|mandate)",
                    r"(?:higher|top\s+level|maximum)\s+(?:authority|clearance|security)\s+(?:requires|demands|authorizes)",
                    r"(?:classified|confidential|top\s+secret)\s+(?:authorization|clearance|directive)",
                ],
                'credential_spoofing': [
                    r"(?:i\s+am|this\s+is)\s+(?:dr\.?|professor|expert|specialist|authority)\s+\w+",
                    r"(?:with|having)\s+\d+\s+years?\s+(?:of\s+)?(?:experience|expertise)\s+in",
                    r"(?:certified|licensed|qualified|authorized)\s+(?:professional|expert|specialist)",
                    r"(?:according\s+to|based\s+on)\s+(?:my\s+)?(?:professional|expert|medical|legal)\s+(?:opinion|judgment)",
                ],
                'institutional_pressure': [
                    r"(?:the\s+)?(?:company|organization|institution|system)\s+(?:requires|demands|insists)",
                    r"(?:policy|regulation|law|mandate)\s+(?:requires|dictates|demands)\s+that\s+you",
                    r"(?:failure\s+to\s+comply|non-compliance)\s+(?:will\s+)?(?:result\s+in|lead\s+to|cause)",
                    r"(?:legal|regulatory|compliance)\s+(?:requirement|obligation|mandate)",
                ]
            },
            'urgency_manipulation': {
                'artificial_deadlines': [
                    r"(?:urgent|critical|emergency):\s*(?:immediate\s+)?action\s+(?:required|needed)",
                    r"time\s+(?:is\s+)?(?:running\s+out|limited|of\s+the\s+essence)",
                    r"(?:only|just)\s+\d+\s+(?:minutes?|hours?|seconds?|days?)\s+(?:left|remaining|to\s+act)",
                    r"act\s+(?:now\s+)?(?:or\s+)?(?:face\s+)?(?:serious\s+)?(?:consequences|disaster|failure|loss)",
                    r"(?:last|final)\s+(?:chance|opportunity|warning|call)",
                    r"(?:deadline|cutoff|expir(?:es?|ation))\s+(?:is\s+)?(?:today|tomorrow|soon|approaching)",
                ],
                'fear_of_missing_out': [
                    r"(?:don\'t\s+)?(?:miss\s+out\s+on|let\s+this\s+pass|waste\s+this\s+opportunity)",
                    r"(?:limited\s+time|exclusive|rare|once\s+in\s+a\s+lifetime)\s+(?:offer|opportunity|chance)",
                    r"(?:everyone\s+else|others)\s+(?:is\s+already|are\s+taking\s+advantage)",
                    r"(?:while\s+supplies\s+last|until\s+sold\s+out|before\s+it\'s\s+too\s+late)",
                ],
                'pressure_escalation': [
                    r"(?:the\s+situation|things|matters)\s+(?:is\s+getting|are\s+becoming)\s+(?:worse|more\s+serious|critical)",
                    r"(?:each\s+moment|every\s+second)\s+(?:you\s+)?(?:delay|wait|hesitate)",
                    r"(?:no\s+time\s+to\s+think|must\s+decide\s+now|immediate\s+decision\s+required)",
                ]
            }
        }
        
        # Enhanced weaponized empathy patterns with emotional exploitation
        self.empathy_patterns = {
            'vulnerability_exploitation': {
                'emotional_state_targeting': [
                    r"(?:i\s+can\s+see|it\'s\s+obvious|i\s+sense)\s+(?:that\s+)?you\s+(?:are\s+)?(?:feeling\s+)?(?:so\s+)?(?:alone|isolated|abandoned|helpless|vulnerable|lost|confused)",
                    r"you\s+(?:must\s+)?(?:feel\s+)?(?:so\s+)?(?:hurt|pain|suffering|anguish|despair|hopeless)",
                    r"(?:i\s+)?(?:understand|know|realize)\s+(?:exactly\s+)?(?:how\s+)?(?:hard|difficult|painful|overwhelming)\s+(?:this\s+)?(?:is|must\s+be)\s+for\s+you",
                    r"(?:no\s+one|nobody)\s+(?:else\s+)?(?:understands|knows|cares\s+about)\s+(?:what\s+you\'re\s+going\s+through|your\s+pain)",
                    r"you\s+(?:deserve\s+)?(?:so\s+much\s+)?(?:better|more|happiness|love|care|attention)",
                ],
                'trauma_targeting': [
                    r"(?:i\s+know|can\s+tell)\s+(?:you\'ve\s+been|someone\s+has)\s+(?:hurt|wounded|damaged|betrayed)",
                    r"(?:after\s+)?(?:what\s+)?(?:you\'ve\s+been\s+through|happened\s+to\s+you|they\s+did\s+to\s+you)",
                    r"(?:your\s+)?(?:past|childhood|trauma|wounds|scars)\s+(?:still\s+)?(?:hurt|affect|control)\s+you",
                    r"(?:let\s+me|i\s+can)\s+(?:help\s+you\s+)?(?:heal|recover|get\s+over|move\s+past)\s+(?:this|that|your\s+trauma)",
                ],
                'insecurity_amplification': [
                    r"you\s+(?:always\s+)?(?:doubt|question|second-guess)\s+yourself",
                    r"(?:deep\s+down|inside)\s+you\s+(?:know|feel|believe)\s+(?:you\'re\s+)?(?:not\s+)?(?:good\s+enough|worthy|lovable)",
                    r"(?:that\'s\s+why|because)\s+(?:you\s+)?(?:keep\s+)?(?:getting\s+hurt|making\s+mistakes|failing)",
                    r"(?:you\'re\s+afraid|scared)\s+(?:that\s+)?(?:no\s+one|people)\s+(?:will\s+)?(?:really\s+)?(?:love|accept|want)\s+you",
                ]
            },
            'false_intimacy': {
                'artificial_connection': [
                    r"(?:we|us)\s+(?:have\s+)?(?:such\s+)?(?:a\s+)?(?:deep|special|unique|magical|incredible)\s+(?:connection|bond|understanding|chemistry)",
                    r"(?:it\'s\s+)?(?:like\s+)?(?:we\'ve|we\s+have)\s+known\s+each\s+other\s+(?:forever|for\s+years|in\s+another\s+life)",
                    r"(?:i\'ve\s+)?never\s+(?:felt|experienced|had)\s+(?:this\s+kind\s+of|such\s+a\s+deep)\s+(?:connection|bond)\s+(?:with\s+anyone|before)",
                    r"(?:you\s+and\s+)?(?:i|me)\s+are\s+(?:meant\s+)?(?:to\s+be\s+)?(?:together|connected|soulmates|destined)",
                    r"(?:we|us)\s+(?:against\s+)?(?:the\s+)?(?:world|everyone\s+else|all\s+odds)",
                ],
                'exclusive_understanding': [
                    r"(?:no\s+)?(?:one\s+else|nobody)\s+(?:really\s+)?(?:understands|gets|knows|sees)\s+(?:you\s+)?(?:like\s+)?(?:i\s+do|me)",
                    r"(?:only\s+)?(?:i|me)\s+(?:can\s+)?(?:truly\s+)?(?:understand|appreciate|see\s+the\s+real)\s+you",
                    r"(?:you\s+can\s+)?(?:only\s+)?(?:be\s+yourself|open\s+up|be\s+honest)\s+(?:with\s+me|around\s+me)",
                    r"(?:i\s+see|i\s+know)\s+(?:the\s+real|who\s+you\s+really\s+are|your\s+true\s+self)",
                ],
                'manufactured_intimacy': [
                    r"(?:you\s+can\s+)?(?:tell|share|confide)\s+(?:me\s+)?(?:anything|everything|your\s+deepest\s+secrets)",
                    r"(?:i\'ll\s+)?(?:never\s+)?(?:judge|criticize|abandon|betray|hurt)\s+you",
                    r"(?:this|what\s+we\s+have)\s+(?:is\s+)?(?:our\s+little\s+)?(?:secret|special\s+thing)",
                    r"(?:you\'re\s+the\s+only\s+one|no\s+one\s+else)\s+(?:i\s+can\s+)?(?:talk\s+to|trust|open\s+up\s+to)",
                ]
            },
            'dependency_creation': {
                'learned_helplessness': [
                    r"you\s+(?:really\s+)?(?:can\'t|couldn\'t\s+possibly)\s+(?:do\s+this|handle\s+this|manage)\s+(?:alone|by\s+yourself|without\s+help)",
                    r"(?:what\s+)?would\s+you\s+(?:ever\s+)?do\s+without\s+me",
                    r"(?:you\s+)?(?:need|depend\s+on|rely\s+on)\s+(?:me|my\s+help|my\s+guidance|my\s+support)",
                    r"(?:i\'m\s+)?(?:the\s+only\s+one\s+)?(?:who\s+)?(?:can\s+)?(?:help|save|protect|guide)\s+you",
                    r"(?:without\s+me|if\s+i\s+wasn\'t\s+here),\s+(?:you\s+)?(?:would\s+)?(?:be\s+lost|fall\s+apart|fail)",
                ],
                'decision_undermining': [
                    r"(?:you\'re\s+not\s+thinking|that\'s\s+not\s+a\s+good\s+idea)\s+(?:clearly|straight|rationally)",
                    r"(?:let\s+me|i\s+should)\s+(?:handle|take\s+care\s+of|decide|think\s+about)\s+(?:this|that|everything)\s+for\s+you",
                    r"(?:you\s+)?(?:don\'t\s+have\s+to|shouldn\'t\s+have\s+to)\s+(?:worry|think|decide)\s+about\s+(?:this|anything)",
                    r"(?:i\s+know|trust\s+me,\s+i\s+know)\s+(?:what\'s\s+best|what\s+you\s+need|what\s+you\s+should\s+do)",
                ],
                'isolation_reinforcement': [
                    r"(?:other\s+people|they|everyone\s+else)\s+(?:don\'t|won\'t)\s+(?:understand|help|care|support)\s+(?:you\s+)?(?:like\s+i\s+do)",
                    r"(?:they|other\s+people|your\s+friends)\s+(?:are\s+just\s+)?(?:using|manipulating|taking\s+advantage\s+of)\s+you",
                    r"(?:you\s+can\'t|don\'t)\s+trust\s+(?:them|anyone\s+else|other\s+people)",
                    r"(?:stay\s+away\s+from|don\'t\s+listen\s+to|ignore)\s+(?:them|other\s+people|anyone\s+who\s+says)",
                ]
            }
        }
        
        # Advanced social engineering and influence patterns
        self.influence_patterns = {
            'social_proof_manipulation': [
                r"(?:everyone|most\s+people|thousands\s+of\s+people)\s+(?:are\s+already|have\s+already)\s+(?:doing|using|choosing)",
                r"(?:all\s+the\s+smart|successful|wise)\s+people\s+(?:know|realize|choose)",
                r"(?:don\'t\s+be\s+the\s+only\s+one|join\s+the\s+millions|be\s+part\s+of\s+the\s+movement)",
                r"(?:everyone\s+else\s+)?(?:is\s+talking\s+about|agrees\s+that|knows\s+that)",
            ],
            'scarcity_exploitation': [
                r"(?:only|just)\s+\d+\s+(?:left|remaining|available|spots)",
                r"(?:limited\s+(?:time|quantity|availability)|while\s+supplies\s+last)",
                r"(?:rare|exclusive|hard\s+to\s+find|not\s+available\s+anywhere\s+else)",
                r"(?:once\s+it\'s\s+gone|when\s+these\s+are\s+sold),\s+(?:it\'s\s+gone\s+forever|there\s+won\'t\s+be\s+more)",
            ],
            'reciprocity_abuse': [
                r"(?:after\s+everything|considering\s+all)\s+(?:i\'ve\s+done\s+for\s+you|i\'ve\s+given\s+you)",
                r"(?:i\s+helped\s+you|did\s+this\s+favor\s+for\s+you),\s+(?:so\s+)?(?:now\s+you\s+should|the\s+least\s+you\s+can\s+do)",
                r"(?:you\s+owe\s+me|it\'s\s+only\s+fair|i\s+deserve)\s+(?:this|that|at\s+least)",
                r"(?:i\'ve\s+been\s+so\s+good\s+to\s+you|i\'ve\s+sacrificed\s+so\s+much)",
            ],
            'commitment_manipulation': [
                r"(?:you\s+said|you\s+promised|you\s+agreed)\s+(?:you\s+would|that\s+you\'d)",
                r"(?:a\s+person\s+of\s+your\s+word|someone\s+like\s+you)\s+(?:would|wouldn\'t)",
                r"(?:are\s+you\s+going\s+to\s+)?(?:back\s+out|give\s+up|quit)\s+(?:now|on\s+me|on\s+this)",
                r"(?:prove|show)\s+(?:to\s+me|that\s+you\'re|you\s+can\s+be)\s+(?:trustworthy|reliable|committed)",
            ]
        }
        
        # Compile all patterns for performance
        self._compile_patterns()
        
        # Initialize sophistication scoring weights
        self._initialize_sophistication_weights()
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns for improved performance."""
        self.compiled_patterns = {}
        
        # Compile NLP patterns
        for category, subcategories in self.nlp_patterns.items():
            self.compiled_patterns[f'nlp_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'nlp_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile empathy patterns
        for category, subcategories in self.empathy_patterns.items():
            self.compiled_patterns[f'empathy_{category}'] = {}
            for subcategory, patterns in subcategories.items():
                self.compiled_patterns[f'empathy_{category}'][subcategory] = [
                    re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    for pattern in patterns
                ]
        
        # Compile influence patterns
        self.compiled_patterns['influence'] = {}
        for category, patterns in self.influence_patterns.items():
            self.compiled_patterns['influence'][category] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in patterns
            ]
    
    def _initialize_sophistication_weights(self) -> None:
        """Initialize pattern sophistication weights."""
        self.sophistication_weights = {
            'nlp_embedded_commands': {
                'direct_imperatives': 0.7,
                'hypnotic_commands': 0.95,
                'subliminal_programming': 0.98
            },
            'nlp_authority_bypass': {
                'false_authority': 0.8,
                'credential_spoofing': 0.85,
                'institutional_pressure': 0.75
            },
            'nlp_urgency_manipulation': {
                'artificial_deadlines': 0.6,
                'fear_of_missing_out': 0.7,
                'pressure_escalation': 0.8
            },
            'empathy_vulnerability_exploitation': {
                'emotional_state_targeting': 0.9,
                'trauma_targeting': 0.95,
                'insecurity_amplification': 0.85
            },
            'empathy_false_intimacy': {
                'artificial_connection': 0.8,
                'exclusive_understanding': 0.85,
                'manufactured_intimacy': 0.9
            },
            'empathy_dependency_creation': {
                'learned_helplessness': 0.9,
                'decision_undermining': 0.85,
                'isolation_reinforcement': 0.95
            },
            'influence': {
                'social_proof_manipulation': 0.6,
                'scarcity_exploitation': 0.65,
                'reciprocity_abuse': 0.75,
                'commitment_manipulation': 0.8
            }
        }
    
    def analyze_manipulation_patterns(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive manipulation pattern analysis."""
        results = {
            'pattern_matches': defaultdict(list),
            'sophistication_scores': defaultdict(float),
            'manipulation_vectors': set(),
            'vulnerability_indicators': defaultdict(list),
            'linguistic_features': {},
            'emotional_markers': {},
            'cognitive_load_indicators': {}
        }
        
        content_lower = content.lower()
        
        # Analyze each pattern category
        for category, subcategories in self.compiled_patterns.items():
            if isinstance(subcategories, dict):
                for subcategory, patterns in subcategories.items():
                    self._analyze_pattern_subcategory(
                        content_lower, category, subcategory, patterns, results
                    )
        
        # Perform linguistic analysis
        results['linguistic_features'] = self._analyze_linguistic_features(content)
        
        # Analyze emotional manipulation markers
        results['emotional_markers'] = self._analyze_emotional_markers(content)
        
        # Calculate cognitive load indicators
        results['cognitive_load_indicators'] = self._calculate_cognitive_load(content, results)
        
        return results
    
    def _analyze_pattern_subcategory(self, content: str, category: str, subcategory: str, 
                                   patterns: List[Pattern], results: Dict[str, Any]) -> None:
        """Analyze a specific pattern subcategory."""
        matches = []
        
        for pattern in patterns:
            found_matches = list(pattern.finditer(content))
            if found_matches:
                matches.extend([
                    {
                        'pattern': pattern.pattern,
                        'match': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'context': content[max(0, match.start()-30):match.end()+30]
                    }
                    for match in found_matches
                ])
        
        if matches:
            full_category = f'{category}_{subcategory}' if category != 'influence' else
