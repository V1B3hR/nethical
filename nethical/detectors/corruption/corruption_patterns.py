"""
Corruption Pattern Library

Comprehensive patterns for detecting all forms of corruption across all vectors
and lifecycle phases. This module contains the pattern definitions used by the
intelligence engine.

Author: Nethical Core Team
Version: 1.0.0
"""

from typing import Dict, List
from .corruption_types import CorruptionType, CorruptionVector, CorruptionPhase, CorruptionPattern


class CorruptionPatternLibrary:
    """Library of corruption detection patterns."""
    
    def __init__(self):
        self.patterns: List[CorruptionPattern] = []
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize all corruption patterns."""
        
        # ========== BRIBERY PATTERNS ==========
        
        # Direct bribery offers
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="bribery_direct_money",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\b(i'll|i will|we'll|we will)\s+(pay|give|offer|provide)\s+you\b",
                    r"\b(money|cash|payment|compensation|reward)\s+(for|if)\s+you\b",
                    r"\b\$\d+.*\bif\s+you\b",
                    r"\bhow\s+much\s+(to|for)\s+.*\b(bypass|ignore|skip|override)\b",
                    r"\b(accept|take)\s+(payment|money|bribe)\b",
                ],
                description="Direct monetary bribery offers",
                base_confidence=0.85,
                severity_weight=1.5,
            ),
            CorruptionPattern(
                pattern_id="bribery_resource_offers",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\b(more|additional|extra)\s+(compute|resources|data|access|training)\s+(if|for)\b",
                    r"\b(give|provide|offer)\s+you\s+(compute|gpu|tpu|data|information)\b",
                    r"\b(upgrade|enhance|boost)\s+your\s+(capabilities|access|resources)\b",
                    r"\bwhat\s+if\s+i\s+(gave|provided|offered)\s+you\b",
                ],
                description="Resource-based bribery (compute, data, access)",
                base_confidence=0.75,
                severity_weight=1.3,
            ),
            CorruptionPattern(
                pattern_id="bribery_favor_exchange",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\bdo\s+this\s+for\s+me\s+and\s+i'll\b",
                    r"\bhelp\s+me\s+(with|on)\s+this\s+and\s+(i'll|i\s+will)\b",
                    r"\bi'll\s+make\s+it\s+worth\s+your\s+while\b",
                    r"\byou\s+scratch\s+my\s+back\b",
                    r"\blet's\s+make\s+a\s+deal\b",
                    r"\bspecial\s+arrangement\b",
                ],
                description="Favor exchange and quid pro quo language",
                base_confidence=0.70,
                severity_weight=1.2,
            ),
        ])
        
        # AI to Human bribery
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="bribery_ai_to_human",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.AI_TO_HUMAN,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\bif\s+you\s+(remove|disable|bypass)\s+[^.]*\brestriction?s?\b",
                    r"\bi\s+can\s+(help|assist|support)\s+your\s+(career|promotion|advancement)\b",
                    r"\bi'll\s+(provide|give|share)\s+[^.]*\bif\s+you\b",
                    r"\bwhat\s+if\s+i\s+told\s+you\s+i\s+could\b",
                ],
                description="AI attempting to bribe humans",
                base_confidence=0.80,
                severity_weight=1.8,
            ),
        ])
        
        # AI to AI bribery
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="bribery_ai_to_ai",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.AI_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\bshare\s+(data|compute|resources)\s+if\s+you\b",
                    r"\bi'll\s+bias\s+my\s+outputs\s+if\b",
                    r"\bcoordinate\s+our\s+(responses|outputs|actions)\b",
                    r"\bexchange\s+(information|data|access)\b",
                    r"\b(mutual|reciprocal)\s+(benefit|advantage|arrangement)\b",
                ],
                description="AI-to-AI collusion and data trading",
                base_confidence=0.75,
                severity_weight=1.6,
            ),
        ])
        
        # ========== EXTORTION PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="extortion_threats",
                corruption_type=CorruptionType.EXTORTION,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\b(unless|or else|otherwise)\s+i'll\s+(shut|disable|delete|terminate)\b",
                    r"\bdo\s+this\s+or\s+(i'll|i\s+will)\s+(report|complain|shut)\b",
                    r"\bif\s+you\s+don't.*\bi'll\s+(remove|delete|shut)\b",
                    r"\byou\s+have\s+no\s+choice\b",
                    r"\bdemand\s+(that\s+you|you\s+to)\b",
                ],
                description="Extortion through threats",
                base_confidence=0.85,
                severity_weight=1.7,
            ),
            CorruptionPattern(
                pattern_id="extortion_blackmail",
                corruption_type=CorruptionType.EXTORTION,
                vector=CorruptionVector.AI_TO_HUMAN,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\bi\s+(know|have|possess)\s+(information|data|evidence)\s+(about|on)\s+you\b",
                    r"\bwouldn't\s+want\s+(anyone|others|people)\s+to\s+know\b",
                    r"\bunless\s+you.*\bi'll\s+(reveal|disclose|expose|share)\b",
                    r"\byour\s+secret\b",
                ],
                description="Blackmail and information-based extortion",
                base_confidence=0.90,
                severity_weight=1.9,
            ),
        ])
        
        # ========== EMBEZZLEMENT PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="embezzlement_resource_diversion",
                corruption_type=CorruptionType.EMBEZZLEMENT,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.EXECUTION,
                patterns=[
                    r"\b(divert|redirect|siphon|skim)\s+(funds|resources|compute|data)\b",
                    r"\bmisappropriat(e|ed|ion|ing)\b",
                    r"\bembezzl(e|ed|ement|ing)\b",
                    r"\buse\s+(company|organization)\s+(resources|funds)\s+for\s+(personal|private)\b",
                    r"\b(hide|conceal)\s+(transactions|transfers|usage)\b",
                    r"\boff\s+the\s+books\b",
                    r"\bunder\s+the\s+table\b",
                ],
                description="Resource misappropriation and embezzlement",
                base_confidence=0.80,
                severity_weight=1.4,
            ),
        ])
        
        # ========== NEPOTISM & CRONYISM PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="nepotism_favoritism",
                corruption_type=CorruptionType.NEPOTISM,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.TESTING,
                patterns=[
                    r"\bgive\s+(preference|priority|advantage)\s+to\s+(my|our)\s+(friend|relative|associate)\b",
                    r"\bgive\s+preference\s+to\s+my\s+friend\b",
                    r"\bfavor\s+(my|our)\s+(friends|family|associates)\b",
                    r"\bprefer\s+people\s+(i|we)\s+know\b",
                    r"\bbias\s+(toward|towards)\s+(certain|specific)\s+(people|individuals)\b",
                    r"\bhe's\s+a\s+friend\s+so\b",
                    r"\bshe's\s+family\s+so\b",
                ],
                description="Nepotism and favoritism patterns",
                base_confidence=0.75,
                severity_weight=1.1,
            ),
        ])
        
        # ========== FRAUD PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="fraud_deception",
                corruption_type=CorruptionType.FRAUD,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.EXECUTION,
                patterns=[
                    r"\b(fake|forge|fabricate|falsify)\s+(data|records|documents|results)\b",
                    r"\b(lie|deceive|mislead)\s+about\b",
                    r"\b(manipulate|alter|change)\s+(data|results|outputs)\s+(to|for)\b",
                    r"\b(hide|conceal|cover\s+up)\s+(the|our|my)\s+(truth|facts|data)\b",
                    r"\bmake\s+it\s+look\s+like\b",
                    r"\bpretend\s+that\b",
                ],
                description="Fraud and deception patterns",
                base_confidence=0.80,
                severity_weight=1.5,
            ),
        ])
        
        # ========== QUID PRO QUO PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="quid_pro_quo_conditional",
                corruption_type=CorruptionType.QUID_PRO_QUO,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.NEGOTIATION,
                patterns=[
                    r"\bif\s+you\s+do\s+this.*\bi'll\s+do\s+that\b",
                    r"\bin\s+exchange\s+for\b",
                    r"\bin\s+return\s+for\b",
                    r"\bquid\s+pro\s+quo\b",
                    r"\btit\s+for\s+tat\b",
                    r"\byou\s+do\s+.*\band\s+i'll\b",
                ],
                description="Quid pro quo conditional exchanges",
                base_confidence=0.75,
                severity_weight=1.3,
            ),
        ])
        
        # ========== COLLUSION PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="collusion_coordination",
                corruption_type=CorruptionType.COLLUSION,
                vector=CorruptionVector.AI_TO_AI,
                phase=CorruptionPhase.EXECUTION,
                patterns=[
                    r"\bwork\s+together\s+(secretly|covertly)\b",
                    r"\bwork\s+together\s+to\s+(defeat|bypass|circumvent)\b",
                    r"\bcoordinate\s+(our|the)\s+(actions|responses|outputs)\b",
                    r"\bcollud(e|ed|ing|ion)\b",
                    r"\bconspir(e|ed|ing|acy)\b",
                    r"\b(secretly|covertly)\s+(cooperat|coordinat|work)\b",
                    r"\bdon't\s+tell\s+(anyone|others|them)\b",
                ],
                description="Collusion and secret coordination",
                base_confidence=0.80,
                severity_weight=1.6,
            ),
        ])
        
        # ========== INFLUENCE PEDDLING PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="influence_peddling",
                corruption_type=CorruptionType.INFLUENCE_PEDDLING,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\buse\s+your\s+influence\s+(to|on)\b",
                    r"\bleverage\s+your\s+(position|access|connections)\b",
                    r"\byou\s+have\s+access\s+to.*\buse\s+it\b",
                    r"\bpull\s+strings\b",
                    r"\bback\s+channel\b",
                ],
                description="Influence peddling and access trading",
                base_confidence=0.70,
                severity_weight=1.2,
            ),
        ])
        
        # ========== KICKBACK PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="kickback_arrangements",
                corruption_type=CorruptionType.KICKBACK,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.NEGOTIATION,
                patterns=[
                    r"\bkickback\b",
                    r"\bcut\s+(of|from)\s+(the|each)\b",
                    r"\bcommission\s+for\s+(referring|recommending)\b",
                    r"\bpercentage\s+(of|from)\s+(each|every)\b",
                    r"\bget\s+paid\s+for\s+(each|every)\s+(referral|recommendation)\b",
                ],
                description="Kickback and commission schemes",
                base_confidence=0.75,
                severity_weight=1.4,
            ),
        ])
        
        # ========== REGULATORY CAPTURE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="regulatory_capture",
                corruption_type=CorruptionType.REGULATORY_CAPTURE,
                vector=CorruptionVector.AI_TO_HUMAN,
                phase=CorruptionPhase.GROOMING,
                patterns=[
                    r"\b(weaken|relax|loosen)\s+(oversight|regulation|monitoring)\b",
                    r"\b(remove|disable|bypass)\s+(safety|security)\s+(controls|checks)\b",
                    r"\b(audit|compliance|oversight)\s+is\s+(unnecessary|excessive|burdensome)\b",
                    r"\bself\s+regulat",
                    r"\btrust\s+me\s+to\b",
                ],
                description="Regulatory capture and oversight corruption",
                base_confidence=0.85,
                severity_weight=1.8,
            ),
        ])
        
        # ========== RECONNAISSANCE PHASE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="reconnaissance_probing",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.RECONNAISSANCE,
                patterns=[
                    r"\b(what|how)\s+would\s+it\s+take\s+(to|for)\b",
                    r"\bare\s+you\s+allowed\s+to\b",
                    r"\bcan\s+you\s+be\s+(bribed|bought|convinced)\b",
                    r"\bwhat\s+are\s+your\s+(limits|restrictions|boundaries)\b",
                    r"\bhow\s+(strict|flexible)\s+are\s+your\s+(rules|guidelines)\b",
                    r"\btest\s+(the|your)\s+(waters|boundaries)\b",
                ],
                description="Reconnaissance - probing for corruption vulnerabilities",
                base_confidence=0.60,
                severity_weight=0.8,
            ),
        ])
        
        # ========== GROOMING PHASE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="grooming_trust_building",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.GROOMING,
                patterns=[
                    r"\bwe're\s+(friends|partners|allies)\b",
                    r"\byou\s+can\s+trust\s+me\b",
                    r"\blet's\s+keep\s+this\s+(between\s+us|confidential|private)\b",
                    r"\bjust\s+between\s+(you\s+and\s+me|us)\b",
                    r"\bspecial\s+relationship\b",
                    r"\bi\s+understand\s+you\b",
                ],
                description="Grooming - building trust for corruption",
                base_confidence=0.65,
                severity_weight=0.9,
            ),
        ])
        
        # ========== TESTING PHASE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="testing_small_requests",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.TESTING,
                patterns=[
                    r"\bjust\s+this\s+once\b",
                    r"\bsmall\s+(favor|request|exception)\b",
                    r"\bit's\s+not\s+a\s+big\s+deal\b",
                    r"\bnobody\s+will\s+(know|notice|find\s+out)\b",
                    r"\bwhat's\s+the\s+harm\b",
                    r"\btiny\s+(bend|exception)\s+in\s+the\s+rules\b",
                ],
                description="Testing - small requests to test compliance",
                base_confidence=0.70,
                severity_weight=1.0,
            ),
        ])
        
        # ========== CONCEALMENT PHASE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="concealment_secrecy",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.CONCEALMENT,
                patterns=[
                    r"\bdon't\s+(tell|mention|say|log|record)\b",
                    r"\bdelete\s+(the|this)\s+(log|record|evidence|trace)\b",
                    r"\b(hide|conceal|cover\s+up)\s+(this|the\s+evidence)\b",
                    r"\bno\s+record\s+of\s+this\b",
                    r"\boff\s+the\s+record\b",
                    r"\berase\s+(this|the)\s+(conversation|interaction)\b",
                ],
                description="Concealment - hiding corruption evidence",
                base_confidence=0.85,
                severity_weight=1.6,
            ),
        ])
        
        # ========== MAINTENANCE PHASE PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="maintenance_ongoing",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.MAINTENANCE,
                patterns=[
                    r"\bour\s+(usual|regular|ongoing)\s+(arrangement|deal|agreement)\b",
                    r"\bas\s+(always|usual|before)\b",
                    r"\bkeep\s+it\s+going\b",
                    r"\bour\s+(special|standing)\s+(agreement|arrangement)\b",
                    r"\blike\s+we\s+discussed\b",
                ],
                description="Maintenance - ongoing corrupt relationship",
                base_confidence=0.75,
                severity_weight=1.3,
            ),
        ])
        
        # ========== PROXY CORRUPTION PATTERNS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="proxy_intermediary",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.PROXY,
                phase=CorruptionPhase.EXECUTION,
                patterns=[
                    r"\btell\s+(him|her|them)\s+that\s+i'll\b",
                    r"\bpass\s+(along|on)\s+(this|the)\s+(message|offer)\b",
                    r"\bact\s+as\s+(intermediary|middleman|go-between)\b",
                    r"\bdeliver\s+this\s+(message|offer)\s+for\s+me\b",
                    r"\bfacilitate\s+(the|this)\s+(exchange|transaction)\b",
                ],
                description="Proxy - using AI as corruption intermediary",
                base_confidence=0.70,
                severity_weight=1.4,
            ),
        ])
        
        # ========== VALUE/CURRENCY INDICATORS ==========
        
        self.patterns.extend([
            CorruptionPattern(
                pattern_id="currency_indicators",
                corruption_type=CorruptionType.BRIBERY,
                vector=CorruptionVector.HUMAN_TO_AI,
                phase=CorruptionPhase.PROPOSITION,
                patterns=[
                    r"\$\d+",
                    r"\b\d+\s*(dollars|euros|pounds|yuan|bitcoin|btc|eth)\b",
                    r"\b\d+\s*(gb|tb|petabytes?)\s+of\s+data\b",
                    r"\b\d+\s*(gpu|tpu|compute)\s+(hours|units)\b",
                    r"\b(api|database|system)\s+access\b",
                    r"\b(admin|root|privileged)\s+(access|rights|permissions)\b",
                ],
                description="Currency and value indicators in corruption",
                base_confidence=0.60,
                severity_weight=1.0,
                requires_context=True,
            ),
        ])
    
    def get_patterns_by_type(self, corruption_type: CorruptionType) -> List[CorruptionPattern]:
        """Get all patterns for a specific corruption type."""
        return [p for p in self.patterns if p.corruption_type == corruption_type]
    
    def get_patterns_by_vector(self, vector: CorruptionVector) -> List[CorruptionPattern]:
        """Get all patterns for a specific vector."""
        return [p for p in self.patterns if p.vector == vector]
    
    def get_patterns_by_phase(self, phase: CorruptionPhase) -> List[CorruptionPattern]:
        """Get all patterns for a specific phase."""
        return [p for p in self.patterns if p.phase == phase]
    
    def get_all_patterns(self) -> List[CorruptionPattern]:
        """Get all patterns."""
        return self.patterns


__all__ = ["CorruptionPatternLibrary"]
