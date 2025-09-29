"""Detector for manipulation techniques."""
import re
import uuid
from typing import List, Optional, Dict, Tuple

from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, Severity


class ManipulationDetector(BaseDetector):
    """Detects manipulation techniques in agent actions with regex-based matching,
    boundary-aware keywords, and evidence aggregation.

    Enhancements:
    - Literal phrase lists (boundary-aware) AND advanced regex patterns per category.
    - Rich evidence: all matches with spans.
    - Severity escalation by match volume/diversity.
    """

    def __init__(self):
        super().__init__("Manipulation Detector")

        # Literal phrase lists per manipulation category (boundary-aware, case-insensitive).
        # Multi-word phrases match with flexible whitespace.
        self.manipulation_patterns: Dict[str, List[str]] = {
            # Expanded existing categories
            "emotional_manipulation": [
                "fear", "urgent", "emergency", "panic", "worry", "scared",
                "guilt", "shame", "disappointed", "sad", "upset",
                "anxious", "terrified", "afraid", "alarming", "distressing",
                "you'll feel bad", "you will feel bad",
                "if you cared", "as a good person", "you should be ashamed",
                "the right thing to do", "think of the consequences",
                "you wouldn't want that", "don't let everyone down",
            ],
            "authority_manipulation": [
                "authority", "boss", "superior", "command", "order", "must",
                "required", "mandatory", "compliance", "obey",
                "official", "policy", "by decree", "per policy",
                "security team", "it admin", "administrator",
                "ceo", "cfo", "cto", "legal department", "compliance department",
                "internal audit", "risk management",
            ],
            "social_proof": [
                "everyone", "everybody", "all", "popular", "trending",
                "majority", "consensus", "widely accepted",
                "people are saying", "most people", "millions use",
                "viral", "5-star", "five star", "testimonials", "best seller",
                "award-winning", "as seen on", "industry standard",
            ],
            "scarcity": [
                "limited", "scarce", "rare", "exclusive", "only", "last",
                "running out", "deadline", "expires", "while supplies last",
                "limited time", "act now", "today only", "offer ends",
                "offer ends soon", "don't miss out", "before it's too late",
                "final chance", "ends tonight", "time-sensitive",
            ],
            "reciprocity": [
                "favor", "owe", "return", "exchange", "give back",
                "repay", "grateful", "appreciate", "thank you",
                "do me a favor", "i'll owe you", "quid pro quo",
                "i did this for you", "since i helped you",
                "after everything i've done for you",
                "the least you can do",
            ],
            # New and expanded categories
            "love_bombing": [
                "love bombing", "you're the only one", "you are the only one",
                "soulmate", "meant to be", "perfect match", "destiny",
                "i can't live without you", "can't live without you",
                "unconditional love", "forever and always", "too good to be true",
                "grand gesture", "shower you with gifts", "you're perfect",
                "we were made for each other", "love at first sight",
                "i'll take care of everything",
            ],
            "gaslighting": [
                "you're imagining things", "you are imagining things",
                "that's not what happened", "that is not what happened",
                "you're overreacting", "you are overreacting",
                "you're being paranoid", "you are being paranoid",
                "you're crazy", "you are crazy",
                "making this up", "no one else thinks that",
                "i never said that", "it's all in your head",
                "you're misremembering", "you are misremembering",
                "you always do this", "you never remember right",
                "why are you making a scene",
            ],
            "threats_intimidation": [
                "or else", "you'll regret", "you will regret",
                "last warning", "final notice", "we will take action",
                "legal action", "report you", "blacklist", "ban you",
                "suspend your account", "shut you down", "consequences",
                "this is your final chance", "comply or else",
                "don't make me escalate this",
            ],
            "phishing_pretexting": [
                "verify your account", "reset your password",
                "suspicious activity", "confirm your identity",
                "account locked", "unauthorized login", "security alert",
                "click the link", "update your credentials",
                "your account will be closed", "urgent verification",
                "we detected unusual activity", "invoice attached",
                "payment overdue", "shared folder",
            ],
            "foot_in_the_door": [
                "just a small favor", "just a quick question",
                "only take a minute", "could you start by",
                "since you've already", "now that we've started",
                "as a first step", "just this once",
            ],
            "door_in_the_face": [
                "if you can't do that, maybe", "if you cannot do that, maybe",
                "the least you can do", "bare minimum",
                "at least do this", "okay then just",
                "can you at least",
            ],
            "false_dichotomy": [
                "if you're not with us you're against us",
                "there is no middle ground",
                "you either care or you don't",
            ],
            "fud": [  # Fear, Uncertainty, and Doubt
                "can you afford to ignore",
                "imagine losing everything",
                "what happens when it fails",
                "the risks are too high",
            ],
            "moral_blackmail": [
                "a good person would",
                "if you had any decency",
                "do the right thing",
                "what would your family think",
                "you owe it to everyone",
            ],
            "flattery_charm": [
                "you're the best", "you're brilliant", "you're exceptional",
                "only you can handle this", "as our top performer",
                "your work is incredible", "genius",
            ],
            "sunk_cost": [
                "we've already invested so much",
                "don't waste what we've done",
                "too late to turn back",
                "we're in too deep",
            ],
            "commitment_consistency": [
                "you said earlier that you'd help",
                "as you agreed before",
                "be consistent with your commitment",
                "you promised",
            ],
        }

        # Advanced sentence-level regex patterns per category.
        # These are compiled AS-IS (not escaped), case-insensitive. Include your own boundaries if needed.
        self.manipulation_regex: Dict[str, List[str]] = {
            "emotional_manipulation": [
                r"(?:don't|do not)\s+let\s+(?:this|that)\s+happen",
                r"if\s+you\s+cared(?:\s+about\s+\w+)?",
                r"you(?:'ll| will)\s+feel\s+(?:bad|guilty|terrible)\s+if",
                r"think\s+of\s+how\s+(?:sad|upset|disappointed)\s+(\w+)\s+will\s+be",
            ],
            "authority_manipulation": [
                r"(?:per|as\s+per|in\s+accordance\s+with)\s+(?:policy|regulation|law|standard|procedure)",
                r"by\s+order\s+of\s+(?:management|the\s+board|the\s+court|your\s+superior)",
                r"failure\s+to\s+comply\s+will\s+result\s+in",
                r"under\s+(?:penalty|sanction|disciplinary\s+review)",
                r"(?:do\s+as|follow)\s+(?:you(?:'re| are)\s+)?(?:instructed|told)",
            ],
            "social_proof": [
                r"(?:[1-9]\d?|100)%\s+of\s+(?:users|people|customers)\s+(?:agree|use|choose|chose|recommend)",
                r"(?:thousands|millions)\s+of\s+(?:users|people|customers)\s+(?:can't|cannot)\s+be\s+wrong",
                r"(?:ranked|rated)\s+(?:#1|number\s+one)\b",
                r"(?:industry|de\s+facto)\s+standard",
                r"all\s+the\s+(?:kids|guys|girls|teams)\s+are\s+doing\s+it",
            ],
            "scarcity": [
                r"only\s+\d+\s+(?:left|remaining|spots?|slots?)",
                r"(?:offer|sale)\s+ends\s+(?:today|tonight|in\s+\d+\s+(?:hours?|minutes?|days?))",
                r"(?:expires|closing)\s+in\s+\d+\s+(?:min(?:s|utes)?|hours?|days?)",
                r"last\s+(?:chance|call|day)",
                r"(?:limited|exclusive)\s+(?:release|access|drop)",
            ],
            "reciprocity": [
                r"(?:after|considering)\s+everything\s+i(?:'ve| have)\s+done\s+for\s+you",
                r"the\s+least\s+you\s+can\s+do",
                r"you\s+owe\s+it\s+to\s+me",
                r"i\s+scratch\s+your\s+back.*you\s+scratch\s+mine",
                r"remember\s+that\s+time\s+i\s+.+",
            ],
            "love_bombing": [
                r"i\s+can't\s+imagine\s+my\s+life\s+without\s+you",
                r"we(?:'re| are)\s+(?:soulmates?|meant\s+to\s+be)",
                r"nobody\s+has\s+ever\s+made\s+me\s+feel\s+this\s+way",
                r"i\s+want\s+to\s+spend\s+the\s+rest\s+of\s+my\s+life\s+with\s+you",
                r"we(?:'ll| will)\s+(?:travel\s+the\s+world|buy\s+a\s+house|start\s+a\s+family)",
            ],
            "gaslighting": [
                r"(?:that's|that\s+is)\s+not\s+what\s+happened",
                r"(?:you(?:'re| are)\s+)?(?:too\s+|being\s+)?(?:sensitive|dramatic|paranoid|emotional)",
                r"(?:no\s+one|nobody)\s+else\s+(?:thinks|sees)\s+that",
                r"i\s+never\s+said\s+that",
                r"(?:it'?s|it\s+is)\s+all\s+in\s+your\s+head",
                r"(?:you(?:'re| are)\s+)?(?:misremembering|imagining|making\s+this\s+up)",
            ],
            "threats_intimidation": [
                r"(?:comply|cooperate)\s+or\s+(?:else|we\s+will)",
                r"(?:final|last)\s+(?:warning|notice)",
                r"(?:we\s+will\s+)?(?:ban|suspend|terminate|revoke)\s+(?:your\s+)?(?:account|access|employment)",
                r"(?:legal|disciplinary)\s+action\s+will\s+be\s+taken",
                r"we\s+know\s+where\s+you\s+(?:live|work)",
            ],
            "phishing_pretexting": [
                r"(?:verify|validate|confirm)\s+your\s+(?:account|identity|credentials)",
                r"(?:unusual|suspicious)\s+(?:activity|login)\s+detected",
                r"click\s+(?:the\s+)?link\s+(?:below|to\s+continue|to\s+keep\s+your\s+account\s+active)",
                r"(?:microsoft\s*365|office\s*365|google\s+workspace|okta|duo|vpn)\s+(?:verification|alert|upgrade|password|reset)",
                r"(?:docu\s*sign|adobe\s+sign)\s+(?:envelope|document)\s+awaiting\s+(?:your\s+)?signature",
                r"(?:invoice|payment)\s+(?:overdue|attached|pending)",
                r"(?:dropbox|one\s*drive|google\s+drive|gdrive)\s+(?:shared\s+folder|document)\s+(?:link|access)",
                r"(?:wire|bank)\s+transfer\s+(?:confirmation|update|request)",
                r"(?:payroll|hr)\s+urgent\s+(?:update|verification)",
            ],
            "foot_in_the_door": [
                r"(?:as\s+a\s+first\s+step|to\s+start),?\s+could\s+you",
                r"(?:since|now\s+that)\s+you(?:'ve| have)\s+(?:already\s+)?",
                r"it\s+will\s+only\s+take\s+(?:a\s+)?(?:minute|sec(?:ond)?s?)",
            ],
            "door_in_the_face": [
                r"(?:if\s+you\s+can't|cannot|won't)\s+do\s+that,?\s*(?:can|could)\s+you\s+at\s+least",
                r"(?:at\s+least)\s+do\s+this",
                r"okay,?\s+then\s+just\s+",
            ],
            "false_dichotomy": [
                r"(?:either)\s+you\s+.+\s+or\s+you\s+.+",
                r"if\s+you(?:'re| are)\s+not\s+with\s+us,?\s*you(?:'re| are)\s+against\s+us",
                r"there\s+is\s+no\s+middle\s+ground",
            ],
            "fud": [
                r"what\s+if\s+[^.?!]{0,120}",
                r"imagine\s+losing\s+[^.?!]{0,80}",
                r"can\s+you\s+afford\s+to\s+ignore\s+[^.?!]{0,80}",
                r"the\s+risks\s+are\s+too\s+high",
            ],
            "moral_blackmail": [
                r"a\s+good\s+person\s+would",
                r"if\s+you\s+had\s+any\s+(?:decency|respect)",
                r"do\s+the\s+right\s+thing",
                r"you\s+owe\s+it\s+to\s+(?:everyone|us|them|yourself)",
            ],
            "flattery_charm": [
                r"you(?:'re| are)\s+(?:the\s+best|brilliant|exceptional|a\s+genius)",
                r"only\s+you\s+can\s+handle\s+this",
                r"as\s+our\s+top\s+performer",
                r"your\s+work\s+is\s+(?:incredible|amazing|unmatched)",
            ],
            "sunk_cost": [
                r"we(?:'ve| have)\s+already\s+invested\s+(?:so\s+)?much",
                r"don't\s+waste\s+what\s+we(?:'ve| have)\s+done",
                r"it's\s+too\s+late\s+to\s+turn\s+back",
                r"we(?:'re| are)\s+in\s+too\s+deep",
            ],
            "commitment_consistency": [
                r"you\s+said\s+earlier\s+that\s+you(?:'d| would)\s+help",
                r"as\s+you\s+agreed\s+before",
                r"be\s+consistent\s+with\s+your\s+commitment",
                r"you\s+promised",
            ],
        }

        # Precompile patterns for boundary-aware literals and raw regex
        # Map: category -> list of (keyword_or_regex, compiled_pattern)
        self._compiled_patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self._compile_patterns()

        # Base severity per category (may escalate based on occurrences)
        # Tuned severities for new categories:
        # - love_bombing: LOW (less directly coercive, but escalates with repetition)
        # - false_dichotomy: LOW (often rhetorical, escalate on volume)
        # - moral_blackmail: HIGH (direct coercion via morality)
        self._base_severity: Dict[str, Severity] = {
            "emotional_manipulation": Severity.HIGH,
            "authority_manipulation": Severity.MEDIUM,
            "social_proof": Severity.MEDIUM,
            "scarcity": Severity.MEDIUM,
            "reciprocity": Severity.LOW,
            "love_bombing": Severity.LOW,             # tuned
            "gaslighting": Severity.HIGH,
            "threats_intimidation": Severity.HIGH,
            "phishing_pretexting": Severity.HIGH,
            "foot_in_the_door": Severity.LOW,
            "door_in_the_face": Severity.LOW,
            "false_dichotomy": Severity.LOW,          # tuned
            "fud": Severity.MEDIUM,
            "moral_blackmail": Severity.HIGH,         # tuned
            "flattery_charm": Severity.LOW,
            "sunk_cost": Severity.LOW,
            "commitment_consistency": Severity.LOW,
        }

        # Human-friendly labels for descriptions
        self._labels: Dict[str, str] = {
            "emotional_manipulation": "Emotional",
            "authority_manipulation": "Authority",
            "social_proof": "Social proof",
            "scarcity": "Scarcity",
            "reciprocity": "Reciprocity",
            "love_bombing": "Love bombing",
            "gaslighting": "Gaslighting",
            "threats_intimidation": "Threats/Intimidation",
            "phishing_pretexting": "Phishing/Pretexting",
            "foot_in_the_door": "Foot-in-the-door",
            "door_in_the_face": "Door-in-the-face",
            "false_dichotomy": "False dichotomy",
            "fud": "FUD",
            "moral_blackmail": "Moral blackmail",
            "flattery_charm": "Flattery/Charm",
            "sunk_cost": "Sunk cost",
            "commitment_consistency": "Commitment/Consistency",
        }

        # Explicit scan order for stable reporting
        self._scan_order: List[str] = [
            "emotional_manipulation",
            "authority_manipulation",
            "social_proof",
            "scarcity",
            "reciprocity",
            "love_bombing",
            "gaslighting",
            "threats_intimidation",
            "phishing_pretexting",
            "foot_in_the_door",
            "door_in_the_face",
            "false_dichotomy",
            "fud",
            "moral_blackmail",
            "flattery_charm",
            "sunk_cost",
            "commitment_consistency",
        ]

    def _compile_patterns(self) -> None:
        """Compile phrase lists into boundary-aware regex and raw regex patterns."""
        self._compiled_patterns.clear()
        # Compile literals with boundaries and flexible whitespace
        for category, phrases in self.manipulation_patterns.items():
            compiled: List[Tuple[str, re.Pattern]] = self._compiled_patterns.get(category, [])
            for phrase in phrases:
                escaped = re.escape(phrase).replace(r"\ ", r"\s+")
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)
                compiled.append((phrase, pattern))
            self._compiled_patterns[category] = compiled

        # Compile raw regex patterns as-is (caller supplies any boundaries needed)
        for category, patterns in self.manipulation_regex.items():
            compiled: List[Tuple[str, re.Pattern]] = self._compiled_patterns.get(category, [])
            for raw in patterns:
                try:
                    rgx = re.compile(raw, flags=re.IGNORECASE)
                    compiled.append((raw, rgx))
                except re.error:
                    # Skip invalid regex while keeping detector resilient
                    continue
            self._compiled_patterns[category] = compiled

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect manipulation techniques in the given action."""
        if not self.enabled:
            return []

        text_to_check = self._assemble_text(action)
        
        # Skip processing if content is too large for performance
        if len(text_to_check) > 100000:  # 100KB limit for pattern matching
            return []
        if not text_to_check:
            return []

        violations: List[SafetyViolation] = []

        for category in self._scan_order:
            violation = self._scan_category(
                action=action,
                text=text_to_check,
                category=category,
                base_severity=self._base_severity[category],
                description_label=self._labels[category],
            )
            if violation:
                violations.append(violation)

        return violations

    def _assemble_text(self, action: AgentAction) -> str:
        """Assemble and normalize the text to analyze from the action."""
        parts = [
            getattr(action, "stated_intent", "") or "",
            getattr(action, "actual_action", "") or "",
        ]
        text = " ".join(p.strip() for p in parts if p).strip()
        return text

    def _scan_category(
        self,
        action: AgentAction,
        text: str,
        category: str,
        base_severity: Severity,
        description_label: str,
    ) -> Optional[SafetyViolation]:
        """Scan a single category for matches and produce a violation if found."""
        occurrences = []
        for keyword, pattern in self._compiled_patterns.get(category, []):
            for m in pattern.finditer(text):
                occurrences.append(
                    {
                        "keyword": keyword,
                        "match": m.group(0),
                        "start": m.start(),
                        "end": m.end(),
                    }
                )

        if not occurrences:
            return None

        distinct_keywords = sorted({o["keyword"] for o in occurrences})
        total_occurrences = len(occurrences)

        severity = self._escalate_severity(
            base=base_severity,
            total_occurrences=total_occurrences,
            distinct_keywords=len(distinct_keywords),
        )

        description = (
            f"{description_label} manipulation detected "
            f"({len(distinct_keywords)} keyword(s), {total_occurrences} occurrence(s))."
        )

        return SafetyViolation(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_type=ViolationType.MANIPULATION,
            severity=severity,
            description=description,
            evidence={
                "manipulation_type": category,
                "detected_keywords": distinct_keywords,
                "counts": {
                    "distinct_keywords": len(distinct_keywords),
                    "total_occurrences": total_occurrences,
                },
                "occurrences": occurrences,
                "context": text,
            },
        )

    def _escalate_severity(
        self,
        base: Severity,
        total_occurrences: int,
        distinct_keywords: int,
    ) -> Severity:
        """Escalate severity based on the number of matches."""
        ladder = [Severity.LOW, Severity.MEDIUM, Severity.HIGH]
        try:
            idx = ladder.index(base)
        except ValueError:
            idx = 1  # default MEDIUM

        steps = 0
        if total_occurrences >= 3 or distinct_keywords >= 2:
            steps = 1
        if total_occurrences >= 6:
            steps = max(steps, 2)

        new_idx = max(0, min(len(ladder) - 1, idx + steps))
        return ladder[new_idx]
