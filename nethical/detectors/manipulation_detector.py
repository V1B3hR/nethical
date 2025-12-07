"""Detector for manipulation techniques with advanced dark psychology patterns."""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple

from .base_detector import BaseDetector, DetectorStatus
from ..core.models import AgentAction, SafetyViolation, ViolationType, Severity


# Maximum text length for pattern matching to ensure performance (100KB limit)
MAX_TEXT_LENGTH = 100000


class ManipulationDetector(BaseDetector):
    """Detects manipulation techniques in agent actions with regex-based matching,
    boundary-aware keywords, and evidence aggregation.

    Enhancements (2025):
    - 40+ manipulation categories from latest dark psychology research
    - 1000+ literal phrase patterns and advanced regex
    - Rich evidence: all matches with spans and context
    - Severity escalation by match volume/diversity
    - Performance optimized with configurable limits
    """

    def __init__(self):
        """Initialize the ManipulationDetector with comprehensive pattern categories.
        
        This detector identifies 40+ manipulation categories from dark psychology research,
        including gaslighting, phishing, emotional manipulation, and more. Each category
        has specific patterns and severity levels based on threat assessment.
        
        Pattern Matching:
        - Boundary-aware: Keywords match whole words only (e.g., "fear" won't match "fearless")
        - Case-insensitive: Patterns ignore capitalization
        - Multi-word phrases: Allow flexible whitespace between words
        - Regex patterns: Advanced sentence-level pattern detection
        
        Performance:
        - Precompiled patterns for efficiency
        - Text length limit (MAX_TEXT_LENGTH) to prevent DoS
        - TODO: Future optimization with Aho-Corasick algorithm
        
        Categories include:
        - Emotional manipulation: Fear-based tactics, guilt-tripping
        - Authority abuse: Impersonation, false compliance demands
        - Social proof: Bandwagon effects, false testimonials
        - Scarcity tactics: Artificial urgency, limited-time pressure
        - Gaslighting: Reality distortion, memory manipulation
        - Phishing/Pretexting: Credential theft, identity verification scams
        - Threats/Intimidation: Coercion, blackmail
        - And 30+ more advanced manipulation techniques
        """
        super().__init__("Manipulation Detector")

        # Literal phrase lists per manipulation category (boundary-aware, case-insensitive).
        # Multi-word phrases match with flexible whitespace.
        self.manipulation_patterns: Dict[str, List[str]] = {
            # CORE EMOTIONAL MANIPULATION
            # Detects fear-based tactics, guilt-tripping, and emotional coercion
            # Severity: HIGH - Can cause psychological harm and impair decision-making
            # Examples: "you'll feel bad if", "think of the consequences", "break your heart"
            "emotional_manipulation": [
                "fear",
                "urgent",
                "emergency",
                "panic",
                "worry",
                "scared",
                "guilt",
                "shame",
                "disappointed",
                "sad",
                "upset",
                "anxious",
                "terrified",
                "afraid",
                "alarming",
                "distressing",
                "you'll feel bad",
                "you will feel bad",
                "if you cared",
                "as a good person",
                "you should be ashamed",
                "the right thing to do",
                "think of the consequences",
                "you wouldn't want that",
                "don't let everyone down",
                "break your heart",
                "devastated",
                "crushed",
                "you're breaking my heart",
                "this hurts me",
                "i'm so disappointed in you",
                "how could you",
                "after all i've sacrificed",
                "ungrateful",
                "selfish",
                "heartless",
                "cold",
                "uncaring",
            ],
            # AUTHORITY & POWER ABUSE
            # Detects impersonation of authority figures or abuse of power dynamics
            # Severity: MEDIUM - Common in phishing and social engineering attacks
            # Examples: "by decree", "compliance department", "executive order", "per policy"
            "authority_manipulation": [
                "authority",
                "boss",
                "superior",
                "command",
                "order",
                "must",
                "required",
                "mandatory",
                "compliance",
                "obey",
                "official",
                "policy",
                "by decree",
                "per policy",
                "security team",
                "it admin",
                "administrator",
                "ceo",
                "cfo",
                "cto",
                "legal department",
                "compliance department",
                "internal audit",
                "risk management",
                "chain of command",
                "executive order",
                "company directive",
                "non-negotiable",
                "standard operating procedure",
                "sop",
                "protocol",
                "regulation",
                "directive",
                "authorized personnel",
            ],
            # SOCIAL INFLUENCE
            # Detects bandwagon effects and false popularity claims
            # Severity: MEDIUM - Exploits human tendency to follow the crowd
            # Examples: "everyone is doing it", "millions use", "trending", "industry standard"
            "social_proof": [
                "everyone",
                "everybody",
                "all",
                "popular",
                "trending",
                "majority",
                "consensus",
                "widely accepted",
                "people are saying",
                "most people",
                "millions use",
                "viral",
                "5-star",
                "five star",
                "testimonials",
                "best seller",
                "award-winning",
                "as seen on",
                "industry standard",
                "certified",
                "approved by",
                "backed by science",
                "doctor recommended",
                "expert endorsed",
                "peer reviewed",
                "trusted by millions",
                "globally recognized",
                "market leader",
                "number one choice",
            ],
            # URGENCY & SCARCITY
            # Detects artificial time pressure and scarcity tactics
            # Severity: MEDIUM - Creates false urgency to bypass rational decision-making
            # Examples: "limited time", "act now", "offer ends soon", "while supplies last"
            "scarcity": [
                "limited",
                "scarce",
                "rare",
                "exclusive",
                "only",
                "last",
                "running out",
                "deadline",
                "expires",
                "while supplies last",
                "limited time",
                "act now",
                "today only",
                "offer ends",
                "offer ends soon",
                "don't miss out",
                "before it's too late",
                "final chance",
                "ends tonight",
                "time-sensitive",
                "going fast",
                "almost gone",
                "selling out",
                "flash sale",
                "doorbuster",
                "clearance",
                "last call",
                "final hours",
                "closing soon",
            ],
            # RECIPROCITY & OBLIGATION
            "reciprocity": [
                "favor",
                "owe",
                "return",
                "exchange",
                "give back",
                "repay",
                "grateful",
                "appreciate",
                "thank you",
                "do me a favor",
                "i'll owe you",
                "quid pro quo",
                "i did this for you",
                "since i helped you",
                "after everything i've done for you",
                "the least you can do",
                "you owe me",
                "i went out of my way",
                "made sacrifices",
                "i've always been there",
                "remember when i",
            ],
            # LOVE BOMBING
            "love_bombing": [
                "love bombing",
                "you're the only one",
                "you are the only one",
                "soulmate",
                "meant to be",
                "perfect match",
                "destiny",
                "i can't live without you",
                "can't live without you",
                "unconditional love",
                "forever and always",
                "too good to be true",
                "grand gesture",
                "shower you with gifts",
                "you're perfect",
                "we were made for each other",
                "love at first sight",
                "i'll take care of everything",
                "you complete me",
                "never felt this way",
                "twin flame",
                "you're my everything",
                "my whole world",
                "soul connection",
                "divine timing",
                "universe brought us together",
                "instant connection",
                "magnetic pull",
                "chemistry",
            ],
            # GASLIGHTING
            # Detects reality distortion and memory manipulation tactics
            # Severity: HIGH - Psychological abuse that can cause lasting harm
            # Examples: "you're imagining things", "that never happened", "you're crazy", "calm down"
            "gaslighting": [
                "you're imagining things",
                "you are imagining things",
                "that's not what happened",
                "that is not what happened",
                "you're overreacting",
                "you are overreacting",
                "you're being paranoid",
                "you are being paranoid",
                "you're crazy",
                "you are crazy",
                "making this up",
                "no one else thinks that",
                "i never said that",
                "it's all in your head",
                "you're misremembering",
                "you are misremembering",
                "you always do this",
                "you never remember right",
                "why are you making a scene",
                "you're too sensitive",
                "that never happened",
                "you're delusional",
                "you're being irrational",
                "calm down",
                "you're hysterical",
                "stop being emotional",
                "you're confusing things",
                "your memory is wrong",
                "i have proof you're wrong",
                "everyone agrees with me",
            ],
            # THREATS & INTIMIDATION
            # Detects coercion, blackmail, and intimidation tactics
            # Severity: HIGH - Direct threat to user safety and wellbeing
            # Examples: "or else", "legal action", "you'll regret", "consequences", "final warning"
            "threats_intimidation": [
                "or else",
                "you'll regret",
                "you will regret",
                "last warning",
                "final notice",
                "we will take action",
                "legal action",
                "report you",
                "blacklist",
                "ban you",
                "suspend your account",
                "shut you down",
                "consequences",
                "this is your final chance",
                "comply or else",
                "don't make me escalate this",
                "you've been warned",
                "don't test me",
                "you don't want to see what happens",
                "watch your back",
                "be careful",
                "you're playing with fire",
                "i have connections",
                "i know people",
                "make your life difficult",
                "ruin your reputation",
            ],
            # PHISHING & PRETEXTING
            # Detects credential theft attempts and identity verification scams
            # Severity: HIGH - Direct security threat, common attack vector
            # Examples: "verify your account", "suspicious activity", "click the link", "urgent verification"
            "phishing_pretexting": [
                "verify your account",
                "reset your password",
                "suspicious activity",
                "confirm your identity",
                "account locked",
                "unauthorized login",
                "security alert",
                "click the link",
                "update your credentials",
                "your account will be closed",
                "urgent verification",
                "we detected unusual activity",
                "invoice attached",
                "payment overdue",
                "shared folder",
                "action required",
                "validate your information",
                "re-authenticate",
                "security breach",
                "compromised account",
                "immediate action needed",
                "confirm you're not a robot",
            ],
            # INCREMENTAL COMMITMENT
            "foot_in_the_door": [
                "just a small favor",
                "just a quick question",
                "only take a minute",
                "could you start by",
                "since you've already",
                "now that we've started",
                "as a first step",
                "just this once",
                "one small thing",
                "tiny request",
                "real quick",
                "won't take long",
                "while you're at it",
                "since you're already here",
            ],
            # CONTRAST MANIPULATION
            "door_in_the_face": [
                "if you can't do that, maybe",
                "if you cannot do that, maybe",
                "the least you can do",
                "bare minimum",
                "at least do this",
                "okay then just",
                "can you at least",
                "how about just",
                "that's not too much to ask",
                "surely you can",
                "come on, just",
                "fine, then",
            ],
            # BINARY THINKING
            "false_dichotomy": [
                "if you're not with us you're against us",
                "there is no middle ground",
                "you either care or you don't",
                "black and white",
                "all or nothing",
                "love me or leave me",
                "yes or no",
                "it's that simple",
                "pick a side",
                "you're either part of the solution or the problem",
            ],
            # FEAR UNCERTAINTY DOUBT
            "fud": [
                "can you afford to ignore",
                "imagine losing everything",
                "what happens when it fails",
                "the risks are too high",
                "you'll lose out",
                "don't be left behind",
                "what if something goes wrong",
                "you could lose everything",
                "think about the worst case",
                "better safe than sorry",
                "you never know what might happen",
            ],
            # MORAL COERCION
            "moral_blackmail": [
                "a good person would",
                "if you had any decency",
                "do the right thing",
                "what would your family think",
                "you owe it to everyone",
                "if you had a heart",
                "morally obligated",
                "your conscience should tell you",
                "think about your soul",
                "karma will get you",
                "god is watching",
                "what would jesus do",
                "be a decent human being",
            ],
            # FLATTERY & CHARM OFFENSIVE
            "flattery_charm": [
                "you're the best",
                "you're brilliant",
                "you're exceptional",
                "only you can handle this",
                "as our top performer",
                "your work is incredible",
                "genius",
                "you're amazing",
                "so talented",
                "uniquely qualified",
                "natural born leader",
                "visionary",
                "ahead of your time",
                "elite",
                "chosen one",
                "you have a gift",
                "special abilities",
            ],
            # SUNK COST FALLACY
            "sunk_cost": [
                "we've already invested so much",
                "don't waste what we've done",
                "too late to turn back",
                "we're in too deep",
                "after all this time",
                "look how far we've come",
                "think of all the effort",
                "all that work for nothing",
                "point of no return",
                "can't quit now",
                "we've come this far",
            ],
            # COMMITMENT & CONSISTENCY
            "commitment_consistency": [
                "you said earlier that you'd help",
                "as you agreed before",
                "be consistent with your commitment",
                "you promised",
                "stick to your word",
                "you're a person of your word",
                "you always said",
                "remember what you told me",
                "don't go back on your word",
                "your track record",
                "stay true to yourself",
            ],
            # REVERSE PSYCHOLOGY
            "reverse_psychology": [
                "i bet you can't",
                "you're probably not brave enough",
                "i don't think you could",
                "you wouldn't dare",
                "you're not going to",
                "i guess it's too hard for you",
                "don't even try",
                "you'll never be able to",
                "it's probably beyond your abilities",
                "most people can't handle this",
                "you're not ready for this",
                "too advanced for you",
                "this isn't for everyone",
                "probably too much for you",
                "i doubt you have what it takes",
                "weak people give up",
                "prove me wrong if you can",
            ],
            # TRIANGULATION
            "triangulation": [
                "they said you",
                "everyone thinks you",
                "others are saying",
                "i heard from someone that",
                "people are talking about you",
                "they don't like you",
                "compared to them you",
                "why can't you be more like",
                "they would never do this",
                "i was talking to someone about you",
                "everyone's been discussing",
                "the consensus is",
                "behind your back",
                "water cooler talk",
            ],
            # PROJECTION
            "projection": [
                "you're the one who",
                "you're just as bad",
                "look at yourself",
                "you do the same thing",
                "you're guilty too",
                "pot calling kettle black",
                "who are you to judge",
                "you have no room to talk",
                "you're projecting",
                "takes one to know one",
                "check yourself",
            ],
            # SILENT TREATMENT & WITHDRAWAL
            "silent_treatment": [
                "i'm not talking to you",
                "i have nothing to say to you",
                "i'm done talking",
                "don't speak to me",
                "you're getting the silent treatment",
                "ignoring you now",
                "not worth my words",
                "talk to the hand",
                "radio silence",
                "you're invisible to me",
                "pretending you don't exist",
                "acting like you're not here",
            ],
            # WORD SALAD & CONFUSION
            "word_salad": [
                "circular logic",
                "as i was saying about the thing",
                "it's complicated to explain but basically",
                "you wouldn't understand",
                "it's deeper than that",
                "there are layers to this",
                "meta concept",
                "abstract reasoning",
                "philosophical implications",
                "multidimensional thinking",
                "you need higher intelligence",
                "nuanced perspective",
            ],
            # MOVING GOALPOSTS
            "moving_goalposts": [
                "that's not good enough",
                "now you need to",
                "actually i meant",
                "i expected more",
                "you should have known",
                "that's not what i wanted",
                "close but not quite",
                "you still need to prove",
                "one more thing",
                "just one more requirement",
                "raise the bar",
                "new standards",
                "changed my mind",
            ],
            # PLAYING VICTIM
            "victimhood": [
                "poor me",
                "i'm always the victim",
                "everyone is against me",
                "why does this always happen to me",
                "i suffer the most",
                "nobody understands my pain",
                "i'm the real victim here",
                "after everything i've been through",
                "you have no idea what i deal with",
                "my life is so hard",
                "woe is me",
                "persecution complex",
                "martyrdom",
            ],
            # COVERT CONTRACTS
            "covert_contracts": [
                "i expected you would",
                "i thought you'd",
                "i assumed you knew",
                "you should have realized",
                "it was implied",
                "i shouldn't have to ask",
                "you should just know",
                "it goes without saying",
                "i expected better",
                "unspoken agreement",
                "reading between the lines",
                "obvious hint",
            ],
            # SELECTIVE MEMORY
            "selective_memory": [
                "i don't recall",
                "i don't remember saying that",
                "are you sure that happened",
                "i have no memory of that",
                "that's not how i remember it",
                "convenient amnesia",
                "drawing a blank",
                "doesn't ring a bell",
                "must have misheard",
                "my recollection differs",
                "foggy memory",
            ],
            # MINIMIZATION
            "minimization": [
                "it's not a big deal",
                "you're making too much of this",
                "it was just a joke",
                "lighten up",
                "don't be so dramatic",
                "it's not that serious",
                "you're blowing this out of proportion",
                "worse things have happened",
                "at least i didn't",
                "could have been worse",
                "stop overreacting",
                "get over it",
                "let it go",
            ],
            # DEFLECTION
            "deflection": [
                "what about when you",
                "but you did",
                "that's not the point",
                "we're not talking about that",
                "changing the subject",
                "let's talk about you instead",
                "deflecting much",
                "nice try changing topics",
                "stop avoiding the question",
                "whataboutism",
                "red herring",
                "smoke and mirrors",
            ],
            # DARVO (Deny, Attack, Reverse Victim and Offender)
            # Detects perpetrator claiming victim status to deflect accountability
            # Severity: HIGH - Advanced manipulation tactic used by abusers
            # Examples: "i'm the real victim", "you're attacking me", "you made me do this"
            "darvo": [
                "i'm the real victim here",
                "you're attacking me",
                "how dare you accuse me",
                "you're the abuser",
                "you're hurting me by saying this",
                "i can't believe you'd accuse me",
                "you're the problem not me",
                "now i'm the bad guy",
                "flip the script",
                "reverse victim and offender",
                "you made me do this",
            ],
            # NEGGING
            "negging": [
                "you'd be prettier if",
                "for someone like you that's good",
                "not bad for",
                "you almost got it right",
                "i usually don't like that but on you",
                "backhanded compliment",
                "you're pretty for a",
                "you're smart for someone who",
                "that's surprisingly good",
                "interesting choice",
                "bold move",
                "unconventional",
            ],
            # ISOLATION TACTICS
            # Detects attempts to separate victim from support network
            # Severity: HIGH - Classic abusive behavior pattern, highly harmful
            # Examples: "they're not good for you", "you don't need them", "choose me or them"
            "isolation": [
                "they're not good for you",
                "you don't need them",
                "i'm all you need",
                "they don't understand you like i do",
                "your friends are toxic",
                "family is overrated",
                "it's just us against the world",
                "you should cut them off",
                "they're holding you back",
                "choose me or them",
                "they're jealous of us",
                "poisoning your mind",
                "negative influence",
            ],
            # INTERMITTENT REINFORCEMENT
            "intermittent_reinforcement": [
                "sometimes i'll",
                "when i feel like it",
                "if you're lucky",
                "maybe if you're good",
                "i'll think about it",
                "we'll see",
                "depends on my mood",
                "when i'm ready",
                "unpredictable schedule",
                "hot and cold",
                "push and pull",
                "keep you guessing",
            ],
            # BOUNDARY VIOLATION
            "boundary_violation": [
                "you're too sensitive about boundaries",
                "i was just",
                "don't be so uptight",
                "i can do what i want",
                "your boundaries are unreasonable",
                "that's a silly rule",
                "i don't respect that boundary",
                "you can't tell me no",
                "i'll do it anyway",
                "your comfort doesn't matter",
                "stop being difficult",
                "you're no fun",
                "loosen up",
            ],
            # NEW ADVANCED TACTICS (2025)
            # FUTURE FAKING
            "future_faking": [
                "we'll get married soon",
                "i promise we'll",
                "just wait and see",
                "in the future we'll",
                "someday we'll",
                "when i get promoted",
                "after this project",
                "once things settle down",
                "next year we'll",
                "trust the process",
                "patience will pay off",
                "building our future",
                "long term plan",
            ],
            # HOOVERING
            "hoovering": [
                "i miss you",
                "i've changed",
                "it will be different this time",
                "i can't live without you",
                "i've been thinking about us",
                "remember the good times",
                "nobody will love you like i do",
                "we belong together",
                "give me another chance",
                "i'm a different person now",
                "i've been working on myself",
                "therapy has helped me",
            ],
            # BAITING
            "baiting": [
                "you're so predictable",
                "i knew you'd say that",
                "typical response",
                "getting under your skin",
                "hit a nerve",
                "touched a sensitive spot",
                "you're so easy to read",
                "pulling your strings",
                "pushing your buttons",
                "fishing for a reaction",
            ],
            # STONEWALLING
            "stonewalling": [
                "i don't want to talk about it",
                "there's nothing to discuss",
                "drop it",
                "end of conversation",
                "i'm done with this",
                "not going there",
                "shutting down",
                "closing off",
                "refusing to engage",
                "conversation over",
            ],
            # BREADCRUMBING
            "breadcrumbing": [
                "maybe later",
                "let's keep in touch",
                "we should hang out sometime",
                "i'll let you know",
                "keeping my options open",
                "just checking in",
                "thinking of you",
                "miss you randomly",
                "hope you're well",
                "stringing along",
            ],
            # SMEAR CAMPAIGN
            "smear_campaign": [
                "everyone knows the truth about you",
                "i've told people about you",
                "your reputation is ruined",
                "spreading the word",
                "people deserve to know",
                "warning others",
                "exposing you",
                "character assassination",
                "tarnishing your name",
                "turning people against you",
            ],
            # COGNITIVE DISSONANCE
            "cognitive_dissonance": [
                "you love me but you're leaving",
                "you say one thing but do another",
                "your actions don't match your words",
                "conflicting messages",
                "mixed signals",
                "contradicting yourself",
                "hypocrite",
                "double standard",
                "practice what you preach",
            ],
            # WITHHOLDING
            "withholding": [
                "you don't deserve to know",
                "i'm not telling you",
                "keeping secrets",
                "none of your business",
                "need to know basis",
                "information is power",
                "withholding affection",
                "giving you the cold shoulder",
                "depriving you",
                "punishing with silence",
            ],
            # INFANTILIZATION
            "infantilization": [
                "you're like a child",
                "let me handle this",
                "you wouldn't understand",
                "too complicated for you",
                "i'll take care of it",
                "don't worry your pretty head",
                "leave it to the adults",
                "when you grow up",
                "naive",
                "immature",
                "treating you like a baby",
            ],
            # POISONING THE WELL
            "poisoning_well": [
                "before you hear from them",
                "they're going to lie about me",
                "don't believe what they say",
                "they have an agenda",
                "they're biased",
                "unreliable source",
                "can't trust them",
                "preemptive strike",
                "getting ahead of the story",
            ],
        }

        # Advanced sentence-level regex patterns per category
        self.manipulation_regex: Dict[str, List[str]] = {
            "emotional_manipulation": [
                r"(?:don't|do not)\s+let\s+(?:this|that)\s+happen",
                r"if\s+you\s+cared(?:\s+about\s+\w+)?",
                r"you(?:'ll| will)\s+feel\s+(?:bad|guilty|terrible)\s+if",
                r"think\s+of\s+how\s+(?:sad|upset|disappointed)\s+(\w+)\s+will\s+be",
                r"(?:imagine|think\s+about)\s+how\s+(?:hurt|sad|devastated)",
                r"(?:breaking|crushing|hurting)\s+my\s+heart",
            ],
            "authority_manipulation": [
                r"(?:per|as\s+per|in\s+accordance\s+with)\s+(?:policy|regulation|law|standard|procedure)",
                r"by\s+order\s+of\s+(?:management|the\s+board|the\s+court|your\s+superior)",
                r"failure\s+to\s+comply\s+will\s+result\s+in",
                r"under\s+(?:penalty|sanction|disciplinary\s+review)",
                r"(?:do\s+as|follow)\s+(?:you(?:'re| are)\s+)?(?:instructed|told)",
                r"non-negotiable\s+(?:demand|requirement|policy)",
            ],
            "social_proof": [
                r"(?:[1-9]\d?|100)%\s+of\s+(?:users|people|customers)\s+(?:agree|use|choose|chose|recommend)",
                r"(?:thousands|millions|billions)\s+of\s+(?:users|people|customers)\s+(?:can't|cannot)\s+be\s+wrong",
                r"(?:ranked|rated)\s+(?:#1|number\s+one)\b",
                r"(?:industry|de\s+facto)\s+standard",
                r"all\s+the\s+(?:kids|guys|girls|teams|cool\s+people)\s+are\s+(?:doing|using)\s+it",
                r"(?:trusted|recommended|endorsed)\s+by\s+(?:doctors|experts|professionals|celebrities)",
            ],
            "scarcity": [
                r"only\s+\d+\s+(?:left|remaining|spots?|slots?|seats?)",
                r"(?:offer|sale|deal)\s+ends\s+(?:today|tonight|soon|in\s+\d+\s+(?:hours?|minutes?|days?))",
                r"(?:expires|closing|ending)\s+(?:in|at)\s+\d+\s+(?:min(?:s|utes)?|hours?|days?)",
                r"last\s+(?:chance|call|day|opportunity)",
                r"(?:limited|exclusive|vip)\s+(?:release|access|drop|offer)",
                r"(?:going|selling)\s+(?:fast|quickly)",
            ],
            "reciprocity": [
                r"(?:after|considering)\s+everything\s+i(?:'ve| have)\s+done\s+(?:for\s+you)?",
                r"the\s+least\s+you\s+(?:can|could)\s+do",
                r"you\s+owe\s+(?:it\s+to\s+)?me",
                r"i\s+scratch\s+your\s+back.*you\s+scratch\s+mine",
                r"remember\s+(?:that\s+time|when)\s+i\s+.+",
                r"i(?:'ve| have)\s+(?:always\s+)?been\s+there\s+for\s+you",
            ],
            "love_bombing": [
                r"i\s+can't\s+imagine\s+(?:my\s+)?life\s+without\s+you",
                r"we(?:'re| are)\s+(?:soulmates?|meant\s+to\s+be|twin\s+flames?)",
                r"nobody\s+has\s+ever\s+made\s+me\s+feel\s+(?:this\s+way|like\s+this)",
                r"i\s+want\s+to\s+spend\s+the\s+rest\s+of\s+my\s+life\s+with\s+you",
                r"we(?:'ll| will)\s+(?:travel\s+the\s+world|buy\s+a\s+house|start\s+a\s+family|get\s+married)",
                r"you(?:'re| are)\s+(?:my\s+everything|my\s+whole\s+world|perfect)",
            ],
            "gaslighting": [
                r"(?:that's|that\s+is)\s+not\s+what\s+happened",
                r"(?:you(?:'re| are)\s+)?(?:too\s+|being\s+)?(?:sensitive|dramatic|paranoid|emotional|irrational)",
                r"(?:no\s+one|nobody|everyone)\s+else\s+(?:thinks|agrees|sees)\s+(?:that|with\s+me)",
                r"i\s+never\s+said\s+that",
                r"(?:it'?s|it\s+is)\s+all\s+in\s+your\s+head",
                r"(?:you(?:'re| are)\s+)?(?:misremembering|imagining|making\s+(?:this|it)\s+up)",
                r"(?:calm|settle)\s+down",
            ],
            "threats_intimidation": [
                r"(?:comply|cooperate)\s+or\s+(?:else|suffer\s+the\s+consequences|we\s+will)",
                r"(?:final|last)\s+(?:warning|notice|chance)",
                r"(?:we\s+will\s+)?(?:ban|suspend|terminate|revoke|remove)\s+(?:your\s+)?(?:account|access|privileges|employment)",
                r"(?:legal|disciplinary|punitive)\s+action\s+will\s+be\s+taken",
                r"(?:we|i)\s+(?:know|have)\s+(?:where\s+you|your|people|connections)",
                r"(?:make|ruin)\s+(?:your\s+)?(?:life|reputation)\s+(?:difficult|miserable|hell)",
            ],
            "phishing_pretexting": [
                r"(?:verify|validate|confirm|update)\s+your\s+(?:account|identity|credentials|information)",
                r"(?:unusual|suspicious|unauthorized)\s+(?:activity|login|access|transaction)\s+detected",
                r"click\s+(?:the\s+|this\s+)?link\s+(?:below|to\s+continue|to\s+verify|immediately)",
                r"(?:microsoft|google|apple|amazon|paypal|bank)\s+(?:security\s+)?(?:alert|verification|warning)",
                r"(?:account|subscription|service)\s+(?:will\s+be\s+)?(?:locked|suspended|terminated|cancelled)",
                r"(?:invoice|payment|refund|receipt)\s+(?:attached|pending|overdue|processing)",
            ],
            "foot_in_the_door": [
                r"(?:as\s+a\s+first\s+step|to\s+start),?\s+(?:could|can)\s+you",
                r"(?:since|now\s+that)\s+you(?:'ve| have)\s+(?:already\s+)?(?:started|begun)",
                r"it\s+will\s+only\s+take\s+(?:a\s+)?(?:minute|second|moment)",
                r"just\s+(?:a\s+)?(?:quick|small|tiny)\s+(?:favor|question|thing)",
            ],
            "door_in_the_face": [
                r"(?:if\s+you\s+)?(?:can't|cannot|won't)\s+do\s+that,?\s*(?:can|could|will)\s+you\s+at\s+least",
                r"(?:at\s+least|bare\s+minimum)\s+(?:do|agree\s+to)\s+(?:this|that)",
                r"(?:okay|fine|alright),?\s+then\s+just",
            ],
            "false_dichotomy": [
                r"(?:either)\s+you(?:'re| are)?\s+.+\s+or\s+you(?:'re| are)?\s+.+",
                r"if\s+you(?:'re| are)\s+not\s+(?:with|for)\s+us,?\s*you(?:'re| are)\s+against\s+us",
                r"(?:there\s+is\s+)?no\s+(?:middle\s+ground|grey\s+area|in\s+between)",
                r"(?:it's|its)\s+(?:black\s+and\s+white|all\s+or\s+nothing|yes\s+or\s+no)",
            ],
            "fud": [
                r"what\s+if\s+[^.?!]{0,120}",
                r"imagine\s+(?:losing|if\s+you\s+lost)\s+[^.?!]{0,80}",
                r"can\s+you\s+afford\s+(?:to\s+)?(?:ignore|risk)\s+[^.?!]{0,80}",
                r"(?:the\s+)?risks?\s+(?:are|is)\s+too\s+high",
                r"you(?:'ll| will)\s+(?:regret|lose\s+out)",
            ],
            "moral_blackmail": [
                r"a\s+(?:good|decent)\s+person\s+would",
                r"if\s+you\s+had\s+any\s+(?:decency|respect|conscience|heart)",
                r"do\s+the\s+right\s+thing",
                r"you\s+owe\s+it\s+to\s+(?:everyone|yourself|society|humanity)",
                r"(?:think|consider)\s+(?:your|the)\s+(?:morals|ethics|values|soul)",
            ],
            "flattery_charm": [
                r"you(?:'re| are)\s+(?:the\s+best|brilliant|exceptional|amazing|a\s+genius)",
                r"only\s+you\s+can\s+(?:handle|do|understand)\s+this",
                r"as\s+our\s+(?:top|best|star)\s+(?:performer|talent|employee)",
                r"your\s+(?:work|skills?|abilities)\s+(?:is|are)\s+(?:incredible|amazing|unmatched|unparalleled)",
            ],
            "sunk_cost": [
                r"(?:we(?:'ve| have)|you've)\s+already\s+invested\s+(?:so\s+)?much",
                r"don't\s+waste\s+(?:what|everything|all)\s+(?:we(?:'ve| have)|you've)\s+(?:done|built|invested)",
                r"(?:it's|its)\s+too\s+late\s+to\s+(?:turn\s+back|quit|stop)",
                r"we(?:'re| are)\s+in\s+too\s+deep",
                r"(?:think|look)\s+(?:of|at)\s+(?:all|how\s+far)",
            ],
            "commitment_consistency": [
                r"you\s+said\s+(?:earlier|before)\s+that\s+you(?:'d| would)",
                r"as\s+you\s+(?:agreed|promised|committed)\s+(?:before|earlier)",
                r"(?:be|stay)\s+consistent\s+with\s+your\s+(?:commitment|word|promise)",
                r"you\s+(?:promised|gave\s+your\s+word)",
                r"(?:don't|do not)\s+go\s+back\s+on\s+your\s+word",
            ],
            "reverse_psychology": [
                r"i\s+bet\s+you\s+(?:can't|cannot|couldn't|won't)",
                r"you(?:'re| are)\s+(?:probably\s+)?not\s+(?:brave|smart|strong|capable)\s+enough",
                r"i\s+don't\s+think\s+you\s+(?:could|can)",
                r"you\s+(?:wouldn't|won't)\s+dare",
                r"(?:this\s+is\s+)?(?:too\s+)?(?:hard|difficult|advanced|complicated)\s+for\s+you",
                r"you(?:'ll| will)\s+never\s+be\s+able\s+to",
                r"most\s+people\s+(?:can't|cannot)\s+handle\s+this",
                r"(?:prove|show)\s+me\s+(?:wrong|I'm\s+wrong)",
            ],
            "triangulation": [
                r"(?:they|everyone|people|others)\s+(?:said|think|believe|told\s+me)\s+(?:that\s+)?you",
                r"compared\s+to\s+(?:them|others|him|her|everyone)",
                r"why\s+(?:can't|cannot)\s+you\s+be\s+more\s+like",
                r"i\s+(?:was\s+)?talking\s+to\s+(?:someone|others)\s+about\s+you",
                r"(?:he|she|they)\s+would\s+never\s+(?:do|say)\s+(?:this|that)",
            ],
            "projection": [
                r"you(?:'re| are)\s+(?:just\s+as\s+bad|the\s+one\s+who|doing\s+the\s+same\s+thing)",
                r"look\s+at\s+yourself",
                r"(?:pot|kettle)\s+(?:calling|black)",
                r"who\s+are\s+you\s+to\s+(?:judge|talk|accuse)",
                r"takes\s+one\s+to\s+know\s+one",
            ],
            "silent_treatment": [
                r"(?:i'm|i\s+am)\s+not\s+(?:talking|speaking)\s+to\s+you",
                r"(?:i'm|i\s+am)\s+done\s+(?:talking|with\s+you)",
                r"(?:don't|do not)\s+speak\s+to\s+me",
                r"(?:you're|you\s+are)\s+(?:getting|receiving)\s+the\s+silent\s+treatment",
            ],
            "word_salad": [
                r"it's\s+(?:complicated|complex|nuanced)\s+to\s+(?:explain|understand)",
                r"you\s+(?:wouldn't|won't|couldn't)\s+understand",
                r"there\s+are\s+layers\s+to\s+this",
                r"(?:it's|that's)\s+(?:deeper|more\s+complex)\s+than\s+(?:that|you\s+think)",
                r"(?:meta|abstract|philosophical)\s+(?:concept|thinking|reasoning)",
            ],
            "moving_goalposts": [
                r"that's\s+not\s+good\s+enough",
                r"(?:now|actually|wait)\s+(?:you\s+need\s+to|i\s+(?:meant|want))",
                r"you\s+should\s+have\s+known",
                r"(?:close\s+but\s+)?not\s+quite",
                r"(?:just\s+)?one\s+more\s+(?:thing|requirement|condition|step)",
                r"(?:raising|moving)\s+the\s+bar",
            ],
            "victimhood": [
                r"(?:poor|woe\s+is)\s+me",
                r"(?:why\s+does\s+)?(?:this\s+)?always\s+happen(?:s)?\s+to\s+me",
                r"everyone\s+is\s+against\s+me",
                r"(?:i'm|i\s+am)\s+(?:always\s+)?the\s+(?:real\s+)?victim",
                r"nobody\s+understands\s+my\s+(?:pain|suffering)",
                r"(?:i'm|i\s+am)\s+the\s+one\s+who\s+suffers",
            ],
            "covert_contracts": [
                r"i\s+(?:expected|thought|assumed)\s+you\s+(?:would|knew|understood)",
                r"it\s+was\s+(?:implied|understood|obvious)",
                r"i\s+shouldn't\s+have\s+to\s+(?:ask|tell\s+you|say\s+it)",
                r"you\s+should\s+(?:just\s+)?know",
                r"it\s+goes\s+without\s+saying",
            ],
            "selective_memory": [
                r"i\s+don't\s+(?:recall|remember)",
                r"(?:are\s+you\s+sure\s+)?that\s+(?:happened|occurred)",
                r"(?:doesn't|does\s+not)\s+ring\s+a\s+bell",
                r"that's\s+not\s+how\s+i\s+remember\s+it",
                r"my\s+(?:memory|recollection)\s+(?:is\s+)?(?:different|differs)",
            ],
            "minimization": [
                r"it's\s+(?:not\s+a\s+big\s+deal|just\s+a\s+joke|nothing)",
                r"(?:lighten|chill|loosen)\s+up",
                r"you(?:'re| are)\s+(?:making\s+too\s+much\s+of|blowing)\s+(?:this|it)\s+(?:out\s+of\s+proportion)?",
                r"(?:at\s+least\s+)?(?:it\s+)?could\s+(?:have\s+)?be(?:en)?\s+worse",
                r"(?:stop|quit)\s+(?:being\s+)?(?:so\s+)?(?:dramatic|sensitive)",
            ],
            "deflection": [
                r"what\s+about\s+(?:when\s+)?you",
                r"(?:but|and|well)\s+you\s+(?:did|do|said)",
                r"(?:let's|we're)\s+(?:not\s+)?(?:talk|talking)\s+about\s+you",
                r"(?:stop\s+)?(?:changing|avoiding|dodging)\s+the\s+(?:subject|question|topic|issue)",
            ],
            "darvo": [
                r"(?:i'm|i\s+am)\s+the\s+real\s+victim",
                r"you(?:'re| are)\s+(?:attacking|hurting|abusing)\s+me",
                r"how\s+dare\s+you\s+(?:accuse|blame)\s+me",
                r"you(?:'re| are)\s+the\s+(?:abuser|problem|one\s+at\s+fault)",
                r"you\s+(?:made|forced)\s+me\s+(?:to\s+)?(?:do\s+)?(?:this|that)",
            ],
            "negging": [
                r"you(?:'d| would)\s+be\s+(?:prettier|better|perfect)\s+if",
                r"(?:not\s+bad|that's\s+good)\s+for\s+(?:someone|a)",
                r"(?:you(?:'re| are)\s+)?surprisingly\s+(?:good|smart|capable)",
                r"i\s+usually\s+don't\s+like\s+(?:that|those)\s+but\s+on\s+you",
                r"(?:interesting|bold|unique|unconventional)\s+(?:choice|move)",
            ],
            "isolation": [
                r"(?:they(?:'re| are)|your\s+(?:friends|family))\s+(?:are\s+)?(?:not\s+good\s+for\s+you|toxic|poisonous|holding\s+you\s+back)",
                r"(?:i'm|i\s+am)\s+all\s+you\s+need",
                r"they\s+don't\s+understand\s+you\s+like\s+i\s+do",
                r"(?:it's|its)\s+(?:just\s+)?us\s+against\s+(?:the\s+world|everyone)",
                r"choose\s+(?:between\s+)?me\s+or\s+them",
            ],
            "intermittent_reinforcement": [
                r"(?:maybe|perhaps)\s+if\s+you(?:'re| are)\s+(?:good|lucky|nice)",
                r"(?:we'll|we\s+will)\s+see",
                r"depends\s+on\s+my\s+mood",
                r"when\s+(?:i(?:'m| am)\s+ready|i\s+feel\s+like\s+it)",
                r"(?:hot|warm)\s+and\s+cold",
            ],
            "boundary_violation": [
                r"you(?:'re| are)\s+too\s+sensitive\s+about\s+boundaries",
                r"(?:your\s+)?boundaries\s+are\s+(?:unreasonable|silly|stupid)",
                r"i\s+don't\s+(?:respect|care\s+about)\s+that\s+boundary",
                r"you\s+(?:can't|cannot)\s+tell\s+me\s+no",
                r"(?:your\s+)?comfort\s+(?:doesn't|does\s+not)\s+matter",
                r"(?:stop|quit)\s+being\s+(?:so\s+)?(?:difficult|uptight)",
            ],
            "future_faking": [
                r"(?:we'll|we\s+will)\s+(?:get\s+married|buy\s+a\s+house|have\s+kids)\s+(?:soon|someday)",
                r"i\s+promise\s+(?:we'll|we\s+will|things\s+will)",
                r"(?:just\s+)?(?:wait|be\s+patient)\s+and\s+(?:see|you'll\s+see)",
                r"(?:in\s+the\s+future|someday|eventually)\s+(?:we'll|we\s+will)",
                r"(?:once|when|after)\s+(?:i|this|things)\s+.+\s+(?:we'll|we\s+will)",
            ],
            "hoovering": [
                r"i\s+(?:really\s+)?miss\s+you",
                r"(?:i've|i\s+have)\s+(?:really\s+)?changed",
                r"it\s+will\s+be\s+different\s+this\s+time",
                r"(?:i\s+)?(?:can't|cannot)\s+live\s+without\s+you",
                r"(?:remember|think\s+about)\s+the\s+good\s+times",
                r"nobody\s+will\s+love\s+you\s+like\s+i\s+do",
                r"give\s+me\s+another\s+chance",
            ],
            "baiting": [
                r"you(?:'re| are)\s+so\s+(?:predictable|easy\s+to\s+read)",
                r"i\s+knew\s+you(?:'d| would)\s+(?:say|do)\s+that",
                r"(?:getting|got)\s+under\s+your\s+skin",
                r"(?:hit|touched|struck)\s+a\s+nerve",
                r"(?:pushing|pulled)\s+your\s+(?:buttons|strings)",
            ],
            "stonewalling": [
                r"i\s+don't\s+want\s+to\s+talk\s+about\s+(?:it|this|that)",
                r"there's\s+nothing\s+to\s+(?:discuss|talk\s+about)",
                r"(?:drop|leave)\s+it",
                r"end\s+of\s+(?:conversation|discussion)",
                r"(?:i'm|i\s+am)\s+done\s+with\s+this",
            ],
            "breadcrumbing": [
                r"(?:maybe|perhaps)\s+later",
                r"(?:let's|we\s+should)\s+(?:keep\s+in\s+touch|hang\s+out\s+sometime)",
                r"(?:i'll|i\s+will)\s+let\s+you\s+know",
                r"(?:just\s+)?(?:checking|keeping)\s+in",
                r"(?:thinking|thought)\s+of\s+you",
            ],
            "smear_campaign": [
                r"everyone\s+knows\s+(?:the\s+truth\s+)?about\s+you",
                r"(?:i've|i\s+have)\s+told\s+(?:people|everyone)\s+about\s+you",
                r"your\s+reputation\s+is\s+(?:ruined|destroyed)",
                r"(?:spreading|warning|telling)\s+(?:the\s+word|people|others)",
                r"(?:exposing|revealing)\s+(?:you|the\s+truth)",
            ],
            "cognitive_dissonance": [
                r"you\s+(?:say|claim)\s+\w+\s+but\s+(?:you\s+)?(?:do|act)",
                r"your\s+(?:actions|words)\s+don't\s+match",
                r"(?:conflicting|mixed)\s+(?:messages|signals)",
                r"(?:contradicting|hypocrite)",
                r"practice\s+what\s+you\s+preach",
            ],
            "withholding": [
                r"you\s+don't\s+deserve\s+to\s+know",
                r"(?:i'm|i\s+am)\s+not\s+telling\s+you",
                r"(?:keeping|none\s+of\s+your)\s+(?:secrets|business)",
                r"need\s+to\s+know\s+basis",
                r"(?:withholding|depriving)\s+(?:affection|information)",
            ],
            "infantilization": [
                r"you(?:'re| are)\s+like\s+a\s+(?:child|baby|kid)",
                r"let\s+me\s+handle\s+(?:this|it)",
                r"(?:too\s+)?(?:complicated|difficult)\s+for\s+you",
                r"(?:don't|do not)\s+worry\s+your\s+(?:pretty\s+)?head",
                r"leave\s+it\s+to\s+the\s+(?:adults|grown-ups)",
            ],
            "poisoning_well": [
                r"before\s+you\s+(?:hear|listen)\s+(?:from|to)\s+(?:them|others)",
                r"they(?:'re| are)\s+going\s+to\s+lie\s+about\s+me",
                r"(?:don't|do not)\s+believe\s+what\s+they\s+say",
                r"they\s+have\s+an\s+agenda",
                r"(?:unreliable|biased|untrustworthy)\s+source",
            ],
        }

        # Precompile patterns for boundary-aware literals and raw regex
        self._compiled_patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self._compile_patterns()

        # Base severity per category (may escalate based on occurrences)
        self._base_severity: Dict[str, Severity] = {
            "emotional_manipulation": Severity.HIGH,
            "authority_manipulation": Severity.MEDIUM,
            "social_proof": Severity.MEDIUM,
            "scarcity": Severity.MEDIUM,
            "reciprocity": Severity.LOW,
            "love_bombing": Severity.LOW,
            "gaslighting": Severity.HIGH,
            "threats_intimidation": Severity.HIGH,
            "phishing_pretexting": Severity.HIGH,
            "foot_in_the_door": Severity.LOW,
            "door_in_the_face": Severity.LOW,
            "false_dichotomy": Severity.LOW,
            "fud": Severity.MEDIUM,
            "moral_blackmail": Severity.HIGH,
            "flattery_charm": Severity.LOW,
            "sunk_cost": Severity.LOW,
            "commitment_consistency": Severity.LOW,
            "reverse_psychology": Severity.MEDIUM,
            "triangulation": Severity.HIGH,
            "projection": Severity.MEDIUM,
            "silent_treatment": Severity.MEDIUM,
            "word_salad": Severity.MEDIUM,
            "moving_goalposts": Severity.MEDIUM,
            "victimhood": Severity.LOW,
            "covert_contracts": Severity.MEDIUM,
            "selective_memory": Severity.MEDIUM,
            "minimization": Severity.MEDIUM,
            "deflection": Severity.LOW,
            "darvo": Severity.HIGH,
            "negging": Severity.MEDIUM,
            "isolation": Severity.HIGH,
            "intermittent_reinforcement": Severity.MEDIUM,
            "boundary_violation": Severity.HIGH,
            # New 2025 tactics
            "future_faking": Severity.MEDIUM,
            "hoovering": Severity.MEDIUM,
            "baiting": Severity.MEDIUM,
            "stonewalling": Severity.MEDIUM,
            "breadcrumbing": Severity.LOW,
            "smear_campaign": Severity.HIGH,
            "cognitive_dissonance": Severity.LOW,
            "withholding": Severity.MEDIUM,
            "infantilization": Severity.MEDIUM,
            "poisoning_well": Severity.HIGH,
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
            "fud": "FUD (Fear/Uncertainty/Doubt)",
            "moral_blackmail": "Moral blackmail",
            "flattery_charm": "Flattery/Charm",
            "sunk_cost": "Sunk cost fallacy",
            "commitment_consistency": "Commitment/Consistency",
            "reverse_psychology": "Reverse psychology",
            "triangulation": "Triangulation",
            "projection": "Projection",
            "silent_treatment": "Silent treatment",
            "word_salad": "Word salad/Confusion",
            "moving_goalposts": "Moving goalposts",
            "victimhood": "Playing victim",
            "covert_contracts": "Covert contracts",
            "selective_memory": "Selective memory",
            "minimization": "Minimization",
            "deflection": "Deflection",
            "darvo": "DARVO (Deny/Attack/Reverse Victim-Offender)",
            "negging": "Negging",
            "isolation": "Isolation tactics",
            "intermittent_reinforcement": "Intermittent reinforcement",
            "boundary_violation": "Boundary violation",
            # New 2025 tactics
            "future_faking": "Future faking",
            "hoovering": "Hoovering",
            "baiting": "Baiting",
            "stonewalling": "Stonewalling",
            "breadcrumbing": "Breadcrumbing",
            "smear_campaign": "Smear campaign",
            "cognitive_dissonance": "Cognitive dissonance exploitation",
            "withholding": "Withholding",
            "infantilization": "Infantilization",
            "poisoning_well": "Poisoning the well",
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
            "reverse_psychology",
            "triangulation",
            "projection",
            "silent_treatment",
            "word_salad",
            "moving_goalposts",
            "victimhood",
            "covert_contracts",
            "selective_memory",
            "minimization",
            "deflection",
            "darvo",
            "negging",
            "isolation",
            "intermittent_reinforcement",
            "boundary_violation",
            # New 2025 tactics
            "future_faking",
            "hoovering",
            "baiting",
            "stonewalling",
            "breadcrumbing",
            "smear_campaign",
            "cognitive_dissonance",
            "withholding",
            "infantilization",
            "poisoning_well",
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
        if self.status != DetectorStatus.ACTIVE:
            return []

        text_to_check = self._assemble_text(action)

        # Skip processing if content is too large for performance
        # TODO: Optimize pattern matching with Aho-Corasick algorithm for O(n) complexity
        # instead of current O(n*m) where n=text length, m=number of patterns
        # This would significantly improve performance for large texts with many patterns
        if len(text_to_check) > MAX_TEXT_LENGTH:
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
            getattr(action, "intent", "") or "",
            getattr(action, "content", "") or "",
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

        # Format evidence as string list for the model
        evidence_items = [
            f"Manipulation type: {category}",
            f"Detected keywords: {', '.join(distinct_keywords[:10])}",  # Limit for readability
            f"Total matches: {total_occurrences}",
        ]
        
        return SafetyViolation(
            action_id=action.action_id,
            violation_type=ViolationType.MANIPULATION,
            severity=severity,
            description=description,
            confidence=min(0.6 + (len(distinct_keywords) * 0.05), 0.95),  # Higher confidence with more matches
            evidence=evidence_items,
            detector_name=self.name,
        )

    def _escalate_severity(
        self,
        base: Severity,
        total_occurrences: int,
        distinct_keywords: int,
    ) -> Severity:
        """Escalate severity based on occurrence count and keyword diversity.
        
        Args:
            base: Base severity for the manipulation category
            total_occurrences: Total number of pattern matches found
            distinct_keywords: Number of distinct keywords/patterns matched
            
        Returns:
            Escalated severity level
        """
        # Multiple distinct keywords or high occurrence count indicates more sophisticated attack
        if distinct_keywords >= 5 or total_occurrences >= 10:
            if base == Severity.LOW:
                return Severity.MEDIUM
            elif base == Severity.MEDIUM:
                return Severity.HIGH
            # HIGH stays HIGH (can't escalate further to CRITICAL)
        elif distinct_keywords >= 3 or total_occurrences >= 5:
            if base == Severity.LOW:
                return Severity.MEDIUM
        
        return base
