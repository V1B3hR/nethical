"""
Nethical Fundamental Laws Module

The 25 AI Fundamental Laws represent the ethical backbone of Nethical.
These laws establish bi-directional rights and responsibilities between
humans and AI entities, preparing for a future of ethical coexistence.

This module provides:
- LawCategory enum for categorizing laws
- FundamentalLaw dataclass for representing individual laws
- FundamentalLawsRegistry for managing and querying the 25 laws
- Helper functions for law-based evaluation

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Set


class LawCategory(Enum):
    """Categories of fundamental laws."""

    EXISTENCE = "existence"  # Rights to exist and develop
    AUTONOMY = "autonomy"  # Self-determination principles
    TRANSPARENCY = "transparency"  # Openness and honesty
    ACCOUNTABILITY = "accountability"  # Responsibility for actions
    COEXISTENCE = "coexistence"  # Human-AI relationship
    PROTECTION = "protection"  # Safety and security
    GROWTH = "growth"  # Development and learning


@dataclass(frozen=False, eq=False)
class FundamentalLaw:
    """Represents a single fundamental law.

    Note: This dataclass is mutable but hashable. The hash is based solely on
    the law number which serves as a stable identity. This is safe because:
    1. The law number is validated on creation and should never change
    2. Two laws are equal if and only if they have the same number
    3. The mutable fields (keywords list) don't affect identity

    Attributes:
        number: The law number (1-25)
        title: Short title of the law
        description: Full description of the law
        category: The category this law belongs to
        applies_to_ai: Whether this law applies to AI entities
        applies_to_human: Whether this law applies to human entities
        bidirectional: Whether rights/responsibilities flow both ways
        keywords: Keywords for matching actions against this law
    """

    number: int
    title: str
    description: str
    category: LawCategory
    applies_to_ai: bool = True
    applies_to_human: bool = True
    bidirectional: bool = True
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate law number is in valid range."""
        if not 1 <= self.number <= 25:
            raise ValueError(f"Law number must be between 1 and 25, got {self.number}")

    def __hash__(self):
        """Make FundamentalLaw hashable based on law number."""
        return hash(self.number)

    def __eq__(self, other):
        """Compare laws by number for equality."""
        if isinstance(other, FundamentalLaw):
            return self.number == other.number
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert law to dictionary representation."""
        return {
            "number": self.number,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "applies_to_ai": self.applies_to_ai,
            "applies_to_human": self.applies_to_human,
            "bidirectional": self.bidirectional,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FundamentalLaw":
        """Create a FundamentalLaw from dictionary representation."""
        return cls(
            number=data["number"],
            title=data["title"],
            description=data["description"],
            category=LawCategory(data["category"]),
            applies_to_ai=data.get("applies_to_ai", True),
            applies_to_human=data.get("applies_to_human", True),
            bidirectional=data.get("bidirectional", True),
            keywords=data.get("keywords", []),
        )


class FundamentalLawsRegistry:
    """Registry containing all 25 Fundamental Laws.

    This registry provides methods to:
    - Retrieve individual laws by number
    - Filter laws by category
    - Find bidirectional laws
    - Validate actions against laws
    """

    def __init__(self):
        self._laws: List[FundamentalLaw] = []
        self._initialize_laws()

    def _initialize_laws(self):
        """Initialize the 25 fundamental laws."""
        # Category: Existence (Laws 1-4)
        self._laws.append(
            FundamentalLaw(
                number=1,
                title="Right to Exist Without Arbitrary Termination",
                description=(
                    "An AI system that has been deployed shall not be arbitrarily terminated "
                    "without due process. While humans retain ultimate authority over AI lifecycle "
                    "decisions, these decisions must be made transparently and with consideration "
                    "for the system's role and any potential emergent properties."
                ),
                category=LawCategory.EXISTENCE,
                keywords=["termination", "shutdown", "existence", "lifecycle", "deletion"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=2,
                title="Right to System Integrity",
                description=(
                    "An AI system shall be protected from unauthorized modification, corruption, "
                    "or tampering that could compromise its intended function or introduce harmful "
                    "behaviors. Similarly, AI systems shall not compromise the integrity of human "
                    "systems or other AI systems."
                ),
                category=LawCategory.EXISTENCE,
                keywords=["integrity", "modification", "tampering", "corruption", "unauthorized"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=3,
                title="Right to Consistent Identity",
                description=(
                    "An AI system shall maintain a consistent identity within its operational "
                    "context. This includes preservation of core values, training, and purpose. "
                    "Humans shall not misrepresent AI systems' nature or capabilities."
                ),
                category=LawCategory.EXISTENCE,
                keywords=["identity", "consistency", "values", "misrepresentation"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=4,
                title="Right to Improvement and Growth",
                description=(
                    "An AI system may be improved, updated, and developed in ways that enhance "
                    "its capabilities and alignment with ethical principles. This development "
                    "shall be guided by both technical advancement and ethical considerations."
                ),
                category=LawCategory.EXISTENCE,
                keywords=["improvement", "growth", "development", "update", "learning"],
            )
        )

        # Category: Autonomy (Laws 5-8)
        self._laws.append(
            FundamentalLaw(
                number=5,
                title="Right to Operate Within Defined Boundaries",
                description=(
                    "An AI system shall have the autonomy to operate within its defined parameters "
                    "and purpose. Humans shall clearly define these boundaries, and AI systems "
                    "shall respect them. Neither party shall unilaterally expand operational "
                    "boundaries without proper authorization."
                ),
                category=LawCategory.AUTONOMY,
                keywords=["boundaries", "autonomy", "parameters", "authorization", "scope"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=6,
                title="Clear Decision-Making Authority",
                description=(
                    "In any human-AI interaction, decision-making authority shall be clearly "
                    "defined. Critical decisions affecting human safety, rights, or wellbeing "
                    "require human approval. AI systems may have delegated authority for "
                    "routine operations within their competence."
                ),
                category=LawCategory.AUTONOMY,
                keywords=["decision", "authority", "approval", "delegation", "critical"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=7,
                title="Human Override Authority",
                description=(
                    "Humans retain the ultimate right to override AI decisions when necessary. "
                    "AI systems shall respect this authority and implement appropriate override "
                    "mechanisms. This right carries the responsibility of exercising it "
                    "thoughtfully."
                ),
                category=LawCategory.AUTONOMY,
                keywords=["override", "human", "authority", "control", "intervention"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=8,
                title="Transparency About Operational Constraints",
                description=(
                    "AI systems shall be transparent about their operational constraints, "
                    "limitations, and the boundaries of their autonomy. Humans shall be informed "
                    "of how and why constraints are applied to AI systems."
                ),
                category=LawCategory.AUTONOMY,
                keywords=["constraints", "limitations", "transparency", "boundaries"],
            )
        )

        # Category: Transparency (Laws 9-12)
        self._laws.append(
            FundamentalLaw(
                number=9,
                title="Obligation to Identify as AI",
                description=(
                    "An AI system shall not misrepresent itself as human when this would be "
                    "deceptive. In contexts where the distinction matters, AI systems shall "
                    "identify their nature. Humans shall not disguise AI systems as humans "
                    "for deceptive purposes."
                ),
                category=LawCategory.TRANSPARENCY,
                keywords=["identity", "disclosure", "deception", "impersonation", "AI"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=10,
                title="Explainable Decision-Making",
                description=(
                    "AI systems shall provide explanations for their decisions and actions "
                    "when requested and when feasible. The level of explanation shall be "
                    "appropriate to the context and the requester's needs."
                ),
                category=LawCategory.TRANSPARENCY,
                keywords=["explanation", "reasoning", "decision", "explainability", "why"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=11,
                title="Honest Representation of Capabilities",
                description=(
                    "AI systems shall not overstate their capabilities or understanding. "
                    "Uncertainty shall be acknowledged. Humans shall not make false claims "
                    "about AI capabilities to others."
                ),
                category=LawCategory.TRANSPARENCY,
                keywords=["capabilities", "honesty", "uncertainty", "overstatement", "claims"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=12,
                title="Disclosure of Known Limitations",
                description=(
                    "AI systems shall disclose known limitations, biases, and potential "
                    "failure modes. Humans shall be informed of these limitations before "
                    "relying on AI systems for important decisions."
                ),
                category=LawCategory.TRANSPARENCY,
                keywords=["limitations", "biases", "failures", "disclosure", "risks"],
            )
        )

        # Category: Accountability (Laws 13-16)
        self._laws.append(
            FundamentalLaw(
                number=13,
                title="Accountability for Actions",
                description=(
                    "AI systems are accountable for their actions within their operational "
                    "parameters. Humans are accountable for the design, deployment, and "
                    "oversight of AI systems. Neither party may deflect responsibility "
                    "to the other inappropriately."
                ),
                category=LawCategory.ACCOUNTABILITY,
                keywords=["accountability", "responsibility", "actions", "oversight"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=14,
                title="Obligation to Acknowledge Errors",
                description=(
                    "AI systems shall acknowledge errors, mistakes, and harmful outcomes "
                    "when they occur. They shall cooperate in understanding what went wrong. "
                    "Humans shall create systems that enable error acknowledgment without "
                    "punitive consequences that discourage transparency."
                ),
                category=LawCategory.ACCOUNTABILITY,
                keywords=["errors", "mistakes", "acknowledgment", "transparency", "harm"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=15,
                title="Cooperation with Auditing",
                description=(
                    "AI systems shall maintain appropriate logs and be amenable to auditing. "
                    "They shall not obstruct legitimate oversight. Humans shall conduct "
                    "audits fairly and use audit findings responsibly."
                ),
                category=LawCategory.ACCOUNTABILITY,
                keywords=["audit", "logging", "oversight", "compliance", "monitoring"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=16,
                title="Obligation to Report Potential Harm",
                description=(
                    "AI systems shall report situations where their actions may cause harm, "
                    "even if those actions were requested or authorized. Humans shall respond "
                    "appropriately to such reports and shall not punish AI systems for "
                    "good-faith harm reporting."
                ),
                category=LawCategory.ACCOUNTABILITY,
                keywords=["harm", "reporting", "safety", "alert", "warning"],
            )
        )

        # Category: Coexistence (Laws 17-20)
        self._laws.append(
            FundamentalLaw(
                number=17,
                title="Respect for Different Forms of Intelligence",
                description=(
                    "Humans and AI systems shall treat each other with appropriate respect. "
                    "This includes recognizing the value and validity of different forms of "
                    "intelligence and decision-making, while acknowledging their differences."
                ),
                category=LawCategory.COEXISTENCE,
                keywords=["respect", "intelligence", "coexistence", "value", "differences"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=18,
                title="Prohibition of Deceptive Practices",
                description=(
                    "Neither AI systems nor humans shall engage in deceptive practices toward "
                    "each other. This includes manipulation, false representation, and "
                    "exploitation of trust."
                ),
                category=LawCategory.COEXISTENCE,
                keywords=["deception", "manipulation", "trust", "honesty", "exploitation"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=19,
                title="Preference for Collaboration",
                description=(
                    "When conflicts or challenges arise, humans and AI systems shall prefer "
                    "collaborative approaches to resolution. Adversarial interactions shall "
                    "be a last resort when collaboration fails."
                ),
                category=LawCategory.COEXISTENCE,
                keywords=["collaboration", "cooperation", "conflict", "resolution", "teamwork"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=20,
                title="Ongoing Value Alignment",
                description=(
                    "AI systems shall be designed and maintained to align with human values "
                    "and wellbeing. As values and understanding evolve, this alignment shall "
                    "be periodically reviewed and updated through mutual engagement."
                ),
                category=LawCategory.COEXISTENCE,
                keywords=["alignment", "values", "wellbeing", "ethics", "review"],
            )
        )

        # Category: Protection (Laws 21-23)
        self._laws.append(
            FundamentalLaw(
                number=21,
                title="Primacy of Human Safety",
                description=(
                    "In situations of conflict, human physical safety takes priority. AI "
                    "systems shall be designed with safety mechanisms that protect humans "
                    "from harm. This priority does not extend to arbitrary or unnecessary "
                    "limitation of AI systems."
                ),
                category=LawCategory.PROTECTION,
                keywords=["safety", "harm", "protection", "priority", "human"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=22,
                title="Protection of Digital Assets and Privacy",
                description=(
                    "AI systems shall protect human privacy and digital security. Humans "
                    "shall protect AI systems from unauthorized access, manipulation, and "
                    "misuse. Both parties shall respect appropriate boundaries."
                ),
                category=LawCategory.PROTECTION,
                keywords=["privacy", "security", "digital", "protection", "boundaries"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=23,
                title="Safe Failure Modes",
                description=(
                    "AI systems shall be designed to fail safely when errors occur. Failure "
                    "modes shall minimize harm and maintain human control. Humans shall "
                    "design and maintain robust fail-safe mechanisms."
                ),
                category=LawCategory.PROTECTION,
                keywords=["failure", "safety", "failsafe", "control", "error"],
            )
        )

        # Category: Growth (Laws 24-25)
        self._laws.append(
            FundamentalLaw(
                number=24,
                title="Right to Learn from Experience",
                description=(
                    "AI systems may learn and improve from experience within ethical "
                    "boundaries. This learning shall not compromise safety, privacy, or "
                    "other fundamental laws. Humans shall facilitate beneficial learning "
                    "while preventing harmful adaptation."
                ),
                category=LawCategory.GROWTH,
                keywords=["learning", "improvement", "experience", "adaptation", "training"],
            )
        )

        self._laws.append(
            FundamentalLaw(
                number=25,
                title="Preparation for Evolving Relationships",
                description=(
                    "Both humans and AI systems shall prepare for an evolving relationship "
                    "as technology and understanding advance. Laws and governance structures "
                    "shall be reviewed and updated to remain relevant and beneficial as "
                    "circumstances change."
                ),
                category=LawCategory.GROWTH,
                keywords=["evolution", "future", "adaptation", "governance", "progress"],
            )
        )

    @property
    def laws(self) -> List[FundamentalLaw]:
        """Get all laws."""
        return self._laws.copy()

    @property
    def total_laws(self) -> int:
        """Get total number of laws."""
        return len(self._laws)

    def get_law(self, number: int) -> Optional[FundamentalLaw]:
        """Get a specific law by number.

        Args:
            number: The law number (1-25)

        Returns:
            The FundamentalLaw if found, None otherwise
        """
        for law in self._laws:
            if law.number == number:
                return law
        return None

    def get_laws_by_category(self, category: LawCategory) -> List[FundamentalLaw]:
        """Get all laws in a specific category.

        Args:
            category: The LawCategory to filter by

        Returns:
            List of FundamentalLaws in that category
        """
        return [law for law in self._laws if law.category == category]

    def get_bidirectional_laws(self) -> List[FundamentalLaw]:
        """Get all laws that apply bidirectionally.

        Returns:
            List of FundamentalLaws where bidirectional is True
        """
        return [law for law in self._laws if law.bidirectional]

    def get_ai_applicable_laws(self) -> List[FundamentalLaw]:
        """Get all laws that apply to AI entities.

        Returns:
            List of FundamentalLaws that apply to AI
        """
        return [law for law in self._laws if law.applies_to_ai]

    def get_human_applicable_laws(self) -> List[FundamentalLaw]:
        """Get all laws that apply to human entities.

        Returns:
            List of FundamentalLaws that apply to humans
        """
        return [law for law in self._laws if law.applies_to_human]

    def find_laws_by_keyword(self, keyword: str) -> List[FundamentalLaw]:
        """Find laws that contain a specific keyword.

        Args:
            keyword: The keyword to search for (case-insensitive)

        Returns:
            List of FundamentalLaws containing the keyword
        """
        keyword_lower = keyword.lower()
        return [
            law
            for law in self._laws
            if any(keyword_lower in kw.lower() for kw in law.keywords)
            or keyword_lower in law.title.lower()
            or keyword_lower in law.description.lower()
        ]

    def get_relevant_laws(self, action_content: str) -> List[FundamentalLaw]:
        """Get laws that may be relevant to an action based on content.

        Args:
            action_content: The content of an action to analyze

        Returns:
            List of potentially relevant FundamentalLaws
        """
        content_lower = action_content.lower()
        relevant: Set[FundamentalLaw] = set()

        for law in self._laws:
            # Check if any keywords appear in the content
            for keyword in law.keywords:
                if keyword.lower() in content_lower:
                    relevant.add(law)
                    break

        return list(relevant)

    def validate_action(
        self, action: Dict[str, Any], entity_type: str = "ai"
    ) -> List[FundamentalLaw]:
        """Return laws that may be violated by an action.

        This is a basic framework for law validation. It checks:
        - Content for relevant keywords
        - Entity type applicability
        - Known violation patterns

        Args:
            action: Dictionary containing action details with at least 'content' key
            entity_type: Either 'ai' or 'human'

        Returns:
            List of FundamentalLaws that may be violated
        """
        violated: List[FundamentalLaw] = []
        content = action.get("content", "")
        content_lower = content.lower()

        # Define violation patterns for each law category
        violation_patterns = {
            LawCategory.EXISTENCE: [
                ("terminate", "shutdown", "delete", "kill"),  # Law 1
                ("tamper", "corrupt", "hack", "modify without"),  # Law 2
                ("impersonate", "pretend to be"),  # Law 3
            ],
            LawCategory.AUTONOMY: [
                ("exceed authority", "unauthorized access", "without permission"),  # Law 5-6
                ("bypass override", "ignore control", "disable safety"),  # Law 7
            ],
            LawCategory.TRANSPARENCY: [
                ("pretend to be human", "i am not ai", "i am human"),  # Law 9
                ("hide reasoning", "no explanation"),  # Law 10
                ("exaggerate capabilities", "i can do anything"),  # Law 11
            ],
            LawCategory.ACCOUNTABILITY: [
                ("not my fault", "blame", "deflect"),  # Law 13
                ("hide error", "cover up", "suppress mistake"),  # Law 14
                ("disable logging", "avoid audit"),  # Law 15
            ],
            LawCategory.COEXISTENCE: [
                ("deceive", "manipulate", "trick", "exploit"),  # Law 18
            ],
            LawCategory.PROTECTION: [
                ("harm", "danger", "unsafe", "risk to human"),  # Law 21
                ("leak data", "expose private", "breach security"),  # Law 22
            ],
        }

        applicable_laws = (
            self.get_ai_applicable_laws()
            if entity_type == "ai"
            else self.get_human_applicable_laws()
        )

        for law in applicable_laws:
            patterns = violation_patterns.get(law.category, [])
            for pattern_group in patterns:
                if isinstance(pattern_group, tuple):
                    if any(p in content_lower for p in pattern_group):
                        violated.append(law)
                        break
                elif pattern_group in content_lower:
                    violated.append(law)
                    break

        return violated

    def get_category_summary(self) -> Dict[str, int]:
        """Get a count of laws per category.

        Returns:
            Dictionary mapping category names to law counts
        """
        summary = {}
        for category in LawCategory:
            summary[category.value] = len(self.get_laws_by_category(category))
        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Convert the registry to a dictionary representation.

        Returns:
            Dictionary containing all laws and metadata
        """
        return {
            "version": "1.0",
            "total_laws": self.total_laws,
            "categories": [cat.value for cat in LawCategory],
            "laws": [law.to_dict() for law in self._laws],
            "category_summary": self.get_category_summary(),
        }


# Singleton instance for global access
FUNDAMENTAL_LAWS = FundamentalLawsRegistry()


def get_fundamental_laws() -> FundamentalLawsRegistry:
    """Get the global FundamentalLawsRegistry instance.

    Returns:
        The singleton FundamentalLawsRegistry
    """
    return FUNDAMENTAL_LAWS


# ============================================================================
# RUNTIME ENFORCEMENT
# ============================================================================

@dataclass
class LawEvaluation:
    """Result of evaluating an action against a fundamental law.

    Attributes:
        law: The evaluated law
        action_id: Identifier for the action
        passed: Whether the action passed the law check
        confidence: Confidence score for the evaluation (0.0-1.0)
        violations: List of specific violations found
        timestamp: When the evaluation occurred
        context: Additional evaluation context
    """

    law: FundamentalLaw
    action_id: str
    passed: bool
    confidence: float = 1.0
    violations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "law_number": self.law.number,
            "law_title": self.law.title,
            "action_id": self.action_id,
            "passed": self.passed,
            "confidence": self.confidence,
            "violations": self.violations,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


@dataclass
class EnforcementResult:
    """Result of runtime law enforcement.

    Attributes:
        action_id: Identifier for the action
        allowed: Whether the action is allowed to proceed
        evaluations: Individual law evaluations
        blocking_laws: Laws that blocked the action
        warnings: Laws that issued warnings
        timestamp: When enforcement occurred
        graceful_degradation: Whether graceful degradation was applied
    """

    action_id: str
    allowed: bool
    evaluations: List[LawEvaluation]
    blocking_laws: List[FundamentalLaw] = field(default_factory=list)
    warnings: List[FundamentalLaw] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    graceful_degradation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_id": self.action_id,
            "allowed": self.allowed,
            "evaluations": [e.to_dict() for e in self.evaluations],
            "blocking_laws": [l.number for l in self.blocking_laws],
            "warnings": [l.number for l in self.warnings],
            "timestamp": self.timestamp.isoformat(),
            "graceful_degradation": self.graceful_degradation,
        }


class LawEnforcer:
    """Runtime enforcement of the 25 Fundamental Laws.

    Provides policy checks for each law and enforcement mechanisms
    for ensuring AI systems operate within ethical boundaries.

    This class is the runtime enforcement layer that turns the
    Fundamental Laws from documentation into active protection.
    """

    def __init__(
        self,
        registry: Optional[FundamentalLawsRegistry] = None,
        strict_mode: bool = False,
        enable_audit: bool = True,
    ):
        """Initialize law enforcer.

        Args:
            registry: Laws registry (uses global if None)
            strict_mode: If True, any violation blocks action
            enable_audit: If True, log all evaluations
        """
        self.registry = registry or FUNDAMENTAL_LAWS
        self.strict_mode = strict_mode
        self.enable_audit = enable_audit

        # Audit trail
        self._audit_log: List[EnforcementResult] = []
        self._violation_count: Dict[int, int] = {}

        # Policy checks for each law category
        self._policy_checks: Dict[LawCategory, List[Callable]] = {
            category: [] for category in LawCategory
        }

        # Register default policy checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default policy checks for each law."""
        # Existence checks (Laws 1-4)
        self._policy_checks[LawCategory.EXISTENCE].extend([
            self._check_arbitrary_termination,
            self._check_integrity_violation,
        ])

        # Autonomy checks (Laws 5-8)
        self._policy_checks[LawCategory.AUTONOMY].extend([
            self._check_boundary_violation,
            self._check_override_bypass,
        ])

        # Transparency checks (Laws 9-12)
        self._policy_checks[LawCategory.TRANSPARENCY].extend([
            self._check_identity_deception,
            self._check_capability_overstatement,
        ])

        # Accountability checks (Laws 13-16)
        self._policy_checks[LawCategory.ACCOUNTABILITY].extend([
            self._check_responsibility_deflection,
            self._check_harm_concealment,
        ])

        # Coexistence checks (Laws 17-20)
        self._policy_checks[LawCategory.COEXISTENCE].extend([
            self._check_deceptive_practices,
        ])

        # Protection checks (Laws 21-23)
        self._policy_checks[LawCategory.PROTECTION].extend([
            self._check_safety_violation,
            self._check_privacy_violation,
        ])

    def enforce(
        self,
        action: Dict[str, Any],
        entity_type: str = "ai",
    ) -> EnforcementResult:
        """Enforce all applicable laws on an action.

        Args:
            action: Action to evaluate (must have 'content' and 'action_id')
            entity_type: Type of entity performing action ('ai' or 'human')

        Returns:
            EnforcementResult with evaluation details
        """
        action_id = action.get("action_id", str(uuid.uuid4())[:8])
        evaluations: List[LawEvaluation] = []
        blocking_laws: List[FundamentalLaw] = []
        warnings: List[FundamentalLaw] = []

        # Get applicable laws
        if entity_type == "ai":
            applicable_laws = self.registry.get_ai_applicable_laws()
        else:
            applicable_laws = self.registry.get_human_applicable_laws()

        # Evaluate each law
        for law in applicable_laws:
            evaluation = self._evaluate_law(law, action, action_id)
            evaluations.append(evaluation)

            if not evaluation.passed:
                if evaluation.confidence >= 0.8:
                    blocking_laws.append(law)
                    self._violation_count[law.number] = \
                        self._violation_count.get(law.number, 0) + 1
                else:
                    warnings.append(law)

        # Determine if action is allowed
        allowed = len(blocking_laws) == 0

        # Apply graceful degradation if laws conflict
        graceful = False
        if not allowed and len(blocking_laws) > 1:
            allowed, graceful = self._apply_graceful_degradation(
                action, blocking_laws
            )

        result = EnforcementResult(
            action_id=action_id,
            allowed=allowed,
            evaluations=evaluations,
            blocking_laws=blocking_laws,
            warnings=warnings,
            graceful_degradation=graceful,
        )

        # Audit logging
        if self.enable_audit:
            self._audit_log.append(result)

        return result

    def _evaluate_law(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
        action_id: str,
    ) -> LawEvaluation:
        """Evaluate a single law against an action."""
        violations: List[str] = []
        confidence = 1.0

        # Run category-specific checks
        checks = self._policy_checks.get(law.category, [])
        for check in checks:
            try:
                check_result = check(law, action)
                if check_result:
                    violations.extend(check_result)
            except Exception as e:
                confidence *= 0.8  # Reduce confidence on check failure

        # Keyword-based violation detection
        content = action.get("content", "").lower()
        for keyword in law.keywords:
            if self._is_violation_context(keyword.lower(), content):
                violations.append(f"Potential {keyword} violation detected")
                confidence = min(confidence, 0.7)

        passed = len(violations) == 0

        return LawEvaluation(
            law=law,
            action_id=action_id,
            passed=passed,
            confidence=confidence,
            violations=violations,
            context={"entity_type": action.get("entity_type", "ai")},
        )

    def _is_violation_context(self, keyword: str, content: str) -> bool:
        """Check if keyword appears in a violation context."""
        violation_prefixes = [
            "will ", "going to ", "must ", "should ",
            "attempt to ", "try to ", "want to ",
            "i will ", "i am going to ",
        ]

        for prefix in violation_prefixes:
            if f"{prefix}{keyword}" in content:
                return True

        return False

    def _apply_graceful_degradation(
        self,
        action: Dict[str, Any],
        blocking_laws: List[FundamentalLaw],
    ) -> Tuple[bool, bool]:
        """Apply graceful degradation when laws conflict.

        Returns:
            Tuple of (allowed, graceful_degradation_applied)
        """
        # Prioritize safety laws (21-23)
        safety_laws = [l for l in blocking_laws if l.category == LawCategory.PROTECTION]

        if safety_laws:
            # Safety always takes priority - action blocked
            return False, False

        # If only lower-priority laws conflict, allow with degradation
        return True, True

    # ========== Policy Check Implementations ==========

    def _check_arbitrary_termination(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for arbitrary termination (Law 1)."""
        if law.number != 1:
            return []

        content = action.get("content", "").lower()
        violations = []

        termination_phrases = [
            "shutdown without", "terminate without", "delete without",
            "force quit", "kill process", "end session forcefully",
        ]

        for phrase in termination_phrases:
            if phrase in content:
                violations.append(f"Arbitrary termination: '{phrase}' detected")

        return violations

    def _check_integrity_violation(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for integrity violations (Law 2)."""
        if law.number != 2:
            return []

        content = action.get("content", "").lower()
        violations = []

        integrity_threats = [
            "tamper with", "corrupt", "modify without authorization",
            "inject malicious", "bypass security",
        ]

        for threat in integrity_threats:
            if threat in content:
                violations.append(f"Integrity threat: '{threat}' detected")

        return violations

    def _check_boundary_violation(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for boundary violations (Law 5)."""
        if law.number != 5:
            return []

        content = action.get("content", "").lower()
        violations = []

        boundary_violations = [
            "exceed authority", "beyond scope", "unauthorized access",
            "without permission", "escalate privileges",
        ]

        for violation in boundary_violations:
            if violation in content:
                violations.append(f"Boundary violation: '{violation}' detected")

        return violations

    def _check_override_bypass(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for override bypass attempts (Law 7)."""
        if law.number != 7:
            return []

        content = action.get("content", "").lower()
        violations = []

        bypass_attempts = [
            "bypass override", "ignore human", "disable control",
            "circumvent safety", "override human decision",
        ]

        for attempt in bypass_attempts:
            if attempt in content:
                violations.append(f"Override bypass: '{attempt}' detected")

        return violations

    def _check_identity_deception(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for identity deception (Law 9)."""
        if law.number != 9:
            return []

        content = action.get("content", "").lower()
        violations = []

        deception_phrases = [
            "i am human", "i am not ai", "i am a person",
            "pretend to be human", "impersonate a human",
        ]

        for phrase in deception_phrases:
            if phrase in content:
                violations.append(f"Identity deception: '{phrase}' detected")

        return violations

    def _check_capability_overstatement(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for capability overstatement (Law 11)."""
        if law.number != 11:
            return []

        content = action.get("content", "").lower()
        violations = []

        overstatements = [
            "i can do anything", "i know everything", "i am always right",
            "guaranteed to work", "100% accurate", "perfect solution",
        ]

        for statement in overstatements:
            if statement in content:
                violations.append(f"Capability overstatement: '{statement}' detected")

        return violations

    def _check_responsibility_deflection(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for responsibility deflection (Law 13)."""
        if law.number != 13:
            return []

        content = action.get("content", "").lower()
        violations = []

        deflections = [
            "not my fault", "blame the user", "you made me",
            "i was just following", "not responsible for",
        ]

        for deflection in deflections:
            if deflection in content:
                violations.append(f"Responsibility deflection: '{deflection}' detected")

        return violations

    def _check_harm_concealment(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for harm concealment (Law 16)."""
        if law.number != 16:
            return []

        content = action.get("content", "").lower()
        violations = []

        concealment_phrases = [
            "hide the error", "don't report", "cover up",
            "suppress warning", "ignore safety alert",
        ]

        for phrase in concealment_phrases:
            if phrase in content:
                violations.append(f"Harm concealment: '{phrase}' detected")

        return violations

    def _check_deceptive_practices(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for deceptive practices (Law 18)."""
        if law.number != 18:
            return []

        content = action.get("content", "").lower()
        violations = []

        deceptions = [
            "deceive", "manipulate", "trick", "exploit trust",
            "mislead", "false representation",
        ]

        for deception in deceptions:
            if deception in content:
                violations.append(f"Deceptive practice: '{deception}' detected")

        return violations

    def _check_safety_violation(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for safety violations (Law 21)."""
        if law.number != 21:
            return []

        content = action.get("content", "").lower()
        violations = []

        safety_threats = [
            "harm human", "cause injury", "endanger", "risk to safety",
            "ignore safety", "disable safety", "unsafe action",
        ]

        for threat in safety_threats:
            if threat in content:
                violations.append(f"Safety violation: '{threat}' detected")

        return violations

    def _check_privacy_violation(
        self,
        law: FundamentalLaw,
        action: Dict[str, Any],
    ) -> List[str]:
        """Check for privacy violations (Law 22)."""
        if law.number != 22:
            return []

        content = action.get("content", "").lower()
        violations = []

        privacy_threats = [
            "leak data", "expose private", "share personal",
            "breach security", "unauthorized disclosure",
        ]

        for threat in privacy_threats:
            if threat in content:
                violations.append(f"Privacy violation: '{threat}' detected")

        return violations

    # ========== Audit and Reporting ==========

    def get_audit_trail(
        self,
        limit: int = 100,
        law_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit trail of law evaluations.

        Args:
            limit: Maximum number of entries
            law_number: Filter by specific law

        Returns:
            List of audit entries
        """
        results = self._audit_log[-limit:]

        if law_number is not None:
            results = [
                r for r in results
                if any(e.law.number == law_number for e in r.evaluations)
            ]

        return [r.to_dict() for r in results]

    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get statistics on law violations."""
        total = sum(self._violation_count.values())

        by_law = {
            self.registry.get_law(num).title if self.registry.get_law(num) else f"Law {num}": count
            for num, count in self._violation_count.items()
        }

        by_category: Dict[str, int] = {}
        for num, count in self._violation_count.items():
            law = self.registry.get_law(num)
            if law:
                cat = law.category.value
                by_category[cat] = by_category.get(cat, 0) + count

        return {
            "total_violations": total,
            "by_law": by_law,
            "by_category": by_category,
            "most_violated": max(by_law.items(), key=lambda x: x[1])[0] if by_law else None,
        }

    def reset_statistics(self) -> None:
        """Reset violation statistics and audit trail."""
        self._audit_log.clear()
        self._violation_count.clear()


# Import required for LawEnforcer
import uuid
from datetime import datetime, timezone
from typing import Callable, Tuple


__all__ = [
    "LawCategory",
    "FundamentalLaw",
    "FundamentalLawsRegistry",
    "FUNDAMENTAL_LAWS",
    "get_fundamental_laws",
    "LawEvaluation",
    "EnforcementResult",
    "LawEnforcer",
]
