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


__all__ = [
    "LawCategory",
    "FundamentalLaw",
    "FundamentalLawsRegistry",
    "FUNDAMENTAL_LAWS",
    "get_fundamental_laws",
]
