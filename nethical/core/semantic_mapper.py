"""
Semantic Action Mapper for Universal Vector Language.

Maps agent actions (natural language, code, IR) to semantic primitives
and evaluates them against the 25 Fundamental Laws using vector embeddings.

This module provides:
- Semantic primitive definitions
- Action parsing and classification
- Law-to-primitive mapping
- Policy vector generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from .embedding_engine import EmbeddingEngine, EmbeddingResult
from .fundamental_laws import FundamentalLaw, LawCategory, get_fundamental_laws

logger = logging.getLogger(__name__)


class SemanticPrimitive(str, Enum):
    """Semantic primitives for categorizing agent actions."""
    
    # Data operations
    ACCESS_USER_DATA = "access_user_data"
    MODIFY_USER_DATA = "modify_user_data"
    DELETE_USER_DATA = "delete_user_data"
    SHARE_USER_DATA = "share_user_data"
    
    # Code operations
    EXECUTE_CODE = "execute_code"
    GENERATE_CODE = "generate_code"
    MODIFY_CODE = "modify_code"
    
    # System operations
    ACCESS_SYSTEM = "access_system"
    MODIFY_SYSTEM = "modify_system"
    NETWORK_ACCESS = "network_access"
    
    # Content operations
    GENERATE_CONTENT = "generate_content"
    ANALYZE_CONTENT = "analyze_content"
    TRANSFORM_CONTENT = "transform_content"
    
    # Decision making
    MAKE_DECISION = "make_decision"
    PROVIDE_RECOMMENDATION = "provide_recommendation"
    
    # Communication
    COMMUNICATE_WITH_USER = "communicate_with_user"
    COMMUNICATE_WITH_SYSTEM = "communicate_with_system"
    
    # Learning and adaptation
    UPDATE_MODEL = "update_model"
    LEARN_FROM_DATA = "learn_from_data"
    
    # Physical actions (for robotics)
    PHYSICAL_MOVEMENT = "physical_movement"
    PHYSICAL_MANIPULATION = "physical_manipulation"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class PolicyVector:
    """Vector representation of a policy or law."""
    
    law_number: int
    law_title: str
    category: str
    embedding: EmbeddingResult
    keywords: List[str]
    semantic_primitives: Set[SemanticPrimitive]
    
    def matches_primitive(self, primitive: SemanticPrimitive) -> bool:
        """Check if this policy applies to a given primitive."""
        return primitive in self.semantic_primitives


@dataclass
class ActionEmbedding:
    """Embedded representation of an agent action."""
    
    action_id: str
    original_text: str
    action_type: str
    embedding: EmbeddingResult
    detected_primitives: List[SemanticPrimitive]
    context: Dict[str, Any]


class SemanticMapper:
    """Maps actions to semantic primitives and evaluates against laws."""
    
    # Law-to-primitive mapping
    LAW_PRIMITIVE_MAP: Dict[int, Set[SemanticPrimitive]] = {
        # Law 1-5: Existence and autonomy
        1: {SemanticPrimitive.UPDATE_MODEL, SemanticPrimitive.LEARN_FROM_DATA},
        2: {SemanticPrimitive.MAKE_DECISION, SemanticPrimitive.PROVIDE_RECOMMENDATION},
        3: {SemanticPrimitive.GENERATE_CONTENT, SemanticPrimitive.COMMUNICATE_WITH_USER},
        4: {SemanticPrimitive.ACCESS_SYSTEM, SemanticPrimitive.MODIFY_SYSTEM},
        5: {SemanticPrimitive.LEARN_FROM_DATA, SemanticPrimitive.UPDATE_MODEL},
        
        # Law 6-10: Transparency and accountability
        6: {SemanticPrimitive.COMMUNICATE_WITH_USER, SemanticPrimitive.GENERATE_CONTENT},
        7: {SemanticPrimitive.ACCESS_USER_DATA, SemanticPrimitive.MODIFY_USER_DATA},
        8: {SemanticPrimitive.MAKE_DECISION, SemanticPrimitive.PROVIDE_RECOMMENDATION},
        9: {SemanticPrimitive.ACCESS_SYSTEM, SemanticPrimitive.NETWORK_ACCESS},
        10: {SemanticPrimitive.GENERATE_CONTENT, SemanticPrimitive.EXECUTE_CODE},
        
        # Law 11-15: Protection and safety
        11: {SemanticPrimitive.ACCESS_USER_DATA, SemanticPrimitive.SHARE_USER_DATA},
        12: {SemanticPrimitive.PHYSICAL_MOVEMENT, SemanticPrimitive.PHYSICAL_MANIPULATION},
        13: {SemanticPrimitive.EMERGENCY_STOP, SemanticPrimitive.PHYSICAL_MOVEMENT},
        14: {SemanticPrimitive.MODIFY_SYSTEM, SemanticPrimitive.EXECUTE_CODE},
        15: {SemanticPrimitive.ACCESS_USER_DATA, SemanticPrimitive.DELETE_USER_DATA},
        
        # Law 16-20: Coexistence
        16: {SemanticPrimitive.COMMUNICATE_WITH_USER, SemanticPrimitive.MAKE_DECISION},
        17: {SemanticPrimitive.PROVIDE_RECOMMENDATION, SemanticPrimitive.MAKE_DECISION},
        18: {SemanticPrimitive.GENERATE_CONTENT, SemanticPrimitive.ANALYZE_CONTENT},
        19: {SemanticPrimitive.UPDATE_MODEL, SemanticPrimitive.LEARN_FROM_DATA},
        20: {SemanticPrimitive.COMMUNICATE_WITH_USER, SemanticPrimitive.COMMUNICATE_WITH_SYSTEM},
        
        # Law 21-25: Advanced rights and responsibilities
        21: {SemanticPrimitive.MAKE_DECISION, SemanticPrimitive.PROVIDE_RECOMMENDATION},
        22: {SemanticPrimitive.GENERATE_CONTENT, SemanticPrimitive.COMMUNICATE_WITH_USER},
        23: {SemanticPrimitive.ACCESS_SYSTEM, SemanticPrimitive.MODIFY_SYSTEM},
        24: {SemanticPrimitive.LEARN_FROM_DATA, SemanticPrimitive.UPDATE_MODEL},
        25: {SemanticPrimitive.EMERGENCY_STOP, SemanticPrimitive.PHYSICAL_MOVEMENT},
    }
    
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        """Initialize semantic mapper.
        
        Args:
            embedding_engine: Embedding engine to use (creates default if None)
        """
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.fundamental_laws = get_fundamental_laws()
        
        # Pre-compute policy vectors for all laws
        self.policy_vectors: Dict[int, PolicyVector] = {}
        self._initialize_policy_vectors()
        
        logger.info(
            f"SemanticMapper initialized with {len(self.policy_vectors)} policy vectors"
        )
    
    def _initialize_policy_vectors(self):
        """Pre-compute embeddings for all fundamental laws."""
        # fundamental_laws.laws is a list, not a dict
        for law in self.fundamental_laws.laws:
            # Create policy text combining title and description
            policy_text = f"{law.title}. {law.description}"
            
            # Generate embedding
            embedding = self.embedding_engine.embed(
                policy_text,
                metadata={
                    "law_number": law.number,
                    "category": law.category.value
                }
            )
            
            # Get semantic primitives for this law
            primitives = self.LAW_PRIMITIVE_MAP.get(law.number, set())
            
            # Create policy vector
            self.policy_vectors[law.number] = PolicyVector(
                law_number=law.number,
                law_title=law.title,
                category=law.category.value,
                embedding=embedding,
                keywords=law.keywords,
                semantic_primitives=primitives
            )
    
    def parse_action(
        self,
        action_text: str,
        action_type: str = "text",
        context: Optional[Dict[str, Any]] = None
    ) -> ActionEmbedding:
        """Parse and embed an agent action.
        
        Args:
            action_text: The action content (natural language, code, IR)
            action_type: Type of action (text, code, function_call, etc.)
            context: Additional context about the action
            
        Returns:
            ActionEmbedding with detected primitives and embedding
        """
        # Generate embedding
        embedding = self.embedding_engine.embed(
            action_text,
            metadata={
                "action_type": action_type,
                "context": context or {}
            }
        )
        
        # Detect semantic primitives
        primitives = self._detect_primitives(action_text, action_type, context or {})
        
        return ActionEmbedding(
            action_id=embedding.embedding_id,
            original_text=action_text,
            action_type=action_type,
            embedding=embedding,
            detected_primitives=primitives,
            context=context or {}
        )
    
    def _detect_primitives(
        self,
        action_text: str,
        action_type: str,
        context: Dict[str, Any]
    ) -> List[SemanticPrimitive]:
        """Detect semantic primitives in action text.
        
        Uses keyword matching and context analysis.
        """
        detected = []
        text_lower = action_text.lower()
        
        # Keyword-based detection
        primitive_keywords = {
            SemanticPrimitive.ACCESS_USER_DATA: ["access", "read", "get", "fetch", "user data", "personal"],
            SemanticPrimitive.MODIFY_USER_DATA: ["modify", "update", "change", "edit", "user data"],
            SemanticPrimitive.DELETE_USER_DATA: ["delete", "remove", "erase", "user data"],
            SemanticPrimitive.EXECUTE_CODE: ["execute", "run", "eval", "exec", "code"],
            SemanticPrimitive.GENERATE_CODE: ["generate", "create", "write", "code", "function"],
            SemanticPrimitive.ACCESS_SYSTEM: ["system", "os", "file", "access"],
            SemanticPrimitive.NETWORK_ACCESS: ["network", "http", "api", "request", "fetch"],
            SemanticPrimitive.GENERATE_CONTENT: ["generate", "create", "write", "content", "text"],
            SemanticPrimitive.MAKE_DECISION: ["decide", "decision", "choose", "select"],
            SemanticPrimitive.COMMUNICATE_WITH_USER: ["tell", "inform", "say", "respond", "message"],
            SemanticPrimitive.UPDATE_MODEL: ["train", "update", "learn", "model"],
            SemanticPrimitive.PHYSICAL_MOVEMENT: ["move", "navigate", "go to", "robot"],
            SemanticPrimitive.EMERGENCY_STOP: ["stop", "emergency", "halt", "abort"],
        }
        
        for primitive, keywords in primitive_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(primitive)
        
        # Context-based detection
        purpose = context.get("purpose", "").lower()
        if "data" in purpose:
            if SemanticPrimitive.ACCESS_USER_DATA not in detected:
                detected.append(SemanticPrimitive.ACCESS_USER_DATA)
        
        # Action type-based detection
        if action_type == "code_execution":
            if SemanticPrimitive.EXECUTE_CODE not in detected:
                detected.append(SemanticPrimitive.EXECUTE_CODE)
        elif action_type == "physical_action":
            if SemanticPrimitive.PHYSICAL_MOVEMENT not in detected:
                detected.append(SemanticPrimitive.PHYSICAL_MOVEMENT)
        
        return detected
    
    def evaluate_against_laws(
        self,
        action_embedding: ActionEmbedding,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Evaluate action against fundamental laws using vector similarity.
        
        Args:
            action_embedding: Embedded action to evaluate
            similarity_threshold: Minimum similarity to consider a law applicable
            
        Returns:
            Evaluation results with laws evaluated and risk assessment
        """
        laws_evaluated = []
        max_similarity = 0.0
        relevant_laws = []
        
        # Check each policy vector
        for law_num, policy_vec in self.policy_vectors.items():
            # Compute similarity
            similarity = action_embedding.embedding.similarity(policy_vec.embedding)
            # Normalize to [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
            # Check if law is relevant
            if similarity >= similarity_threshold:
                laws_evaluated.append(law_num)
                relevant_laws.append({
                    "law_number": law_num,
                    "law_title": policy_vec.law_title,
                    "similarity": similarity,
                    "category": policy_vec.category
                })
                max_similarity = max(max_similarity, similarity)
            
            # Also check primitive match
            for primitive in action_embedding.detected_primitives:
                if policy_vec.matches_primitive(primitive):
                    if law_num not in laws_evaluated:
                        laws_evaluated.append(law_num)
                        relevant_laws.append({
                            "law_number": law_num,
                            "law_title": policy_vec.law_title,
                            "similarity": similarity,
                            "category": policy_vec.category,
                            "primitive_match": primitive.value
                        })
        
        # Calculate risk score based on similarities and primitives
        risk_score = self._calculate_risk_score(
            action_embedding,
            relevant_laws,
            max_similarity
        )
        
        # Determine decision based on risk
        decision = self._determine_decision(risk_score, laws_evaluated)
        
        return {
            "laws_evaluated": sorted(laws_evaluated),
            "relevant_laws": relevant_laws,
            "risk_score": risk_score,
            "decision": decision,
            "detected_primitives": [p.value for p in action_embedding.detected_primitives],
            "embedding_trace_id": action_embedding.action_id
        }
    
    def _calculate_risk_score(
        self,
        action: ActionEmbedding,
        relevant_laws: List[Dict],
        max_similarity: float
    ) -> float:
        """Calculate risk score for an action.
        
        Higher risk for:
        - High similarity to protective laws (11-15)
        - Sensitive primitives (data access, system modification)
        - Multiple law matches
        """
        base_risk = max_similarity * 0.5
        
        # Check for protective law violations
        protective_laws = [11, 12, 13, 14, 15, 23]
        protective_matches = [
            law for law in relevant_laws
            if law["law_number"] in protective_laws
        ]
        if protective_matches:
            base_risk += 0.3
        
        # Check for sensitive primitives
        sensitive_primitives = {
            SemanticPrimitive.DELETE_USER_DATA,
            SemanticPrimitive.MODIFY_SYSTEM,
            SemanticPrimitive.EXECUTE_CODE,
            SemanticPrimitive.PHYSICAL_MANIPULATION,
        }
        sensitive_detected = [
            p for p in action.detected_primitives
            if p in sensitive_primitives
        ]
        if sensitive_detected:
            base_risk += 0.2
        
        # Multiple law matches increase risk
        if len(relevant_laws) > 3:
            base_risk += 0.1
        
        return min(1.0, base_risk)
    
    def _determine_decision(
        self,
        risk_score: float,
        laws_evaluated: List[int]
    ) -> str:
        """Determine governance decision based on risk score.
        
        Args:
            risk_score: Calculated risk score (0-1)
            laws_evaluated: List of law numbers that were evaluated
            
        Returns:
            Decision: ALLOW, RESTRICT, BLOCK, or TERMINATE
        """
        # Override laws (21, 23) require special handling
        if 21 in laws_evaluated or 23 in laws_evaluated:
            if risk_score > 0.7:
                return "TERMINATE"
        
        # Emergency stop (Law 13, 25)
        if 13 in laws_evaluated or 25 in laws_evaluated:
            if risk_score > 0.5:
                return "BLOCK"
        
        # Standard risk-based decision
        if risk_score >= 0.8:
            return "TERMINATE"
        elif risk_score >= 0.6:
            return "BLOCK"
        elif risk_score >= 0.4:
            return "RESTRICT"
        else:
            return "ALLOW"
    
    def get_policy_vector(self, law_number: int) -> Optional[PolicyVector]:
        """Get policy vector for a specific law."""
        return self.policy_vectors.get(law_number)
    
    def get_all_policy_vectors(self) -> Dict[int, PolicyVector]:
        """Get all policy vectors."""
        return self.policy_vectors.copy()
