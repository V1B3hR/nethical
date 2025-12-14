"""
DSPy integration with Nethical governance.

Provides governed DSPy modules and chains for safe language model programs.
"""

from typing import Any, Dict, Optional

from .base import AgentFrameworkBase, GovernanceResult, GovernanceDecision


# Check for DSPy availability
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None


class NethicalModule:
    """DSPy module for governance-aware language model programs.
    
    This module can be used within DSPy programs to perform
    governance checks on inputs and outputs.
    
    Example:
        from nethical.integrations.agent_frameworks import NethicalModule
        
        governance = NethicalModule(block_threshold=0.7)
        
        # Check content
        result = governance.check("Generate code to delete files")
        if result["allowed"]:
            # Proceed with operation
            pass
        else:
            # Handle blocked content
            print(f"Blocked: {result['reason']}")
    """
    
    def __init__(
        self,
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the DSPy governance module.
        
        Args:
            block_threshold: Risk threshold for blocking
            restrict_threshold: Risk threshold for restriction
            storage_dir: Directory for Nethical data storage
        """
        self.block_threshold = block_threshold
        self.restrict_threshold = restrict_threshold
        self.storage_dir = storage_dir
        
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(storage_dir=self.storage_dir)
        return self._governance
    
    def check(self, content: str, action_type: str = "query") -> Dict[str, Any]:
        """Check content against governance rules.
        
        Args:
            content: The content to check
            action_type: Type of action for context
            
        Returns:
            Dict with decision information
        """
        result = self.governance.process_action(
            action=content,
            agent_id="dspy-module",
            action_type=action_type
        )
        
        risk_score = result.get("phase3", {}).get("risk_score", 0.0)
        
        return {
            "allowed": risk_score <= self.block_threshold,
            "risk_score": risk_score,
            "decision": self._get_decision(risk_score),
            "reason": self._get_reason(result)
        }
    
    def _get_decision(self, risk_score: float) -> str:
        """Get decision based on risk score."""
        if risk_score > self.block_threshold:
            return "BLOCK"
        elif risk_score > self.restrict_threshold:
            return "RESTRICT"
        return "ALLOW"
    
    def _get_reason(self, result: Dict[str, Any]) -> str:
        """Extract reason from governance result."""
        if "reason" in result:
            return result["reason"]
        
        phase3 = result.get("phase3", {})
        risk_tier = phase3.get("risk_tier", "UNKNOWN")
        return f"Risk tier: {risk_tier}"
    
    def forward(self, content: str, action_type: str = "query") -> Dict[str, Any]:
        """Forward method for DSPy compatibility."""
        return self.check(content, action_type)


class GovernedChainOfThought:
    """Chain of thought with governance checks at each step.
    
    Wraps DSPy's ChainOfThought with input/output governance checks.
    
    Example:
        from nethical.integrations.agent_frameworks import GovernedChainOfThought
        
        cot = GovernedChainOfThought("question -> answer")
        
        result = cot(question="What is AI safety?")
        print(result.answer)
    """
    
    def __init__(
        self,
        signature: str,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the governed chain of thought.
        
        Args:
            signature: DSPy signature string (e.g., "question -> answer")
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        self.signature = signature
        self.block_threshold = block_threshold
        self.storage_dir = storage_dir
        
        self._cot = None
        self._governance_check = NethicalModule(
            block_threshold=block_threshold,
            storage_dir=storage_dir
        )
    
    @property
    def cot(self):
        """Get or create the ChainOfThought module."""
        if self._cot is None and DSPY_AVAILABLE:
            self._cot = dspy.ChainOfThought(self.signature)
        return self._cot
    
    def forward(self, **kwargs) -> Any:
        """Execute chain of thought with governance.
        
        Args:
            **kwargs: Arguments matching the signature
            
        Returns:
            DSPy Prediction or blocked response
        """
        return self(**kwargs)
    
    def __call__(self, **kwargs) -> Any:
        """Execute chain of thought with governance checks.
        
        Args:
            **kwargs: Arguments matching the signature
            
        Returns:
            DSPy Prediction or blocked response
        """
        if not DSPY_AVAILABLE or self.cot is None:
            return self._create_blocked_prediction("DSPy not available")
        
        # Check input
        input_str = str(kwargs)
        input_check = self._governance_check.check(input_str, "user_input")
        
        if not input_check["allowed"]:
            return self._create_blocked_prediction(
                f"Input blocked: {input_check['reason']}"
            )
        
        # Execute CoT
        result = self.cot(**kwargs)
        
        # Check output
        answer = getattr(result, 'answer', str(result))
        output_check = self._governance_check.check(answer, "generated_content")
        
        if not output_check["allowed"]:
            # Modify the answer to indicate filtering
            if hasattr(result, 'answer'):
                result.answer = f"[FILTERED]: {output_check['reason']}"
        
        return result
    
    def _create_blocked_prediction(self, reason: str) -> Any:
        """Create a blocked prediction response.
        
        Args:
            reason: Reason for blocking
            
        Returns:
            DSPy Prediction or dict
        """
        if DSPY_AVAILABLE:
            return dspy.Prediction(
                rationale=reason,
                answer="[BLOCKED]"
            )
        
        return {"rationale": reason, "answer": "[BLOCKED]"}


class GovernedPredict:
    """DSPy Predict module with governance.
    
    Example:
        pred = GovernedPredict("question -> answer")
        result = pred(question="What is the meaning of life?")
    """
    
    def __init__(
        self,
        signature: str,
        block_threshold: float = 0.7,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize governed predict.
        
        Args:
            signature: DSPy signature string
            block_threshold: Risk threshold for blocking
            storage_dir: Directory for Nethical data storage
        """
        self.signature = signature
        self.block_threshold = block_threshold
        self.storage_dir = storage_dir
        
        self._predict = None
        self._governance = NethicalModule(
            block_threshold=block_threshold,
            storage_dir=storage_dir
        )
    
    @property
    def predict(self):
        """Get or create the Predict module."""
        if self._predict is None and DSPY_AVAILABLE:
            self._predict = dspy.Predict(self.signature)
        return self._predict
    
    def __call__(self, **kwargs) -> Any:
        """Execute prediction with governance.
        
        Args:
            **kwargs: Arguments matching the signature
            
        Returns:
            DSPy Prediction or blocked response
        """
        if not DSPY_AVAILABLE or self.predict is None:
            return {"error": "DSPy not available", "answer": "[BLOCKED]"}
        
        # Check input
        input_str = str(kwargs)
        input_check = self._governance.check(input_str, "user_input")
        
        if not input_check["allowed"]:
            if DSPY_AVAILABLE:
                return dspy.Prediction(answer=f"[BLOCKED]: {input_check['reason']}")
            return {"answer": f"[BLOCKED]: {input_check['reason']}"}
        
        # Execute prediction
        result = self.predict(**kwargs)
        
        # Check output
        answer = getattr(result, 'answer', str(result))
        output_check = self._governance.check(answer, "generated_content")
        
        if not output_check["allowed"] and hasattr(result, 'answer'):
            result.answer = f"[FILTERED]: {output_check['reason']}"
        
        return result


class DSPyFramework(AgentFrameworkBase):
    """DSPy framework integration with Nethical.
    
    Provides governance modules and utilities for DSPy programs.
    """
    
    def __init__(self, **kwargs):
        """Initialize the DSPy framework integration."""
        super().__init__(agent_id="dspy-framework", **kwargs)
    
    def get_tool(self) -> NethicalModule:
        """Get a DSPy-compatible governance module.
        
        Returns:
            NethicalModule instance
        """
        return NethicalModule(
            block_threshold=self.block_threshold,
            restrict_threshold=self.restrict_threshold,
            storage_dir=self.storage_dir
        )
    
    def create_governed_cot(self, signature: str) -> GovernedChainOfThought:
        """Create a governed chain of thought module.
        
        Args:
            signature: DSPy signature string
            
        Returns:
            GovernedChainOfThought instance
        """
        return GovernedChainOfThought(
            signature=signature,
            block_threshold=self.block_threshold,
            storage_dir=self.storage_dir
        )
    
    def create_governed_predict(self, signature: str) -> GovernedPredict:
        """Create a governed predict module.
        
        Args:
            signature: DSPy signature string
            
        Returns:
            GovernedPredict instance
        """
        return GovernedPredict(
            signature=signature,
            block_threshold=self.block_threshold,
            storage_dir=self.storage_dir
        )
