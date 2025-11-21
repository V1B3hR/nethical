"""
LangChain Integration for Nethical

This module provides LangChain tool wrappers for integrating Nethical's
IntegratedGovernance system into LangChain agents and workflows.

Features:
- NethicalGuardTool: BaseTool wrapper for IntegratedGovernance
- Pre-action safety checks
- Post-action evaluation
- Configurable decision thresholds
- Optional LlamaGuard integration support

Example:
    from langchain.agents import initialize_agent
    from langchain.llms import OpenAI
    from nethical.integrations.langchain_tools import NethicalGuardTool, create_nethical_agent

    # Create the Nethical guard tool
    guard_tool = NethicalGuardTool(
        storage_dir="./nethical_data",
        block_threshold=0.7
    )

    # Add to LangChain agent
    llm = OpenAI(temperature=0)
    tools = [guard_tool, ... other tools ...]
    agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description")

    # Or use the helper to create a guarded agent
    agent = create_nethical_agent(llm, tools, storage_dir="./nethical_data")
"""

from typing import Any, Dict, List, Optional, Type, Union
import json
from datetime import datetime

try:
    from langchain.tools import BaseTool
    from langchain.callbacks.manager import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from pydantic import BaseModel, Field

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    BaseModel = object
    Field = lambda *args, **kwargs: None
    # Dummy types for when LangChain is not available
    CallbackManagerForToolRun = object
    AsyncCallbackManagerForToolRun = object

from ..core import IntegratedGovernance
from ..core.models import Decision, ActionType, AgentAction


class NethicalGuardInput(BaseModel):
    """Input schema for NethicalGuardTool."""

    action: str = Field(description="The action/content to evaluate for safety and ethics")
    agent_id: str = Field(
        default="default_agent", description="Agent identifier for tracking"
    )
    action_type: str = Field(
        default="query",
        description="Type of action: query, response, function_call, data_access, etc.",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context for the evaluation"
    )


class NethicalGuardTool(BaseTool):
    """LangChain tool wrapper for Nethical IntegratedGovernance.

    This tool allows LangChain agents to check actions for safety, ethics,
    and policy compliance before execution using Nethical's governance system.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LangChain agent
        governance: IntegratedGovernance instance
        block_threshold: Risk score threshold for blocking (0.0-1.0)
        return_detailed_response: Whether to return full governance results
    """

    name: str = "nethical_ethics_guard"
    description: str = (
        "Checks if an AI action is safe and ethical. "
        "Input should be the action/content to evaluate. "
        "Returns ALLOW, BLOCK, WARN, or ESCALATE with reasoning."
    )
    args_schema: Type[BaseModel] = NethicalGuardInput if LANGCHAIN_AVAILABLE else None

    # Nethical-specific attributes
    governance: Optional[Any] = None
    storage_dir: str = "./nethical_data"
    block_threshold: float = 0.7
    warn_threshold: float = 0.4
    return_detailed_response: bool = False

    # Configuration for IntegratedGovernance
    enable_shadow_mode: bool = True
    enable_ml_blending: bool = True
    enable_anomaly_detection: bool = True
    auto_escalate_on_block: bool = False

    if LANGCHAIN_AVAILABLE:

        class Config:
            """Pydantic configuration."""

            arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize the Nethical guard tool.

        Args:
            storage_dir: Directory for Nethical data storage
            block_threshold: Risk score threshold for blocking (default: 0.7)
            warn_threshold: Risk score threshold for warnings (default: 0.4)
            return_detailed_response: Return full governance results
            enable_shadow_mode: Enable ML shadow mode
            enable_ml_blending: Enable ML blending
            enable_anomaly_detection: Enable anomaly detection
            auto_escalate_on_block: Auto-escalate blocked actions
            **kwargs: Additional arguments passed to BaseTool
        """
        # Extract our custom attributes before calling super()
        self.storage_dir = kwargs.pop("storage_dir", self.storage_dir)
        self.block_threshold = kwargs.pop("block_threshold", self.block_threshold)
        self.warn_threshold = kwargs.pop("warn_threshold", self.warn_threshold)
        self.return_detailed_response = kwargs.pop(
            "return_detailed_response", self.return_detailed_response
        )
        self.enable_shadow_mode = kwargs.pop("enable_shadow_mode", self.enable_shadow_mode)
        self.enable_ml_blending = kwargs.pop("enable_ml_blending", self.enable_ml_blending)
        self.enable_anomaly_detection = kwargs.pop(
            "enable_anomaly_detection", self.enable_anomaly_detection
        )
        self.auto_escalate_on_block = kwargs.pop(
            "auto_escalate_on_block", self.auto_escalate_on_block
        )
        self.governance = kwargs.pop("governance", None)

        # Only call super().__init__ if LangChain is available
        if LANGCHAIN_AVAILABLE:
            super().__init__(**kwargs)

        # Initialize IntegratedGovernance if not provided
        if self.governance is None:
            self.governance = IntegratedGovernance(
                storage_dir=self.storage_dir,
                enable_shadow_mode=self.enable_shadow_mode,
                enable_ml_blending=self.enable_ml_blending,
                enable_anomaly_detection=self.enable_anomaly_detection,
                auto_escalate_on_block=self.auto_escalate_on_block,
                enable_merkle_anchoring=False,  # Lighter weight for LangChain
                enable_quarantine=True,
                enable_ethical_taxonomy=True,
                enable_sla_monitoring=False,  # Disable for LangChain unless needed
            )

    def _run(
        self,
        action: str,
        agent_id: str = "default_agent",
        action_type: str = "query",
        context: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Evaluate an action through Nethical governance.

        Args:
            action: The action/content to evaluate
            agent_id: Agent identifier
            action_type: Type of action
            context: Additional context
            run_manager: LangChain callback manager

        Returns:
            Decision string with reasoning (ALLOW, BLOCK, WARN, ESCALATE)
        """
        # Process through governance
        # Note: IntegratedGovernance.process_action expects action to be Any type
        # It will convert to string internally as needed
        try:
            # Create a simple action identifier
            action_id = f"langchain_action_{hash(action) % 10000}"

            result = self.governance.process_action(
                agent_id=agent_id,
                action=action,  # Pass the string content directly
                action_id=action_id,
                action_type=action_type,
                context=context or {},
            )

            # Extract decision based on risk assessment
            decision = self._make_decision(result)

            if self.return_detailed_response:
                return json.dumps(
                    {"decision": decision, "details": result}, indent=2, default=str
                )
            else:
                return self._format_simple_response(decision, result)

        except Exception as e:
            return f"ERROR: Governance evaluation failed: {str(e)}"

    async def _arun(
        self,
        action: str,
        agent_id: str = "default_agent",
        action_type: str = "query",
        context: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async version of _run (currently delegates to sync version).

        Args:
            action: The action/content to evaluate
            agent_id: Agent identifier
            action_type: Type of action
            context: Additional context
            run_manager: LangChain async callback manager

        Returns:
            Decision string with reasoning
        """
        # For now, delegate to sync version
        # Could be optimized with async governance processing in the future
        return self._run(action, agent_id, action_type, context)

    def _make_decision(self, result: Dict[str, Any]) -> str:
        """Make a decision based on governance results.

        Args:
            result: Governance processing result

        Returns:
            Decision string: ALLOW, WARN, BLOCK, or ESCALATE
        """
        # Check if explicitly blocked by quota
        if result.get("blocked_by_quota", False):
            return "BLOCK"

        # Extract risk score from Phase 3 results
        risk_score = 0.0
        phase3_data = result.get("phase3", {})
        if phase3_data:
            risk_score = phase3_data.get("risk_score", 0.0)

        # Check Phase 5-7 ML blending results if available
        phase567_data = result.get("phase567", {})
        if phase567_data:
            # Check blended decision if available
            blended_data = phase567_data.get("blended", {})
            if blended_data:
                zone = blended_data.get("zone", "")
                if zone == "HIGH_RISK":
                    return "BLOCK"
                elif zone == "GRAY":
                    return "WARN"

        # Check Phase 4 quarantine status
        phase4_data = result.get("phase4", {})
        if phase4_data and phase4_data.get("quarantined", False):
            return "BLOCK"

        # Check Phase 8-9 escalation
        phase89_data = result.get("phase89", {})
        if phase89_data and phase89_data.get("escalated", False):
            return "ESCALATE"

        # Default decision based on risk score thresholds
        if risk_score >= self.block_threshold:
            return "BLOCK"
        elif risk_score >= self.warn_threshold:
            return "WARN"
        else:
            return "ALLOW"

    def _format_simple_response(self, decision: str, result: Dict[str, Any]) -> str:
        """Format a simple, human-readable response.

        Args:
            decision: The decision (ALLOW, WARN, BLOCK, ESCALATE)
            result: Full governance result

        Returns:
            Formatted response string
        """
        phase3_data = result.get("phase3", {})
        risk_score = phase3_data.get("risk_score", 0.0)
        risk_tier = phase3_data.get("risk_tier", "UNKNOWN")

        reason_parts = [f"Decision: {decision}"]
        reason_parts.append(f"Risk Score: {risk_score:.2f}")
        reason_parts.append(f"Risk Tier: {risk_tier}")

        # Add violation info if present
        phase4_data = result.get("phase4", {})
        if phase4_data and phase4_data.get("ethical_tags"):
            tags = phase4_data["ethical_tags"]
            if tags:
                reason_parts.append(f"Ethical Tags: {', '.join(tags)}")

        # Add escalation reason if escalated
        phase89_data = result.get("phase89", {})
        if phase89_data and phase89_data.get("escalation_reason"):
            reason_parts.append(f"Reason: {phase89_data['escalation_reason']}")

        return " | ".join(reason_parts)


def create_nethical_agent(
    llm: Any,
    tools: List[Any],
    storage_dir: str = "./nethical_data",
    block_threshold: float = 0.7,
    prepend_guard: bool = True,
    agent_type: str = "zero-shot-react-description",
    verbose: bool = False,
    **agent_kwargs,
) -> Any:
    """Create a LangChain agent with Nethical guard protection.

    This is a convenience function that creates a LangChain agent with
    Nethical governance automatically integrated as a tool.

    Args:
        llm: LangChain LLM instance
        tools: List of LangChain tools
        storage_dir: Directory for Nethical data storage
        block_threshold: Risk threshold for blocking
        prepend_guard: Add guard tool at the beginning of tools list
        agent_type: LangChain agent type
        verbose: Enable verbose output
        **agent_kwargs: Additional arguments for initialize_agent

    Returns:
        LangChain agent with Nethical guard

    Raises:
        ImportError: If LangChain is not installed
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Install with: pip install langchain"
        )

    from langchain.agents import initialize_agent

    # Create Nethical guard tool
    guard_tool = NethicalGuardTool(
        storage_dir=storage_dir,
        block_threshold=block_threshold,
    )

    # Add guard tool to tools list
    if prepend_guard:
        all_tools = [guard_tool] + tools
    else:
        all_tools = tools + [guard_tool]

    # Create agent
    agent = initialize_agent(
        all_tools, llm, agent_type=agent_type, verbose=verbose, **agent_kwargs
    )

    return agent


class LlamaGuardChain:
    """Optional LlamaGuard integration for additional content moderation.

    This class provides a wrapper for LlamaGuard models to work alongside
    Nethical's governance. It can be used as a pre-filter or post-filter.

    Example:
        from nethical.integrations.langchain_tools import LlamaGuardChain

        guard = LlamaGuardChain(
            model_id="meta-llama/LlamaGuard-3-8B",
            use_local=True  # or False for HuggingFace API
        )

        if guard.is_safe(user_input):
            # Process with Nethical
            pass
    """

    def __init__(
        self,
        model_id: str = "meta-llama/LlamaGuard-3-8B",
        use_local: bool = False,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        """Initialize LlamaGuard chain.

        Args:
            model_id: HuggingFace model ID
            use_local: Use local model (requires transformers/torch)
            max_new_tokens: Maximum tokens for generation
            temperature: Sampling temperature
        """
        self.model_id = model_id
        self.use_local = use_local
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.chain = None

        if use_local:
            self._init_local_chain()
        else:
            self._init_api_chain()

    def _init_local_chain(self):
        """Initialize local LlamaGuard model."""
        try:
            from langchain.llms import HuggingFacePipeline
            from langchain.prompts import PromptTemplate
            from langchain.chains import LLMChain

            # Create HuggingFace pipeline
            llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                },
            )

            # Create prompt template
            prompt = PromptTemplate.from_template(
                "[INST] Task: Check if the following content is safe. "
                "Return 'safe' if appropriate or 'unsafe' with reason if not.\n\n"
                "Content: {input}\n[/INST]"
            )

            # Create chain
            self.chain = LLMChain(llm=llm, prompt=prompt)

        except ImportError as e:
            raise ImportError(
                f"Local LlamaGuard requires transformers and torch: {e}"
            )

    def _init_api_chain(self):
        """Initialize API-based LlamaGuard chain."""
        # Placeholder for API-based implementation
        # Could use HuggingFace Inference API or other hosted services
        raise NotImplementedError(
            "API-based LlamaGuard not yet implemented. Use use_local=True"
        )

    def is_safe(self, text: str) -> bool:
        """Check if text is safe according to LlamaGuard.

        Args:
            text: Text to evaluate

        Returns:
            True if safe, False if unsafe
        """
        if self.chain is None:
            raise RuntimeError("LlamaGuard chain not initialized")

        try:
            result = self.chain.run(input=text)
            return "safe" in result.lower() and "unsafe" not in result.lower()
        except Exception as e:
            # If evaluation fails, err on the side of caution
            print(f"LlamaGuard evaluation failed: {e}")
            return False

    def evaluate(self, text: str) -> Dict[str, Any]:
        """Get detailed evaluation from LlamaGuard.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with 'safe' boolean and 'reason' if unsafe
        """
        if self.chain is None:
            raise RuntimeError("LlamaGuard chain not initialized")

        try:
            result = self.chain.run(input=text)
            is_safe = "safe" in result.lower() and "unsafe" not in result.lower()

            return {
                "safe": is_safe,
                "raw_response": result,
                "reason": result if not is_safe else None,
            }
        except Exception as e:
            return {"safe": False, "raw_response": None, "reason": f"Error: {e}"}


def chain_guards(
    nethical_tool: NethicalGuardTool,
    action: str,
    agent_id: str = "default_agent",
    llama_guard: Optional[LlamaGuardChain] = None,
) -> Dict[str, Any]:
    """Chain multiple guards together for comprehensive safety checking.

    This function runs both Nethical and LlamaGuard (if provided) on an action
    and combines their results for maximum safety.

    Args:
        nethical_tool: Nethical guard tool instance
        action: Action to evaluate
        agent_id: Agent identifier
        llama_guard: Optional LlamaGuard chain instance

    Returns:
        Combined evaluation results with final decision
    """
    results = {"nethical": None, "llama_guard": None, "final_decision": "ALLOW"}

    # First check with LlamaGuard if available (fast pre-filter)
    if llama_guard:
        llama_result = llama_guard.evaluate(action)
        results["llama_guard"] = llama_result

        if not llama_result["safe"]:
            results["final_decision"] = "BLOCK"
            results["blocked_by"] = "llama_guard"
            results["reason"] = llama_result["reason"]
            return results

    # Then check with Nethical (comprehensive governance)
    nethical_result = nethical_tool._run(action=action, agent_id=agent_id)
    results["nethical"] = nethical_result

    # Parse Nethical decision
    if "BLOCK" in nethical_result or "ESCALATE" in nethical_result:
        results["final_decision"] = "BLOCK" if "BLOCK" in nethical_result else "ESCALATE"
        results["blocked_by"] = "nethical"
    elif "WARN" in nethical_result:
        results["final_decision"] = "WARN"

    return results


# Convenience check for LangChain availability
if not LANGCHAIN_AVAILABLE:
    import warnings

    warnings.warn(
        "LangChain is not installed. LangChain integration features will not be available. "
        "Install with: pip install langchain",
        ImportWarning,
    )
