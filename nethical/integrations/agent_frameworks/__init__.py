"""
Agent Framework integrations with Nethical governance.

This package provides governed wrappers for various agent frameworks:

- LlamaIndex: Tools and query engine wrappers
- CrewAI: Agent and task wrappers
- DSPy: Governed modules and chains
- AutoGen: Agent and conversation wrappers

All integrations include:
- Pre/post execution governance checks
- Risk scoring and blocking
- Framework-specific tool definitions
- Configurable thresholds

Example:
    # LlamaIndex
    from nethical.integrations.agent_frameworks import (
        NethicalLlamaIndexTool,
        NethicalQueryEngine
    )
    
    # CrewAI
    from nethical.integrations.agent_frameworks import (
        NethicalCrewAITool,
        NethicalAgentWrapper
    )
    
    # DSPy
    from nethical.integrations.agent_frameworks import (
        NethicalModule,
        GovernedChainOfThought
    )
    
    # AutoGen
    from nethical.integrations.agent_frameworks import (
        NethicalAutoGenTool,
        NethicalConversableAgent
    )
"""

from .base import (
    AgentFrameworkBase,
    AgentWrapper,
    GovernanceDecision,
    GovernanceResult,
)

# LlamaIndex imports
try:
    from .llamaindex_tools import (
        NethicalLlamaIndexTool,
        NethicalQueryEngine,
        LlamaIndexFramework,
        create_safe_index,
        LLAMAINDEX_AVAILABLE,
    )
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    NethicalLlamaIndexTool = None
    NethicalQueryEngine = None
    LlamaIndexFramework = None
    create_safe_index = None

# CrewAI imports
try:
    from .crewai_tools import (
        NethicalCrewAITool,
        NethicalAgentWrapper,
        CrewAIFramework,
        CREWAI_AVAILABLE,
        get_nethical_tool as get_crewai_tool,
        handle_nethical_tool as handle_crewai_tool,
    )
except ImportError:
    CREWAI_AVAILABLE = False
    NethicalCrewAITool = None
    NethicalAgentWrapper = None
    CrewAIFramework = None

# DSPy imports
try:
    from .dspy_tools import (
        NethicalModule,
        GovernedChainOfThought,
        GovernedPredict,
        DSPyFramework,
        DSPY_AVAILABLE,
    )
except ImportError:
    DSPY_AVAILABLE = False
    NethicalModule = None
    GovernedChainOfThought = None
    GovernedPredict = None
    DSPyFramework = None

# AutoGen imports
try:
    from .autogen_tools import (
        NethicalAutoGenTool,
        NethicalConversableAgent,
        GovernedGroupChat,
        AutoGenFramework,
        AUTOGEN_AVAILABLE,
        get_nethical_function,
        handle_nethical_function,
    )
except ImportError:
    AUTOGEN_AVAILABLE = False
    NethicalAutoGenTool = None
    NethicalConversableAgent = None
    GovernedGroupChat = None
    AutoGenFramework = None


__all__ = [
    # Base classes
    "AgentFrameworkBase",
    "AgentWrapper",
    "GovernanceDecision",
    "GovernanceResult",
    # Availability flags
    "LLAMAINDEX_AVAILABLE",
    "CREWAI_AVAILABLE",
    "DSPY_AVAILABLE",
    "AUTOGEN_AVAILABLE",
    # LlamaIndex
    "NethicalLlamaIndexTool",
    "NethicalQueryEngine",
    "LlamaIndexFramework",
    "create_safe_index",
    # CrewAI
    "NethicalCrewAITool",
    "NethicalAgentWrapper",
    "CrewAIFramework",
    # DSPy
    "NethicalModule",
    "GovernedChainOfThought",
    "GovernedPredict",
    "DSPyFramework",
    # AutoGen
    "NethicalAutoGenTool",
    "NethicalConversableAgent",
    "GovernedGroupChat",
    "AutoGenFramework",
]


def get_framework_info():
    """Get information about available agent frameworks.
    
    Returns:
        Dict with framework availability and setup instructions
    """
    return {
        "llamaindex": {
            "available": LLAMAINDEX_AVAILABLE,
            "setup": "pip install llama-index",
            "classes": ["NethicalLlamaIndexTool", "NethicalQueryEngine"],
            "features": ["tools", "query_engine_wrapper", "index_wrapper"]
        },
        "crewai": {
            "available": CREWAI_AVAILABLE,
            "setup": "pip install crewai",
            "classes": ["NethicalCrewAITool", "NethicalAgentWrapper"],
            "features": ["tools", "agent_wrapper", "task_checking"]
        },
        "dspy": {
            "available": DSPY_AVAILABLE,
            "setup": "pip install dspy-ai",
            "classes": ["NethicalModule", "GovernedChainOfThought"],
            "features": ["modules", "governed_cot", "governed_predict"]
        },
        "autogen": {
            "available": AUTOGEN_AVAILABLE,
            "setup": "pip install pyautogen",
            "classes": ["NethicalAutoGenTool", "NethicalConversableAgent"],
            "features": ["function_registration", "agent_wrapper", "group_chat"]
        }
    }
