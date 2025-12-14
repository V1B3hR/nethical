"""Observability integrations for Nethical governance.

This module provides integration with multiple ML observability platforms:
- Langfuse: LLM tracing and monitoring
- LangSmith: LangChain observability
- Arize AI: ML monitoring and explainability
- WhyLabs: Data and ML monitoring
- Helicone: LLM observability
- TruLens: LLM evaluation and monitoring
"""

from typing import Dict, Any, List, Optional
import logging

from .base import ObservabilityProvider, TraceSpan, GovernanceMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "ObservabilityProvider",
    "TraceSpan",
    "GovernanceMetrics",
    "ObservabilityManager",
    "create_observability_stack",
]

# Import connectors with graceful fallback
try:
    from .langfuse_connector import LangfuseConnector, NethicalLangfuseCallback
    LANGFUSE_AVAILABLE = True
    __all__.extend(["LangfuseConnector", "NethicalLangfuseCallback"])
except ImportError:
    LANGFUSE_AVAILABLE = False
    LangfuseConnector = None

try:
    from .langsmith_connector import LangSmithConnector
    LANGSMITH_AVAILABLE = True
    __all__.append("LangSmithConnector")
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithConnector = None

try:
    from .arize_connector import ArizeConnector
    ARIZE_AVAILABLE = True
    __all__.append("ArizeConnector")
except ImportError:
    ARIZE_AVAILABLE = False
    ArizeConnector = None

try:
    from .whylabs_connector import WhyLabsConnector
    WHYLABS_AVAILABLE = True
    __all__.append("WhyLabsConnector")
except ImportError:
    WHYLABS_AVAILABLE = False
    WhyLabsConnector = None

try:
    from .helicone_connector import HeliconeConnector
    HELICONE_AVAILABLE = True
    __all__.append("HeliconeConnector")
except ImportError:
    HELICONE_AVAILABLE = False
    HeliconeConnector = None

try:
    from .trulens_connector import TruLensConnector
    TRULENS_AVAILABLE = True
    __all__.append("TruLensConnector")
except ImportError:
    TRULENS_AVAILABLE = False
    TruLensConnector = None


class ObservabilityManager:
    """Manage multiple observability providers."""
    
    def __init__(self):
        """Initialize observability manager."""
        self.providers: Dict[str, ObservabilityProvider] = {}
        logger.info("ObservabilityManager initialized")
    
    def add_provider(self, name: str, provider: ObservabilityProvider) -> None:
        """Add an observability provider.
        
        Args:
            name: Provider name (for reference)
            provider: ObservabilityProvider instance
        """
        self.providers[name] = provider
        logger.info(f"Added observability provider: {name}")
    
    def remove_provider(self, name: str) -> None:
        """Remove an observability provider.
        
        Args:
            name: Provider name
        """
        if name in self.providers:
            self.providers.pop(name)
            logger.info(f"Removed observability provider: {name}")
    
    def log_trace_all(self, span: TraceSpan) -> None:
        """Log trace to all providers.
        
        Args:
            span: TraceSpan to log
        """
        for name, provider in self.providers.items():
            try:
                provider.log_trace(span)
            except Exception as e:
                logger.error(f"Failed to log trace to {name}: {e}")
    
    def log_governance_event_all(
        self,
        action: str,
        decision: str,
        risk_score: float,
        metadata: Dict[str, Any]
    ) -> None:
        """Log governance event to all providers.
        
        Args:
            action: The action being evaluated
            decision: Governance decision (ALLOW, BLOCK, RESTRICT)
            risk_score: Risk score (0.0-1.0)
            metadata: Additional event metadata
        """
        for name, provider in self.providers.items():
            try:
                provider.log_governance_event(
                    action=action,
                    decision=decision,
                    risk_score=risk_score,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to log governance event to {name}: {e}")
    
    def log_metrics_all(self, metrics: GovernanceMetrics) -> None:
        """Log metrics to all providers.
        
        Args:
            metrics: GovernanceMetrics to log
        """
        for name, provider in self.providers.items():
            try:
                provider.log_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to log metrics to {name}: {e}")
    
    def flush_all(self) -> None:
        """Flush all providers."""
        for name, provider in self.providers.items():
            try:
                provider.flush()
            except Exception as e:
                logger.error(f"Failed to flush {name}: {e}")
    
    def close_all(self) -> None:
        """Close all providers."""
        for name, provider in self.providers.items():
            try:
                provider.close()
            except Exception as e:
                logger.error(f"Failed to close {name}: {e}")
    
    def get_provider(self, name: str) -> Optional[ObservabilityProvider]:
        """Get a specific provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            ObservabilityProvider or None if not found
        """
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all provider names.
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())


def create_observability_stack(
    langfuse_config: Optional[Dict[str, str]] = None,
    langsmith_config: Optional[Dict[str, str]] = None,
    arize_config: Optional[Dict[str, str]] = None,
    whylabs_config: Optional[Dict[str, str]] = None,
    helicone_config: Optional[Dict[str, str]] = None,
    trulens_config: Optional[Dict[str, Any]] = None
) -> ObservabilityManager:
    """Create a pre-configured observability stack.
    
    Args:
        langfuse_config: Langfuse configuration (public_key, secret_key, host)
        langsmith_config: LangSmith configuration (api_key, project_name)
        arize_config: Arize configuration (api_key, space_key, model_id, model_version)
        whylabs_config: WhyLabs configuration (api_key, org_id, dataset_id)
        helicone_config: Helicone configuration (api_key, base_url)
        trulens_config: TruLens configuration (database_url, app_id)
    
    Returns:
        Configured ObservabilityManager
        
    Example:
        >>> manager = create_observability_stack(
        ...     langfuse_config={
        ...         "public_key": "pk-...",
        ...         "secret_key": "sk-...",
        ...         "host": "https://cloud.langfuse.com"
        ...     },
        ...     langsmith_config={
        ...         "api_key": "ls-...",
        ...         "project_name": "my-project"
        ...     }
        ... )
        >>> manager.log_governance_event_all(
        ...     action="test action",
        ...     decision="ALLOW",
        ...     risk_score=0.1,
        ...     metadata={}
        ... )
    """
    manager = ObservabilityManager()
    
    if langfuse_config and LANGFUSE_AVAILABLE and LangfuseConnector:
        try:
            connector = LangfuseConnector(**langfuse_config)
            manager.add_provider("langfuse", connector)
            logger.info("Added Langfuse to observability stack")
        except Exception as e:
            logger.error(f"Failed to add Langfuse: {e}")
    
    if langsmith_config and LANGSMITH_AVAILABLE and LangSmithConnector:
        try:
            connector = LangSmithConnector(**langsmith_config)
            manager.add_provider("langsmith", connector)
            logger.info("Added LangSmith to observability stack")
        except Exception as e:
            logger.error(f"Failed to add LangSmith: {e}")
    
    if arize_config and ARIZE_AVAILABLE and ArizeConnector:
        try:
            connector = ArizeConnector(**arize_config)
            manager.add_provider("arize", connector)
            logger.info("Added Arize to observability stack")
        except Exception as e:
            logger.error(f"Failed to add Arize: {e}")
    
    if whylabs_config and WHYLABS_AVAILABLE and WhyLabsConnector:
        try:
            connector = WhyLabsConnector(**whylabs_config)
            manager.add_provider("whylabs", connector)
            logger.info("Added WhyLabs to observability stack")
        except Exception as e:
            logger.error(f"Failed to add WhyLabs: {e}")
    
    if helicone_config and HELICONE_AVAILABLE and HeliconeConnector:
        try:
            connector = HeliconeConnector(**helicone_config)
            manager.add_provider("helicone", connector)
            logger.info("Added Helicone to observability stack")
        except Exception as e:
            logger.error(f"Failed to add Helicone: {e}")
    
    if trulens_config and TRULENS_AVAILABLE and TruLensConnector:
        try:
            connector = TruLensConnector(**trulens_config)
            manager.add_provider("trulens", connector)
            logger.info("Added TruLens to observability stack")
        except Exception as e:
            logger.error(f"Failed to add TruLens: {e}")
    
    return manager


def get_observability_info() -> Dict[str, Any]:
    """Get information about available observability integrations.
    
    Returns:
        Dictionary with integration availability
    """
    return {
        "langfuse": {
            "available": LANGFUSE_AVAILABLE,
            "setup": "pip install langfuse",
            "docs": "https://langfuse.com/docs"
        },
        "langsmith": {
            "available": LANGSMITH_AVAILABLE,
            "setup": "pip install langsmith",
            "docs": "https://docs.smith.langchain.com/"
        },
        "arize": {
            "available": ARIZE_AVAILABLE,
            "setup": "pip install arize",
            "docs": "https://docs.arize.com/"
        },
        "whylabs": {
            "available": WHYLABS_AVAILABLE,
            "setup": "pip install whylogs",
            "docs": "https://docs.whylabs.ai/"
        },
        "helicone": {
            "available": HELICONE_AVAILABLE,
            "setup": "API only, no SDK required",
            "docs": "https://docs.helicone.ai/"
        },
        "trulens": {
            "available": TRULENS_AVAILABLE,
            "setup": "pip install trulens-eval",
            "docs": "https://www.trulens.org/trulens_eval/getting_started/"
        }
    }


__all__.append("get_observability_info")


if __name__ == "__main__":
    info = get_observability_info()
    print("Nethical Observability Integrations:")
    for name, details in info.items():
        status = "✓ Available" if details["available"] else "✗ Not Available"
        print(f"\n{name.upper()}: {status}")
        print(f"  Setup: {details['setup']}")
        print(f"  Docs: {details['docs']}")
