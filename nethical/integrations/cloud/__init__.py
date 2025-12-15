"""Cloud ML platform integrations for Nethical governance.

This module provides integration with major cloud ML platforms:
- Google Vertex AI: Complete MLOps platform
- Databricks: Unified analytics platform with MLflow
- Snowflake Cortex: AI/ML functions in Snowflake
"""

from typing import Dict, Any, Optional
import logging

from .base import CloudMLProvider, ExperimentRun, RunStatus

logger = logging.getLogger(__name__)

__all__ = [
    "CloudMLProvider",
    "ExperimentRun",
    "RunStatus",
]

# Import connectors with graceful fallback
try:
    from .vertex_ai_connector import VertexAIConnector
    VERTEX_AI_AVAILABLE = True
    __all__.append("VertexAIConnector")
except ImportError:
    VERTEX_AI_AVAILABLE = False
    VertexAIConnector = None

try:
    from .databricks_connector import DatabricksConnector
    DATABRICKS_AVAILABLE = True
    __all__.append("DatabricksConnector")
except ImportError:
    DATABRICKS_AVAILABLE = False
    DatabricksConnector = None

try:
    from .snowflake_cortex_connector import SnowflakeCortexConnector
    SNOWFLAKE_AVAILABLE = True
    __all__.append("SnowflakeCortexConnector")
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    SnowflakeCortexConnector = None


def get_cloud_integration_info() -> Dict[str, Any]:
    """Get information about available cloud integrations.
    
    Returns:
        Dictionary with integration availability and setup info
    """
    return {
        "vertex_ai": {
            "available": VERTEX_AI_AVAILABLE,
            "setup": "pip install google-cloud-aiplatform",
            "docs": "https://cloud.google.com/vertex-ai/docs",
            "features": ["experiments", "predictions", "endpoints", "governance"]
        },
        "databricks": {
            "available": DATABRICKS_AVAILABLE,
            "setup": "pip install databricks-sdk mlflow",
            "docs": "https://docs.databricks.com/",
            "features": ["mlflow", "serving", "model_registry", "governance"]
        },
        "snowflake_cortex": {
            "available": SNOWFLAKE_AVAILABLE,
            "setup": "pip install snowflake-connector-python",
            "docs": "https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions",
            "features": ["llm_complete", "classify", "sentiment", "governance"]
        }
    }


__all__.append("get_cloud_integration_info")


if __name__ == "__main__":
    info = get_cloud_integration_info()
    print("Nethical Cloud ML Platform Integrations:")
    for name, details in info.items():
        status = "✓ Available" if details["available"] else "✗ Not Available"
        print(f"\n{name.upper()}: {status}")
        print(f"  Setup: {details['setup']}")
        print(f"  Features: {', '.join(details['features'])}")
        print(f"  Docs: {details['docs']}")
