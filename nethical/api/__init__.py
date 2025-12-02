"""API modules for Nethical governance system.

This package provides the HTTP API for Nethical governance:
- v2: Enhanced API with full governance features (Phase 2)
- taxonomy_api: Ethical taxonomy management
- explainability_api: Decision explanation endpoints
- hitl_api: Human-in-the-loop review API
- kill_switch_api: Emergency shutdown capabilities
- middleware: Request/response middleware

All APIs adhere to the 25 Fundamental Laws of AI Ethics.
"""

from .taxonomy_api import TaxonomyAPI
from .explainability_api import ExplainabilityAPI
from .hitl_api import HITLReviewAPI
from .kill_switch_api import router as kill_switch_router

# Import v2 API module
try:
    from .v2 import create_v2_app, router as v2_router
except ImportError:
    create_v2_app = None
    v2_router = None

# Import middleware
try:
    from .middleware import (
        RequestContextMiddleware,
        ResponseHeadersMiddleware,
        ErrorHandlerMiddleware,
    )
except ImportError:
    RequestContextMiddleware = None
    ResponseHeadersMiddleware = None
    ErrorHandlerMiddleware = None

# Re-export app from the main API module (nethical/api.py is shadowed by this package)
# Import the main FastAPI app for convenience
try:
    import importlib.util
    from pathlib import Path

    # Load the api.py module directly since it's shadowed by this package
    api_py_path = Path(__file__).parent.parent / "api.py"
    if api_py_path.exists():
        spec = importlib.util.spec_from_file_location("nethical_api_main", api_py_path)
        if spec and spec.loader:
            api_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_module)
            app = api_module.app
            API_VERSION = api_module.API_VERSION
except Exception:
    app = None
    API_VERSION = "2.3.0"

__all__ = [
    # Core API components
    "TaxonomyAPI",
    "ExplainabilityAPI",
    "HITLReviewAPI",
    "kill_switch_router",
    "app",
    "API_VERSION",
    # v2 API
    "create_v2_app",
    "v2_router",
    # Middleware
    "RequestContextMiddleware",
    "ResponseHeadersMiddleware",
    "ErrorHandlerMiddleware",
]
