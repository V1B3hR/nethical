# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

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

try:
    from .taxonomy_api import TaxonomyAPI
except ImportError:
    TaxonomyAPI = None

try:
    from .explainability_api import ExplainabilityAPI
except ImportError:
    ExplainabilityAPI = None

try:
    from .hitl_api import HITLReviewAPI
except ImportError:
    HITLReviewAPI = None

try:
    from .kill_switch_api import router as kill_switch_router
except ImportError:
    kill_switch_router = None

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

# Core FastAPI application and global state instances
try:
    import sys
    if "nethical.api.app" in sys.modules:
        import importlib
        importlib.reload(sys.modules["nethical.api.app"])
    from .app import (
        app,
        API_VERSION,
        rbac_manager_instance,
        tenant_manager_instance,
        gateway_instance,
    )
except Exception:
    app = None
    API_VERSION = "2.3.0"
    rbac_manager_instance = None
    tenant_manager_instance = None
    gateway_instance = None

__all__ = [
    # Core API components
    "TaxonomyAPI",
    "ExplainabilityAPI",
    "HITLReviewAPI",
    "kill_switch_router",
    "app",
    "API_VERSION",
    "rbac_manager_instance",
    "tenant_manager_instance",
    "gateway_instance",
    # v2 API
    "create_v2_app",
    "v2_router",
    # Middleware
    "RequestContextMiddleware",
    "ResponseHeadersMiddleware",
    "ErrorHandlerMiddleware",
]
