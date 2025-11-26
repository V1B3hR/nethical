"""API modules for Nethical governance system."""

from .taxonomy_api import TaxonomyAPI
from .explainability_api import ExplainabilityAPI
from .hitl_api import HITLReviewAPI

# Re-export app from the main API module (nethical/api.py is shadowed by this package)
# Import the main FastAPI app for convenience
try:
    import sys
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

__all__ = ["TaxonomyAPI", "ExplainabilityAPI", "HITLReviewAPI", "app", "API_VERSION"]
