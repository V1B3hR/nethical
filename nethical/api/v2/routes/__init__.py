"""API v2 route modules.

Each module handles a specific domain of the API:
- evaluate: Action evaluation with latency metrics
- decisions: Decision lookup and history
- policies: Policy CRUD operations
- metrics: Prometheus-compatible metrics
- fairness: Fairness metrics and bias detection
- appeals: Appeal submission and tracking
- audit: Audit trail access

All routes implement the 25 Fundamental Laws of AI Ethics.
"""

from . import appeals, audit, decisions, evaluate, fairness, metrics, policies

__all__ = [
    "evaluate",
    "decisions",
    "policies",
    "metrics",
    "fairness",
    "appeals",
    "audit",
]
