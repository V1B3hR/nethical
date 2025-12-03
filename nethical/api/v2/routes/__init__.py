"""API v2 route modules.

Each module handles a specific domain of the API:
- evaluate: Action evaluation with latency metrics
- decisions: Decision lookup and history
- policies: Policy CRUD operations
- metrics: Prometheus-compatible metrics
- fairness: Fairness metrics and bias detection
- appeals: Appeal submission and tracking
- audit: Audit trail access
- explanations: GDPR Article 22 Right to Explanation
- human_oversight: EU AI Act Article 14 Human Oversight
- transparency: EU AI Act Article 13 Transparency

All routes implement the 25 Fundamental Laws of AI Ethics.
"""

from . import (
    appeals,
    audit,
    decisions,
    evaluate,
    explanations,
    fairness,
    human_oversight,
    metrics,
    policies,
    transparency,
)

__all__ = [
    "evaluate",
    "decisions",
    "policies",
    "metrics",
    "fairness",
    "appeals",
    "audit",
    "explanations",
    "human_oversight",
    "transparency",
]
