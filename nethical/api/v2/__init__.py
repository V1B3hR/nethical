"""Nethical Governance API v2.

Enhanced REST API with full governance features, latency metrics,
batch processing, and comprehensive audit capabilities.

This module implements Phase 2 of the Nethical roadmap:
- POST /v2/evaluate - Enhanced evaluation with latency metrics
- POST /v2/batch-evaluate - Batch processing
- GET /v2/decisions/{id} - Decision lookup
- POST /v2/policies - Policy management
- GET /v2/policies - List policies
- GET /v2/metrics - Prometheus metrics
- GET /v2/fairness - Fairness metrics
- POST /v2/appeals - Appeals submission
- GET /v2/audit/{id} - Audit trail lookup

Adheres to the 25 Fundamental Laws of AI Ethics.
"""

from .app import create_v2_app, router

__all__ = ["create_v2_app", "router"]
