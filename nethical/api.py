# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Nethical Governance API Module (Compatibility Bridge).

This module re-exports the core FastAPI application and endpoints from `nethical.api.app`
for backward compatibility with legacy imports and standalone deployment entry points.
The canonical location is `nethical.api.app` or `nethical.api`.
"""

from __future__ import annotations

# Forward all symbols from the modular nethical.api.app implementation
from nethical.api.app import *  # noqa: F401, F403
from nethical.api.app import (
    app,
    API_VERSION,
    rbac_manager_instance,
    tenant_manager_instance,
    gateway_instance,
)

__all__ = [
    "app",
    "API_VERSION",
    "rbac_manager_instance",
    "tenant_manager_instance",
    "gateway_instance",
]
