# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Database module for Nethical.

Provides database connectivity, models, and persistence layer for:
- Agents
- Policies
- Audit logs
- Users and roles (RBAC)
"""

from __future__ import annotations

__all__ = [
    "get_db",
    "get_async_db",
    "init_db",
    "init_async_db",
    "Base",
    "Agent",
    "Policy",
    "AuditLog",
    "User",
    "Tenant",
    "ApiKey",
    "RevokedToken",
    "SessionLocal",
    "AsyncSessionLocal",
    "engine",
    "async_engine",
]

try:
    from .database import (
        AsyncSessionLocal,
        SessionLocal,
        async_engine,
        engine,
        get_async_db,
        get_db,
        init_async_db,
        init_db,
    )
    from .models import (
        Agent,
        ApiKey,
        AuditLog,
        Base,
        Policy,
        RevokedToken,
        Tenant,
        User,
    )
except ImportError:
    # Graceful fallback if SQLAlchemy not installed
    SessionLocal = None
    AsyncSessionLocal = None
    engine = None
    async_engine = None
    get_db = None
    get_async_db = None
    init_db = None
    init_async_db = None
    Base = None
    Agent = None
    Policy = None
    AuditLog = None
    User = None
    Tenant = None
    ApiKey = None
    RevokedToken = None

