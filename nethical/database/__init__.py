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

from typing import Any, AsyncGenerator, Generator


def _fallback_get_db() -> Generator[Any, None, None]:
    """Fallback generator for database session dependency."""
    yield None


async def _fallback_get_async_db() -> AsyncGenerator[Any, None]:
    """Fallback async generator for async database session dependency."""
    yield None


# Models
try:
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
except (ImportError, ModuleNotFoundError):
    Base = None
    Agent = None
    Policy = None
    AuditLog = None
    User = None
    Tenant = None
    ApiKey = None
    RevokedToken = None

# Database connection & session management
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
except (ImportError, ModuleNotFoundError):
    SessionLocal = None
    AsyncSessionLocal = None
    engine = None
    async_engine = None
    get_db = _fallback_get_db
    get_async_db = _fallback_get_async_db
    init_db = None
    init_async_db = None

if get_db is None:
    get_db = _fallback_get_db

if get_async_db is None:
    get_async_db = _fallback_get_async_db

