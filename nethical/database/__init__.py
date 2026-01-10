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
    "init_db",
    "Base",
    "Agent",
    "Policy",
    "AuditLog",
    "User",
    "SessionLocal",
    "engine",
]

try:
    from .database import SessionLocal, engine, get_db, init_db
    from .models import Agent, AuditLog, Base, Policy, User
except ImportError:
    # Graceful fallback if SQLAlchemy not installed
    SessionLocal = None
    engine = None
    get_db = None
    init_db = None
    Base = None
    Agent = None
    Policy = None
    AuditLog = None
    User = None
