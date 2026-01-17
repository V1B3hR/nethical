"""Database models for Nethical."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    """User model for RBAC."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(String(50), nullable=False, default="operator", index=True)  # admin, auditor, operator
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Agent(Base):
    """Agent model for configuration management."""
    
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    agent_type = Column(String(100), default="general")
    description = Column(Text)
    trust_level = Column(Float, default=0.5)
    status = Column(String(50), default="active", index=True)  # active, suspended, terminated, quarantine
    configuration = Column(JSON, default=dict)
    meta_data = Column(JSON, default=dict)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    region_id = Column(String(50))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    created_by = Column(String(255))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self.description,
            "trust_level": self.trust_level,
            "status": self.status,
            "configuration": self.configuration,
            "metadata": self.meta_data,  # Expose as 'metadata' in API
            "region_id": self.region_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
        }


class Policy(Base):
    """Policy model for governance rules management."""
    
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, index=True)
    policy_id = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    policy_type = Column(String(100), default="governance")
    priority = Column(Integer, default=100)
    status = Column(String(50), default="active", index=True)  # active, deprecated, quarantine
    rules = Column(JSON, default=list)
    scope = Column(String(100), default="global")
    fundamental_laws = Column(JSON, default=list)
    meta_data = Column(JSON, default=dict)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    activated_at = Column(DateTime)
    deprecated_at = Column(DateTime)
    created_by = Column(String(255))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "policy_type": self.policy_type,
            "priority": self.priority,
            "status": self.status,
            "rules": self.rules,
            "scope": self.scope,
            "fundamental_laws": self.fundamental_laws,
            "metadata": self.meta_data,  # Expose as 'metadata' in API
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "deprecated_at": self.deprecated_at.isoformat() if self.deprecated_at else None,
            "created_by": self.created_by,
        }


class AuditLog(Base):
    """Audit log model for immutable event tracking."""
    
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)  # decision, policy_change, threat_detected, etc.
    agent_id = Column(String(255), index=True)
    action = Column(String(255))
    outcome = Column(String(100))
    threat_type = Column(String(100), index=True)
    threat_level = Column(String(50), index=True)  # low, medium, high, critical
    risk_score = Column(Float)
    details = Column(JSON, default=dict)
    merkle_hash = Column(String(128))
    previous_hash = Column(String(128))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    verified = Column(Boolean, default=True)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "log_id": self.log_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "action": self.action,
            "outcome": self.outcome,
            "threat_type": self.threat_type,
            "threat_level": self.threat_level,
            "risk_score": self.risk_score,
            "details": self.details,
            "merkle_hash": self.merkle_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "verified": self.verified,
        }
