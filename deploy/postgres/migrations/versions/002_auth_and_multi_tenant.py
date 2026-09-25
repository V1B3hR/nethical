# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Migration for Multi-Tenant Isolation, Users, MFA, and Token Blacklist.

Revision ID: 002_auth_and_multi_tenant
Revises: 001_initial_schema
Create Date: 2026-09-25

This migration synchronizes PostgreSQL schema with Nethical's unified ORM:
- Creates `tenants` table for sovereign multi-tenant isolation
- Creates `users` table for RBAC, MFA secrets, and audit attribution
- Creates `revoked_tokens` table for permanent JWT blacklist persistence
- Adds `tenant_id` foreign identifier to `agents`, `policy_versions`, and `audit_events`
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Revision identifiers
revision = "002_auth_and_multi_tenant"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply migration: add tenants, users, revoked_tokens, and tenant_id."""
    
    # 1. Create tenants table
    op.create_table(
        "tenants",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("jurisdiction", sa.String(50), server_default="GLOBAL"),
        sa.Column("classification", sa.String(50), server_default="UNCLASSIFIED"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("config", JSONB, server_default="{}"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        schema="nethical",
    )
    op.create_index("idx_tenants_tenant_id", "tenants", ["tenant_id"], schema="nethical")
    op.create_index("idx_tenants_jurisdiction", "tenants", ["jurisdiction"], schema="nethical")

    # Seed default tenant
    op.execute("""
        INSERT INTO nethical.tenants (tenant_id, name, description, jurisdiction, classification)
        VALUES ('default_tenant', 'Nethical Global Sovereign Node', 'Primary default tenant', 'GLOBAL', 'UNCLASSIFIED')
        ON CONFLICT (tenant_id) DO NOTHING
    """)

    # 2. Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("role", sa.String(50), nullable=False, server_default="operator"),
        sa.Column("tenant_id", sa.String(100), nullable=False, server_default="default_tenant"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("mfa_enabled", sa.Boolean, server_default="false"),
        sa.Column("mfa_secret", sa.String(255)),
        sa.Column("mfa_backup_codes", JSONB, server_default="[]"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        schema="nethical",
    )
    op.create_index("idx_users_username", "users", ["username"], schema="nethical")
    op.create_index("idx_users_email", "users", ["email"], schema="nethical")
    op.create_index("idx_users_tenant_id", "users", ["tenant_id"], schema="nethical")

    # 3. Create revoked_tokens table
    op.create_table(
        "revoked_tokens",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("jti", sa.String(128), nullable=False, unique=True),
        sa.Column("token_type", sa.String(50), server_default="access"),
        sa.Column("user_id", sa.String(255)),
        sa.Column("revoked_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("reason", sa.String(255)),
        schema="nethical",
    )
    op.create_index("idx_revoked_tokens_jti", "revoked_tokens", ["jti"], schema="nethical")
    op.create_index("idx_revoked_tokens_expires_at", "revoked_tokens", ["expires_at"], schema="nethical")

    # 4. Add tenant_id to existing tables
    op.add_column("agents", sa.Column("tenant_id", sa.String(100), server_default="default_tenant"), schema="nethical")
    op.create_index("idx_agents_tenant_id", "agents", ["tenant_id"], schema="nethical")

    op.add_column("policy_versions", sa.Column("tenant_id", sa.String(100), server_default="default_tenant"), schema="nethical")
    op.create_index("idx_policy_versions_tenant_id", "policy_versions", ["tenant_id"], schema="nethical")

    op.add_column("audit_events", sa.Column("tenant_id", sa.String(100), server_default="default_tenant"), schema="nethical")
    op.create_index("idx_audit_events_tenant_id", "audit_events", ["tenant_id"], schema="nethical")


def downgrade() -> None:
    """Reverse migration: remove tenant_id columns and drop new tables."""
    op.drop_column("audit_events", "tenant_id", schema="nethical")
    op.drop_column("policy_versions", "tenant_id", schema="nethical")
    op.drop_column("agents", "tenant_id", schema="nethical")
    
    op.drop_table("revoked_tokens", schema="nethical")
    op.drop_table("users", schema="nethical")
    op.drop_table("tenants", schema="nethical")
