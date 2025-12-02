"""
Initial migration for Nethical core tables.

Revision ID: 001
Revises: None
Create Date: 2025-12-02

This migration creates the core Nethical database schema including:
- Agent management tables
- Model registry tables
- Policy management tables
- Audit trail tables (TimescaleDB hypertables)
- Quota and rate limiting tables
- Security and access control tables
- Metrics and performance tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

# Revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply the migration: create all core tables."""
    
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "timescaledb"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    
    # Create schema
    op.execute('CREATE SCHEMA IF NOT EXISTS nethical')
    
    # Agents table
    op.create_table(
        'agents',
        sa.Column('id', UUID, primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('agent_id', sa.String(255), nullable=False, unique=True),
        sa.Column('name', sa.String(255)),
        sa.Column('agent_type', sa.String(100), server_default='general'),
        sa.Column('trust_level', sa.Float, server_default='0.5'),
        sa.Column('status', sa.String(50), server_default='active'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('metadata', JSONB, server_default='{}'),
        sa.Column('region_id', sa.String(50)),
        sa.Column('logical_domain', sa.String(100), server_default='default'),
        sa.CheckConstraint('trust_level >= 0 AND trust_level <= 1', name='trust_level_range'),
        sa.CheckConstraint("status IN ('active', 'suspended', 'terminated', 'quarantine')", name='valid_status'),
        schema='nethical'
    )
    
    op.create_index('idx_agents_agent_id', 'agents', ['agent_id'], schema='nethical')
    op.create_index('idx_agents_status', 'agents', ['status'], schema='nethical')
    op.create_index('idx_agents_region', 'agents', ['region_id'], schema='nethical')
    
    # Model versions table
    op.create_table(
        'model_versions',
        sa.Column('id', UUID, primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('artifact_path', sa.Text),
        sa.Column('artifact_hash', sa.String(128)),
        sa.Column('model_type', sa.String(100)),
        sa.Column('framework', sa.String(100)),
        sa.Column('metrics', JSONB, server_default='{}'),
        sa.Column('metadata', JSONB, server_default='{}'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('promoted_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('deprecated_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('status', sa.String(50), server_default='staging'),
        sa.Column('created_by', sa.String(255)),
        sa.UniqueConstraint('model_name', 'version', name='unique_model_version'),
        sa.CheckConstraint("status IN ('staging', 'canary', 'production', 'deprecated', 'quarantine')", name='valid_model_status'),
        schema='nethical'
    )
    
    op.create_index('idx_model_versions_name', 'model_versions', ['model_name'], schema='nethical')
    op.create_index('idx_model_versions_status', 'model_versions', ['status'], schema='nethical')
    
    # Policy versions table
    op.create_table(
        'policy_versions',
        sa.Column('id', UUID, primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('policy_id', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('version_hash', sa.String(64), nullable=False),
        sa.Column('content', JSONB, nullable=False),
        sa.Column('policy_type', sa.String(100), server_default='governance'),
        sa.Column('priority', sa.Integer, server_default='100'),
        sa.Column('status', sa.String(20), server_default='quarantine'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('activated_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('deprecated_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('created_by', sa.String(255)),
        sa.Column('approved_by', sa.String(255)),
        sa.Column('metadata', JSONB, server_default='{}'),
        sa.UniqueConstraint('policy_id', 'version', name='unique_policy_version'),
        sa.CheckConstraint("status IN ('quarantine', 'staging', 'active', 'deprecated')", name='valid_policy_status'),
        schema='nethical'
    )
    
    op.create_index('idx_policy_versions_policy_id', 'policy_versions', ['policy_id'], schema='nethical')
    op.create_index('idx_policy_versions_status', 'policy_versions', ['status'], schema='nethical')
    
    # Audit events table (TimescaleDB hypertable)
    op.create_table(
        'audit_events',
        sa.Column('time', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('event_id', UUID, nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('action_type', sa.String(100)),
        sa.Column('action_content', sa.Text),
        sa.Column('decision', sa.String(20), nullable=False),
        sa.Column('risk_score', sa.Float),
        sa.Column('latency_ms', sa.Float),
        sa.Column('violations', JSONB, server_default='[]'),
        sa.Column('policies_applied', JSONB, server_default='[]'),
        sa.Column('context', JSONB, server_default='{}'),
        sa.Column('metadata', JSONB, server_default='{}'),
        sa.Column('region_id', sa.String(50)),
        sa.Column('logical_domain', sa.String(100)),
        sa.Column('request_id', sa.String(255)),
        sa.Column('merkle_hash', sa.String(128)),
        sa.PrimaryKeyConstraint('time', 'event_id'),
        schema='nethical'
    )
    
    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable('nethical.audit_events', 'time', 
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)
    
    op.create_index('idx_audit_events_agent', 'audit_events', ['agent_id', sa.text('time DESC')], schema='nethical')
    op.create_index('idx_audit_events_decision', 'audit_events', ['decision', sa.text('time DESC')], schema='nethical')
    op.create_index('idx_audit_events_request', 'audit_events', ['request_id'], schema='nethical')
    
    # Security events table
    op.create_table(
        'security_events',
        sa.Column('time', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('event_id', UUID, nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('source_ip', sa.String(50)),
        sa.Column('agent_id', sa.String(255)),
        sa.Column('user_id', sa.String(255)),
        sa.Column('description', sa.Text),
        sa.Column('details', JSONB, server_default='{}'),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('resolution_notes', sa.Text),
        sa.PrimaryKeyConstraint('time', 'event_id'),
        schema='nethical'
    )
    
    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable('nethical.security_events', 'time', 
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
    """)
    
    op.create_index('idx_security_events_type', 'security_events', ['event_type', sa.text('time DESC')], schema='nethical')
    op.create_index('idx_security_events_severity', 'security_events', ['severity', sa.text('time DESC')], schema='nethical')
    
    # Governance metrics table
    op.create_table(
        'governance_metrics',
        sa.Column('time', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('agent_id', sa.String(255), nullable=False),
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('metric_value', sa.Float),
        sa.Column('metric_type', sa.String(50), server_default='gauge'),
        sa.Column('tags', JSONB, server_default='{}'),
        sa.Column('region_id', sa.String(50)),
        sa.Column('logical_domain', sa.String(100)),
        sa.PrimaryKeyConstraint('time', 'agent_id', 'metric_name'),
        schema='nethical'
    )
    
    # Convert to hypertable
    op.execute("""
        SELECT create_hypertable('nethical.governance_metrics', 'time', 
            chunk_time_interval => INTERVAL '1 hour',
            if_not_exists => TRUE
        )
    """)
    
    op.create_index('idx_metrics_agent', 'governance_metrics', ['agent_id', sa.text('time DESC')], schema='nethical')
    op.create_index('idx_metrics_name', 'governance_metrics', ['metric_name', sa.text('time DESC')], schema='nethical')
    
    # API keys table
    op.create_table(
        'api_keys',
        sa.Column('id', UUID, primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('key_hash', sa.String(128), nullable=False, unique=True),
        sa.Column('key_prefix', sa.String(10), nullable=False),
        sa.Column('name', sa.String(255)),
        sa.Column('agent_id', sa.String(255)),
        sa.Column('scopes', JSONB, server_default='[]'),
        sa.Column('rate_limit', sa.Integer, server_default='1000'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('last_used_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('revoked_at', sa.TIMESTAMP(timezone=True)),
        sa.Column('revoked_reason', sa.Text),
        sa.Column('metadata', JSONB, server_default='{}'),
        schema='nethical'
    )
    
    op.create_index('idx_api_keys_agent', 'api_keys', ['agent_id'], schema='nethical')
    op.create_index('idx_api_keys_prefix', 'api_keys', ['key_prefix'], schema='nethical')
    
    # Set retention policies
    op.execute("SELECT add_retention_policy('nethical.audit_events', INTERVAL '365 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('nethical.security_events', INTERVAL '730 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('nethical.governance_metrics', INTERVAL '30 days', if_not_exists => TRUE)")


def downgrade() -> None:
    """Reverse the migration: drop all tables."""
    
    # Drop tables in reverse order of dependencies
    op.drop_table('api_keys', schema='nethical')
    op.drop_table('governance_metrics', schema='nethical')
    op.drop_table('security_events', schema='nethical')
    op.drop_table('audit_events', schema='nethical')
    op.drop_table('policy_versions', schema='nethical')
    op.drop_table('model_versions', schema='nethical')
    op.drop_table('agents', schema='nethical')
    
    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS nethical CASCADE')
