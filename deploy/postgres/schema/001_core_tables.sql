-- Nethical Core Database Schema
-- Version: 1.0.0
-- Description: Core tables for the Nethical AI Safety Governance Platform
-- Compatible with: PostgreSQL 14+ with TimescaleDB extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create schema for organization
CREATE SCHEMA IF NOT EXISTS nethical;

SET search_path TO nethical, public;

-- ============================================================================
-- AGENT MANAGEMENT TABLES
-- ============================================================================

-- Agents table: Stores registered AI agents
CREATE TABLE IF NOT EXISTS nethical.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    agent_type VARCHAR(100) DEFAULT 'general',
    trust_level FLOAT DEFAULT 0.5,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    region_id VARCHAR(50),
    logical_domain VARCHAR(100) DEFAULT 'default',
    CONSTRAINT trust_level_range CHECK (trust_level >= 0 AND trust_level <= 1),
    CONSTRAINT valid_status CHECK (status IN ('active', 'suspended', 'terminated', 'quarantine'))
);

CREATE INDEX idx_agents_agent_id ON nethical.agents(agent_id);
CREATE INDEX idx_agents_status ON nethical.agents(status);
CREATE INDEX idx_agents_region ON nethical.agents(region_id);
CREATE INDEX idx_agents_metadata ON nethical.agents USING GIN(metadata);

-- ============================================================================
-- MODEL REGISTRY TABLES
-- ============================================================================

-- Model versions: Tracks all model versions in the registry
CREATE TABLE IF NOT EXISTS nethical.model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    artifact_path TEXT,
    artifact_hash VARCHAR(128),
    model_type VARCHAR(100),
    framework VARCHAR(100),
    metrics JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'staging',
    created_by VARCHAR(255),
    CONSTRAINT unique_model_version UNIQUE (model_name, version),
    CONSTRAINT valid_model_status CHECK (status IN ('staging', 'canary', 'production', 'deprecated', 'quarantine'))
);

CREATE INDEX idx_model_versions_name ON nethical.model_versions(model_name);
CREATE INDEX idx_model_versions_status ON nethical.model_versions(status);
CREATE INDEX idx_model_versions_created ON nethical.model_versions(created_at DESC);

-- Model lineage: Tracks parent-child relationships between models
CREATE TABLE IF NOT EXISTS nethical.model_lineage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_model_id UUID REFERENCES nethical.model_versions(id),
    child_model_id UUID REFERENCES nethical.model_versions(id),
    relationship_type VARCHAR(50) DEFAULT 'derived',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_lineage UNIQUE (parent_model_id, child_model_id)
);

CREATE INDEX idx_model_lineage_parent ON nethical.model_lineage(parent_model_id);
CREATE INDEX idx_model_lineage_child ON nethical.model_lineage(child_model_id);

-- ============================================================================
-- POLICY MANAGEMENT TABLES
-- ============================================================================

-- Policy versions: Stores all policy versions
CREATE TABLE IF NOT EXISTS nethical.policy_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    version_hash VARCHAR(64) NOT NULL,
    content JSONB NOT NULL,
    policy_type VARCHAR(100) DEFAULT 'governance',
    priority INTEGER DEFAULT 100,
    status VARCHAR(20) DEFAULT 'quarantine',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    deprecated_at TIMESTAMPTZ,
    created_by VARCHAR(255),
    approved_by VARCHAR(255),
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_policy_version UNIQUE (policy_id, version),
    CONSTRAINT valid_policy_status CHECK (status IN ('quarantine', 'staging', 'active', 'deprecated'))
);

CREATE INDEX idx_policy_versions_policy_id ON nethical.policy_versions(policy_id);
CREATE INDEX idx_policy_versions_status ON nethical.policy_versions(status);
CREATE INDEX idx_policy_versions_type ON nethical.policy_versions(policy_type);
CREATE INDEX idx_policy_versions_content ON nethical.policy_versions USING GIN(content);

-- Policy assignments: Links policies to agents/domains
CREATE TABLE IF NOT EXISTS nethical.policy_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_version_id UUID REFERENCES nethical.policy_versions(id),
    target_type VARCHAR(50) NOT NULL,  -- 'agent', 'domain', 'region', 'global'
    target_id VARCHAR(255),
    priority INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT unique_assignment UNIQUE (policy_version_id, target_type, target_id)
);

CREATE INDEX idx_policy_assignments_target ON nethical.policy_assignments(target_type, target_id);
CREATE INDEX idx_policy_assignments_policy ON nethical.policy_assignments(policy_version_id);

-- ============================================================================
-- AUDIT TRAIL TABLES (TimescaleDB Hypertables)
-- ============================================================================

-- Audit events: Immutable log of all governance decisions
CREATE TABLE IF NOT EXISTS nethical.audit_events (
    time TIMESTAMPTZ NOT NULL,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    action_type VARCHAR(100),
    action_content TEXT,
    decision VARCHAR(20) NOT NULL,
    risk_score FLOAT,
    latency_ms FLOAT,
    violations JSONB DEFAULT '[]'::jsonb,
    policies_applied JSONB DEFAULT '[]'::jsonb,
    context JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    region_id VARCHAR(50),
    logical_domain VARCHAR(100),
    request_id VARCHAR(255),
    merkle_hash VARCHAR(128),
    PRIMARY KEY (time, event_id)
);

-- Convert to TimescaleDB hypertable for time-series optimization
SELECT create_hypertable('nethical.audit_events', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for common query patterns
CREATE INDEX idx_audit_events_agent ON nethical.audit_events(agent_id, time DESC);
CREATE INDEX idx_audit_events_decision ON nethical.audit_events(decision, time DESC);
CREATE INDEX idx_audit_events_request ON nethical.audit_events(request_id);
CREATE INDEX idx_audit_events_region ON nethical.audit_events(region_id, time DESC);
CREATE INDEX idx_audit_events_merkle ON nethical.audit_events(merkle_hash);

-- ============================================================================
-- QUOTA AND RATE LIMITING TABLES
-- ============================================================================

-- Quota definitions
CREATE TABLE IF NOT EXISTS nethical.quota_definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quota_name VARCHAR(255) NOT NULL UNIQUE,
    quota_type VARCHAR(50) NOT NULL,  -- 'requests', 'tokens', 'cost', 'actions'
    limit_value BIGINT NOT NULL,
    window_seconds INTEGER NOT NULL DEFAULT 3600,
    burst_limit BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Quota assignments to agents
CREATE TABLE IF NOT EXISTS nethical.quota_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    quota_definition_id UUID REFERENCES nethical.quota_definitions(id),
    override_limit BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    CONSTRAINT unique_quota_assignment UNIQUE (agent_id, quota_definition_id)
);

CREATE INDEX idx_quota_assignments_agent ON nethical.quota_assignments(agent_id);

-- Quota usage tracking (hypertable for time-series)
CREATE TABLE IF NOT EXISTS nethical.quota_usage (
    time TIMESTAMPTZ NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    quota_name VARCHAR(255) NOT NULL,
    usage_count BIGINT DEFAULT 0,
    window_start TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (time, agent_id, quota_name)
);

SELECT create_hypertable('nethical.quota_usage', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX idx_quota_usage_agent ON nethical.quota_usage(agent_id, time DESC);

-- ============================================================================
-- SECURITY AND ACCESS CONTROL TABLES
-- ============================================================================

-- API keys
CREATE TABLE IF NOT EXISTS nethical.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash VARCHAR(128) NOT NULL UNIQUE,
    key_prefix VARCHAR(10) NOT NULL,
    name VARCHAR(255),
    agent_id VARCHAR(255),
    scopes JSONB DEFAULT '[]'::jsonb,
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    revoked_reason TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_api_keys_agent ON nethical.api_keys(agent_id);
CREATE INDEX idx_api_keys_prefix ON nethical.api_keys(key_prefix);

-- Security events
CREATE TABLE IF NOT EXISTS nethical.security_events (
    time TIMESTAMPTZ NOT NULL,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    source_ip VARCHAR(50),
    agent_id VARCHAR(255),
    user_id VARCHAR(255),
    description TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,
    PRIMARY KEY (time, event_id)
);

SELECT create_hypertable('nethical.security_events', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX idx_security_events_type ON nethical.security_events(event_type, time DESC);
CREATE INDEX idx_security_events_severity ON nethical.security_events(severity, time DESC);
CREATE INDEX idx_security_events_agent ON nethical.security_events(agent_id, time DESC);

-- ============================================================================
-- METRICS AND PERFORMANCE TABLES
-- ============================================================================

-- Governance metrics (hypertable for time-series)
CREATE TABLE IF NOT EXISTS nethical.governance_metrics (
    time TIMESTAMPTZ NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value DOUBLE PRECISION,
    metric_type VARCHAR(50) DEFAULT 'gauge',  -- 'gauge', 'counter', 'histogram'
    tags JSONB DEFAULT '{}'::jsonb,
    region_id VARCHAR(50),
    logical_domain VARCHAR(100),
    PRIMARY KEY (time, agent_id, metric_name)
);

SELECT create_hypertable('nethical.governance_metrics', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE INDEX idx_metrics_agent ON nethical.governance_metrics(agent_id, time DESC);
CREATE INDEX idx_metrics_name ON nethical.governance_metrics(metric_name, time DESC);
CREATE INDEX idx_metrics_region ON nethical.governance_metrics(region_id, time DESC);

-- Latency tracking
CREATE TABLE IF NOT EXISTS nethical.latency_metrics (
    time TIMESTAMPTZ NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    latency_p50_ms FLOAT,
    latency_p95_ms FLOAT,
    latency_p99_ms FLOAT,
    request_count BIGINT DEFAULT 0,
    error_count BIGINT DEFAULT 0,
    region_id VARCHAR(50),
    PRIMARY KEY (time, endpoint)
);

SELECT create_hypertable('nethical.latency_metrics', 'time',
    chunk_time_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

CREATE INDEX idx_latency_endpoint ON nethical.latency_metrics(endpoint, time DESC);

-- ============================================================================
-- CONTINUOUS AGGREGATES (Pre-computed metrics)
-- ============================================================================

-- Hourly audit summary
CREATE MATERIALIZED VIEW IF NOT EXISTS nethical.audit_hourly_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    agent_id,
    decision,
    region_id,
    COUNT(*) AS decision_count,
    AVG(risk_score) AS avg_risk_score,
    AVG(latency_ms) AS avg_latency_ms,
    MAX(latency_ms) AS max_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
FROM nethical.audit_events
GROUP BY bucket, agent_id, decision, region_id
WITH NO DATA;

-- Daily audit summary
CREATE MATERIALIZED VIEW IF NOT EXISTS nethical.audit_daily_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    agent_id,
    decision,
    region_id,
    COUNT(*) AS decision_count,
    AVG(risk_score) AS avg_risk_score,
    AVG(latency_ms) AS avg_latency_ms,
    MAX(latency_ms) AS max_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms
FROM nethical.audit_events
GROUP BY bucket, agent_id, decision, region_id
WITH NO DATA;

-- ============================================================================
-- DATA RETENTION POLICIES
-- ============================================================================

-- Set retention policies for hypertables
-- Audit events: 1 year retention
SELECT add_retention_policy('nethical.audit_events', INTERVAL '365 days', if_not_exists => TRUE);

-- Security events: 2 years retention
SELECT add_retention_policy('nethical.security_events', INTERVAL '730 days', if_not_exists => TRUE);

-- Quota usage: 90 days retention
SELECT add_retention_policy('nethical.quota_usage', INTERVAL '90 days', if_not_exists => TRUE);

-- Metrics: 30 days retention for raw data
SELECT add_retention_policy('nethical.governance_metrics', INTERVAL '30 days', if_not_exists => TRUE);

-- Latency metrics: 7 days retention for high-frequency data
SELECT add_retention_policy('nethical.latency_metrics', INTERVAL '7 days', if_not_exists => TRUE);

-- ============================================================================
-- COMPRESSION POLICIES
-- ============================================================================

-- Enable compression on older chunks for space efficiency
SELECT alter_table_set_access_method('nethical.audit_events', 'columnar');

ALTER TABLE nethical.audit_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'agent_id, decision, region_id',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('nethical.audit_events', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE nethical.governance_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'agent_id, metric_name',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('nethical.governance_metrics', INTERVAL '3 days', if_not_exists => TRUE);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION nethical.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_agents_updated_at
    BEFORE UPDATE ON nethical.agents
    FOR EACH ROW
    EXECUTE FUNCTION nethical.update_updated_at_column();

-- Function to generate audit event hash for Merkle tree
CREATE OR REPLACE FUNCTION nethical.generate_audit_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.merkle_hash = encode(
        sha256(
            (NEW.time::text || NEW.agent_id || NEW.decision || COALESCE(NEW.action_content, ''))::bytea
        ),
        'hex'
    );
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER generate_audit_hash_trigger
    BEFORE INSERT ON nethical.audit_events
    FOR EACH ROW
    EXECUTE FUNCTION nethical.generate_audit_hash();

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA nethical TO nethical_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA nethical TO nethical_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA nethical TO nethical_app;
