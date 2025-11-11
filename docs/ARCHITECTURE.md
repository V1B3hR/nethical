# Nethical Architecture

This document provides a comprehensive overview of the Nethical system architecture, design principles, and key components.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Integration Patterns](#integration-patterns)
- [Security Architecture](#security-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Design Principles](#design-principles)

## Overview

Nethical is a comprehensive safety governance system for AI agents that provides real-time monitoring, ethical compliance enforcement, and advanced security capabilities. The system is designed with modularity, extensibility, and production-readiness in mind.

### Key Characteristics

- **Modular Design**: Components are loosely coupled and independently deployable
- **Plugin Architecture**: Extensible through a plugin marketplace (F6)
- **Multi-Tenant**: Supports isolation and quota management across tenants
- **Production-Ready**: Includes observability, monitoring, and deployment automation
- **Security-First**: Military-grade security with quantum-resistant cryptography

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Web API  │  │ CLI Tool │  │ Webhooks │  │ Plugins  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                  Integrated Governance Layer                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         IntegratedGovernance (Unified Interface)      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                      Core Engine Layer                      │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Safety    │  │   Ethical    │  │   Compliance     │   │
│  │  Judge     │  │   Taxonomy   │  │   Framework      │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Anomaly    │  │ Correlation  │  │   Privacy &      │   │
│  │ Detection  │  │   Engine     │  │   Redaction      │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                   Security & Audit Layer                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   RBAC     │  │     JWT      │  │    Merkle        │   │
│  │            │  │   Auth       │  │   Anchoring      │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Quantum   │  │   Audit      │  │   Encryption     │   │
│  │  Crypto    │  │   Trail      │  │                  │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Redis    │  │  PostgreSQL  │  │   Time Series   │   │
│  │   Cache    │  │   Database   │  │   (Metrics)     │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Integrated Governance

**Location**: `nethical/core/integrated_governance.py`

The `IntegratedGovernance` class serves as the unified interface for all governance functionality. It consolidates:
- Phase 3: Compliance & Audit Framework
- Phase 4: Advanced Security & Policy Management
- Phases 5-7: Threat Protection & AI/ML Security
- Phases 8-9: Advanced Integration

**Key Responsibilities**:
- Action evaluation and judgment
- Policy enforcement
- Escalation management
- Plugin integration
- Unified API surface

### 2. Safety Judge

**Location**: `nethical/core/safety_judge.py`

The Safety Judge evaluates actions against safety constraints and ethical guidelines.

**Components**:
- Intent vs Action analysis
- Ethical violation detection (harm, deception, privacy, discrimination)
- Safety constraint enforcement
- Manipulation recognition
- Decision system (ALLOW, RESTRICT, BLOCK, TERMINATE)

### 3. Ethical Taxonomy

**Location**: `nethical/core/ethical_taxonomy.py`

Multi-dimensional ethical impact classification system.

**Features**:
- Ethical dimension tagging (privacy, manipulation, fairness, safety)
- Violation-to-dimension mapping
- Coverage tracking (>90% target)
- Industry-specific taxonomies

**Configuration**: `policies/ethics_taxonomy.json`

### 4. Anomaly Detection

**Location**: `nethical/ml/anomaly_classifier.py`

ML-based anomaly detection with trainable models.

**Capabilities**:
- Real-time anomaly scoring
- Distribution drift monitoring (PSI, KL divergence)
- Model training and optimization
- Feature engineering

### 5. Correlation Engine

**Location**: `nethical/core/correlation_engine.py`

Multi-agent pattern detection for sophisticated attack patterns.

**Features**:
- Sliding window correlation
- Pattern matching across agents
- Risk score aggregation
- Redis-backed persistence

**Configuration**: `policies/correlation_rules.yaml`

### 6. Privacy & Redaction

**Location**: `nethical/privacy/` modules

Privacy-preserving data handling (F3).

**Components**:
- PII detection (10+ types: SSN, email, credit cards, etc.)
- Redaction pipeline with configurable policies
- Differential privacy mechanisms
- Data minimization
- Right-to-be-Forgotten (RTBF) support

### 7. Compliance Framework

**Location**: `nethical/compliance/` modules

Enterprise-grade compliance support.

**Standards Supported**:
- NIST 800-53 control mapping
- HIPAA Privacy Rule
- FedRAMP continuous monitoring
- GDPR/CCPA requirements
- PCI-DSS Level 1

### 8. Audit Trail

**Location**: `nethical/audit/` modules

Immutable audit logging with cryptographic verification.

**Features**:
- Merkle tree anchoring
- Blockchain-style tamper detection
- Chain-of-custody tracking
- Forensic analysis tools

### 9. Plugin System

**Location**: `nethical/plugins/` modules

Extensibility framework (F6).

**Components**:
- Plugin loader and registry
- Plugin isolation and sandboxing
- Healthcare pack (example)
- Plugin marketplace integration

## Data Flow

### Action Evaluation Flow

```
┌──────────────┐
│ Agent Action │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ IntegratedGovernance │
└──────┬───────────────┘
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
┌──────────────┐            ┌──────────────┐
│ Safety Judge │            │   Privacy    │
│  Evaluation  │            │  Redaction   │
└──────┬───────┘            └──────┬───────┘
       │                           │
       ▼                           ▼
┌──────────────┐            ┌──────────────┐
│   Ethical    │            │  Correlation │
│   Taxonomy   │            │    Engine    │
└──────┬───────┘            └──────┬───────┘
       │                           │
       ├───────────┬───────────────┘
       │           │
       ▼           ▼
┌──────────────────────┐
│  Anomaly Detection   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Decision (ALLOW,   │
│ RESTRICT, BLOCK, etc)│
└──────┬───────────────┘
       │
       ├────────────────┬──────────────┐
       │                │              │
       ▼                ▼              ▼
┌──────────┐    ┌──────────┐   ┌──────────┐
│  Audit   │    │  Metrics │   │ Escalate │
│   Log    │    │ Recording│   │ (if HITL)│
└──────────┘    └──────────┘   └──────────┘
```

### Training & Optimization Flow

```
┌──────────────┐
│ Training Data│
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Training Orchestrator│
└──────┬───────────────┘
       │
       ├────────────────┬──────────────┐
       │                │              │
       ▼                ▼              ▼
┌──────────┐    ┌──────────┐   ┌──────────┐
│ Anomaly  │    │  Drift   │   │Correlation│
│ Training │    │ Tracking │   │  Tuning  │
└──────┬───┘    └──────┬───┘   └──────┬───┘
       │                │              │
       └────────┬───────┴──────────────┘
                │
                ▼
         ┌──────────────┐
         │Model Registry│
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │ Promotion    │
         │ Gate (>90%   │
         │ accuracy)    │
         └──────────────┘
```

## Integration Patterns

### 1. Synchronous Integration

Direct API calls for real-time action evaluation:

```python
from nethical.core.integrated_governance import IntegratedGovernance

governance = IntegratedGovernance()
decision = governance.evaluate_action(action)
```

### 2. Asynchronous Integration

Webhook-based integration for non-blocking evaluation:

```python
from nethical.webhooks import WebhookManager

webhook_mgr = WebhookManager()
webhook_mgr.register_endpoint(
    "https://your-app.com/nethical/callback"
)
```

### 3. Plugin Integration

Extending functionality through plugins:

```python
governance = IntegratedGovernance(
    plugins=["healthcare_pack"]
)
governance.load_plugin("custom_validator")
```

## Security Architecture

### Defense in Depth

1. **Authentication Layer**
   - JWT tokens with short expiration
   - Refresh token rotation
   - API key management
   - Multi-factor authentication (MFA)
   - SSO/SAML support

2. **Authorization Layer**
   - Role-Based Access Control (RBAC)
   - 4-tier role hierarchy (Admin, Operator, Auditor, Viewer)
   - Fine-grained permissions
   - Policy-based access control

3. **Encryption Layer**
   - Data at rest encryption
   - TLS 1.3 for data in transit
   - Quantum-resistant algorithms (CRYSTALS-Kyber, Dilithium)
   - Hybrid classical-quantum TLS

4. **Audit Layer**
   - Immutable audit logs
   - Merkle tree anchoring
   - Tamper detection
   - Cryptographic verification

### Threat Protection

- **Adversarial Attack Detection**: FGSM, PGD, DeepFool, C&W
- **Prompt Injection Prevention**: 36 comprehensive tests
- **PII Protection**: 10+ PII types with configurable redaction
- **Resource Protection**: Quota enforcement, rate limiting, backpressure
- **Model Security**: Poisoning detection, differential privacy

## Deployment Architecture

### Docker Deployment

```yaml
services:
  nethical-api:
    image: nethical:latest
    environment:
      - REDIS_URL=redis://cache:6379
      - POSTGRES_URL=postgresql://db:5432/nethical
    depends_on:
      - cache
      - db
      - otel-collector

  cache:
    image: redis:7-alpine

  db:
    image: postgres:15-alpine

  otel-collector:
    image: otel/opentelemetry-collector-contrib
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nethical-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nethical
        image: nethical:latest
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Observability Stack

- **Metrics**: Prometheus + Grafana
- **Traces**: OpenTelemetry
- **Logs**: Structured logging with ELK stack
- **Alerting**: Prometheus AlertManager

## Design Principles

### 1. Modularity

Each component is independently deployable and testable. Components communicate through well-defined interfaces.

### 2. Extensibility

Plugin system allows extending functionality without modifying core code.

### 3. Security by Default

All security features are enabled by default. Opt-out rather than opt-in.

### 4. Privacy First

PII detection and redaction are automatic. Differential privacy is available for sensitive operations.

### 5. Observability

All components emit metrics, traces, and logs for comprehensive observability.

### 6. Testability

427+ tests cover all major functionality. Includes unit, integration, and adversarial tests.

### 7. Performance

- Sub-100ms action evaluation latency
- >10,000 actions/sec throughput
- Horizontal scalability
- Efficient caching and batching

### 8. Backward Compatibility

Legacy APIs (Phase3/4/5-7/8-9 integrations) remain available for smooth migration.

## Technology Stack

### Core Technologies

- **Language**: Python 3.8+
- **Frameworks**: Pydantic for data validation
- **ML**: NumPy, Pandas, Scikit-learn
- **Async**: asyncio, aiohttp

### Data Storage

- **Cache**: Redis (correlation state, session data)
- **Database**: PostgreSQL (audit logs, user data)
- **Time Series**: Prometheus (metrics)

### Security

- **Cryptography**: CRYSTALS-Kyber (key encapsulation), CRYSTALS-Dilithium (signatures)
- **Authentication**: JWT (HS256/RS256)
- **Secrets Management**: HashiCorp Vault integration

### Deployment

- **Containers**: Docker
- **Orchestration**: Kubernetes
- **CI/CD**: GitHub Actions
- **Observability**: OpenTelemetry, Prometheus, Grafana

## Performance Characteristics

### Latency Targets

- Action evaluation: <100ms (p95)
- Anomaly detection: <50ms (p95)
- Audit log write: <10ms (p95)
- Query response: <200ms (p95)

### Throughput Targets

- Actions/sec: >10,000
- Concurrent users: >1,000
- Plugin load time: <1s

### Scalability

- Horizontal scaling: Linear to 10+ nodes
- Vertical scaling: Up to 16 CPU cores
- Data retention: 5+ years of audit logs

## Future Architecture Enhancements

See [advancedplan.md](../advancedplan.md) for detailed roadmap:

- Enhanced federation for multi-cloud deployment
- Advanced ML model serving with TensorFlow/PyTorch
- Graph database for complex relationship tracking
- Real-time streaming with Apache Kafka
- Service mesh integration (Istio)

## References

- [Implementation History](./archive/IMPLEMENTATION_HISTORY.md)
- [Compliance Mapping](./compliance/NIST_RMF_MAPPING.md)
- [Security Guide](./security/AI_ML_SECURITY_GUIDE.md)
- [Performance Sizing](./ops/PERFORMANCE_SIZING.md)
- [Plugin Developer Guide](./PLUGIN_DEVELOPER_GUIDE.md)

---

Last Updated: 2024-11-11
