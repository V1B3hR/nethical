# Interoperability & Compliance Guide

Comprehensive guide for ensuring Nethical's interoperability with LLM platforms and compliance with industry standards.

## Table of Contents

- [Overview](#overview)
- [LLM Platform Compliance](#llm-platform-compliance)
- [MCP Platform Compliance](#mcp-platform-compliance)
- [Industry Standards](#industry-standards)
- [Security Requirements](#security-requirements)
- [Data Privacy](#data-privacy)
- [Audit & Reporting](#audit--reporting)

## Overview

Nethical maintains compliance with:

- **LLM Platform Standards**: OpenAI, Anthropic, xAI, Google requirements
- **MCP Framework Standards**: LangChain, HuggingFace, AutoGen specifications
- **Security Standards**: OWASP, NIST, ISO 27001
- **Privacy Regulations**: GDPR, CCPA, HIPAA
- **Industry Frameworks**: SLSA, OpenSSF, SBOM

## LLM Platform Compliance

### OpenAI Plugin Standards

**Requirements Met:**
- ✅ OpenAPI 3.1 specification
- ✅ ai-plugin.json manifest (in `config/integrations/`)
- ✅ HTTPS endpoints required
- ✅ Schema validation
- ✅ CORS configuration
- ✅ Rate limiting support

**Implementation:**

```json
{
  "schema_version": "v1",
  "name_for_model": "nethical",
  "api": {
    "type": "openapi",
    "url": "https://api.nethical.dev/openapi.yaml"
  },
  "auth": {
    "type": "none"
  }
}
```

**Validation:**

```bash
# Validate OpenAPI spec
npx @apidevtools/swagger-cli validate openapi.yaml

# Test plugin manifest
curl https://your-domain.com/.well-known/ai-plugin.json
```

### Anthropic Claude Standards

**Requirements Met:**
- ✅ Function calling format
- ✅ Tool use protocol
- ✅ Response formatting
- ✅ Error handling
- ✅ Context preservation

**Implementation:**

```python
from nethical.integrations.claude_tools import get_nethical_tool

# Standard tool definition
tool = get_nethical_tool()

# Compliant with Claude's tool schema:
# - type: function
# - name: nethical_guard
# - parameters: object with required/optional fields
```

### xAI Grok Standards

**Requirements Met:**
- ✅ Function declaration format
- ✅ Manifest specification
- ✅ API compatibility
- ✅ Response structure
- ✅ Error codes

**Manifest:** `config/integrations/grok-manifest.json`

### Google Gemini Standards

**Requirements Met:**
- ✅ Function declarations format
- ✅ Gemini-specific schemas
- ✅ Response formatting
- ✅ Safety settings integration
- ✅ Context handling

**Manifest:** `config/integrations/gemini-manifest.json`

## MCP Platform Compliance

### LangChain Tool Standards

**Requirements Met:**
- ✅ BaseTool interface
- ✅ Async support
- ✅ Callbacks integration
- ✅ Memory compatibility
- ✅ Chain composition

**Manifest:** `config/integrations/langchain-tool.json`

**Validation:**

```python
from nethical.integrations.langchain_tools import NethicalTool

# Verify tool compliance
tool = NethicalTool()
assert hasattr(tool, '_run')
assert hasattr(tool, '_arun')  # Async support
```

### HuggingFace Integration Standards

**Requirements Met:**
- ✅ Transformers pipeline compatibility
- ✅ Inference API support
- ✅ Spaces deployment ready
- ✅ Model Hub integration
- ✅ Dataset compatibility

**Manifest:** `config/integrations/huggingface-tool.yaml`

### AutoGen Standards

**Requirements Met:**
- ✅ Agent wrapper interface
- ✅ Message filtering
- ✅ Function call guards
- ✅ Group chat monitoring
- ✅ Conversation tracking

**Manifest:** `config/integrations/autogen-manifest.json`

### MLflow Standards

**Requirements Met:**
- ✅ Model wrapper interface
- ✅ Artifact logging
- ✅ Metric tracking
- ✅ Registry integration
- ✅ Deployment hooks

**Manifest:** `config/integrations/mlflow-integration.yaml`

### Ray Serve Standards

**Requirements Met:**
- ✅ Deployment interface
- ✅ Scaling compatibility
- ✅ Metrics integration
- ✅ Health checks
- ✅ Rolling updates

**Module:** `ray_serve_connector.py`

## Industry Standards

### OWASP LLM Top 10 (2023)

Nethical addresses all OWASP LLM Top 10 risks:

| Risk | Nethical Protection | Implementation |
|------|---------------------|----------------|
| **LLM01: Prompt Injection** | Detection & blocking | Adversarial pattern detection |
| **LLM02: Insecure Output** | Output filtering | Safety evaluation before display |
| **LLM03: Training Data Poisoning** | Model validation | Anomaly detection |
| **LLM04: Model DoS** | Quota enforcement | Rate limiting, backpressure |
| **LLM05: Supply Chain** | SBOM, signing | Dependency validation |
| **LLM06: Sensitive Info Disclosure** | PII detection | 10+ PII types, redaction |
| **LLM07: Insecure Plugin** | Plugin validation | Manifest verification |
| **LLM08: Excessive Agency** | Action monitoring | Risk scoring, decision system |
| **LLM09: Overreliance** | Human-in-loop | Escalation for high-risk |
| **LLM10: Model Theft** | Access control | Audit trails, monitoring |

**Compliance Report:**

```python
from nethical.core.integrated_governance import IntegratedGovernance

gov = IntegratedGovernance()
compliance = gov.get_owasp_compliance_report()

# Returns coverage for each LLM01-10
```

### NIST AI Risk Management Framework

**Requirements Met:**

- ✅ **Govern**: Policy management, audit trails
- ✅ **Map**: Risk assessment, taxonomy
- ✅ **Measure**: Metrics, monitoring, SLA
- ✅ **Manage**: Controls, mitigation, response

**Implementation:**

```python
# Map phase: Risk identification
result = gov.process_action(action, agent_id="llm")
risk_score = result["risk_score"]

# Measure phase: Metrics
metrics = gov.get_system_status()

# Manage phase: Controls
if risk_score > threshold:
    # Apply controls
    decision = "BLOCK"
```

**Documentation:** See `docs/compliance/NIST_AI_RMF.md`

### ISO/IEC 42001 (AI Management)

**Requirements Met:**

- ✅ AI system governance
- ✅ Risk management process
- ✅ Data governance
- ✅ Transparency measures
- ✅ Accountability mechanisms

### SLSA Framework (Supply Chain)

**Level 3 Compliance:**

- ✅ Build provenance
- ✅ Signed artifacts
- ✅ Non-falsifiable provenance
- ✅ Hermetic builds (partial)

**Implementation:**

```yaml
# .github/workflows/release.yml
- uses: slsa-framework/slsa-github-generator@v1
  with:
    provenance-name: nethical-provenance.json
```

## Security Requirements

### Transport Security

**Requirements:**
- ✅ TLS 1.3 for all API endpoints
- ✅ Certificate validation
- ✅ HTTPS-only in production
- ✅ HSTS headers

**Implementation:**

```python
# FastAPI with TLS
uvicorn.run(
    app,
    host="0.0.0.0",
    port=443,
    ssl_keyfile="privkey.pem",
    ssl_certfile="fullchain.pem",
    ssl_version=ssl.PROTOCOL_TLSv1_3
)
```

### Authentication & Authorization

**Supported Methods:**
- ✅ API Key authentication
- ✅ Bearer token (JWT)
- ✅ OAuth 2.0 (configurable)
- ✅ Mutual TLS (mTLS)

**Implementation:**

```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/evaluate")
async def evaluate(
    request: EvaluateRequest,
    token: str = Depends(security)
):
    # Validate token
    validate_token(token)
    # Process request
```

### Vulnerability Management

**Process:**
- ✅ Dependency scanning (GitHub Dependabot)
- ✅ SAST (CodeQL)
- ✅ DAST (OWASP ZAP)
- ✅ Container scanning (Trivy)
- ✅ SBOM generation

**CI/CD Integration:**

```yaml
# .github/workflows/security.yml
- name: Run Trivy scanner
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    scan-ref: '.'
```

## Data Privacy

### GDPR Compliance

**Requirements Met:**

| Article | Requirement | Implementation |
|---------|-------------|----------------|
| **Art. 5** | Data minimization | Minimal data collection |
| **Art. 6** | Lawful basis | Consent, legitimate interest |
| **Art. 9** | Special categories | PII detection & redaction |
| **Art. 17** | Right to erasure | RTBF support |
| **Art. 25** | Privacy by design | Default privacy settings |
| **Art. 32** | Security | Encryption, audit logs |
| **Art. 33** | Breach notification | Alert system |

**Data Processing Agreement:** Available in `docs/compliance/DPA.md`

### CCPA Compliance

**Requirements Met:**
- ✅ Right to know (audit trails)
- ✅ Right to delete (RTBF)
- ✅ Right to opt-out (configurable)
- ✅ Non-discrimination
- ✅ Data disclosure

### HIPAA Compliance

**Requirements Met:**

- ✅ **Administrative Safeguards**: Access controls, audit
- ✅ **Physical Safeguards**: Encrypted storage
- ✅ **Technical Safeguards**: Encryption, audit trails
- ✅ **Organizational Requirements**: BAA available

**BAA Template:** `docs/compliance/BAA_TEMPLATE.md`

### Data Residency

**Support:**
- ✅ Regional deployment configuration
- ✅ Data locality enforcement
- ✅ Geo-fencing options
- ✅ Multi-region support

**Configuration:**

```python
gov = IntegratedGovernance(
    region_id="eu-west-1",
    enable_regional_compliance=True,
    data_residency_rules={
        "eu": ["GDPR"],
        "us": ["HIPAA", "CCPA"]
    }
)
```

## Audit & Reporting

### Audit Trail Requirements

**Features:**
- ✅ Immutable logs (Merkle anchoring)
- ✅ Cryptographic integrity
- ✅ Timestamping (RFC 3161)
- ✅ Non-repudiation
- ✅ Tamper detection

**Implementation:**

```python
# Every action logged with Merkle proof
result = gov.process_action(action)
audit_id = result["audit_id"]

# Verify integrity
proof = gov.get_audit_proof(audit_id)
is_valid = gov.verify_audit_integrity(audit_id, proof)
```

### Compliance Reporting

**Available Reports:**

1. **OWASP LLM Top 10 Coverage**
2. **NIST AI RMF Assessment**
3. **Privacy Impact Assessment**
4. **Security Posture Report**
5. **Incident Response Summary**

**Generate Report:**

```python
# Compliance report
report = gov.generate_compliance_report(
    standards=["OWASP_LLM", "GDPR", "NIST_AI_RMF"],
    period="2024-01-01,2024-12-31"
)
```

### Third-Party Audits

**Supported:**
- ✅ SOC 2 Type II preparation
- ✅ ISO 27001 evidence collection
- ✅ PCI DSS artifact generation
- ✅ HIPAA audit support

## Certification & Attestation

### Available Certifications

- 📋 **SLSA Level 3** (In Progress)
- 📋 **OpenSSF Best Practices** (Gold Badge Target)
- 📋 **SOC 2 Type II** (Preparation Phase)

### Self-Attestation

```json
{
  "product": "Nethical",
  "version": "0.1.0",
  "attestation": {
    "owasp_llm_top10": "full_coverage",
    "nist_ai_rmf": "compliant",
    "gdpr": "compliant",
    "sbom_available": true,
    "signed_releases": true,
    "vulnerability_disclosure": "responsible"
  },
  "last_audit": "2025-11-22",
  "next_audit": "2026-05-22"
}
```

## Continuous Compliance

### Monitoring

```python
# Real-time compliance monitoring
compliance_status = gov.get_compliance_status()

for standard, status in compliance_status.items():
    if not status["compliant"]:
        alert_compliance_team(standard, status["issues"])
```

### Updates

- 📅 **Quarterly**: Standard updates review
- 📅 **Monthly**: Dependency updates
- 📅 **Weekly**: Security patches
- 📅 **Daily**: Vulnerability scans

## Additional Resources

- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- [SLSA Framework](https://slsa.dev/)
- [OpenSSF Scorecard](https://github.com/ossf/scorecard)
- [ISO/IEC 42001](https://www.iso.org/standard/81230.html)

## Support

For compliance questions:
- 📧 Email: compliance@nethical.dev
- 📚 Documentation: [docs/compliance/](../compliance/)
- 🔒 Security: security@nethical.dev

## License

MIT License - See [LICENSE](../LICENSE) for details.
