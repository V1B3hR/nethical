# Model Deployment Guide

## Overview

This guide provides step-by-step procedures for safely deploying machine learning models in the Nethical framework. All deployments follow a strict validation and rollout process to ensure model quality and system stability.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Modes](#deployment-modes)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Rollback Procedures](#rollback-procedures)
- [Emergency Response](#emergency-response)
- [Production Monitoring](#production-monitoring)

## Prerequisites

### Required Tools

- Python 3.9+
- Access to GitHub repository
- Permissions to trigger workflows
- Access to production environment (if deploying to production)

### Required Artifacts

- Trained model files in `models/current/` or `models/candidates/`
- Model metadata files in `models/metadata/`
- Performance metrics JSON files
- Audit logs (if audit enabled)

### Pre-Deployment Checklist

- [ ] Model passes all quality gates (accuracy ≥ 0.85, ECE ≤ 0.08)
- [ ] Performance comparison with baseline shows improvement or no regression
- [ ] Governance validation passed (no safety violations)
- [ ] Audit trail complete and verified
- [ ] Drift analysis shows acceptable levels
- [ ] Stakeholders notified of pending deployment

## Deployment Modes

### 1. Shadow Mode

**Purpose**: Validate model in production environment without affecting traffic

**Characteristics**:
- Model loads and runs inference
- No traffic routing to model
- Performance logged for analysis
- Zero impact on users

**Use When**:
- Initial production deployment
- Major model architecture changes
- Unvalidated training data sources
- Testing new feature sets

**Duration**: Typically 24-48 hours

**Command**:
```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode shadow \
  --environment staging
```

### 2. Canary Deployment

**Purpose**: Gradual rollout to subset of traffic

**Characteristics**:
- Routes small percentage of traffic (10% default)
- Monitor for issues before full rollout
- Easy rollback if problems detected
- Minimal user impact if issues occur

**Use When**:
- Model passes shadow mode validation
- Incremental rollout desired
- High-confidence in model quality
- Testing under real traffic patterns

**Traffic Percentages**:
- Conservative: 5-10%
- Standard: 10-25%
- Aggressive: 25-50%

**Duration**: 24-72 hours per stage

**Command**:
```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode canary \
  --rollout-percentage 10 \
  --environment production
```

### 3. Full Deployment

**Purpose**: Complete rollout to all traffic

**Characteristics**:
- 100% traffic routing
- Full production load
- Complete model replacement

**Use When**:
- Canary deployment successful
- All monitoring metrics healthy
- Stakeholder approval obtained
- Emergency rollback plan in place

**Command**:
```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode full \
  --environment production
```

## Step-by-Step Deployment

### Phase 1: Pre-Deployment Validation

**Step 1: Verify Model Quality**

```bash
# Check model metrics
cat models/current/logistic_metrics_*.json

# Expected output should show:
# - accuracy >= 0.85
# - ece <= 0.08
# - f1, precision, recall > 0.80
```

**Step 2: Run Validation**

```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --validate-only
```

Expected output:
```
✓ Found N model files
✓ Found N metadata files
✓ Model successfully loaded
✓ Performance thresholds met
Overall status: passed
```

**Step 3: Review Model Card**

```bash
# Check model metadata
cat models/metadata/logistic_model_*_card.json

# Verify:
# - training_timestamp is recent
# - governance_summary shows no violations
# - audit_merkle_root is present
# - metrics meet requirements
```

### Phase 2: Shadow Deployment

**Step 1: Deploy to Shadow**

```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode shadow \
  --environment staging
```

**Step 2: Monitor Shadow Performance (24-48 hours)**

```bash
# Monitor drift metrics
python scripts/monitor_models.py \
  --model-path models/current/ \
  --calculate-drift \
  --num-samples 5000
```

Check for:
- Drift score < 0.15
- No accuracy degradation
- Stable prediction distribution
- No runtime errors

**Step 3: Analyze Results**

```bash
# Review monitoring results
cat drift_metrics.json

# Key checks:
# - drift_detected: false
# - drift_score: < 0.15
# - all models showing "performing normally"
```

### Phase 3: Canary Deployment

**Step 1: Deploy Canary (10% Traffic)**

```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode canary \
  --rollout-percentage 10 \
  --environment production
```

**Step 2: Monitor Canary Performance (24 hours minimum)**

Set up automated monitoring via GitHub Actions:
```bash
# Trigger monitoring workflow
gh workflow run model-monitoring.yml
```

Monitor for:
- User-reported issues
- Error rate changes
- Latency increases
- Accuracy metrics
- Drift signals

**Step 3: Increase Canary Percentage (Optional)**

If initial canary is stable, gradually increase:

```bash
# Increase to 25%
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode canary \
  --rollout-percentage 25 \
  --environment production

# Wait 24 hours, then 50%
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode canary \
  --rollout-percentage 50 \
  --environment production
```

### Phase 4: Full Deployment

**Step 1: Get Stakeholder Approval**

- Review canary performance metrics
- Present no-regression evidence
- Get sign-off from team lead
- Schedule deployment window

**Step 2: Deploy Full**

```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode full \
  --environment production
```

**Step 3: Archive Old Production Model**

```bash
# Move old production model to archive
mv models/current/old_model_*.json models/archived/

# Update metadata
echo "Replaced by: new_model_timestamp" >> models/archived/old_model_metadata.txt
```

**Step 4: Verify Deployment**

```bash
# Run validation tests
python scripts/deploy_model.py \
  --model-path models/current/ \
  --validate-only

# Check production inference
python scripts/monitor_models.py \
  --model-path models/current/ \
  --inference-mode \
  --num-samples 1000
```

### Phase 5: Post-Deployment Monitoring

**Immediate (First 1 hour)**:
- Monitor error rates every 15 minutes
- Check inference latency
- Watch for alerts

**Short-term (First 24 hours)**:
- Review performance metrics every 2 hours
- Check drift scores
- Monitor user feedback

**Long-term (First week)**:
- Daily performance reviews
- Drift monitoring (automated every 6 hours)
- Weekly performance comparison

## Rollback Procedures

### Scenario 1: Performance Regression Detected

**Indicators**:
- Accuracy drops below 0.85
- ECE increases above 0.08
- High drift score (> 0.15)

**Rollback Steps**:

1. **Stop New Deployments**
```bash
# Immediately halt any in-progress deployments
```

2. **Identify Previous Stable Model**
```bash
# List archived models
ls -lt models/archived/

# Check metadata for last stable model
cat models/archived/logistic_model_TIMESTAMP_card.json
```

3. **Redeploy Previous Model**
```bash
# Copy archived model back to current
cp models/archived/logistic_model_TIMESTAMP.json models/current/

# Deploy in shadow mode first
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode shadow \
  --environment production

# If shadow looks good, deploy full
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode full \
  --environment production
```

4. **Create Incident Report**
```bash
# Document issue in GitHub
gh issue create \
  --title "Model Rollback: Performance Regression" \
  --body "Details: [describe issue]" \
  --label "ml-ops,incident,needs-investigation"
```

### Scenario 2: Critical Safety Violation

**Indicators**:
- Governance violations in production
- Harmful predictions detected
- Security vulnerabilities identified

**Immediate Actions**:

1. **Emergency Lockdown**
```bash
# Immediately stop all model inference
# (Implementation specific to your deployment platform)
```

2. **Rollback to Last Safe Model**
```bash
# Use most recent model with zero violations
cp models/archived/SAFE_MODEL.json models/current/
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode full \
  --environment production
```

3. **Investigate Root Cause**
```bash
# Review audit logs
cat training_audit_logs/training_summary.json

# Check governance metrics
grep -r "violation" training_audit_logs/
```

4. **Alert Team**
```bash
# Create high-priority incident
gh issue create \
  --title "CRITICAL: Safety Violation in Production Model" \
  --body "Immediate rollback executed. Investigation required." \
  --label "ml-ops,security,critical"
```

### Scenario 3: System Errors or Crashes

**Indicators**:
- Model loading failures
- Runtime exceptions
- Out of memory errors

**Recovery Steps**:

1. **Verify Model Integrity**
```bash
python scripts/deploy_model.py \
  --model-path models/current/ \
  --validate-only
```

2. **If Validation Fails, Rollback**
```bash
cp models/archived/LAST_WORKING_MODEL.json models/current/
python scripts/deploy_model.py \
  --model-path models/current/ \
  --deployment-mode full \
  --environment production
```

3. **Debug Original Model**
```bash
# Try loading model locally
python -c "import pickle; pickle.load(open('models/current/MODEL.pkl', 'rb'))"

# Check for corrupted files
md5sum models/current/*.pkl
```

## Emergency Response

### Emergency Contact Protocol

1. **On-Call Engineer**: Notified via GitHub issue labels
2. **Team Lead**: Escalate if issue persists > 30 minutes
3. **Security Team**: Contact if safety violations detected

### Emergency Decision Tree

```
Issue Detected
    │
    ├─ Safety Violation? ──YES──> Immediate Lockdown + Rollback
    │
    ├─ Performance < 80% Accuracy? ──YES──> Rollback + Investigation
    │
    ├─ System Error? ──YES──> Validate Model + Rollback if Failed
    │
    └─ Minor Drift? ──NO──> Continue Monitoring
```

### Post-Incident Actions

1. **Document Incident**: Create detailed post-mortem
2. **Root Cause Analysis**: Identify what went wrong
3. **Update Procedures**: Improve deployment process
4. **Test Fixes**: Validate before next deployment
5. **Team Review**: Share learnings with team

## Production Monitoring

### Automated Monitoring

The model-monitoring.yml workflow runs every 6 hours:

```yaml
Schedule: "0 */6 * * *"  # Every 6 hours

Checks:
  - Model drift
  - Performance degradation
  - Prediction distribution changes
  
Actions:
  - Create GitHub issues on alerts
  - Auto-trigger retraining if drift > threshold
  - Log all monitoring results
```

### Manual Monitoring

**Daily Checks** (5 minutes):
```bash
# Quick health check
python scripts/monitor_models.py \
  --model-path models/current/ \
  --list-models

python scripts/monitor_models.py \
  --model-path models/current/ \
  --calculate-drift \
  --num-samples 1000
```

**Weekly Reviews** (30 minutes):
```bash
# Comprehensive analysis
python scripts/compare_models.py \
  --current-path models/current/ \
  --baseline-path models/archived/ \
  --history-path performance_history/

# Review GitHub issues
gh issue list --label ml-ops --state open
```

### Metrics to Monitor

**Model Performance**:
- Accuracy, precision, recall, F1
- Expected Calibration Error (ECE)
- Prediction distribution

**System Health**:
- Inference latency
- Error rates
- Memory usage
- CPU utilization

**Business Metrics**:
- User satisfaction
- Task completion rates
- False positive/negative rates

## Best Practices

1. **Always Start with Shadow**: Never skip shadow deployment
2. **Monitor Continuously**: Use automated monitoring workflows
3. **Document Everything**: Maintain deployment logs and incident reports
4. **Test Rollbacks**: Practice rollback procedures regularly
5. **Gradual Rollout**: Use canary deployments for risk mitigation
6. **Maintain History**: Keep archived models for at least 1 year
7. **Stakeholder Communication**: Keep team informed of deployments
8. **Quality Gates**: Never bypass validation steps

## Related Documentation

- [MLOps Architecture](./mlops-architecture.md)
- [Training Pipeline Documentation](../training/README.md)
- [Model Directory Structure](../models/README.md)
