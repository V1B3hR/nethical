# Security Operations Runbook

This runbook provides step-by-step procedures for managing and responding to security events in the Nethical platform.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Security Incident Response](#security-incident-response)
3. [Vulnerability Management](#vulnerability-management)
4. [Secret Rotation](#secret-rotation)
5. [Network Policy Updates](#network-policy-updates)
6. [mTLS Certificate Management](#mtls-certificate-management)
7. [Security Monitoring](#security-monitoring)
8. [Audit Log Review](#audit-log-review)
9. [Emergency Procedures](#emergency-procedures)

---

## Daily Operations

### Morning Security Checklist

**Time Required:** 15-20 minutes  
**Frequency:** Daily

1. **Check Security Dashboards**
   ```bash
   # Access Grafana security dashboard
   kubectl port-forward -n monitoring svc/grafana 3000:3000
   # Navigate to http://localhost:3000/d/security-overview
   ```

2. **Review Overnight Alerts**
   ```bash
   # Check for critical security alerts
   kubectl get alerts -n nethical -l severity=critical
   
   # Review WAF blocked requests
   kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --tail=100 | grep "ModSecurity"
   ```

3. **Verify Network Policy Status**
   ```bash
   # Run automated verification
   ./scripts/verify-security-controls.sh
   ```

4. **Check Secret Rotation Status**
   ```bash
   # Check last rotation timestamps
   kubectl exec -n nethical deploy/nethical -- python -c "
   from datetime import datetime
   import json
   # Check rotation timestamps in secrets
   "
   
   # Or run the monitoring script
   kubectl get cronjob check-secret-rotation -n nethical
   kubectl logs -n nethical $(kubectl get pods -n nethical -l job=secret-rotation-check --sort-by=.metadata.creationTimestamp -o name | tail -1)
   ```

5. **Review Vulnerability Scan Results**
   ```bash
   # Check latest vulnerability scan
   gh run list --workflow=vuln-sla.yml --limit 1
   
   # Download and review if needed
   gh run view $(gh run list --workflow=vuln-sla.yml --limit 1 --json databaseId -q '.[0].databaseId')
   ```

### Weekly Security Tasks

**Time Required:** 1-2 hours  
**Frequency:** Weekly

1. **Network Policy Audit**
   ```bash
   # Test network isolation
   ./scripts/test-network-isolation.sh
   
   # Review network policy changes
   kubectl get networkpolicies -n nethical -o yaml > /tmp/netpol-$(date +%Y%m%d).yaml
   diff /tmp/netpol-last-week.yaml /tmp/netpol-$(date +%Y%m%d).yaml
   ```

2. **Review Access Logs**
   ```bash
   # Extract and analyze access patterns
   kubectl logs -n nethical -l app=nethical --since=168h | grep "authentication" > access-logs.txt
   
   # Look for suspicious patterns
   grep -i "failed\|denied\|blocked" access-logs.txt | sort | uniq -c | sort -rn
   ```

3. **Update Threat Intelligence**
   ```bash
   # Update WAF rules with latest threat patterns
   kubectl get configmap nethical-waf-rules -n nethical -o yaml > waf-rules-backup.yaml
   
   # Review OWASP ModSecurity CRS updates
   # Update custom rules if needed
   ```

4. **Security Metrics Review**
   - False positive rate for WAF rules
   - Blocked requests analysis
   - Authentication failure rate
   - Network policy violations
   - Certificate expiration dates

---

## Security Incident Response

### P1: Critical Security Incident

**Indicators:**
- Active breach detected
- Critical vulnerability exploitation
- Unauthorized data access
- Service compromise

**Immediate Actions (0-15 minutes):**

1. **Activate Incident Response Team**
   ```bash
   # Send alert to security team
   curl -X POST $SLACK_SECURITY_WEBHOOK -d '{"text":"P1 Security Incident - All hands on deck!"}'
   
   # Page on-call security engineer
   curl -X POST $PAGERDUTY_WEBHOOK -d '{"incident_key":"security-p1","description":"Critical security incident"}'
   ```

2. **Isolate Affected Systems**
   ```bash
   # Enable quarantine mode
   kubectl set env statefulset/nethical -n nethical NETHICAL_ENABLE_QUARANTINE=true
   
   # Block external traffic
   kubectl apply -f - <<EOF
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: emergency-lockdown
     namespace: nethical
   spec:
     podSelector: {}
     policyTypes:
     - Ingress
     - Egress
   EOF
   ```

3. **Capture Evidence**
   ```bash
   # Take pod snapshots
   kubectl get pods -n nethical -o yaml > incident-pods-$(date +%s).yaml
   
   # Capture logs
   kubectl logs -n nethical -l app=nethical --all-containers --since=1h > incident-logs-$(date +%s).txt
   
   # Export audit logs
   kubectl exec -n nethical nethical-0 -- tar czf /tmp/audit-$(date +%s).tar.gz /data/audit/
   kubectl cp nethical/nethical-0:/tmp/audit-*.tar.gz ./incident-audit-logs.tar.gz
   ```

4. **Enable Enhanced Monitoring**
   ```bash
   # Increase log verbosity
   kubectl set env statefulset/nethical -n nethical LOG_LEVEL=DEBUG
   
   # Enable detailed audit logging
   kubectl set env statefulset/nethical -n nethical NETHICAL_AUDIT_VERBOSE=true
   ```

**Investigation (15-60 minutes):**

1. **Analyze Attack Vector**
   ```bash
   # Review WAF logs for attack patterns
   kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx --since=2h | grep -i "attack\|injection\|exploit"
   
   # Check authentication logs
   kubectl logs -n nethical -l app=nethical --since=2h | grep "auth_failed\|unauthorized"
   
   # Review network connections
   kubectl exec -n nethical nethical-0 -- netstat -tulpn
   ```

2. **Identify Compromised Assets**
   ```bash
   # Check for modified files
   kubectl exec -n nethical nethical-0 -- find /app -type f -mtime -1
   
   # Review running processes
   kubectl exec -n nethical nethical-0 -- ps aux
   
   # Check for suspicious connections
   kubectl exec -n nethical nethical-0 -- lsof -i
   ```

3. **Assess Data Impact**
   ```bash
   # Query audit logs for data access
   kubectl exec -n nethical nethical-0 -- python -c "
   from nethical.governance.phase4_security import AuditLogger
   logger = AuditLogger()
   recent_access = logger.query_recent_access(hours=2)
   print(recent_access)
   "
   ```

**Containment (1-4 hours):**

1. **Patch Vulnerability (if known)**
   ```bash
   # Apply emergency patch
   kubectl set image statefulset/nethical -n nethical nethical=nethical:patched-$(date +%s)
   
   # Or rollback to last known good version
   kubectl rollout undo statefulset/nethical -n nethical
   ```

2. **Rotate Compromised Credentials**
   ```bash
   # Force immediate secret rotation
   kubectl create job --from=cronjob/rotate-secrets emergency-rotation-$(date +%s) -n nethical
   
   # Monitor rotation progress
   kubectl logs -f -n nethical $(kubectl get pods -n nethical -l job=emergency-rotation --sort-by=.metadata.creationTimestamp -o name | tail -1)
   ```

3. **Update Security Rules**
   ```bash
   # Add emergency WAF rule
   kubectl patch configmap nethical-waf-rules -n nethical --patch '
   data:
     emergency-rules.conf: |
       # Block specific attack pattern
       SecRule REQUEST_URI "@rx /malicious/path" "id:9999,deny,status:403"
   '
   
   # Reload ingress controller
   kubectl rollout restart deployment/ingress-nginx-controller -n ingress-nginx
   ```

**Recovery (4-24 hours):**

1. **Restore Normal Operations**
   ```bash
   # Remove emergency lockdown
   kubectl delete networkpolicy emergency-lockdown -n nethical
   
   # Disable quarantine mode
   kubectl set env statefulset/nethical -n nethical NETHICAL_ENABLE_QUARANTINE=false
   
   # Restore normal log level
   kubectl set env statefulset/nethical -n nethical LOG_LEVEL=INFO
   ```

2. **Post-Incident Analysis**
   - Root cause identification
   - Timeline reconstruction
   - Impact assessment
   - Lessons learned documentation

3. **Implement Preventive Measures**
   - Update security policies
   - Enhance monitoring rules
   - Improve detection capabilities
   - Conduct security training

---

## Vulnerability Management

### Critical Vulnerability Response (SLA: 24 hours)

**Automated Detection:**
```bash
# Vulnerability scan runs automatically every 6 hours
# Check latest scan results
gh run view $(gh run list --workflow=vuln-sla.yml --limit 1 --json databaseId -q '.[0].databaseId')
```

**Manual Verification:**
```bash
# Run on-demand scan
trivy fs --severity CRITICAL,HIGH --format json --output vulns.json .

# Check SLA compliance
python scripts/check-vuln-sla.py vulns.json
```

**Remediation Process:**

1. **Assess Vulnerability**
   ```bash
   # Get vulnerability details
   trivy fs --severity CRITICAL --format table .
   
   # Check if exploit is publicly available
   searchsploit <CVE-ID>
   ```

2. **Prioritize and Plan**
   - **Immediate:** Publicly exploited, internet-facing
   - **Urgent:** Exploitable but not publicly exploited
   - **Scheduled:** Requires complex conditions

3. **Apply Fix**
   ```bash
   # Update dependency
   pip install --upgrade <package>==<fixed-version>
   
   # Or update requirements
   echo "<package>==<fixed-version>" >> requirements.txt
   pip-compile requirements.in
   
   # Rebuild and test
   docker build -t nethical:patched-$(date +%s) .
   ```

4. **Deploy Patch**
   ```bash
   # Deploy to staging first
   kubectl set image statefulset/nethical -n nethical-staging nethical=nethical:patched-$(date +%s)
   
   # Verify no issues
   kubectl logs -n nethical-staging -l app=nethical --tail=100
   
   # Deploy to production
   kubectl set image statefulset/nethical -n nethical nethical=nethical:patched-$(date +%s)
   ```

5. **Verify Fix**
   ```bash
   # Re-scan to confirm vulnerability is resolved
   trivy image nethical:patched-$(date +%s) --severity CRITICAL,HIGH
   
   # Update tracking issue
   gh issue comment <issue-number> --body "Vulnerability patched and verified in version patched-$(date +%s)"
   ```

### High Vulnerability Response (SLA: 72 hours)

Follow same process as critical but with extended timeline for testing and validation.

---

## Secret Rotation

### Routine Rotation (Every 30 Days)

**Automated Process:**
```bash
# CronJob runs automatically every 30 days
kubectl get cronjob rotate-secrets -n nethical

# Check last run
kubectl get jobs -n nethical -l cronjob=rotate-secrets --sort-by=.metadata.creationTimestamp
```

**Manual Rotation (Emergency):**

1. **Trigger Rotation**
   ```bash
   # Create one-time job from CronJob
   kubectl create job --from=cronjob/rotate-secrets manual-rotation-$(date +%s) -n nethical
   ```

2. **Monitor Progress**
   ```bash
   # Watch job status
   kubectl get job manual-rotation-* -n nethical -w
   
   # View logs
   kubectl logs -f -n nethical $(kubectl get pods -n nethical -l job-name=manual-rotation-* -o name | head -1)
   ```

3. **Verify Success**
   ```bash
   # Check external secrets sync status
   kubectl get externalsecrets -n nethical
   
   # Verify pods picked up new secrets
   kubectl get pods -n nethical -l app=nethical -o jsonpath='{.items[*].status.containerStatuses[*].restartCount}'
   ```

### Individual Secret Rotation

**JWT Signing Keys:**
```bash
# Generate new RSA key pair
openssl genrsa -out jwt-private.key 4096
openssl rsa -in jwt-private.key -pubout -out jwt-public.key

# Update in Vault
vault kv put nethical/jwt \
  private_key="$(cat jwt-private.key | base64 -w0)" \
  public_key="$(cat jwt-public.key | base64 -w0)" \
  rotated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Trigger external secret refresh
kubectl annotate externalsecret nethical-jwt-keys -n nethical \
  force-sync="$(date +%s)" --overwrite

# Wait for propagation
sleep 30

# Restart pods
kubectl rollout restart statefulset/nethical -n nethical
```

**Database Password:**
```bash
# Generate new password
NEW_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

# Update in database
kubectl exec -n nethical postgresql-0 -- psql -U postgres -c \
  "ALTER USER nethical WITH PASSWORD '$NEW_PASSWORD';"

# Update in Vault
vault kv put nethical/database \
  username="nethical" \
  password="$NEW_PASSWORD" \
  rotated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Trigger refresh and restart
kubectl annotate externalsecret nethical-database -n nethical \
  force-sync="$(date +%s)" --overwrite
sleep 30
kubectl rollout restart statefulset/nethical -n nethical
```

---

## Network Policy Updates

### Adding New Service

1. **Create Allow Policy**
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: nethical-to-newservice
     namespace: nethical
   spec:
     podSelector:
       matchLabels:
         app: nethical
     policyTypes:
     - Egress
     egress:
     - to:
       - podSelector:
           matchLabels:
             app: newservice
       ports:
       - protocol: TCP
         port: 8080
   ```

2. **Apply and Test**
   ```bash
   # Apply policy
   kubectl apply -f new-service-policy.yaml
   
   # Test connectivity
   kubectl run test -it --rm --image=nicolaka/netshoot -n nethical -- \
     curl http://newservice.nethical.svc.cluster.local:8080
   ```

3. **Monitor Impact**
   ```bash
   # Check for denied connections
   kubectl logs -n nethical -l app=nethical | grep "connection refused\|network unreachable"
   ```

### Troubleshooting Network Issues

```bash
# Check applied policies
kubectl get networkpolicies -n nethical

# Describe specific policy
kubectl describe networkpolicy <policy-name> -n nethical

# Test connectivity from pod
kubectl exec -n nethical nethical-0 -- curl -v http://service:port

# Check if traffic is being blocked
kubectl logs -n kube-system -l k8s-app=cilium | grep "Policy denied"
```

---

## mTLS Certificate Management

### Certificate Rotation

**Automatic (cert-manager):**
```bash
# Certificates auto-renew 30 days before expiration
# Check certificate status
kubectl get certificates -n nethical

# Force renewal if needed
kubectl delete certificate nethical-tls -n nethical
# cert-manager will recreate it automatically
```

**Manual Verification:**
```bash
# Check certificate expiration
kubectl get secret nethical-tls -n nethical -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -enddate

# Verify certificate chain
kubectl get secret nethical-tls -n nethical -o jsonpath='{.data.tls\.crt}' | \
  base64 -d | openssl x509 -noout -text
```

### Testing mTLS

```bash
# Verify mTLS is enabled
kubectl get peerauthentication default -n nethical -o yaml

# Test service-to-service mTLS
kubectl exec -n nethical nethical-0 -- \
  curl -v --cacert /etc/certs/root-cert.pem \
       --cert /etc/certs/cert-chain.pem \
       --key /etc/certs/key.pem \
       https://redis.nethical.svc.cluster.local:6379
```

---

## Security Monitoring

### Key Metrics to Monitor

1. **Authentication Failures**
   ```promql
   rate(nethical_auth_failures_total[5m]) > 10
   ```

2. **WAF Blocks**
   ```promql
   rate(nginx_ingress_controller_requests{status="403"}[5m]) > 5
   ```

3. **Network Policy Violations**
   ```promql
   rate(cilium_policy_denied_total[5m]) > 0
   ```

4. **Suspicious Activity**
   ```promql
   rate(nethical_blocked_actions_total[5m]) > 2
   ```

### Alert Configuration

See Prometheus alerting rules in `deploy/monitoring/alerts.yaml`

---

## Audit Log Review

### Daily Review

```bash
# Export last 24 hours of audit logs
kubectl exec -n nethical nethical-0 -- python -c "
from nethical.governance.phase4_security import AuditLogger
import json
logger = AuditLogger()
logs = logger.get_logs(hours=24)
print(json.dumps(logs, indent=2))
" > audit-$(date +%Y%m%d).json

# Analyze patterns
cat audit-$(date +%Y%m%d).json | jq '.[] | select(.action_type=="BLOCK")'
cat audit-$(date +%Y%m%d).json | jq '.[] | select(.severity=="CRITICAL")'
```

### Compliance Reporting

```bash
# Generate monthly audit report
kubectl exec -n nethical nethical-0 -- python -c "
from nethical.governance.phase4_security import AuditLogger
logger = AuditLogger()
report = logger.generate_compliance_report(days=30)
print(report)
" > compliance-report-$(date +%Y%m).txt
```

---

## Emergency Procedures

### Complete Service Lockdown

```bash
# 1. Block all external traffic
kubectl apply -f deploy/kubernetes/emergency-lockdown.yaml

# 2. Enable quarantine mode
kubectl set env statefulset/nethical -n nethical NETHICAL_ENABLE_QUARANTINE=true

# 3. Notify stakeholders
./scripts/send-emergency-alert.sh "Service lockdown activated"

# 4. Create snapshot
kubectl exec -n nethical nethical-0 -- tar czf /tmp/emergency-backup-$(date +%s).tar.gz /data
```

### Data Breach Response

1. **Immediately stop data access**
2. **Isolate affected systems**
3. **Notify legal/compliance teams**
4. **Preserve evidence**
5. **Follow regulatory requirements (GDPR, CCPA, etc.)**

### Rollback Procedure

```bash
# Quick rollback to previous version
kubectl rollout undo statefulset/nethical -n nethical

# Rollback to specific revision
kubectl rollout undo statefulset/nethical -n nethical --to-revision=<revision>

# Verify rollback
kubectl rollout status statefulset/nethical -n nethical
```

---

## Contact Information

**Security Team:**
- Email: security@nethical.io
- Slack: #security-incidents
- PagerDuty: Security On-Call

**Escalation:**
- L1: Security Engineer (response time: 15 min)
- L2: Security Lead (response time: 30 min)
- L3: CISO (response time: 1 hour)

**External Resources:**
- [NIST Incident Response Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf)
- [OWASP Incident Response](https://owasp.org/www-project-incident-response/)
- [SANS Incident Handler's Handbook](https://www.sans.org/reading-room/whitepapers/incident/incident-handlers-handbook-33901)

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-24  
**Next Review:** 2025-12-24
