#!/bin/bash
# Comprehensive security controls verification script
# Verifies all mandatory security hardening controls

set -euo pipefail

NAMESPACE="${NAMESPACE:-nethical}"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BOLD}Nethical Security Controls Verification${NC}"
echo "========================================"
echo

# Track overall status
PASSED=0
FAILED=0
WARNINGS=0

# Function to print status
print_status() {
    local status=$1
    local message=$2
    
    if [ "$status" == "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
        ((PASSED++))
    elif [ "$status" == "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
        ((FAILED++))
    elif [ "$status" == "WARN" ]; then
        echo -e "${YELLOW}⚠${NC} $message"
        ((WARNINGS++))
    else
        echo "  $message"
    fi
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}Error: kubectl is not installed${NC}"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${YELLOW}Warning: Namespace '$NAMESPACE' does not exist. Some checks will be skipped.${NC}"
    echo
fi

# 1. Network Policies
echo -e "${BOLD}1. Zero Trust Network Segmentation${NC}"
echo "-----------------------------------"

# Check for default deny-all policy
if kubectl get networkpolicy default-deny-all -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "Default deny-all network policy exists"
else
    print_status "FAIL" "Default deny-all network policy NOT found"
fi

# Check for explicit allow policies
NP_COUNT=$(kubectl get networkpolicy -n "$NAMESPACE" 2>/dev/null | wc -l)
if [ "$NP_COUNT" -gt 5 ]; then
    print_status "PASS" "Multiple network policies configured ($NP_COUNT policies)"
else
    print_status "WARN" "Few network policies found ($NP_COUNT policies)"
fi

echo

# 2. mTLS Configuration
echo -e "${BOLD}2. mTLS Between Services${NC}"
echo "------------------------"

# Check for Istio PeerAuthentication
if kubectl get peerauthentication default -n "$NAMESPACE" &> /dev/null; then
    MODE=$(kubectl get peerauthentication default -n "$NAMESPACE" -o jsonpath='{.spec.mtls.mode}' 2>/dev/null)
    if [ "$MODE" == "STRICT" ]; then
        print_status "PASS" "Istio mTLS configured with STRICT mode"
    else
        print_status "WARN" "Istio mTLS mode is $MODE (expected STRICT)"
    fi
else
    print_status "WARN" "Istio PeerAuthentication not found (check if service mesh is installed)"
fi

# Check for cert-manager certificates
if command -v kubectl &> /dev/null && kubectl get crd certificates.cert-manager.io &> /dev/null; then
    CERT_COUNT=$(kubectl get certificates -n "$NAMESPACE" 2>/dev/null | wc -l)
    if [ "$CERT_COUNT" -gt 0 ]; then
        print_status "PASS" "TLS certificates managed by cert-manager ($CERT_COUNT certificates)"
    else
        print_status "WARN" "No certificates found in cert-manager"
    fi
fi

echo

# 3. Secret Management
echo -e "${BOLD}3. Secret Management & Rotation${NC}"
echo "-------------------------------"

# Check for External Secrets
if kubectl get crd externalsecrets.external-secrets.io &> /dev/null; then
    ES_COUNT=$(kubectl get externalsecrets -n "$NAMESPACE" 2>/dev/null | wc -l)
    if [ "$ES_COUNT" -gt 0 ]; then
        print_status "PASS" "External Secrets configured ($ES_COUNT secrets)"
    else
        print_status "FAIL" "No External Secrets found"
    fi
else
    print_status "WARN" "External Secrets CRD not installed"
fi

# Check for secret rotation CronJob
if kubectl get cronjob rotate-secrets -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "Secret rotation CronJob configured"
    
    # Check schedule
    SCHEDULE=$(kubectl get cronjob rotate-secrets -n "$NAMESPACE" -o jsonpath='{.spec.schedule}')
    print_status "INFO" "Rotation schedule: $SCHEDULE"
else
    print_status "FAIL" "Secret rotation CronJob NOT found"
fi

# Check for inline secrets (should not exist in production)
INLINE_SECRETS=$(kubectl get secrets -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.items[] | select(.type == "Opaque") | select(.metadata.annotations["external-secrets.io/secret-store"] == null) | .metadata.name' | wc -l)
if [ "$INLINE_SECRETS" -gt 5 ]; then
    print_status "WARN" "Multiple inline secrets found ($INLINE_SECRETS) - consider externalizing"
else
    print_status "PASS" "Minimal inline secrets ($INLINE_SECRETS)"
fi

echo

# 4. Pod Security Context
echo -e "${BOLD}4. Runtime Security${NC}"
echo "-------------------"

# Check if pods run as non-root
if kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical &> /dev/null; then
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$POD_NAME" ]; then
        RUN_AS_USER=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.runAsUser}' 2>/dev/null)
        if [ "$RUN_AS_USER" != "0" ] && [ -n "$RUN_AS_USER" ]; then
            print_status "PASS" "Pods run as non-root user (UID: $RUN_AS_USER)"
        else
            print_status "FAIL" "Pods may be running as root"
        fi
        
        # Check for read-only root filesystem
        READ_ONLY=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.readOnlyRootFilesystem}' 2>/dev/null)
        if [ "$READ_ONLY" == "true" ]; then
            print_status "PASS" "Read-only root filesystem enabled"
        else
            print_status "WARN" "Read-only root filesystem NOT enabled"
        fi
        
        # Check for privilege escalation
        ALLOW_PRIV=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.allowPrivilegeEscalation}' 2>/dev/null)
        if [ "$ALLOW_PRIV" == "false" ]; then
            print_status "PASS" "Privilege escalation disabled"
        else
            print_status "WARN" "Privilege escalation not explicitly disabled"
        fi
        
        # Check for dropped capabilities
        DROPPED_CAPS=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].securityContext.capabilities.drop}' 2>/dev/null)
        if echo "$DROPPED_CAPS" | grep -q "ALL"; then
            print_status "PASS" "All capabilities dropped"
        else
            print_status "WARN" "Not all capabilities dropped"
        fi
        
        # Check for seccomp profile
        SECCOMP=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.securityContext.seccompProfile.type}' 2>/dev/null)
        if [ -n "$SECCOMP" ]; then
            print_status "PASS" "Seccomp profile configured: $SECCOMP"
        else
            print_status "WARN" "Seccomp profile not configured"
        fi
        
        # Check for AppArmor annotation
        APPARMOR=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.container\.apparmor\.security\.beta\.kubernetes\.io/nethical}' 2>/dev/null)
        if [ -n "$APPARMOR" ]; then
            print_status "PASS" "AppArmor profile configured: $APPARMOR"
        else
            print_status "WARN" "AppArmor profile not configured"
        fi
    fi
else
    print_status "WARN" "No Nethical pods found to verify"
fi

echo

# 5. Ingress & WAF
echo -e "${BOLD}5. Perimeter Security (WAF)${NC}"
echo "----------------------------"

# Check for ingress with WAF annotations
if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
    INGRESS_COUNT=$(kubectl get ingress -n "$NAMESPACE" | wc -l)
    print_status "INFO" "Ingress resources found: $INGRESS_COUNT"
    
    # Check for ModSecurity annotations
    MODSEC=$(kubectl get ingress -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.items[].metadata.annotations["nginx.ingress.kubernetes.io/enable-modsecurity"]' | grep -c "true" || echo 0)
    if [ "$MODSEC" -gt 0 ]; then
        print_status "PASS" "ModSecurity WAF enabled on ingress"
    else
        print_status "WARN" "ModSecurity WAF not enabled on ingress"
    fi
    
    # Check for rate limiting
    RATE_LIMIT=$(kubectl get ingress -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.items[].metadata.annotations["nginx.ingress.kubernetes.io/limit-rps"]' | grep -v null | wc -l)
    if [ "$RATE_LIMIT" -gt 0 ]; then
        print_status "PASS" "Rate limiting configured on ingress"
    else
        print_status "WARN" "Rate limiting not configured"
    fi
    
    # Check for TLS
    TLS_ENABLED=$(kubectl get ingress -n "$NAMESPACE" -o json 2>/dev/null | jq -r '.items[].spec.tls' | grep -v null | wc -l)
    if [ "$TLS_ENABLED" -gt 0 ]; then
        print_status "PASS" "TLS configured on ingress"
    else
        print_status "FAIL" "TLS NOT configured on ingress"
    fi
else
    print_status "WARN" "No ingress resources found"
fi

echo

# 6. RBAC
echo -e "${BOLD}6. Authentication & Authorization${NC}"
echo "----------------------------------"

# Check for ServiceAccount
if kubectl get serviceaccount nethical -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "ServiceAccount 'nethical' exists"
else
    print_status "WARN" "ServiceAccount 'nethical' not found"
fi

# Check for Role
if kubectl get role nethical -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "Role 'nethical' exists"
    
    # Count rules
    RULE_COUNT=$(kubectl get role nethical -n "$NAMESPACE" -o json 2>/dev/null | jq '.rules | length')
    print_status "INFO" "RBAC rules configured: $RULE_COUNT"
else
    print_status "WARN" "Role 'nethical' not found"
fi

# Check for RoleBinding
if kubectl get rolebinding nethical -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "RoleBinding 'nethical' exists"
else
    print_status "WARN" "RoleBinding 'nethical' not found"
fi

echo

# 7. Monitoring & Audit
echo -e "${BOLD}7. Logging & Monitoring${NC}"
echo "-----------------------"

# Check for audit logging
if kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical &> /dev/null; then
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$POD_NAME" ]; then
        # Check if OTEL is configured
        OTEL_ENDPOINT=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].env[?(@.name=="OTEL_EXPORTER_OTLP_ENDPOINT")].value}' 2>/dev/null)
        if [ -n "$OTEL_ENDPOINT" ]; then
            print_status "PASS" "OpenTelemetry configured: $OTEL_ENDPOINT"
        else
            print_status "WARN" "OpenTelemetry not configured"
        fi
    fi
fi

# Check for prometheus scraping
SCRAPE_ANNOTATION=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical -o jsonpath='{.items[0].metadata.annotations.prometheus\.io/scrape}' 2>/dev/null)
if [ "$SCRAPE_ANNOTATION" == "true" ]; then
    print_status "PASS" "Prometheus scraping enabled"
else
    print_status "WARN" "Prometheus scraping not configured"
fi

echo

# 8. Supply Chain Security
echo -e "${BOLD}8. Supply Chain Security${NC}"
echo "-------------------------"

# Check for image signatures (requires cosign)
if command -v cosign &> /dev/null; then
    if kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical &> /dev/null; then
        IMAGE=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=nethical -o jsonpath='{.items[0].spec.containers[0].image}' 2>/dev/null)
        if [ -n "$IMAGE" ]; then
            print_status "INFO" "Container image: $IMAGE"
            # Note: Actual signature verification requires access to the registry
            print_status "INFO" "Run 'cosign verify $IMAGE' to verify signature"
        fi
    fi
else
    print_status "WARN" "cosign not installed - cannot verify image signatures"
fi

# Check for SBOM
if [ -f "sbom.spdx.json" ] || [ -f "sbom.cyclonedx.json" ]; then
    print_status "PASS" "SBOM file found in repository"
else
    print_status "WARN" "SBOM file not found (check CI/CD artifacts)"
fi

echo

# Summary
echo -e "${BOLD}Verification Summary${NC}"
echo "===================="
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo

if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✓ All critical security controls are in place!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some critical security controls are missing!${NC}"
    echo "Please review the failed checks and implement the missing controls."
    exit 1
fi
