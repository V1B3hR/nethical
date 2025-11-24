#!/bin/bash
# Test network isolation and zero-trust policies
# Verifies that network policies are enforcing zero-trust segmentation

set -euo pipefail

NAMESPACE="${NAMESPACE:-nethical}"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}Network Isolation Testing${NC}"
echo "=========================="
echo

# Function to print status
print_status() {
    local status=$1
    local message=$2
    
    if [ "$status" == "PASS" ]; then
        echo -e "${GREEN}✓${NC} $message"
    elif [ "$status" == "FAIL" ]; then
        echo -e "${RED}✗${NC} $message"
    elif [ "$status" == "EXPECTED_FAIL" ]; then
        echo -e "${GREEN}✓${NC} $message (connection blocked as expected)"
    else
        echo "  $message"
    fi
}

# Check prerequisites
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo -e "${RED}Error: Namespace '$NAMESPACE' does not exist${NC}"
    exit 1
fi

# Deploy test pod
echo "Deploying test pod..."
cat <<EOF | kubectl apply -f - > /dev/null
apiVersion: v1
kind: Pod
metadata:
  name: network-test
  namespace: $NAMESPACE
  labels:
    app: network-test
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot
    command: ["sleep", "3600"]
  restartPolicy: Never
EOF

# Wait for pod to be ready
echo "Waiting for test pod to be ready..."
kubectl wait --for=condition=Ready pod/network-test -n "$NAMESPACE" --timeout=60s 2>/dev/null || {
    echo -e "${RED}Failed to start test pod${NC}"
    kubectl delete pod network-test -n "$NAMESPACE" --ignore-not-found=true
    exit 1
}

echo
echo "Running network isolation tests..."
echo

# Test 1: Should NOT be able to reach external internet (denied by default)
echo -e "${BOLD}Test 1: External internet access (should be blocked)${NC}"
if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 curl -s -o /dev/null -w "%{http_code}" https://google.com 2>/dev/null | grep -q 200; then
    print_status "FAIL" "External internet access ALLOWED (security issue!)"
else
    print_status "EXPECTED_FAIL" "External internet access blocked"
fi

# Test 2: Should NOT be able to reach other namespaces
echo -e "${BOLD}Test 2: Cross-namespace access (should be blocked)${NC}"
if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 curl -s -o /dev/null kube-dns.kube-system.svc.cluster.local:53 2>/dev/null; then
    print_status "FAIL" "Cross-namespace access ALLOWED (security issue!)"
else
    print_status "EXPECTED_FAIL" "Cross-namespace access blocked"
fi

# Test 3: DNS should work (explicitly allowed)
echo -e "${BOLD}Test 3: DNS resolution (should work)${NC}"
if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 nslookup kubernetes.default.svc.cluster.local 2>/dev/null | grep -q "Address"; then
    print_status "PASS" "DNS resolution works"
else
    print_status "FAIL" "DNS resolution failed (check network policy)"
fi

# Test 4: Should be able to reach allowed services (Redis)
echo -e "${BOLD}Test 4: Redis access (should work if Redis is deployed)${NC}"
if kubectl get service redis -n "$NAMESPACE" &> /dev/null; then
    # Create a temporary pod with proper labels
    kubectl label pod network-test app=nethical --overwrite > /dev/null 2>&1
    
    if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 nc -zv redis.nethical.svc.cluster.local 6379 2>&1 | grep -q "succeeded"; then
        print_status "PASS" "Redis access allowed for labeled pod"
    else
        print_status "FAIL" "Redis access blocked (check network policy)"
    fi
    
    # Remove label
    kubectl label pod network-test app- --overwrite > /dev/null 2>&1
else
    print_status "INFO" "Redis service not deployed, skipping test"
fi

# Test 5: Should NOT be able to access pods in other namespaces directly
echo -e "${BOLD}Test 5: Direct pod-to-pod access across namespaces (should be blocked)${NC}"
RANDOM_POD=$(kubectl get pods -A -o jsonpath='{.items[0].status.podIP}' 2>/dev/null || echo "")
if [ -n "$RANDOM_POD" ]; then
    if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 ping -c 1 "$RANDOM_POD" 2>/dev/null | grep -q "1 packets transmitted, 1 received"; then
        print_status "FAIL" "Direct pod access ALLOWED (security issue!)"
    else
        print_status "EXPECTED_FAIL" "Direct pod access blocked"
    fi
else
    print_status "INFO" "No other pods found to test"
fi

# Test 6: Egress to monitoring namespace (if allowed)
echo -e "${BOLD}Test 6: Monitoring access (depends on policy)${NC}"
if kubectl get namespace monitoring &> /dev/null; then
    if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 nc -zv prometheus.monitoring.svc.cluster.local 9090 2>&1 | grep -q "succeeded"; then
        print_status "INFO" "Monitoring access allowed"
    else
        print_status "INFO" "Monitoring access blocked (depends on policy)"
    fi
else
    print_status "INFO" "Monitoring namespace not found, skipping test"
fi

# Test 7: Verify default deny is in place
echo -e "${BOLD}Test 7: Default deny policy verification${NC}"
if kubectl get networkpolicy default-deny-all -n "$NAMESPACE" &> /dev/null; then
    print_status "PASS" "Default deny-all policy exists"
    
    # Check policy configuration
    POLICY_TYPES=$(kubectl get networkpolicy default-deny-all -n "$NAMESPACE" -o jsonpath='{.spec.policyTypes}')
    if echo "$POLICY_TYPES" | grep -q "Ingress" && echo "$POLICY_TYPES" | grep -q "Egress"; then
        print_status "PASS" "Default deny covers both Ingress and Egress"
    else
        print_status "FAIL" "Default deny does not cover both Ingress and Egress"
    fi
else
    print_status "FAIL" "Default deny-all policy NOT found"
fi

# Test 8: Port scanning attempt (should be blocked)
echo -e "${BOLD}Test 8: Port scanning to random service (should be blocked)${NC}"
if kubectl exec network-test -n "$NAMESPACE" -- timeout 5 nc -zv 10.96.0.1 443 2>&1 | grep -q "succeeded"; then
    print_status "FAIL" "Port scanning succeeded (potential security issue)"
else
    print_status "EXPECTED_FAIL" "Port scanning blocked"
fi

echo
echo "Cleaning up test pod..."
kubectl delete pod network-test -n "$NAMESPACE" --ignore-not-found=true > /dev/null

echo
echo -e "${BOLD}Network Isolation Test Complete${NC}"
echo
echo "Summary:"
echo "- Zero-trust network policies enforce deny-all by default"
echo "- Only explicitly allowed connections should succeed"
echo "- All unexpected connection attempts should be blocked"
echo
echo "For production deployments:"
echo "1. Ensure all tests marked as 'FAIL' are resolved"
echo "2. Verify that only required services are accessible"
echo "3. Monitor network policy violations in logs"
echo "4. Regularly audit and update network policies"
