# OpenShift Migration Guide

This guide documents the path for migrating Nethical from managed Kubernetes (GKE/EKS/AKS) to Red Hat OpenShift, with considerations for regulated enterprise sectors.

## Table of Contents

- [OpenShift Compatibility Overview](#openshift-compatibility-overview)
- [Key Differences](#key-differences)
- [Migration Checklist](#migration-checklist)
- [Security Context Constraints](#security-context-constraints)
- [Routes and Ingress](#routes-and-ingress)
- [Service Mesh Integration](#service-mesh-integration)
- [Advanced RBAC](#advanced-rbac)
- [Audit and Compliance](#audit-and-compliance)
- [Regulated Sector Considerations](#regulated-sector-considerations)
- [Migration Procedure](#migration-procedure)

## OpenShift Compatibility Overview

Nethical is designed to be compatible with OpenShift with minimal modifications:

| Feature | Kubernetes | OpenShift | Migration Effort |
|---------|------------|-----------|------------------|
| Container Runtime | Any CRI | CRI-O | None (transparent) |
| Ingress | Ingress | Routes | Low |
| Security Context | PSP/PSA | SCC | Medium |
| RBAC | Roles/ClusterRoles | + Projects | Low |
| Service Mesh | Istio/Linkerd | OpenShift Service Mesh | Medium |
| Secrets | Secrets/ESO | + Vault Integration | Low |
| Monitoring | Prometheus | OpenShift Monitoring | Low |
| Logging | Various | OpenShift Logging | Low |

## Key Differences

### 1. Security Context Constraints (SCC)

OpenShift uses SCCs instead of Pod Security Policies/Standards:

**Current Kubernetes Configuration:**
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  seccompProfile:
    type: RuntimeDefault
```

**OpenShift SCC Equivalent:**
```yaml
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: nethical-scc
allowPrivilegedContainer: false
runAsUser:
  type: MustRunAsNonRoot
seLinuxContext:
  type: MustRunAs
fsGroup:
  type: MustRunAs
seccompProfiles:
  - runtime/default
```

### 2. Routes vs Ingress

OpenShift uses Routes natively, though Ingress is also supported:

**Kubernetes Ingress:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nethical
spec:
  rules:
    - host: nethical.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: nethical
                port:
                  number: 8000
```

**OpenShift Route:**
```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: nethical
spec:
  host: nethical.apps.example.com
  to:
    kind: Service
    name: nethical
    weight: 100
  port:
    targetPort: http
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
```

### 3. Projects vs Namespaces

OpenShift extends namespaces with Projects:

```yaml
apiVersion: project.openshift.io/v1
kind: Project
metadata:
  name: nethical
  annotations:
    openshift.io/description: "Nethical AI Safety Governance"
    openshift.io/display-name: "Nethical"
    openshift.io/requester: "admin"
```

## Migration Checklist

### Pre-Migration

- [ ] Review OpenShift version compatibility (4.10+)
- [ ] Audit container image for OpenShift compatibility
- [ ] Review SCC requirements
- [ ] Verify network policy compatibility
- [ ] Plan for Route configuration
- [ ] Review RBAC and project structure

### Infrastructure

- [ ] Create OpenShift project
- [ ] Configure SCC for Nethical
- [ ] Set up persistent storage (OpenShift Data Foundation)
- [ ] Configure Routes or Ingress
- [ ] Set up OpenShift Monitoring integration
- [ ] Configure OpenShift Logging

### Application

- [ ] Update Helm values for OpenShift
- [ ] Migrate secrets to OpenShift Secrets/Vault
- [ ] Configure Service Mesh (if using)
- [ ] Update CI/CD pipelines
- [ ] Validate compliance controls

### Post-Migration

- [ ] Verify all pods running
- [ ] Test Routes/Ingress
- [ ] Validate monitoring and alerting
- [ ] Confirm audit logging
- [ ] Run compliance validation
- [ ] Update documentation

## Security Context Constraints

### Create Nethical SCC

```yaml
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraints
metadata:
  name: nethical-restricted
  annotations:
    kubernetes.io/description: "Nethical restricted SCC for production workloads"
# Run as specific non-root user
runAsUser:
  type: MustRunAsNonRoot
  uidRangeMin: 1000
  uidRangeMax: 65535
# Specific supplemental groups
supplementalGroups:
  type: MustRunAs
  ranges:
    - min: 1000
      max: 65535
# Specific FSGroup
fsGroup:
  type: MustRunAs
  ranges:
    - min: 1000
      max: 65535
# SELinux context
seLinuxContext:
  type: MustRunAs
# Read-only root filesystem (commented for Nethical compatibility)
# readOnlyRootFilesystem: true
# Seccomp profiles
seccompProfiles:
  - runtime/default
# Capabilities
allowedCapabilities: []
requiredDropCapabilities:
  - ALL
# Volumes
volumes:
  - configMap
  - emptyDir
  - persistentVolumeClaim
  - projected
  - secret
# Network
allowHostNetwork: false
allowHostPorts: false
allowHostPID: false
allowHostIPC: false
allowPrivilegedContainer: false
allowPrivilegeEscalation: false
# Users
users:
  - system:serviceaccount:nethical:nethical
groups: []
```

### Apply SCC

```bash
# Create SCC
oc apply -f nethical-scc.yaml

# Add service account to SCC
oc adm policy add-scc-to-user nethical-restricted \
  -z nethical -n nethical
```

## Routes and Ingress

### Basic Route

```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: nethical-api
  namespace: nethical
  labels:
    app.kubernetes.io/name: nethical
spec:
  host: nethical-api.apps.cluster.example.com
  to:
    kind: Service
    name: nethical
  port:
    targetPort: http
  tls:
    termination: edge
    insecureEdgeTerminationPolicy: Redirect
    certificate: |
      -----BEGIN CERTIFICATE-----
      ...
      -----END CERTIFICATE-----
    key: |
      -----BEGIN RSA PRIVATE KEY-----
      ...
      -----END RSA PRIVATE KEY-----
```

### A/B Testing with Routes

```yaml
# Primary route (90% traffic)
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: nethical-primary
spec:
  host: nethical.apps.example.com
  to:
    kind: Service
    name: nethical-stable
    weight: 90
  alternateBackends:
    - kind: Service
      name: nethical-canary
      weight: 10
```

## Service Mesh Integration

### OpenShift Service Mesh Configuration

```yaml
# Service Mesh Member Roll
apiVersion: maistra.io/v1
kind: ServiceMeshMemberRoll
metadata:
  name: default
  namespace: istio-system
spec:
  members:
    - nethical

---
# Destination Rule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: nethical
  namespace: nethical
spec:
  host: nethical
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
  subsets:
    - name: stable
      labels:
        version: stable
    - name: canary
      labels:
        version: canary

---
# Virtual Service for traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: nethical
  namespace: nethical
spec:
  hosts:
    - nethical
  http:
    - match:
        - headers:
            x-canary:
              exact: "true"
      route:
        - destination:
            host: nethical
            subset: canary
    - route:
        - destination:
            host: nethical
            subset: stable
          weight: 90
        - destination:
            host: nethical
            subset: canary
          weight: 10
```

## Advanced RBAC

### OpenShift RBAC Extensions

```yaml
# Project Admin Role
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nethical-project-admin
rules:
  - apiGroups: [""]
    resources: ["*"]
    verbs: ["*"]
  - apiGroups: ["apps", "batch", "networking.k8s.io"]
    resources: ["*"]
    verbs: ["*"]
  - apiGroups: ["route.openshift.io"]
    resources: ["routes"]
    verbs: ["*"]
  - apiGroups: ["security.openshift.io"]
    resources: ["securitycontextconstraints"]
    verbs: ["get", "list", "watch"]

---
# Project View Role (Auditors)
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: nethical-project-view
rules:
  - apiGroups: ["*"]
    resources: ["*"]
    verbs: ["get", "list", "watch"]
```

## Audit and Compliance

### OpenShift Audit Logging

OpenShift provides built-in audit logging:

```yaml
# Audit Policy
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  # Log all requests at Metadata level
  - level: Metadata
    namespaces: ["nethical"]
    
  # Log request/response bodies for secret access
  - level: RequestResponse
    namespaces: ["nethical"]
    resources:
      - group: ""
        resources: ["secrets"]
        
  # Log all write operations in detail
  - level: RequestResponse
    namespaces: ["nethical"]
    verbs: ["create", "update", "patch", "delete"]
```

### Compliance Operator Integration

```yaml
# ScanSettingBinding for compliance scanning
apiVersion: compliance.openshift.io/v1alpha1
kind: ScanSettingBinding
metadata:
  name: nethical-compliance
  namespace: openshift-compliance
profiles:
  - name: rhcos4-moderate
    kind: Profile
    apiGroup: compliance.openshift.io/v1alpha1
  - name: ocp4-moderate
    kind: Profile
    apiGroup: compliance.openshift.io/v1alpha1
settingsRef:
  name: default
  kind: ScanSetting
  apiGroup: compliance.openshift.io/v1alpha1
```

## Regulated Sector Considerations

### Healthcare (HIPAA)

OpenShift-specific HIPAA considerations:

- Enable OpenShift File Integrity Operator
- Configure encrypted etcd
- Use OpenShift Compliance Operator with HIPAA profiles
- Enable PCI-DSS-compliant logging

### Government (FedRAMP)

OpenShift-specific FedRAMP considerations:

- Use FIPS-enabled OpenShift
- Configure compliant TLS settings
- Enable Compliance Operator with NIST profiles
- Implement network segmentation with NetworkPolicy

### Defense (IL4/IL5)

OpenShift-specific considerations:

- Use Red Hat OpenShift on AWS GovCloud
- Enable STIG compliance profiles
- Configure CAC/PIV authentication
- Implement Zero Trust architecture

## Migration Procedure

### Step 1: Prepare OpenShift Environment

```bash
# Create project
oc new-project nethical \
  --display-name="Nethical AI Governance" \
  --description="AI Safety Governance System"

# Apply SCC
oc apply -f openshift/nethical-scc.yaml
oc adm policy add-scc-to-user nethical-restricted -z nethical -n nethical
```

### Step 2: Deploy Storage

```bash
# Create PVC (OpenShift Data Foundation)
oc apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nethical-data
  namespace: nethical
spec:
  storageClassName: ocs-storagecluster-ceph-rbd
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF
```

### Step 3: Deploy Application

```bash
# Apply OpenShift-specific values
helm install nethical deploy/helm/nethical \
  -f values-openshift.yaml \
  -n nethical

# Create Route
oc apply -f openshift/route.yaml
```

### Step 4: Validate

```bash
# Check pods
oc get pods -n nethical

# Check route
oc get route -n nethical

# Test endpoint
curl https://nethical.apps.cluster.example.com/health

# Verify SCC
oc describe pod nethical-0 -n nethical | grep scc
```

### Step 5: Enable Monitoring

```bash
# Enable user workload monitoring
oc apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-monitoring-config
  namespace: openshift-monitoring
data:
  config.yaml: |
    enableUserWorkload: true
EOF

# Create ServiceMonitor
oc apply -f openshift/servicemonitor.yaml
```

## OpenShift Values File

Create `values-openshift.yaml`:

```yaml
# OpenShift-specific values
replicaCount: 3

image:
  repository: nethical
  tag: "0.1.0"
  pullPolicy: Always

# OpenShift security context
podSecurityContext:
  runAsNonRoot: true
  # Let OpenShift assign UID from range
  # runAsUser: null

securityContext:
  capabilities:
    drop:
      - ALL
  allowPrivilegeEscalation: false
  # readOnlyRootFilesystem: true  # Uncomment if compatible

# Use OpenShift Route instead of Ingress
ingress:
  enabled: false

# OpenShift-specific storage class
persistence:
  enabled: true
  storageClass: "ocs-storagecluster-ceph-rbd"
  size: 100Gi

# Service for Route
service:
  type: ClusterIP
  port: 8000

# Enable OpenShift monitoring integration
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8888"

# OpenShift tolerations
tolerations:
  - key: "node-role.kubernetes.io/infra"
    operator: "Exists"
```

## Support

For OpenShift-specific issues:
- Red Hat Support Portal
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
