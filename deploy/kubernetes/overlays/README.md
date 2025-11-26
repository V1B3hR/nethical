# Kubernetes Overlays for Nethical

This directory contains Kustomize overlays for deploying Nethical with specific compliance and cloud provider configurations.

## Directory Structure

```
overlays/
├── base/                    # Base configuration (common resources)
├── compliance/              # Compliance-specific overlays
│   ├── hipaa/              # HIPAA (US Healthcare)
│   ├── nhs-dspt/           # NHS Data Security and Protection Toolkit
│   ├── eu-mdr/             # EU Medical Device Regulation
│   ├── fips/               # FIPS 140-2/140-3 compliance
│   ├── nist/               # NIST 800-53 controls
│   └── ferpa/              # FERPA (US Education)
└── cloud-providers/         # Cloud-specific overlays
    ├── gke/                # Google Kubernetes Engine
    ├── eks/                # Amazon Elastic Kubernetes Service
    └── aks/                # Azure Kubernetes Service
```

## Usage

### Deploy with a Compliance Overlay

```bash
# Deploy with HIPAA compliance
kubectl apply -k overlays/compliance/hipaa/

# Deploy with NHS DSPT compliance
kubectl apply -k overlays/compliance/nhs-dspt/

# Deploy with FIPS compliance
kubectl apply -k overlays/compliance/fips/
```

### Deploy with Cloud Provider Overlay

```bash
# Deploy on GKE
kubectl apply -k overlays/cloud-providers/gke/

# Deploy on EKS
kubectl apply -k overlays/cloud-providers/eks/

# Deploy on AKS
kubectl apply -k overlays/cloud-providers/aks/
```

### Combine Compliance and Cloud Provider Overlays

Create a custom overlay that references both:

```yaml
# custom-overlay/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../compliance/hipaa
  - ../cloud-providers/gke
```

Then apply:

```bash
kubectl apply -k custom-overlay/
```

## Overlay Details

### Compliance Overlays

| Overlay | Sector | Key Features |
|---------|--------|--------------|
| `hipaa` | Healthcare (US) | PHI encryption, audit logging, BAA requirements |
| `nhs-dspt` | Healthcare (UK) | NHS Data Security toolkit compliance |
| `eu-mdr` | Medical Devices (EU) | EU MDR Article 62 requirements |
| `fips` | Government/Defense | FIPS 140-2/3 cryptographic modules |
| `nist` | Government (US) | NIST 800-53 controls implementation |
| `ferpa` | Education (US) | Student data protection |

### Cloud Provider Overlays

| Overlay | Provider | Key Features |
|---------|----------|--------------|
| `gke` | Google Cloud | Workload Identity, GCP Secret Manager |
| `eks` | AWS | IRSA, AWS Secrets Manager, EBS CSI |
| `aks` | Azure | Workload Identity, Azure Key Vault |

## Configuration

Each overlay contains:

- `kustomization.yaml` - Kustomize configuration
- `patches/` - JSON patches for resource modifications
- `resources/` - Additional resources specific to the overlay
- `configmap.yaml` - Environment-specific configuration
