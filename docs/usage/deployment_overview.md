# Nethical Deployment

This directory contains deployment configurations for Nethical across Kubernetes and Terraform.

## Quick Links

- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Kubernetes Manifests](kubernetes/README.md)
- [Helm Chart](helm/nethical/README.md)
- [Terraform Multi-Region](terraform/multi-region/)

## Secret Provisioning

**IMPORTANT**: Secret manifests in this repository do NOT contain actual secret values. You must provision secrets before deployment using one of the methods below.

### Kubernetes Secrets

#### Option 1: kubectl create secret (Development/Testing)

```bash
# Nethical application secrets
kubectl create secret generic nethical-secrets \
  --namespace nethical \
  --from-literal=DB_PASSWORD='<your-db-password>' \
  --from-literal=API_KEY='<your-api-key>' \
  --from-literal=REDIS_PASSWORD='<your-redis-password>'

# PostgreSQL credentials
kubectl create secret generic postgres-credentials \
  --namespace nethical-db \
  --from-literal=POSTGRES_USER='nethical' \
  --from-literal=POSTGRES_PASSWORD='<your-strong-password>' \
  --from-literal=POSTGRES_DB='nethical' \
  --from-literal=REPLICATION_USER='replicator' \
  --from-literal=REPLICATION_PASSWORD='<your-replication-password>' \
  --from-literal=DATA_SOURCE_NAME='postgresql://nethical:<password>@localhost:5432/nethical?sslmode=disable'

# MinIO credentials
kubectl create secret generic minio-credentials \
  --namespace nethical-storage \
  --from-literal=MINIO_ROOT_USER='<your-admin-user>' \
  --from-literal=MINIO_ROOT_PASSWORD='<your-strong-password>' \
  --from-literal=ACCESS_KEY='<your-app-access-key>' \
  --from-literal=SECRET_KEY='<your-app-secret-key>'
```

#### Option 2: External Secrets Operator (Production Recommended)

Install the [External Secrets Operator](https://external-secrets.io/):

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  --namespace external-secrets \
  --create-namespace
```

Example ExternalSecret for AWS Secrets Manager:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: nethical-secrets
  namespace: nethical
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: nethical-secrets
    creationPolicy: Owner
  data:
    - secretKey: DB_PASSWORD
      remoteRef:
        key: prod/nethical/database
        property: password
    - secretKey: API_KEY
      remoteRef:
        key: prod/nethical/api
        property: key
```

#### Option 3: Sealed Secrets

Install [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets):

```bash
# Install controller
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Seal your secrets
kubeseal --format yaml < my-secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

### Terraform Variables

Sensitive Terraform variables should be passed via CI secrets or environment variables. **Never commit secret values to terraform.tfvars files.**

#### Required Variables by Module

**RDS Module (`deploy/terraform/multi-region/modules/rds/`):**
- `kms_admin_iam_arn` - IAM ARN of the KMS key administrator
- `rds_db_identifier` - (Optional) RDS DB identifier for KMS policy constraints

**EKS Module (`deploy/terraform/multi-region/modules/eks/`):**
- `kms_admin_iam_arn` - IAM ARN of the KMS key administrator
- `eks_cluster_arn` - (Optional) EKS cluster ARN for KMS policy constraints

**Redis Global Module (`deploy/terraform/multi-region/modules/redis-global/`):**
- `elasticache_kms_key_arn` - (Optional) KMS key ARN for at-rest encryption
- `redis_auth_token` - (Sensitive) Redis AUTH token (min 16 characters)

#### Passing Variables via CI

**GitHub Actions Example:**

```yaml
- name: Terraform Apply
  env:
    TF_VAR_kms_admin_iam_arn: ${{ secrets.KMS_ADMIN_IAM_ARN }}
    TF_VAR_redis_auth_token: ${{ secrets.REDIS_AUTH_TOKEN }}
  run: |
    terraform apply -auto-approve
```

**GitLab CI Example:**

```yaml
terraform_apply:
  script:
    - export TF_VAR_kms_admin_iam_arn="$KMS_ADMIN_IAM_ARN"
    - export TF_VAR_redis_auth_token="$REDIS_AUTH_TOKEN"
    - terraform apply -auto-approve
```

#### Using terraform.tfvars.example

Create a `terraform.tfvars` file from the example (do not commit actual values):

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
# Add terraform.tfvars to .gitignore
```

## Security Best Practices

1. **Never commit secrets** - All secret values must be provisioned externally
2. **Use external secret management** - HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager
3. **Rotate secrets regularly** - Implement automated secret rotation
4. **Least privilege** - KMS key policies are configured for least-privilege access
5. **Encryption at rest** - All databases and caches use encryption at rest
6. **Encryption in transit** - TLS/SSL enabled for all network communication

## Directory Structure

```
deploy/
├── README.md                    # This file
├── DEPLOYMENT_GUIDE.md          # Detailed deployment guide
├── kubernetes/                  # Kubernetes manifests
│   ├── secret.yaml             # Secret template (no values)
│   ├── configmap.yaml
│   ├── statefulset.yaml
│   └── ...
├── postgres/                    # PostgreSQL HA cluster
│   └── ha-cluster.yaml         # Includes secret template
├── minio/                       # MinIO object storage
│   └── cluster.yaml            # Includes secret template
├── helm/                        # Helm charts
│   └── nethical/
├── terraform/                   # Terraform configurations
│   └── multi-region/
│       └── modules/
│           ├── rds/            # RDS with KMS encryption
│           ├── eks/            # EKS with KMS encryption
│           └── redis-global/   # Redis with encryption
└── ...
```

## Support

- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical
