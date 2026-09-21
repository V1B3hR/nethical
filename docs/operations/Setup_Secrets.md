# Secret Management Setup Guide

## Overview

Nethical uses environment-driven secrets for all sensitive configuration.  **Never hard-code secrets in code or commit them to version control.**

## Quick Start (Development)

### 1. Create Local Environment File

```bash
# Copy the template
cp .env.example .env

# Generate a secure JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Edit .env and paste the generated secret
nano .env
```

### 2. Load Environment Variables

**Option A: Using python-dotenv (Recommended)**

```bash
pip install python-dotenv
```

```python
# At the start of your application
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

# Now environment variables are available
import os
jwt_secret = os.environ.get("JWT_SECRET")
```

**Option B: Export manually**

```bash
export JWT_SECRET="your-secret-here"
export DB_PASSWORD="your-password"
# ...  etc
```

### 3. Verify Setup

```python
from nethical.security. auth import AuthManager

# This will use JWT_SECRET from environment
auth = AuthManager()

# Create a test token
token, payload = auth.create_access_token("test_user")
print(f"✓ Token created successfully: {token[: 20]}...")
```

## Production Deployment

### GitHub Actions

Add secrets in **Settings → Secrets and variables → Actions**:

1. `JWT_SECRET` - Your production JWT signing key
2. `DB_PASSWORD` - Database password
3. `REDIS_PASSWORD` - Redis password
4. `API_KEY` - External API keys

**Workflow example:**

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    env:
      JWT_SECRET: ${{ secrets.JWT_SECRET }}
      DB_PASSWORD: ${{ secrets.DB_PASSWORD }}
    steps:
      - uses: actions/checkout@v4
      - name: Run application
        run: python -m nethical.api
```

### Kubernetes

**Option 1: kubectl (Quick)**

```bash
kubectl create secret generic nethical-secrets \
  --namespace nethical \
  --from-literal=JWT_SECRET="$(python3 -c 'import secrets; print(secrets. token_urlsafe(32))')" \
  --from-literal=DB_PASSWORD='your-db-password'
```

**Option 2: External Secrets Operator (Production)**

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
  data:
    - secretKey: JWT_SECRET
      remoteRef: 
        key: prod/nethical/jwt
        property: secret
```

**Deployment reference:**

```yaml
apiVersion: apps/v1
kind:  Deployment
metadata:
  name:  nethical
spec:
  template:
    spec:
      containers:
        - name: nethical
          env:
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: nethical-secrets
                  key: JWT_SECRET
```

### Docker / Docker Compose

```yaml
# docker-compose.yml
services:
  nethical:
    image: nethical: latest
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - DB_PASSWORD=${DB_PASSWORD}
    env_file:
      - .env  # Loads from .env file
```

```bash
# Run with environment variables
docker run -e JWT_SECRET="your-secret" nethical: latest
```

## Secret Rotation

### JWT Secret Rotation

1. **Generate new secret:**
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Update in secret store** (Kubernetes/Vault/etc.)

3. **Restart application** to pick up new secret

4. **Invalidate old tokens** (users will need to re-authenticate)

### Automated Rotation (Kubernetes CronJob)

See `deploy/kubernetes/secret-rotation-cronjob.yaml` for automated rotation setup.

## Security Checklist

- [ ] Never commit `.env` files (check `.gitignore`)
- [ ] Use secrets at least 32 characters long
- [ ] Rotate secrets every 90 days
- [ ] Use different secrets for dev/staging/production
- [ ] Store production secrets in a secret manager (Vault, AWS Secrets Manager, etc.)
- [ ] Enable audit logging for secret access
- [ ] Restrict secret access with RBAC

## Troubleshooting

### Error: "Refusing to use insecure literal secret"

**Cause:** AuthManager detected the hard-coded string `"secret"` being used.

**Fix:**
```bash
# Set JWT_SECRET environment variable
export JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

### Error: "secret_key too short"

**Cause:** JWT secret is less than 16 characters.

**Fix:** Generate a proper secret (32+ chars recommended):
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Tokens invalid after restart

**Cause:** Using auto-generated ephemeral secret (not persisted).

**Fix:** Set `JWT_SECRET` environment variable to a fixed value. 

## Additional Resources

- [python-dotenv documentation](https://github.com/theskumar/python-dotenv)
- [External Secrets Operator](https://external-secrets.io/)
- [HashiCorp Vault](https://www.vaultproject.io/)
- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
