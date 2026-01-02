# Plugin Signing & Security Guide

Complete guide for securing and signing Nethical plugins, manifests, and integrations for marketplace distribution.

## Table of Contents

- [Overview](#overview)
- [Why Sign Plugins](#why-sign-plugins)
- [Signing Methods](#signing-methods)
- [Implementation](#implementation)
- [Verification](#verification)
- [Best Practices](#best-practices)
- [Compliance](#compliance)

## Overview

Plugin signing provides cryptographic proof of authenticity and integrity for:

- **Manifest Files**: JSON/YAML plugin descriptors
- **Python Packages**: Nethical distribution packages
- **Container Images**: Docker images for deployment
- **API Endpoints**: REST API with TLS/HTTPS

### Security Benefits

- ✅ **Authenticity**: Verify publisher identity
- ✅ **Integrity**: Detect tampering or modifications
- ✅ **Trust**: Build user confidence
- ✅ **Compliance**: Meet marketplace requirements
- ✅ **Supply Chain**: Secure software supply chain

## Why Sign Plugins

### Marketplace Requirements

Many marketplaces require signed plugins:

- **OpenAI Plugin Store**: HTTPS endpoints, verified domain
- **Enterprise Platforms**: Code signing certificates
- **Cloud Marketplaces**: AWS, Azure, GCP signing
- **Package Repositories**: PyPI, npm signatures

### Security Threats Prevented

- **Tampering**: Detect unauthorized modifications
- **Impersonation**: Prevent fake plugins
- **MITM Attacks**: Secure transmission
- **Supply Chain**: Verify dependencies

## Signing Methods

### 1. GPG/PGP Signatures

For manifest files and documentation.

#### Generate GPG Key

```bash
# Generate key
gpg --full-generate-key

# Select:
# - RSA and RSA (default)
# - 4096 bits
# - No expiration or set expiration
# - Your name and email

# List keys
gpg --list-keys

# Export public key
gpg --armor --export your-email@example.com > nethical-public-key.asc
```

#### Sign Manifests

```bash
# Sign individual manifest
gpg --armor --detach-sign ai-plugin.json

# This creates ai-plugin.json.asc

# Sign all manifests
for file in *.json *.yaml; do
    gpg --armor --detach-sign "$file"
done
```

#### Verify Signature

```bash
# Import public key
gpg --import nethical-public-key.asc

# Verify signature
gpg --verify ai-plugin.json.asc ai-plugin.json
```

### 2. Python Package Signing

For PyPI distribution.

#### Using Twine

```bash
# Install twine
pip install twine

# Build package
python -m build

# Sign and upload
twine upload --sign dist/*

# Or sign separately
gpg --detach-sign -a dist/nethical-0.1.0.tar.gz
gpg --detach-sign -a dist/nethical-0.1.0-py3-none-any.whl
```

#### Verification

```bash
# Download and verify
pip download nethical --no-binary :all:
gpg --verify nethical-0.1.0.tar.gz.asc nethical-0.1.0.tar.gz
```

### 3. Container Image Signing

For Docker images.

#### Using Cosign (Sigstore)

```bash
# Install cosign
wget https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign

# Generate key pair
cosign generate-key-pair

# Sign image
docker build -t nethical:latest .
cosign sign --key cosign.key nethical:latest

# Sign and push
docker push nethical:latest
cosign sign --key cosign.key $(docker inspect --format='{{index .RepoDigests 0}}' nethical:latest)
```

#### Verification

```bash
# Verify image
cosign verify --key cosign.pub nethical:latest
```

### 4. Code Signing Certificates

For enterprise distribution.

#### Obtain Certificate

1. **Commercial CA**: DigiCert, Sectigo, GlobalSign
2. **Platform Specific**: 
   - Apple Developer Certificate (macOS)
   - Microsoft Authenticode (Windows)
   - AWS Certificate Manager (AWS)

#### Sign Code

```bash
# Example: Sign with jarsigner (Java)
jarsigner -keystore myKeyStore.jks myApp.jar myAlias

# Example: Sign with signtool (Windows)
signtool sign /f certificate.pfx /p password /t http://timestamp.digicert.com myApp.exe

# Example: Sign with codesign (macOS)
codesign --sign "Developer ID Application: Your Name" myApp.app
```

### 5. HTTPS/TLS for APIs

Essential for REST API deployment.

#### Let's Encrypt (Free)

```bash
# Install certbot
sudo apt install certbot

# Obtain certificate
sudo certbot certonly --standalone -d api.nethical.dev

# Certificates stored in:
# /etc/letsencrypt/live/api.nethical.dev/fullchain.pem
# /etc/letsencrypt/live/api.nethical.dev/privkey.pem

# Auto-renewal
sudo certbot renew --dry-run
```

#### Use with FastAPI

```python
import uvicorn
from nethical.integrations.rest_api import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,
        ssl_keyfile="/etc/letsencrypt/live/api.nethical.dev/privkey.pem",
        ssl_certfile="/etc/letsencrypt/live/api.nethical.dev/fullchain.pem"
    )
```

## Implementation

### Complete Signing Workflow

```bash
#!/bin/bash
# sign-release.sh - Sign all release artifacts

set -e

VERSION="0.1.0"
GPG_KEY="your-email@example.com"

echo "Signing Nethical Release v${VERSION}"

# 1. Sign manifests
echo "Signing manifests..."
for file in *.json *.yaml; do
    gpg --armor --detach-sign "$file"
done

# 2. Build and sign Python package
echo "Building and signing Python package..."
python -m build
twine upload --sign --identity "$GPG_KEY" dist/*

# 3. Build and sign Docker image
echo "Building and signing Docker image..."
docker build -t nethical:${VERSION} .
docker tag nethical:${VERSION} nethical:latest

# Sign with cosign
cosign sign --key cosign.key nethical:${VERSION}
cosign sign --key cosign.key nethical:latest

# 4. Push to registry
docker push nethical:${VERSION}
docker push nethical:latest

# 5. Generate checksums
echo "Generating checksums..."
sha256sum *.json *.yaml > SHA256SUMS
gpg --armor --detach-sign SHA256SUMS

# 6. Create release bundle
echo "Creating release bundle..."
mkdir -p release-${VERSION}
cp *.json *.yaml release-${VERSION}/
cp *.asc release-${VERSION}/
cp SHA256SUMS* release-${VERSION}/
tar czf nethical-release-${VERSION}.tar.gz release-${VERSION}/

echo "Release signing complete!"
```

### Verification Script

```bash
#!/bin/bash
# verify-release.sh - Verify signed artifacts

set -e

VERSION="0.1.0"
PUBLIC_KEY="nethical-public-key.asc"

echo "Verifying Nethical Release v${VERSION}"

# 1. Import public key
gpg --import "$PUBLIC_KEY" 2>/dev/null || true

# 2. Verify checksums
echo "Verifying checksums..."
gpg --verify SHA256SUMS.asc SHA256SUMS
sha256sum -c SHA256SUMS

# 3. Verify manifests
echo "Verifying manifests..."
for file in *.json *.yaml; do
    if [ -f "${file}.asc" ]; then
        gpg --verify "${file}.asc" "$file"
        echo "✓ $file verified"
    fi
done

# 4. Verify Docker image
echo "Verifying Docker image..."
cosign verify --key cosign.pub nethical:${VERSION}

echo "Verification complete! All signatures valid."
```

## Verification

### Public Key Distribution

Distribute public keys via multiple channels:

```markdown
# In README.md

## Verify Downloads

All releases are signed with GPG. Verify authenticity:

1. Import public key:
   ```bash
   curl https://nethical.dev/pgp-key.asc | gpg --import
   ```

2. Verify signature:
   ```bash
   gpg --verify nethical-0.1.0.tar.gz.asc
   ```

Fingerprint: `1234 5678 90AB CDEF 1234 5678 90AB CDEF 1234 5678`
```

### Automated Verification

GitHub Actions workflow:

```yaml
# .github/workflows/verify-signatures.yml
name: Verify Signatures

on:
  pull_request:
  push:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Import GPG key
        run: |
          gpg --import nethical-public-key.asc
      
      - name: Verify manifest signatures
        run: |
          for file in *.json *.yaml; do
            if [ -f "${file}.asc" ]; then
              gpg --verify "${file}.asc" "$file"
            fi
          done
      
      - name: Verify checksums
        run: |
          gpg --verify SHA256SUMS.asc SHA256SUMS
          sha256sum -c SHA256SUMS
```

## Best Practices

### Key Management

1. **Keep Private Keys Secure**
   - Never commit to version control
   - Use hardware security keys (YubiKey, etc.)
   - Encrypt key storage
   - Use key rotation

2. **Key Backup**
   ```bash
   # Backup private key
   gpg --export-secret-keys --armor your-email@example.com > private-key-backup.asc
   
   # Store securely (encrypted)
   gpg --symmetric private-key-backup.asc
   ```

3. **Key Rotation**
   - Rotate keys annually
   - Revoke compromised keys immediately
   - Publish revocation certificates

### Signature Verification

1. **Always Verify** before using plugins
2. **Check Fingerprints** from multiple sources
3. **Trust Chain** - verify entire chain
4. **Automated Checks** in CI/CD

### Documentation

Document signing process:

```markdown
# SECURITY.md

## Signature Verification

All releases are cryptographically signed:

### GPG Signatures
- Manifest files: `*.json.asc`, `*.yaml.asc`
- Python packages: Available on PyPI

### Public Key
- Fingerprint: 1234...5678
- Download: https://nethical.dev/pgp-key.asc
- Keyserver: keys.openpgp.org

### Docker Images
- Signed with Cosign
- Public key: cosign.pub
- Verification: `cosign verify --key cosign.pub nethical:latest`
```

## Compliance

### Industry Standards

| Standard | Requirement | Implementation |
|----------|-------------|----------------|
| **NIST SP 800-204** | API Security | HTTPS, TLS 1.3, Certificate validation |
| **SLSA Framework** | Supply Chain | Build provenance, Signed artifacts |
| **OpenSSF Scorecard** | Best Practices | Signed releases, Dependency pinning |
| **SOC 2** | Trust Services | Integrity verification, Audit trails |

### Supply Chain Security

#### SLSA (Supply-chain Levels for Software Artifacts)

```yaml
# .github/workflows/slsa-build.yml
name: SLSA Build

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write
    
    steps:
      - uses: actions/checkout@v3
      - uses: slsa-framework/slsa-github-generator@v1
        with:
          provenance-name: nethical-provenance.json
```

#### SBOM (Software Bill of Materials)

```bash
# Generate SBOM
pip install cyclonedx-bom
cyclonedx-py -o sbom.json

# Sign SBOM
gpg --armor --detach-sign sbom.json
```

### Marketplace Requirements

| Marketplace | Requirements |
|-------------|--------------|
| **OpenAI** | HTTPS, Domain verification |
| **AWS Marketplace** | Code signing, SBOM, Vulnerability scanning |
| **Azure Marketplace** | Certificate signing, Security assessment |
| **Google Cloud Marketplace** | Container signing, Binary authorization |

## Additional Resources

- [SLSA Framework](https://slsa.dev/)
- [Sigstore/Cosign](https://www.sigstore.dev/)
- [OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GPG Documentation](https://gnupg.org/documentation/)

## Support

For security issues:
- 📧 Email: security@nethical.dev
- 🔒 Security Policy: [SECURITY.md](../SECURITY.md)
- 🐛 Report Vulnerabilities: [GitHub Security](https://github.com/V1B3hR/nethical/security)

## License

MIT License - See [LICENSE](../LICENSE) for details.
