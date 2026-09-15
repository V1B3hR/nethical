# Cryptographic Release Signing & Verification Guide

**Effective Date:** 2026-09-15  
**Version:** 2.0  
**Standards:** SLSA Level 3, Sigstore Cosign, NIST SP 800-218 (SSDF)  

---

## 1. Overview

To guarantee supply chain integrity and prevent man-in-the-middle tampering, all official Nethical release assets, Python wheels, container images, and git tags are cryptographically signed.

We employ two complementary signing mechanisms:
1. **GPG / SSH Git Tag Signatures:** Anchored to maintainer hardware security keys (YubiKey / OpenPGP).
2. **Keyless Sigstore / Cosign Signatures:** Bound to GitHub Actions OIDC identity with automated transparency logging on the Rekor public ledger.

---

## 2. Verifying Official Release Assets with Cosign

Download the target wheel, signature, and certificate from GitHub Releases:

```bash
# Verify Python Wheel signature with Sigstore Cosign
cosign verify-blob \
  --certificate dist/nethical-2.7.0-py3-none-any.whl.pem \
  --signature dist/nethical-2.7.0-py3-none-any.whl.sig \
  --certificate-identity "https://github.com/V1B3hR/nethical/.github/workflows/sbom-sign.yml@refs/tags/v2.7.0" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  dist/nethical-2.7.0-py3-none-any.whl
```

---

## 3. Verifying Git Tags

All official release tags (e.g., `v2.7.0`) are signed:

```bash
# Verify tag signature
git tag -v v2.7.0
```

To configure automatic tag signing for core maintainers:
```bash
git config --global user.signingkey <YOUR_GPG_OR_SSH_KEY_ID>
git config --global tag.gpgSign true
git tag -s v2.7.0 -m "Release v2.7.0: Enterprise Governance & NATO Hardening"
```

---

## 4. Checksum Verification

Every release includes a cryptographically attested `SHA256SUMS` manifest:
```bash
sha256sum -c SHA256SUMS
```
