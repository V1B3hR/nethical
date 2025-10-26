# Supply Chain Security - Hash Verification TODO

## Current Status
- ✅ Dependency version pinning implemented in requirements.txt
- ✅ SLSA compliance assessment tooling created
- ✅ Dependabot configured for automated updates
- ✅ SBOM generation capability

## Pending Implementation

### Hash Verification for Dependencies

To implement full hash verification (as noted in code review), we need to add `--hash` flags to requirements.txt:

```bash
# Generate requirements with hashes
pip-compile --generate-hashes requirements.in -o requirements.txt

# Or manually add hashes
pip hash pydantic==2.12.3
```

Example format:
```
pydantic==2.12.3 \
    --hash=sha256:abcd1234... \
    --hash=sha256:efgh5678...
```

### SLSA Level 3 Complete Compliance

For full SLSA Level 3 compliance, we need:

1. **Build Provenance Generation**
   - Generate SLSA provenance during CI builds
   - Use slsa-github-generator for GitHub Actions
   - Include builder identity and build parameters

2. **Provenance Verification**
   - Verify provenance signatures
   - Check builder identity matches expectations
   - Validate build parameters

3. **Non-falsifiable Provenance**
   - Use hosted CI/CD with ephemeral environments
   - Implement hermetic builds
   - Use isolated build runners

4. **Dependency Verification**
   - Verify all dependency checksums
   - Validate dependency provenance
   - Check for known vulnerabilities

## Implementation Steps

1. Install pip-tools:
   ```bash
   pip install pip-tools
   ```

2. Create requirements.in (source file):
   ```
   pydantic==2.12.3
   typing-extensions==4.15.0
   # ... other dependencies
   ```

3. Generate hashed requirements:
   ```bash
   pip-compile --generate-hashes requirements.in
   ```

4. Update GitHub Actions to use slsa-github-generator:
   ```yaml
   - uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
     with:
       subjects: "artifacts/*"
   ```

5. Add verification step:
   ```yaml
   - name: Verify SLSA provenance
     uses: slsa-framework/slsa-verifier/actions/installer@v2.4.0
   ```

## References
- [pip hash documentation](https://pip.pypa.io/en/stable/topics/secure-installs/)
- [SLSA Level 3 requirements](https://slsa.dev/spec/v1.0/levels#level-3)
- [slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)

## Priority
Medium - Current implementation provides strong supply chain security with version pinning and automated updates. Hash verification and full SLSA L3 are enhancements for production deployments.
