# Supply Chain Security Guide (Updated)

## 1. Hash Verification

We use a two-file model:
- `requirements.txt` – human-maintained input (pinned versions, no hashes).
- `requirements-hashed.txt` – machine-generated lock with `--hash` entries (`pip-compile --generate-hashes`).

### Generate/Update
```bash
python -m pip install --upgrade pip pip-tools
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt
```

### Install (production / CI)
```bash
pip install --require-hashes -r requirements-hashed.txt
```

### Policy
- Every dependency must be pinned (==).
- Hash file must be regenerated whenever versions change.
- CI enforces sync; drift blocks merges (or auto-fixes if same-repo PR).

## 2. Recommended Workflow

| Step | Action | Tool |
|------|--------|------|
| Edit | Bump version in `requirements.txt` | Manual |
| Lock | Run `pip-compile --generate-hashes` | pip-tools |
| Verify | `pip install --require-hashes -r requirements-hashed.txt` | pip |
| Commit | Add both files | git |
| CI | Hash drift check | GitHub Actions |

## 3. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Hashes in `requirements.txt` | Remove; hashes only in lock file |
| Duplicate pins | Keep one instance |
| Unpinned transitive added manually | Either pin explicitly or let pip-compile resolve |
| Placeholder hashes | Never commit placeholders; let CI or pip-compile populate |

## 4. Dependency Updates Example
```bash
# Update pydantic
sed -i '' 's/pydantic==2.12.3/pydantic==2.13.0/' requirements.txt
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt
git add requirements.txt requirements-hashed.txt
git commit -m "feat: bump pydantic to 2.13.0 with updated hash lock"
```

## 5. Security Layers
- ✅ Version pinning
- ✅ Hash verification (`--require-hashes`)
- ✅ Drift enforcement (CI)
- ✅ SBOM generation (planned full release integration)
- 🔄 Provenance (SLSA Level 3 WIP)
- ✅ Artifact signing (design docs prepared)
- 🔄 Provenance verification (future job)

## 6. Future Enhancements
- Multi-wheel hash completeness auditing script
- Automatic vulnerability diff on lock regeneration
- Provenance attachment to release artifacts

## 7. Quick Commands
```bash
# Regenerate all
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt

# Dry-run install test
pip install --require-hashes -r requirements-hashed.txt --no-deps --dry-run

# Show outdated (without changing lock)
pip list --outdated
```

## 8. References
- [pip-tools](https://github.com/jazzband/pip-tools)
- [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/)
- [SLSA Framework](https://slsa.dev/)
