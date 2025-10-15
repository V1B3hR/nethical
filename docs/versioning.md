# Versioning and Stability Policy

## Semantic Versioning
Nethical follows [Semantic Versioning 2.0.0](https://semver.org/)

Format: **MAJOR.MINOR.PATCH**

### Version Increments
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Current Version: 0.1.0
- Status: **Alpha** - API may change
- Stability: Development
- Production Ready: Not yet

## API Stability

### Stable APIs (v1.0+)
Will maintain backward compatibility:
- `IntegratedGovernance.process_action()`
- Core data models (AgentAction, SafetyViolation)
- Configuration parameters

### Experimental APIs (0.x)
May change without notice:
- Plugin system internals
- ML model interfaces
- Performance optimization heuristics

### Deprecated APIs
- `SafetyGovernance` (legacy) - Use `IntegratedGovernance`
- Phase-specific integration classes - Use `IntegratedGovernance`

## Deprecation Policy
1. Mark as deprecated (with warnings)
2. Maintain for 2 minor versions
3. Remove in next major version
4. Document in CHANGELOG.md

Example:
```python
import warnings
warnings.warn(
    "SafetyGovernance is deprecated. Use IntegratedGovernance instead.",
    DeprecationWarning,
    stacklevel=2
)
```

## Release Cadence
- **Patches**: As needed (security/critical bugs)
- **Minor**: Monthly
- **Major**: Annually (or when breaking changes needed)

## Version Support
- **Current Major**: Full support
- **Previous Major**: Security fixes only (6 months)
- **Older**: Community support only

## Changelog
All changes documented in `CHANGELOG.md` with:
- Version number and date
- Breaking changes (if any)
- New features
- Bug fixes
- Security updates

---
Last Updated: 2025-10-15
