# Changelog

All notable changes to the Nethical project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive repository audit in `AUDIT.md` documenting all 90 Python files and organizational structure
- This CHANGELOG.md to track project changes
- Organized examples into subdirectories: `basic/`, `governance/`, `training/`, `advanced/`
- Created comprehensive `examples/README.md` with categorized example descriptions
- **Deprecation notices** for phase integration files:
  - `phase3_integration.py` - Added deprecation warnings and migration guide
  - `phase4_integration.py` - Added deprecation warnings and migration guide
  - `phase567_integration.py` - Added deprecation warnings and migration guide
  - `phase89_integration.py` - Added deprecation warnings and migration guide
  - All users are directed to use `IntegratedGovernance` for unified access to all phases

### Fixed
- Import error in `tests/unit/test_governance.py` - changed incorrect `JudgmentDecision` import to correct `Decision` enum
- Resolved test collection failure that was blocking test execution
- **Test suite issues** in `tests/unit/test_governance.py`:
  - All 9 tests now pass successfully with pytest-asyncio support
  - Async test functions properly supported via pytest-asyncio plugin
  - pytest-asyncio already included in requirements-dev.txt

### Completed
- **Phase Integration Deprecation** (Technical Debt #2):
  - All phase integration files now include deprecation warnings
  - Files maintained for backward compatibility
  - Clear migration paths documented in docstrings
  - Runtime warnings guide users to IntegratedGovernance
- **Test Import Errors** (Technical Debt #3):
  - All unit tests passing successfully
  - Proper async test support configured
  - Test infrastructure modernized
- **MLOps Stub Implementations** (Q2 2025 milestone):
  - `mlops/data_pipeline.py` - Full implementation with validation, versioning, and preprocessing (371 lines)
  - `mlops/model_registry.py` - Complete model registry with versioning and lifecycle management (417 lines)
  - `mlops/monitoring.py` - Comprehensive monitoring with alerts, metrics, and dashboard (446 lines)
- **Documentation Fragmentation** resolved:
  - All implementation summary files consolidated into `docs/implementation/`
  - Root directory now contains only README, CHANGELOG, AUDIT, and roadmap as intended
- **Example Script Organization**:
  - 21 example scripts organized into 4 logical categories
  - Created unified documentation with usage guides and best practices
  - Examples remain fully functional after reorganization

### Removed
- Duplicate `training/test_model.py` (109 lines) - consolidated into `scripts/test_model.py` (383 lines)
  - The scripts version is more feature-complete with comprehensive metrics, baseline comparison, and evaluation reports
  - The scripts version is actively referenced in documentation and examples
  - The training version was a simpler CLI tool with no external references

### Changed
- Updated test assertions in `tests/unit/test_governance.py` to use correct enum values:
  - `JudgmentDecision.ALLOW` → `Decision.ALLOW`
  - `JudgmentDecision.BLOCK` → `Decision.BLOCK`
  - `JudgmentDecision.TERMINATE` → `Decision.TERMINATE`
  - `JudgmentDecision.RESTRICT` → `Decision.WARN` (mapping to available enum value)
- Roadmap updated to mark technical debt items 4, 5, and 6 as COMPLETE

## Repository Structure Overview

### Current Organization

```
nethical/
├── nethical/              # Core package (44 files)
│   ├── core/             # Governance components (23 files, ~11.5K lines)
│   ├── detectors/        # Safety detectors (8 files, ~6.2K lines)
│   ├── judges/           # Judgment system (3 files, ~515 lines)
│   ├── monitors/         # Monitoring (3 files, ~1.6K lines)
│   └── mlops/            # ML operations (7 files, ~1K lines)
├── scripts/              # Training/testing scripts
│   ├── dataset_processors/  # Data processors (5 files)
│   └── test_model.py     # Consolidated testing script
├── training/             # Training utilities
│   └── train_any_model.py   # Main training script (789 lines)
├── tests/                # Test suite (22 files, 192 tests)
├── examples/             # Example scripts (21 files, organized in 4 categories)
│   ├── basic/           # Getting started examples (3 files)
│   ├── governance/      # Phase demos (7 files)
│   ├── training/        # ML training examples (5 files)
│   └── advanced/        # Enterprise features (6 files)
└── docs/                 # Documentation
```

### Key Files

#### Core Components
- **governance.py** (1732 lines) - Main governance system
- **models.py** (740 lines) - Data models and schemas
- **integrated_governance.py** (552 lines) - Unified interface for all phases

#### Training & Testing
- **scripts/test_model.py** (383 lines) - Primary model testing script
- **training/train_any_model.py** (789 lines) - Comprehensive training pipeline
- **scripts/dataset_processors/** - Dataset processing modules

#### Testing
- 192 tests across 22 test files
- Comprehensive coverage of all phases (3, 4, 5, 6, 7, 8-9)
- Integration and end-to-end tests

## Known Issues

### Test Status
- ~~`tests/unit/test_governance.py`~~ ✅ COMPLETED
  - ~~Tests now pass import but may fail assertions due to evolved API~~
  - ~~System now has 14 detectors instead of expected 3~~
  - ~~`MonitoringConfig` missing some expected attributes~~
  - ~~`AgentAction` requires additional fields~~
  - All 9 tests now pass successfully with pytest-asyncio support
  - Test infrastructure modernized and working correctly

### Areas for Future Improvement

1. **Large Files**
   - `governance.py` (1732 lines) could benefit from modularization
   - Consider breaking into smaller, focused modules

2. ~~**Phase Integration Files**~~ ✅ COMPLETED
   - ~~Multiple phase integration files (phase3_integration.py, phase4_integration.py, etc.)~~
   - ~~Now superseded by `integrated_governance.py`~~
   - ~~Consider adding deprecation notices or documenting as compatibility layers~~
   - All phase integration files now include deprecation warnings and migration guides

3. ~~**Stub Implementations**~~ ✅ COMPLETED
   - ~~`mlops/data_pipeline.py` (26 lines)~~
   - ~~`mlops/model_registry.py` (25 lines)~~
   - ~~`mlops/monitoring.py` (35 lines)~~
   - All MLOps modules now fully implemented with production-ready features

4. ~~**Documentation Fragmentation**~~ ✅ COMPLETED
   - ~~Multiple implementation summary files in root directory~~
   - ~~Training documentation in multiple locations~~
   - All consolidated into `docs/implementation/` directory

5. ~~**Example Organization**~~ ✅ COMPLETED
   - ~~14 example files, some with overlapping functionality~~
   - Examples now organized into 4 logical categories with comprehensive documentation

## Migration Guide

### For Users of Phase Integration Files

The phase-specific integration files are now deprecated in favor of the unified `IntegratedGovernance` class. While the old files remain for backward compatibility, you should migrate to the new interface:

**Phase 3 Migration:**
```python
# Old (deprecated):
from nethical.core.phase3_integration import Phase3IntegratedGovernance
governance = Phase3IntegratedGovernance(
    redis_client=redis,
    enable_performance_optimization=True
)

# New (recommended):
from nethical.core.integrated_governance import IntegratedGovernance
governance = IntegratedGovernance(
    redis_client=redis,
    enable_performance_optimization=True
)
```

**Phase 4 Migration:**
```python
# Old (deprecated):
from nethical.core.phase4_integration import Phase4IntegratedGovernance
governance = Phase4IntegratedGovernance(
    storage_dir="./data",
    enable_merkle_anchoring=True
)

# New (recommended):
from nethical.core.integrated_governance import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True
)
```

**Phase 5-7 Migration:**
```python
# Old (deprecated):
from nethical.core.phase567_integration import Phase567IntegratedGovernance
governance = Phase567IntegratedGovernance(
    storage_dir="./data",
    enable_shadow_mode=True
)

# New (recommended):
from nethical.core.integrated_governance import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True
)
```

**Phase 8-9 Migration:**
```python
# Old (deprecated):
from nethical.core.phase89_integration import Phase89IntegratedGovernance
governance = Phase89IntegratedGovernance(
    storage_dir="./data",
    triage_sla_seconds=3600
)

# New (recommended):
from nethical.core.integrated_governance import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_escalation=True,
    enable_optimization=True,
    triage_sla_seconds=3600
)
```

**Benefits of IntegratedGovernance:**
- Unified interface for all phases (3, 4, 5-7, 8-9)
- Simplified configuration and initialization
- Single method for processing actions across all phases
- Comprehensive system status and monitoring
- Easy feature toggling with individual enable flags

### For Users of training/test_model.py

If you were using `training/test_model.py`, please switch to `scripts/test_model.py`:

**Old (removed):**
```bash
python training/test_model.py --model-type logistic --test-data-path data/test_data.json
```

**New (recommended):**
```bash
python scripts/test_model.py
```

The scripts version provides:
- Automatic model type detection
- Comprehensive metrics (Precision, Recall, F1, ROC-AUC, ECE)
- Baseline comparison
- Evaluation report generation
- Support for both BaselineMLClassifier and MLShadowClassifier

### For Test Writers

When writing tests that interact with the governance system:

**Import the correct enum:**
```python
from nethical.core.models import Decision  # Not JudgmentDecision
```

**Use correct enum values:**
```python
assert judgment.decision == Decision.ALLOW
assert judgment.decision == Decision.BLOCK
assert judgment.decision == Decision.TERMINATE
```

## Versioning Strategy

This project follows [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

## Contributing

When adding new features or making changes:
1. Update this CHANGELOG.md in the [Unreleased] section
2. Follow the existing code organization patterns
3. Add tests for new functionality
4. Update relevant documentation
5. Check for duplicate functionality before adding new files

## References

- [AUDIT.md](AUDIT.md) - Comprehensive repository audit and analysis
- [README.md](README.md) - Project overview and quick start
- [roadmap.md](roadmap.md) - Development roadmap
- [docs/](docs/) - Detailed documentation

---

**Note:** This changelog started with the repository refactoring initiative. Historical changes before this point are documented in git commit history and various implementation summary files.
