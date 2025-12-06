"""
Validation Modules Package

Provides structured validation modules for:
- Ethics/Fairness benchmarking
- Performance validation
- Data integrity checks
- Explainability validation
- Drift detection
"""

__version__ = "1.0.0"

from pathlib import Path

# Package root directory
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
