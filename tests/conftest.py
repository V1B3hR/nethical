"""
Pytest configuration for Nethical tests

Adds custom command line options for extended tests
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="Run extended/long-running tests (e.g., 10-minute load tests)"
    )
    
    parser.addoption(
        "--run-soak",
        action="store_true",
        default=False,
        help="Run soak tests (2-hour tests)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "extended: marks tests that require --run-extended flag"
    )
    config.addinivalue_line(
        "markers", "soak: marks tests that require --run-soak flag"
    )
