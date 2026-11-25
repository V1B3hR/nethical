# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def fuzzer(monkeypatch):
    atheris = types.ModuleType("atheris")
    atheris.instrument_imports = contextlib.nullcontext

    class Provider:
        def __init__(self, data):
            self.data = data

        def ConsumeUnicodeNoSurrogates(self, length):
            assert length == 256
            return "hello"

    atheris.FuzzedDataProvider = Provider
    monkeypatch.setitem(sys.modules, "atheris", atheris)
    path = Path(__file__).resolve().parents[2] / ".clusterfuzzlite" / "governance_fuzzer.py"
    spec = importlib.util.spec_from_file_location("governance_fuzzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.gov.close()
    module.loop.close()


def test_fuzz_input_evaluates_action_and_clears_timing(fuzzer):
    fuzzer.TestOneInput(b"hello")
    assert fuzzer.gov.metrics["total_actions_processed"] == 1
    assert fuzzer.gov.metrics["detector_timing"] == {}


def test_fuzz_input_propagates_detector_errors(fuzzer, monkeypatch):
    async def fail(_action):
        raise RuntimeError("detector failure")

    detector = fuzzer.gov.detectors[0]
    monkeypatch.setattr(detector, "detect_violations", fail)
    monkeypatch.setattr(fuzzer.gov, "detectors", [detector])
    with pytest.raises(RuntimeError, match="detector failure"):
        fuzzer.TestOneInput(b"hello")
    assert fuzzer.gov.metrics["detector_timing"] == {}
