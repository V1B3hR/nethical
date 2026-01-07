"""Profiling module for Nethical threat detection system.

This module provides production-safe profiling with flamegraph generation:
- py-spy for minimal overhead (<1% CPU)
- cProfile fallback for development
- Flamegraph generation
"""

from nethical.profiling.flamegraph_profiler import FlamegraphProfiler

__all__ = ["FlamegraphProfiler"]
