"""
Tests for Performance Profiling Module

Tests the PerformanceProfiler class and related functionality for
profiling, benchmarking, and regression detection.
"""

import pytest
import time
from pathlib import Path
import tempfile
import shutil
from nethical.performanceprofiling import (
    PerformanceProfiler,
    ProfileResult,
    Benchmark,
    profile,
    get_profiler,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test results"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def profiler(temp_dir):
    """Create a profiler instance for testing"""
    return PerformanceProfiler(results_dir=temp_dir)


class TestProfileResult:
    """Test ProfileResult dataclass"""

    def test_profile_result_creation(self):
        """Test creating a ProfileResult"""
        result = ProfileResult(
            function_name="test_func", execution_time_ms=10.5, call_count=1
        )
        assert result.function_name == "test_func"
        assert result.execution_time_ms == 10.5
        assert result.call_count == 1

    def test_profile_result_to_dict(self):
        """Test converting ProfileResult to dict"""
        result = ProfileResult(
            function_name="test_func", execution_time_ms=10.5, call_count=1
        )
        result_dict = result.to_dict()
        assert result_dict["function_name"] == "test_func"
        assert result_dict["execution_time_ms"] == 10.5
        assert result_dict["call_count"] == 1
        assert "timestamp" in result_dict


class TestBenchmark:
    """Test Benchmark class"""

    def test_benchmark_creation(self):
        """Test creating a Benchmark"""
        bench = Benchmark(name="test_benchmark", baseline_ms=10.0, tolerance_pct=15.0)
        assert bench.name == "test_benchmark"
        assert bench.baseline_ms == 10.0
        assert bench.tolerance_pct == 15.0

    def test_check_regression_no_regression(self):
        """Test regression check when performance is good"""
        bench = Benchmark(name="test", baseline_ms=10.0, tolerance_pct=10.0)
        is_regression, pct_change = bench.check_regression(10.5)
        assert not is_regression
        assert pct_change == 5.0

    def test_check_regression_with_regression(self):
        """Test regression check when performance degrades"""
        bench = Benchmark(name="test", baseline_ms=10.0, tolerance_pct=10.0)
        is_regression, pct_change = bench.check_regression(12.0)
        assert is_regression
        assert pct_change == 20.0

    def test_update_baseline(self):
        """Test updating benchmark baseline"""
        bench = Benchmark(name="test", baseline_ms=10.0)
        bench.add_sample(11.0)
        assert len(bench.samples) == 1

        bench.update_baseline(12.0)
        assert bench.baseline_ms == 12.0
        assert len(bench.samples) == 0

    def test_add_sample(self):
        """Test adding samples to benchmark"""
        bench = Benchmark(name="test", baseline_ms=10.0)
        for i in range(10):
            bench.add_sample(10.0 + i * 0.1)
        assert len(bench.samples) == 10

    def test_sample_limit(self):
        """Test that sample list is limited to 100"""
        bench = Benchmark(name="test", baseline_ms=10.0)
        for i in range(150):
            bench.add_sample(10.0 + i * 0.1)
        assert len(bench.samples) == 100


class TestPerformanceProfiler:
    """Test PerformanceProfiler class"""

    def test_profiler_initialization(self, profiler, temp_dir):
        """Test profiler initialization"""
        assert profiler.results_dir == temp_dir
        assert temp_dir.exists()
        assert isinstance(profiler.results, dict)
        assert isinstance(profiler.benchmarks, dict)

    def test_profile_decorator_simple(self, profiler):
        """Test profiling a simple function"""

        @profiler.profile()
        def test_function():
            time.sleep(0.01)
            return 42

        result = test_function()
        assert result == 42
        assert "test_function" in profiler.results
        assert len(profiler.results["test_function"]) == 1
        assert profiler.results["test_function"][0].execution_time_ms > 10.0

    def test_profile_decorator_with_name(self, profiler):
        """Test profiling with custom name"""

        @profiler.profile(name="custom_name")
        def test_function():
            return 42

        test_function()
        assert "custom_name" in profiler.results

    def test_profile_decorator_detailed(self, profiler):
        """Test detailed profiling"""

        @profiler.profile(detailed=True)
        def test_function():
            total = 0
            for i in range(1000):
                total += i
            return total

        test_function()
        assert "test_function" in profiler.results
        result = profiler.results["test_function"][0]
        assert result.stats is not None
        assert "profile_output" in result.stats

    def test_benchmark_function(self, profiler):
        """Test benchmarking a function"""

        def test_func():
            time.sleep(0.001)

        stats = profiler.benchmark_function(test_func, iterations=10)

        assert "function" in stats
        assert stats["function"] == "test_func"
        assert stats["iterations"] == 10
        assert stats["mean_ms"] > 1.0
        assert "median_ms" in stats
        assert "std_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats

    def test_set_benchmark(self, profiler, temp_dir):
        """Test setting a benchmark"""
        profiler.set_benchmark("test_bench", 10.0, tolerance_pct=15.0)

        assert "test_bench" in profiler.benchmarks
        bench = profiler.benchmarks["test_bench"]
        assert bench.baseline_ms == 10.0
        assert bench.tolerance_pct == 15.0

        # Check that it was saved to file
        benchmark_file = temp_dir / "benchmarks.json"
        assert benchmark_file.exists()

    def test_get_results_single_function(self, profiler):
        """Test getting results for a single function"""

        @profiler.profile()
        def test_function():
            time.sleep(0.01)

        # Run multiple times
        for _ in range(5):
            test_function()

        results = profiler.get_results("test_function")
        assert results["function"] == "test_function"
        assert results["samples"] == 5
        assert results["mean_ms"] > 10.0
        assert "median_ms" in results
        assert "std_ms" in results

    def test_get_results_all_functions(self, profiler):
        """Test getting results for all functions"""

        @profiler.profile()
        def func1():
            time.sleep(0.01)

        @profiler.profile()
        def func2():
            time.sleep(0.005)

        func1()
        func2()

        results = profiler.get_results()
        assert "func1" in results
        assert "func2" in results
        assert results["func1"]["samples"] == 1
        assert results["func2"]["samples"] == 1

    def test_regression_detection(self, profiler, capsys):
        """Test automatic regression detection"""
        # Set a benchmark
        profiler.set_benchmark("slow_func", baseline_ms=10.0, tolerance_pct=10.0)

        @profiler.profile(name="slow_func")
        def slow_func():
            time.sleep(0.015)  # 15ms - should trigger regression

        slow_func()

        # Check that warning was printed
        captured = capsys.readouterr()
        assert (
            "regression detected" in captured.out.lower() or "WARNING" in captured.out
        )

    def test_generate_report(self, profiler, temp_dir):
        """Test generating a performance report"""

        # Add some profiling data
        @profiler.profile()
        def test_function():
            time.sleep(0.01)

        test_function()
        test_function()

        # Set a benchmark
        profiler.set_benchmark("test_function", 10.0)

        # Generate report
        output_file = temp_dir / "report.txt"
        report = profiler.generate_report(output_file)

        assert "PERFORMANCE PROFILING REPORT" in report
        assert "BENCHMARKS" in report or "PROFILING RESULTS" in report
        assert "test_function" in report
        assert output_file.exists()

    def test_benchmark_persistence(self, temp_dir):
        """Test that benchmarks persist across profiler instances"""
        # Create first profiler and set benchmark
        profiler1 = PerformanceProfiler(results_dir=temp_dir)
        profiler1.set_benchmark("persist_test", 20.0, tolerance_pct=12.0)

        # Create second profiler with same directory
        profiler2 = PerformanceProfiler(results_dir=temp_dir)

        # Check that benchmark was loaded
        assert "persist_test" in profiler2.benchmarks
        assert profiler2.benchmarks["persist_test"].baseline_ms == 20.0
        assert profiler2.benchmarks["persist_test"].tolerance_pct == 12.0


class TestGlobalProfiler:
    """Test global profiler convenience functions"""

    def test_global_profile_decorator(self):
        """Test using the global profile decorator"""

        @profile()
        def global_func():
            return 42

        result = global_func()
        assert result == 42

        # Check that it was recorded in global profiler
        global_profiler = get_profiler()
        assert "global_func" in global_profiler.results

    def test_get_profiler(self):
        """Test getting the global profiler instance"""
        profiler = get_profiler()
        assert isinstance(profiler, PerformanceProfiler)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_results(self, profiler):
        """Test getting results when there are none"""
        results = profiler.get_results("nonexistent")
        assert results == {}

    def test_profile_with_exception(self, profiler):
        """Test that profiling propagates exceptions correctly"""

        @profiler.profile()
        def error_func():
            raise ValueError("Test error")

        # Exception should be propagated
        with pytest.raises(ValueError, match="Test error"):
            error_func()

        # Note: The profiler doesn't record failed executions,
        # which is acceptable behavior

    def test_benchmark_with_zero_samples(self, profiler):
        """Test benchmark behavior with no samples"""
        bench = Benchmark(name="test", baseline_ms=10.0)
        bench_dict = bench.to_dict()

        assert bench_dict["samples_count"] == 0
        assert bench_dict["recent_avg_ms"] is None
        assert bench_dict["recent_std_ms"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
