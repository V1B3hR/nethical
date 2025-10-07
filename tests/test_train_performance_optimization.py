#!/usr/bin/env python3
"""Test performance optimization in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_performance_optimization():
    """Test train_any_model.py with performance optimization enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Performance Optimization")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training with performance optimization
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-performance-optimization",
            "--performance-target-reduction", "30.0"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=tmpdir)
        
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that performance optimization was enabled
        assert "Performance optimization tracking enabled" in result.stdout
        assert "Target CPU reduction: 30.0%" in result.stdout
        
        # Check that performance metrics were tracked
        assert "Performance Metrics:" in result.stdout
        assert "data_loading:" in result.stdout
        assert "preprocessing:" in result.stdout
        assert "training:" in result.stdout
        assert "validation:" in result.stdout
        
        # Check that performance report was generated
        assert "Performance report saved to:" in result.stdout
        
        print("\n✓ Performance optimization integration test passed")


def test_train_without_performance_optimization():
    """Test train_any_model.py without performance optimization (default behavior)."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without Performance Optimization")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training WITHOUT performance optimization
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=tmpdir)
        
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that performance optimization was NOT enabled
        assert "Performance optimization tracking enabled" not in result.stdout
        assert "Performance Metrics:" not in result.stdout
        
        print("\n✓ Training without performance optimization works correctly")


def test_performance_report_content():
    """Test that performance report contains expected data."""
    print("\n" + "=" * 70)
    print("  Test: Performance Report Content")
    print("=" * 70)
    
    # Run training with performance optimization
    script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--model-type", "logistic",
        "--epochs", "2",
        "--num-samples", "100",
        "--seed", "123",
        "--enable-performance-optimization"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    # Check that training succeeded
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"
    
    # Find the generated report file
    report_dir = Path("training_performance_reports")
    assert report_dir.exists(), "Performance report directory not created"
    
    report_files = list(report_dir.glob("perf_report_logistic_*.json"))
    assert len(report_files) > 0, "No performance report file generated"
    
    # Read and validate the report
    report_file = report_files[-1]  # Get the most recent report
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    print(f"\nValidating report: {report_file}")
    
    # Check report structure
    assert 'timestamp' in report
    assert 'action_metrics' in report
    assert 'detector_stats' in report
    assert 'optimization' in report
    
    # Check detector stats
    detector_stats = report['detector_stats']
    assert 'detectors' in detector_stats
    assert 'summary' in detector_stats
    
    detectors = detector_stats['detectors']
    assert 'data_loading' in detectors
    assert 'preprocessing' in detectors
    assert 'training' in detectors
    assert 'validation' in detectors
    
    # Check that each detector has expected fields
    for detector_name, detector_info in detectors.items():
        assert 'tier' in detector_info
        assert 'total_invocations' in detector_info
        assert 'total_cpu_time_ms' in detector_info
        assert 'avg_cpu_time_ms' in detector_info
        assert detector_info['total_invocations'] >= 1, f"{detector_name} should be invoked at least once"
        assert detector_info['total_cpu_time_ms'] >= 0, f"{detector_name} should have non-negative CPU time"
    
    print("\n✓ Performance report content validation passed")
    
    # Clean up
    shutil.rmtree(report_dir)


if __name__ == "__main__":
    test_train_with_performance_optimization()
    test_train_without_performance_optimization()
    test_performance_report_content()
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
