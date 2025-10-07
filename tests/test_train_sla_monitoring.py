#!/usr/bin/env python3
"""Test SLA monitoring in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_sla_monitoring():
    """Test train_any_model.py with SLA monitoring enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with SLA Monitoring")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        sla_report_dir = tmpdir_path / "test_sla_reports"
        
        # Run training with SLA monitoring
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-sla-monitoring",
            "--sla-report-dir", str(sla_report_dir)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that SLA report directory was created
        assert sla_report_dir.exists(), "SLA report directory not created"
        
        # Check for SLA report file
        report_files = list(sla_report_dir.glob("sla_report_*.json"))
        assert len(report_files) > 0, "No SLA report files created"
        
        # Check report contents
        with open(report_files[0]) as f:
            sla_report = json.load(f)
        
        print("\nSLA Report:")
        print(json.dumps(sla_report, indent=2))
        
        # Verify required fields in SLA report
        assert 'overall_status' in sla_report, "overall_status not in report"
        assert 'sla_met' in sla_report, "sla_met not in report"
        assert 'p95_latency_ms' in sla_report, "p95_latency_ms not in report"
        assert 'p95_target_ms' in sla_report, "p95_target_ms not in report"
        assert 'metrics' in sla_report, "metrics not in report"
        assert 'targets' in sla_report, "targets not in report"
        
        # Verify metrics
        metrics = sla_report['metrics']
        assert 'p95_latency_ms' in metrics, "p95_latency_ms not in metrics"
        assert 'p99_latency_ms' in metrics, "p99_latency_ms not in metrics"
        assert 'avg_latency_ms' in metrics, "avg_latency_ms not in metrics"
        assert 'sample_count' in metrics, "sample_count not in metrics"
        assert metrics['sample_count'] > 0, "No samples recorded"
        
        # Verify targets
        targets = sla_report['targets']
        assert 'p95_latency' in targets, "p95_latency target not in report"
        assert 'p99_latency' in targets, "p99_latency target not in report"
        assert 'avg_latency' in targets, "avg_latency target not in report"
        
        # Check for SLA documentation file
        doc_files = list(sla_report_dir.glob("sla_documentation_*.md"))
        assert len(doc_files) > 0, "No SLA documentation files created"
        
        # Check documentation contents
        with open(doc_files[0]) as f:
            doc_content = f.read()
        
        print("\nSLA Documentation (first 500 chars):")
        print(doc_content[:500])
        
        # Verify documentation contains key sections
        assert "# SLA Documentation" in doc_content, "Missing SLA Documentation header"
        assert "## Performance Targets" in doc_content, "Missing Performance Targets section"
        assert "## Current Performance" in doc_content, "Missing Current Performance section"
        assert "P95" in doc_content, "Missing P95 information"
        
        print("\n✓ All SLA monitoring tests passed!")
        return True


def test_train_without_sla_monitoring():
    """Test train_any_model.py without SLA monitoring."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without SLA Monitoring")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training WITHOUT SLA monitoring
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "30",
            "--seed", "42"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nSTDOUT (last 20 lines):")
        stdout_lines = result.stdout.split('\n')
        print('\n'.join(stdout_lines[-20:]))
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that no SLA monitoring was mentioned
        assert "SLA monitoring enabled" not in result.stdout, "SLA monitoring should not be active"
        assert "SLA performance report" not in result.stdout, "SLA report should not be generated"
        
        print("\n✓ Training without SLA monitoring works correctly!")
        return True


if __name__ == "__main__":
    try:
        test_train_with_sla_monitoring()
        test_train_without_sla_monitoring()
        print("\n" + "=" * 70)
        print("  All Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
