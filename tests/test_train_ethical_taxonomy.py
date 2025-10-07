"""Tests for ethical taxonomy integration in train_any_model.py"""

import subprocess
import sys
import json
import tempfile
from pathlib import Path
import pytest


class TestEthicalTaxonomyTraining:
    """Test ethical taxonomy integration in training pipeline."""

    def test_training_with_ethical_taxonomy(self):
        """Test training with ethical taxonomy enabled."""
        train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-ethical-taxonomy"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that the command succeeded
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Check that ethical taxonomy was enabled
        assert "Ethical taxonomy enabled" in result.stdout
        assert "ethical dimensions" in result.stdout
        
        # Check for coverage stats only if violations were detected
        # (only shown when promotion gate fails)
        if "Promotion result: FAIL" in result.stdout:
            assert "Ethical Taxonomy Coverage" in result.stdout
        
    def test_training_without_ethical_taxonomy(self):
        """Test training without ethical taxonomy (backward compatibility)."""
        train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check that the command succeeded
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Check that ethical taxonomy was NOT mentioned
        assert "Ethical taxonomy enabled" not in result.stdout
        
    def test_taxonomy_report_generation(self):
        """Test that taxonomy reports are generated correctly."""
        train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-ethical-taxonomy"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        
        # Find the taxonomy report file
        candidates_dir = Path("models/candidates")
        taxonomy_files = list(candidates_dir.glob("heuristic_taxonomy_*.json"))
        
        assert len(taxonomy_files) > 0, "No taxonomy report generated"
        
        # Read the most recent file
        latest_file = max(taxonomy_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            report = json.load(f)
        
        # Verify report structure
        assert 'model_type' in report
        assert 'timestamp' in report
        assert 'promoted' in report
        assert 'coverage_stats' in report
        assert 'dimensions' in report
        
        assert report['model_type'] == 'heuristic'
        
        # Verify dimensions
        assert 'privacy' in report['dimensions']
        assert 'manipulation' in report['dimensions']
        assert 'fairness' in report['dimensions']
        assert 'safety' in report['dimensions']
        
        # Verify coverage stats
        coverage = report['coverage_stats']
        assert 'total_violation_types' in coverage
        assert 'coverage_percentage' in coverage
        assert 'meets_target' in coverage
        
    def test_violation_tagging(self):
        """Test that violations are tagged with ethical dimensions."""
        train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        # Use parameters that likely fail promotion gate
        cmd = [
            sys.executable,
            str(train_script),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-ethical-taxonomy"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        
        # Check that violations were analyzed
        if "Promotion result: FAIL" in result.stdout:
            assert "Analyzing violations with ethical taxonomy" in result.stdout
            
            # Check for dimension reporting
            output_has_dimension = (
                "fairness" in result.stdout or
                "safety" in result.stdout or
                "manipulation" in result.stdout or
                "privacy" in result.stdout
            )
            assert output_has_dimension, "No ethical dimensions found in output"
    
    def test_custom_taxonomy_path(self):
        """Test using custom taxonomy path."""
        train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(train_script),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-ethical-taxonomy",
            "--taxonomy-path", "ethics_taxonomy.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Using taxonomy from: ethics_taxonomy.json" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
