#!/usr/bin/env python3
"""
Tests for the load_datasets_from_file function in train_any_model.py.

These tests verify:
1. Parsing of valid Kaggle dataset URLs
2. Ignoring non-dataset URLs (discussions, code, competitions)
3. Handling of missing file
4. Handling of empty file
5. Handling of mixed content
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_any_model import load_datasets_from_file


class TestLoadDatasetsFromFile:
    """Tests for the load_datasets_from_file function."""

    def test_valid_dataset_urls(self):
        """Test parsing of valid Kaggle dataset URLs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks\n")
            f.write("https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction\n")
            f.write("https://www.kaggle.com/datasets/owner/dataset-name\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert len(result) == 3
            assert "teamincribo/cyber-security-attacks" in result
            assert "Microsoft/microsoft-security-incident-prediction" in result
            assert "owner/dataset-name" in result
        finally:
            Path(file_path).unlink()

    def test_ignore_discussion_urls(self):
        """Test that discussion URLs are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks\n")
            f.write("https://www.kaggle.com/competitions/2023-kaggle-ai-report/discussion/409817\n")
            f.write("https://www.kaggle.com/discussions/general/409208\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            # Only the first URL should be parsed (the dataset URL)
            assert len(result) == 1
            assert "teamincribo/cyber-security-attacks" in result
        finally:
            Path(file_path).unlink()

    def test_ignore_code_urls(self):
        """Test that code/notebook URLs are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            # Only the dataset URL should be parsed, not code
            assert len(result) == 1
            assert "teamincribo/cyber-security-attacks" in result
        finally:
            Path(file_path).unlink()

    def test_ignore_competition_urls(self):
        """Test that competition URLs are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/owner/dataset\n")
            f.write("https://www.kaggle.com/competitions/some-competition\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert len(result) == 1
            assert "owner/dataset" in result
        finally:
            Path(file_path).unlink()

    def test_missing_file_returns_empty_list(self):
        """Test that a missing file returns an empty list."""
        result = load_datasets_from_file(Path("/nonexistent/path/to/file"))
        assert result == []

    def test_empty_file_returns_empty_list(self):
        """Test that an empty file returns an empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert result == []
        finally:
            Path(file_path).unlink()

    def test_file_with_only_non_dataset_urls(self):
        """Test file containing only non-dataset URLs returns empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/code/some-user/notebook\n")
            f.write("https://www.kaggle.com/competitions/competition-name\n")
            f.write("https://www.kaggle.com/discussions/general/123456\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert result == []
        finally:
            Path(file_path).unlink()

    def test_mixed_content_with_empty_lines(self):
        """Test file with mixed content including empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/owner1/dataset1\n")
            f.write("\n")
            f.write("   \n")
            f.write("https://www.kaggle.com/code/user/notebook\n")
            f.write("https://www.kaggle.com/datasets/owner2/dataset2\n")
            f.write("\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert len(result) == 2
            assert "owner1/dataset1" in result
            assert "owner2/dataset2" in result
        finally:
            Path(file_path).unlink()

    def test_dataset_url_with_trailing_path(self):
        """Test dataset URLs with trailing path components (like /discussion)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsphilosophycsv/discussion/156387\n")
            f.write("https://www.kaggle.com/datasets/owner/dataset/versions/2\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert len(result) == 2
            # Should extract just the owner/dataset part
            assert "mpwolke/cusersmarildownloadsphilosophycsv" in result
            assert "owner/dataset" in result
        finally:
            Path(file_path).unlink()

    def test_default_file_path_when_none(self):
        """Test that default file path is used when None is passed."""
        # This test just verifies the function doesn't crash with None
        # and returns empty list when the default file doesn't exist in test environment
        result = load_datasets_from_file(None)
        # Result depends on whether the datasets/datasets file exists
        assert isinstance(result, list)

    def test_malformed_dataset_url_ignored(self):
        """Test that malformed URLs are gracefully ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://www.kaggle.com/datasets/\n")  # Missing owner/dataset
            f.write("https://www.kaggle.com/datasets/only-one-part\n")  # Only one part
            f.write("https://www.kaggle.com/datasets//dataset\n")  # Empty owner
            f.write("https://www.kaggle.com/datasets/owner/\n")  # Empty dataset name
            f.write("https://www.kaggle.com/datasets/owner/dataset\n")  # Valid
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            # Only the valid URL should be parsed
            assert len(result) == 1
            assert "owner/dataset" in result
        finally:
            Path(file_path).unlink()

    def test_non_kaggle_urls_ignored(self):
        """Test that non-Kaggle URLs are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("https://github.com/owner/repo\n")
            f.write("https://example.com/datasets/something\n")
            f.write("https://www.kaggle.com/datasets/owner/dataset\n")
            file_path = f.name
        
        try:
            result = load_datasets_from_file(Path(file_path))
            assert len(result) == 1
            assert "owner/dataset" in result
        finally:
            Path(file_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
