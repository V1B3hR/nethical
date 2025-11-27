#!/usr/bin/env python3
"""
Comprehensive tests for all MLops classifiers.

Tests BaselineMLClassifier, AnomalyMLClassifier, and CorrelationMLClassifier
for:
- Data loading
- Training
- Model saving and loading
- Prediction
- Metrics computation

These tests validate the end-to-end training workflow documented in
docs/TRAINING_GUIDE.md.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.mlops.baseline import BaselineMLClassifier
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier
from nethical.mlops.correlation_classifier import CorrelationMLClassifier


class TestBaselineMLClassifier:
    """Tests for BaselineMLClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization with default and custom parameters."""
        # Default initialization
        clf = BaselineMLClassifier()
        assert clf.threshold == 0.5
        assert clf.trained is False
        assert clf.training_samples == 0
        
        # Custom initialization
        clf_custom = BaselineMLClassifier(threshold=0.7, learning_rate=0.05)
        assert clf_custom.threshold == 0.7
        assert clf_custom.learning_rate == 0.05
    
    def test_train(self):
        """Test training with sample data."""
        clf = BaselineMLClassifier()
        
        train_data = [
            {'features': {'violation_count': 0.1, 'severity_max': 0.2}, 'label': 0},
            {'features': {'violation_count': 0.2, 'severity_max': 0.3}, 'label': 0},
            {'features': {'violation_count': 0.8, 'severity_max': 0.9}, 'label': 1},
            {'features': {'violation_count': 0.9, 'severity_max': 0.95}, 'label': 1},
        ]
        
        clf.train(train_data)
        
        assert clf.trained is True
        assert clf.training_samples == 4
        assert len(clf.feature_weights) > 0
        assert clf.timestamp is not None
    
    def test_train_empty_data_raises(self):
        """Test that training with empty data raises ValueError."""
        clf = BaselineMLClassifier()
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            clf.train([])
    
    def test_predict(self):
        """Test prediction on new data."""
        clf = BaselineMLClassifier()
        
        train_data = [
            {'features': {'f1': 0.1, 'f2': 0.2}, 'label': 0},
            {'features': {'f1': 0.9, 'f2': 0.8}, 'label': 1},
        ] * 10  # Repeat for better training
        
        clf.train(train_data)
        
        # Predict on low-risk features
        result_low = clf.predict({'f1': 0.1, 'f2': 0.2})
        assert 'label' in result_low
        assert 'score' in result_low
        assert 'confidence' in result_low
        assert 0 <= result_low['score'] <= 1
        assert 0 <= result_low['confidence'] <= 1
        
        # Predict on high-risk features
        result_high = clf.predict({'f1': 0.9, 'f2': 0.8})
        assert result_high['score'] > result_low['score']
    
    def test_save_and_load(self):
        """Test model save and load."""
        clf = BaselineMLClassifier(threshold=0.6)
        
        train_data = [
            {'features': {'f1': 0.2, 'f2': 0.3}, 'label': 0},
            {'features': {'f1': 0.8, 'f2': 0.9}, 'label': 1},
        ] * 5
        
        clf.train(train_data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            clf.save(filepath)
            assert os.path.exists(filepath)
            
            loaded_clf = BaselineMLClassifier.load(filepath)
            
            assert loaded_clf.threshold == clf.threshold
            assert loaded_clf.trained == clf.trained
            assert loaded_clf.training_samples == clf.training_samples
            assert loaded_clf.feature_weights == clf.feature_weights
            
            # Verify predictions match
            test_features = {'f1': 0.5, 'f2': 0.5}
            result1 = clf.predict(test_features)
            result2 = loaded_clf.predict(test_features)
            assert abs(result1['score'] - result2['score']) < 0.01
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        clf = BaselineMLClassifier()
        
        predictions = [0, 0, 1, 1, 0, 1]
        labels = [0, 0, 1, 0, 1, 1]
        
        metrics = clf.compute_metrics(predictions, labels)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'ece' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1


class TestAnomalyMLClassifierBasic:
    """Tests for AnomalyMLClassifier basic functionality."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        clf = AnomalyMLClassifier()
        assert clf.n == 3
        assert clf.anomaly_threshold == 0.5
        assert clf.trained is False
        
        clf_custom = AnomalyMLClassifier(n=4, anomaly_threshold=0.6)
        assert clf_custom.n == 4
        assert clf_custom.anomaly_threshold == 0.6
    
    def test_train_with_sequences(self):
        """Test training with sequence data."""
        clf = AnomalyMLClassifier()
        
        train_data = []
        # Normal patterns
        for _ in range(20):
            train_data.append({
                'features': {'sequence': ['read', 'process', 'write']},
                'label': 0
            })
        # Anomalous patterns
        for _ in range(20):
            train_data.append({
                'features': {'sequence': ['delete', 'exfiltrate', 'cover_tracks']},
                'label': 1
            })
        
        clf.train(train_data)
        
        assert clf.trained is True
        assert clf.training_samples == 40
        assert len(clf.normal_ngrams) > 0
        assert len(clf.anomalous_ngrams) > 0
    
    def test_predict_normal_sequence(self):
        """Test prediction on normal sequence."""
        clf = AnomalyMLClassifier()
        
        train_data = [
            {'features': {'sequence': ['read', 'process', 'write']}, 'label': 0}
        ] * 20 + [
            {'features': {'sequence': ['delete', 'exfiltrate', 'cover']}, 'label': 1}
        ] * 20
        
        clf.train(train_data)
        
        result = clf.predict({'sequence': ['read', 'process', 'write']})
        
        assert result['label'] == 0
        assert 0 <= result['score'] <= 1
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_anomalous_sequence(self):
        """Test prediction on anomalous sequence."""
        clf = AnomalyMLClassifier()
        
        train_data = [
            {'features': {'sequence': ['read', 'process', 'write']}, 'label': 0}
        ] * 20 + [
            {'features': {'sequence': ['delete', 'exfiltrate', 'cover']}, 'label': 1}
        ] * 20
        
        clf.train(train_data)
        
        result = clf.predict({'sequence': ['delete', 'exfiltrate', 'cover']})
        
        assert result['label'] == 1
        assert 0 <= result['score'] <= 1
    
    def test_save_and_load(self):
        """Test model save and load."""
        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.4)
        
        train_data = [
            {'features': {'sequence': ['a', 'b', 'c']}, 'label': 0}
        ] * 10 + [
            {'features': {'sequence': ['x', 'y', 'z']}, 'label': 1}
        ] * 10
        
        clf.train(train_data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            clf.save(filepath)
            assert os.path.exists(filepath)
            
            loaded_clf = AnomalyMLClassifier.load(filepath)
            
            assert loaded_clf.n == clf.n
            assert loaded_clf.anomaly_threshold == clf.anomaly_threshold
            assert loaded_clf.trained == clf.trained
            assert loaded_clf.training_samples == clf.training_samples
            
            # Verify predictions match
            test_seq = {'sequence': ['a', 'b', 'c']}
            result1 = clf.predict(test_seq)
            result2 = loaded_clf.predict(test_seq)
            assert result1['label'] == result2['label']
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestCorrelationMLClassifierBasic:
    """Tests for CorrelationMLClassifier basic functionality."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        clf = CorrelationMLClassifier()
        assert clf.pattern_threshold == 0.5
        assert clf.trained is False
        assert len(clf.feature_weights) == 5
        
        clf_custom = CorrelationMLClassifier(pattern_threshold=0.6)
        assert clf_custom.pattern_threshold == 0.6
    
    def test_train_with_correlation_data(self):
        """Test training with correlation data."""
        clf = CorrelationMLClassifier()
        
        train_data = []
        # Normal patterns
        for _ in range(20):
            train_data.append({
                'features': {
                    'agent_count': 2,
                    'action_rate': 5,
                    'entropy_variance': 0.2,
                    'time_correlation': 0.1,
                    'payload_similarity': 0.2
                },
                'label': 0
            })
        # Correlation patterns
        for _ in range(20):
            train_data.append({
                'features': {
                    'agent_count': 10,
                    'action_rate': 50,
                    'entropy_variance': 0.7,
                    'time_correlation': 0.8,
                    'payload_similarity': 0.7
                },
                'label': 1
            })
        
        clf.train(train_data)
        
        assert clf.trained is True
        assert clf.training_samples == 40
    
    def test_predict_normal_pattern(self):
        """Test prediction on normal pattern."""
        clf = CorrelationMLClassifier()
        
        train_data = [
            {'features': {'agent_count': 2, 'action_rate': 5, 'entropy_variance': 0.2,
                         'time_correlation': 0.1, 'payload_similarity': 0.2}, 'label': 0}
        ] * 20 + [
            {'features': {'agent_count': 10, 'action_rate': 50, 'entropy_variance': 0.7,
                         'time_correlation': 0.8, 'payload_similarity': 0.7}, 'label': 1}
        ] * 20
        
        clf.train(train_data)
        
        result = clf.predict({
            'agent_count': 2, 'action_rate': 5, 'entropy_variance': 0.2,
            'time_correlation': 0.1, 'payload_similarity': 0.2
        })
        
        assert result['label'] == 0
        assert 0 <= result['score'] <= 1
    
    def test_predict_correlation_pattern(self):
        """Test prediction on correlation pattern."""
        clf = CorrelationMLClassifier()
        
        train_data = [
            {'features': {'agent_count': 2, 'action_rate': 5, 'entropy_variance': 0.2,
                         'time_correlation': 0.1, 'payload_similarity': 0.2}, 'label': 0}
        ] * 20 + [
            {'features': {'agent_count': 10, 'action_rate': 50, 'entropy_variance': 0.7,
                         'time_correlation': 0.8, 'payload_similarity': 0.7}, 'label': 1}
        ] * 20
        
        clf.train(train_data)
        
        result = clf.predict({
            'agent_count': 10, 'action_rate': 50, 'entropy_variance': 0.7,
            'time_correlation': 0.8, 'payload_similarity': 0.7
        })
        
        assert result['label'] == 1
        assert 0 <= result['score'] <= 1
    
    def test_save_and_load(self):
        """Test model save and load."""
        clf = CorrelationMLClassifier(pattern_threshold=0.55)
        
        train_data = [
            {'features': {'agent_count': 2, 'action_rate': 5, 'entropy_variance': 0.2,
                         'time_correlation': 0.1, 'payload_similarity': 0.2}, 'label': 0}
        ] * 10 + [
            {'features': {'agent_count': 10, 'action_rate': 50, 'entropy_variance': 0.7,
                         'time_correlation': 0.8, 'payload_similarity': 0.7}, 'label': 1}
        ] * 10
        
        clf.train(train_data)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            clf.save(filepath)
            assert os.path.exists(filepath)
            
            loaded_clf = CorrelationMLClassifier.load(filepath)
            
            assert loaded_clf.pattern_threshold == clf.pattern_threshold
            assert loaded_clf.trained == clf.trained
            assert loaded_clf.training_samples == clf.training_samples
            
            # Verify predictions match
            test_features = {'agent_count': 5, 'action_rate': 25, 'entropy_variance': 0.4,
                           'time_correlation': 0.4, 'payload_similarity': 0.4}
            result1 = clf.predict(test_features)
            result2 = loaded_clf.predict(test_features)
            assert abs(result1['score'] - result2['score']) < 0.01
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestAllClassifiersIntegration:
    """Integration tests for all classifiers with train_any_model.py workflow."""
    
    def test_all_classifiers_train_predict_save_load(self):
        """Test complete workflow for all classifiers."""
        classifiers = [
            ('baseline', BaselineMLClassifier, 
             [{'features': {'f1': 0.2, 'f2': 0.3}, 'label': 0}] * 20 +
             [{'features': {'f1': 0.8, 'f2': 0.9}, 'label': 1}] * 20,
             {'f1': 0.5, 'f2': 0.6}),
            
            ('anomaly', AnomalyMLClassifier,
             [{'features': {'sequence': ['a', 'b', 'c']}, 'label': 0}] * 20 +
             [{'features': {'sequence': ['x', 'y', 'z']}, 'label': 1}] * 20,
             {'sequence': ['a', 'b', 'c']}),
            
            ('correlation', CorrelationMLClassifier,
             [{'features': {'agent_count': 2, 'action_rate': 5, 'entropy_variance': 0.2,
                           'time_correlation': 0.1, 'payload_similarity': 0.2}, 'label': 0}] * 20 +
             [{'features': {'agent_count': 10, 'action_rate': 50, 'entropy_variance': 0.7,
                           'time_correlation': 0.8, 'payload_similarity': 0.7}, 'label': 1}] * 20,
             {'agent_count': 5, 'action_rate': 25, 'entropy_variance': 0.4,
              'time_correlation': 0.4, 'payload_similarity': 0.4}),
        ]
        
        for name, cls, train_data, test_features in classifiers:
            print(f"\nTesting {name} classifier...")
            
            # Initialize
            clf = cls()
            assert clf.trained is False
            
            # Train
            clf.train(train_data)
            assert clf.trained is True
            assert clf.training_samples == len(train_data)
            
            # Predict
            result = clf.predict(test_features)
            assert 'label' in result
            assert 'score' in result
            assert 0 <= result['score'] <= 1
            
            # Save and load
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                filepath = f.name
            
            try:
                clf.save(filepath)
                loaded_clf = cls.load(filepath)
                
                assert loaded_clf.trained == clf.trained
                assert loaded_clf.training_samples == clf.training_samples
                
                # Verify predictions match
                result2 = loaded_clf.predict(test_features)
                assert result['label'] == result2['label']
                
                print(f"  ✓ {name} classifier: all tests passed")
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
