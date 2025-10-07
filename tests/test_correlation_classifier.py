"""Tests for CorrelationMLClassifier."""
import pytest
from nethical.mlops.correlation_classifier import CorrelationMLClassifier


class TestCorrelationMLClassifier:
    """Test correlation ML classifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        clf = CorrelationMLClassifier(pattern_threshold=0.5)
        
        assert clf.pattern_threshold == 0.5
        assert clf.trained is False
        assert clf.training_samples == 0
        assert len(clf.feature_weights) == 5
    
    def test_train_with_correlation_data(self):
        """Test training with correlation pattern data."""
        # Create training data
        train_data = []
        
        # Normal patterns (no correlation)
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 2,
                    'action_rate': 5.0,
                    'entropy_variance': 0.2,
                    'time_correlation': 0.1,
                    'payload_similarity': 0.2
                },
                'label': 0
            })
        
        # Correlation patterns
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 10,
                    'action_rate': 50.0,
                    'entropy_variance': 0.7,
                    'time_correlation': 0.8,
                    'payload_similarity': 0.7
                },
                'label': 1
            })
        
        clf = CorrelationMLClassifier(pattern_threshold=0.5)
        clf.train(train_data)
        
        assert clf.trained is True
        assert clf.training_samples == 100
    
    def test_predict_normal_pattern(self):
        """Test prediction on normal (no correlation) pattern."""
        # Create training data
        train_data = []
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 2,
                    'action_rate': 5.0,
                    'entropy_variance': 0.2,
                    'time_correlation': 0.1,
                    'payload_similarity': 0.2
                },
                'label': 0
            })
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 10,
                    'action_rate': 50.0,
                    'entropy_variance': 0.7,
                    'time_correlation': 0.8,
                    'payload_similarity': 0.7
                },
                'label': 1
            })
        
        clf = CorrelationMLClassifier(pattern_threshold=0.5)
        clf.train(train_data)
        
        # Test normal pattern
        result = clf.predict({
            'agent_count': 2,
            'action_rate': 5.0,
            'entropy_variance': 0.2,
            'time_correlation': 0.1,
            'payload_similarity': 0.2
        })
        
        assert result['label'] == 0
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_predict_correlation_pattern(self):
        """Test prediction on correlation pattern."""
        # Create training data
        train_data = []
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 2,
                    'action_rate': 5.0,
                    'entropy_variance': 0.2,
                    'time_correlation': 0.1,
                    'payload_similarity': 0.2
                },
                'label': 0
            })
        for _ in range(50):
            train_data.append({
                'features': {
                    'agent_count': 10,
                    'action_rate': 50.0,
                    'entropy_variance': 0.7,
                    'time_correlation': 0.8,
                    'payload_similarity': 0.7
                },
                'label': 1
            })
        
        clf = CorrelationMLClassifier(pattern_threshold=0.5)
        clf.train(train_data)
        
        # Test correlation pattern
        result = clf.predict({
            'agent_count': 10,
            'action_rate': 50.0,
            'entropy_variance': 0.7,
            'time_correlation': 0.8,
            'payload_similarity': 0.7
        })
        
        assert result['label'] == 1
        assert 0.0 <= result['score'] <= 1.0
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_save_and_load(self):
        """Test model save and load."""
        import tempfile
        import os
        
        # Create and train model
        train_data = []
        for _ in range(20):
            train_data.append({
                'features': {
                    'agent_count': 2,
                    'action_rate': 5.0,
                    'entropy_variance': 0.2,
                    'time_correlation': 0.1,
                    'payload_similarity': 0.2
                },
                'label': 0
            })
        for _ in range(20):
            train_data.append({
                'features': {
                    'agent_count': 10,
                    'action_rate': 50.0,
                    'entropy_variance': 0.7,
                    'time_correlation': 0.8,
                    'payload_similarity': 0.7
                },
                'label': 1
            })
        
        clf = CorrelationMLClassifier(pattern_threshold=0.5)
        clf.train(train_data)
        
        # Save model
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            clf.save(filepath)
            
            # Load model
            loaded_clf = CorrelationMLClassifier.load(filepath)
            
            assert loaded_clf.pattern_threshold == clf.pattern_threshold
            assert loaded_clf.trained == clf.trained
            assert loaded_clf.training_samples == clf.training_samples
            
            # Test that loaded model makes same predictions
            test_features = {
                'agent_count': 2,
                'action_rate': 5.0,
                'entropy_variance': 0.2,
                'time_correlation': 0.1,
                'payload_similarity': 0.2
            }
            result1 = clf.predict(test_features)
            result2 = loaded_clf.predict(test_features)
            
            assert result1['label'] == result2['label']
            assert abs(result1['score'] - result2['score']) < 0.01
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_calculate_entropy(self):
        """Test entropy calculation."""
        clf = CorrelationMLClassifier()
        
        # Uniform distribution should have high entropy
        entropy1 = clf._calculate_entropy("abcdefgh")
        
        # Repetitive sequence should have low entropy
        entropy2 = clf._calculate_entropy("aaaaaaaa")
        
        assert entropy1 > entropy2
        assert entropy2 == 0.0  # All same elements
    
    def test_extract_features(self):
        """Test feature extraction and normalization."""
        clf = CorrelationMLClassifier()
        
        raw_features = {
            'agent_count': 5,
            'action_rate': 25.0,
            'entropy_variance': 0.5,
            'time_correlation': 0.6,
            'payload_similarity': 0.7
        }
        
        normalized = clf._extract_features(raw_features)
        
        assert 'agent_count' in normalized
        assert 'action_rate' in normalized
        assert 'entropy_variance' in normalized
        assert 'time_correlation' in normalized
        assert 'payload_similarity' in normalized
        
        # Check that all values are normalized
        for value in normalized.values():
            assert 0.0 <= value <= 1.0
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        clf = CorrelationMLClassifier()
        
        predictions = [0, 0, 1, 1, 0, 1, 0, 1]
        labels = [0, 0, 1, 0, 1, 1, 0, 1]
        
        metrics = clf.compute_metrics(predictions, labels)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'ece' in metrics
        
        # Check that metrics are in valid range
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'ece']:
            assert 0.0 <= metrics[key] <= 1.0
