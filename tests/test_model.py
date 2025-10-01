# tests/test_model.py
import os
import joblib
import json
import pytest

def test_model_file_exists():
    """Test that model file exists"""
    assert os.path.exists('models/sentiment_model.pkl'), "Model file not found"

def test_metrics_file_exists():
    """Test that metrics file exists"""
    assert os.path.exists('metrics.json'), "Metrics file not found"

def test_model_loads():
    """Test that model can be loaded"""
    model = joblib.load('models/sentiment_model.pkl')
    assert model is not None, "Model failed to load"

def test_model_predicts():
    """Test that model can make predictions"""
    model = joblib.load('models/sentiment_model.pkl')
    prediction = model.predict(["This is a test"])
    assert len(prediction) == 1, "Prediction failed"
    assert prediction[0] in ['positive', 'negative', 'neutral'], "Invalid prediction"

def test_metrics_valid():
    """Test that metrics are valid"""
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)
    
    assert 'accuracy' in metrics, "Accuracy metric missing"
    assert 0 <= metrics['accuracy'] <= 1, "Invalid accuracy value"
    assert metrics['accuracy'] > 0.5, "Accuracy too low"

if __name__ == "__main__":
    pytest.main([__file__])