# tests/test_basic.py
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_project_structure():
    """Test basic project structure"""
    assert os.path.exists('src/'), "src/ directory missing"
    assert os.path.exists('src/app.py'), "app.py missing"
    assert os.path.exists('src/train.py'), "train.py missing"
    assert os.path.exists('requirements.txt'), "requirements.txt missing"

def test_requirements_file():
    """Test requirements.txt is not empty"""
    with open('requirements.txt', 'r') as f:
        content = f.read().strip()
    assert len(content) > 0, "requirements.txt is empty"
    assert 'flask' in content.lower(), "Flask not in requirements"
    assert 'scikit-learn' in content.lower(), "scikit-learn not in requirements"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])