import pytest
import sys
import os
import pickle
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="module")
def model_fixture():
    """Load model once for all tests"""
    model_paths = ['./models/best_model.pkl', '../models/best_model.pkl']
    model = None
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            break
    if model is None:
        pytest.skip("Model file not found")
    return model

def test_predict_single_passenger(model_fixture):
    """Test predict function works with single passenger"""
    model = model_fixture
    data = pd.DataFrame([{
        'Pclass': 3, 'Sex': 'male', 'Age': 22.0, 
        'SibSp': 1, 'Parch': 0, 'Fare': 7.25, 'Embarked': 'S'
    }])
    prediction = model.predict(data)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]

def test_predict_multiple_passengers(model_fixture):
    """Test predict function works with multiple passengers"""
    model = model_fixture
    data = pd.DataFrame([
        {'Pclass': 1, 'Sex': 'female', 'Age': 35.0, 'SibSp': 0, 'Parch': 0, 'Fare': 80.0, 'Embarked': 'C'},
        {'Pclass': 3, 'Sex': 'male', 'Age': 25.0, 'SibSp': 0, 'Parch': 0, 'Fare': 8.0, 'Embarked': 'S'}
    ])
    predictions = model.predict(data)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)

def test_predict_proba(model_fixture):
    """Test predict_proba function returns valid probabilities"""
    model = model_fixture
    data = pd.DataFrame([{
        'Pclass': 1, 'Sex': 'female', 'Age': 30.0, 
        'SibSp': 0, 'Parch': 0, 'Fare': 50.0, 'Embarked': 'C'
    }])
    probabilities = model.predict_proba(data)
    assert probabilities.shape == (1, 2)
    assert abs(probabilities[0].sum() - 1.0) < 1e-5
