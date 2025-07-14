import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

@pytest.fixture
def client():
    """Test client fixture"""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Test health check works"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_predict_single_passenger(client):
    """Test prediction endpoint works"""
    passenger_data = [{
        "Pclass": 1,
        "Sex": "female", 
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": "C"
    }]
    
    response = client.post("/predict", json=passenger_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    
    prediction = data["predictions"][0]
    assert prediction["survival_prediction"] in [0, 1]

def test_predict_multiple_passengers(client):
    """Test prediction endpoint works with multiple passengers"""
    passenger_data = [
        {
            "Pclass": 1,
            "Sex": "female", 
            "Age": 25.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80.0,
            "Embarked": "C"
        },
        {
            "Pclass": 3,
            "Sex": "male", 
            "Age": 30.0,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 8.0,
            "Embarked": "S"
        }
    ]
    
    response = client.post("/predict", json=passenger_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    for prediction in data["predictions"]:
        assert prediction["survival_prediction"] in [0, 1]

def test_input_validation(client):
    """Test input validation works"""
    # Invalid passenger class
    invalid_data = [{
        "Pclass": 4,  # Invalid
        "Sex": "female",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Embarked": "C"
    }]
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
