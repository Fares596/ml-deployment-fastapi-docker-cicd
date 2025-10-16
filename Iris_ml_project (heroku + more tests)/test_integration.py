import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_endpoint():
    """Test if the home page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200, "Failed to load the home page."

def test_predict_page_endpoint():
    """Test if the prediction page loads successfully."""
    response = client.get("/predict")
    assert response.status_code == 200, "Failed to load the prediction page."

def test_predict_species_endpoint():
    """Test if the predict endpoint works."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", data=payload)
    assert response.status_code == 200, "Failed to predict species."
    assert "prediction" in response.json(), "Response missing prediction field." 