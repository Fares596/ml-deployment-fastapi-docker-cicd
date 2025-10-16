import pytest
from fastapi.testclient import TestClient
from main import app  


@pytest.fixture
def client():
    """Create a TestClient instance for making requests."""
    with TestClient(app) as client:
        yield client

# Test home route
def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Welcome to Meta Brain's Prediction Model" in response.text  # Check if the static index page content is correct


# Test predict page route
def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Predict Test Score" in response.text  # Check if predict page content is present

def test_predict_invalid_input(client):
    response = client.post("/predict", json={"hours_studied": -2.0})
    assert response.status_code == 400
    assert "Hours studied should be a positive number" in response.json()["detail"]

