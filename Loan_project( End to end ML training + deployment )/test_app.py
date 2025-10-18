import pytest
from fastapi.testclient import TestClient
from main import app  

# Create a TestClient for the app
client = TestClient(app)

def test_home():
    """Test the home endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Loan Approval Prediction API" in response.text  # Verify content in the response

def test_predict_endpoint_valid_data():
    data = {
        'Gender': 0,
        'Married': 1,
        'Dependents': 0,
        'Education': 0,
        'Self_Employed': 0,
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 100,
        'Loan_Amount_Term': 360,
        'Credit_History': 1,
        'Property_Area': 2
    }
    response = client.post("/predict", data=data)
    
    assert response.status_code == 200
    assert 'Loan Status' in response.text  # Check that Loan Status is rendered in the response
    assert 'Approved' in response.text or 'Not Approved' in response.text

def test_predict_endpoint_invalid_data():
    data = {
        "Gender": "invalid",  # Invalid data type
        "Married": 1,
        "Dependents": 0,
        "Education": 0,
        "Self_Employed": 0,
        "ApplicantIncome": 5000.0,
        "CoapplicantIncome": 2000.0,
        "LoanAmount": 100.0,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": 2
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 422  # FastAPI returns 422 for validation errors
