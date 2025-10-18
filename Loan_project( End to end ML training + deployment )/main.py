from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import os
import uvicorn


# Initialize FastAPI app
app = FastAPI()

# Load the saved model and scaler
model = joblib.load('random_forest_model.pkl')  # Load your trained model
scaler = joblib.load('scaler.pkl')  # Load your saved scaler

# Mount the static directory to serve static files like CSS, HTML, etc.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 template engine
templates = Jinja2Templates(directory="static")


# Column names used when the scaler was fitted
column_names = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area'
]


# Home endpoint serving the index page
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=open("static/index.html").read(), status_code=200)

# Endpoint to predict loan approval
@app.get("/predict", response_class=HTMLResponse)
@app.post("/predict", response_class=HTMLResponse)
def predict_loan_status(request: Request, 
                 Gender: int = Form(None),
                 Married: int = Form(None),
                 Dependents: int = Form(None),
                 Education: int = Form(None),
                 Self_Employed: int = Form(None),
                 ApplicantIncome: float = Form(None),
                 CoapplicantIncome: float = Form(None),
                 LoanAmount: float = Form(None),
                 Loan_Amount_Term: int = Form(None),
                 Credit_History: int = Form(None),
                 Property_Area: int = Form(None)):

    prediction = None
    loan_status = None

    if Gender is not None:
        try:
            # Prepare the data as a DataFrame to keep column names
            data = pd.DataFrame([{
                'Gender': Gender,
                'Married': Married,
                'Dependents': Dependents,
                'Education': Education,
                'Self_Employed': Self_Employed,
                'ApplicantIncome': ApplicantIncome,
                'CoapplicantIncome': CoapplicantIncome,
                'LoanAmount': LoanAmount,
                'Loan_Amount_Term': Loan_Amount_Term,
                'Credit_History': Credit_History,
                'Property_Area': Property_Area
            }])

            # Scale the data using the loaded scaler
            scaled_data = scaler.transform(data)

            # Make the prediction using the model
            prediction = model.predict(scaled_data)
            loan_status = "Approved" if prediction[0] == 1 else "Not Approved"
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            loan_status = "There was an issue with the prediction request. Please check the input data."
        
    return templates.TemplateResponse("predict.html", {"request": request, "loan_status": loan_status})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Heroku's PORT 
    uvicorn.run(app, host="0.0.0.0", port=port)