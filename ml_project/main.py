from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import logging
from fastapi import HTTPException
from dotenv import load_dotenv
import os
import uvicorn

# Load the saved model and scaler
model = None
scaler = None

# Load environment variables from .env file in the same directory
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Serve the static files from the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
   # Use environment variables to load model and scaler paths
        model_path = os.getenv("MODEL_PATH", "linear_regression_model.pkl")
        scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        logging.info("Model and scaler have been loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        raise HTTPException(status_code=500, detail="Error loading model")

# Define a request model for the input
class PredictionRequest(BaseModel):
    hours_studied: float

# Home endpoint 
@app.get("/")
def home():
    return FileResponse("static/index.html")

# Prediction page endpoint
@app.get("/predict")
async def predict_page():
    return FileResponse("static/predict.html")

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or scaler is None:
        logging.error("Model is not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded, please try again later")
    
  # Input validation: Ensure hours_studied is a positive number
    if request.hours_studied <= 0:
        logging.warning("Received invalid input: Hours studied should be positive.")
        raise HTTPException(status_code=400, detail="Hours studied should be a positive number")

    hours = request.hours_studied     # Extract the hours studied from the request
    data = pd.DataFrame([[hours]], columns=['Hours_Studied'])   # Prepare the data for prediction
    scaled_data = scaler.transform(data) # Scale the input data

    try:
        prediction = model.predict(scaled_data)  # Make prediction using the model
        logging.info(f"Prediction for {hours} hours: {prediction[0]}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
    
    return {"predicted_test_score": prediction[0]} # Return the predicted test score



if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Heroku's PORT 
    uvicorn.run(app, host="0.0.0.0", port=port)
