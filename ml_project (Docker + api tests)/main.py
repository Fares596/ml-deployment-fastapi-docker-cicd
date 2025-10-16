from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables from the .env file
load_dotenv()

# Global variables for model and scaler
model = None
scaler = None

# Define the application lifespan (startup and shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    try:
        model_path = os.getenv("MODEL_PATH", "linear_regression_model.pkl")
        scaler_path = os.getenv("SCALER_PATH", "scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        logging.info("Model and scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")
        raise RuntimeError("Error loading model or scaler")

    # --- Startup complete ---
    yield
    # --- Application shutdown ---
    logging.info("Application shutting down .")


# Create FastAPI app with lifespan context
app = FastAPI(lifespan=lifespan)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request model for prediction input
class PredictionRequest(BaseModel):
    hours_studied: float


# Home page endpoint
@app.get("/")
def home():
    return FileResponse("static/index.html")


# Prediction page endpoint
@app.get("/predict")
async def predict_page():
    return FileResponse("static/predict.html")


# Prediction API endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    global model, scaler

    if model is None or scaler is None:
        logging.error("Model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded, please try again later")

    if request.hours_studied <= 0:
        logging.warning("Invalid input: hours_studied must be positive.")
        raise HTTPException(status_code=400, detail="Hours studied must be a positive number")

    # Prepare input data for prediction
    hours = request.hours_studied
    data = pd.DataFrame([[hours]], columns=['Hours_Studied'])
    scaled_data = scaler.transform(data)

    try:
        prediction = model.predict(scaled_data)
        logging.info(f"Prediction for {hours} hours: {prediction[0]}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")

    return {"predicted_test_score": float(prediction[0])}


# Run the application (useful for local execution)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Compatible with Heroku or Render
    uvicorn.run(app, host="0.0.0.0", port=port)

