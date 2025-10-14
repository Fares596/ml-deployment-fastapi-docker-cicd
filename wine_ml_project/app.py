import streamlit as st
import requests

# FastAPI backend URL
BASE_URL = "http://127.0.0.1:8000"

# Define pages
def home_page():
    st.title("Wine Quality Prediction API")
    st.write("This is the home page of the Wine Quality Prediction App.")
    
    # Communicate with the FastAPI `/` endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            st.success(data.get("message", "Welcome to the API!"))
        else:
            st.error("Failed to fetch data from the backend.")
    except Exception as e:
        st.error(f"Error: {e}")

def prediction_page():
    st.title("Wine Quality Prediction")
    st.write("Enter the wine features to predict its quality:")

    # Input fields for wine features
    fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.27)
    citric_acid = st.number_input("Citric Acid", value=0.36)
    residual_sugar = st.number_input("Residual Sugar", value=20.7)
    chlorides = st.number_input("Chlorides", value=0.045)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=45.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=170.0)
    density = st.number_input("Density", value=1.001)
    pH = st.number_input("pH", value=3.0)
    sulphates = st.number_input("Sulphates", value=0.45)
    alcohol = st.number_input("Alcohol", value=8.8)

    # Predict button
    if st.button("Predict Quality"):
        # Prepare the payload
        payload = {
            "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity,
            "citric_acid": citric_acid,
            "residual_sugar": residual_sugar,
            "chlorides": chlorides,
            "free_sulfur_dioxide": free_sulfur_dioxide,
            "total_sulfur_dioxide": total_sulfur_dioxide,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol
        }

        # Communicate with the FastAPI `/predict` endpoint
        try:
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Wine Quality: {result['predicted_quality']}")
            else:
                st.error("Failed to get prediction from the backend.")
        except Exception as e:
            st.error(f"Error: {e}")

# Streamlit page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Wine Quality"])

if page == "Home":
    home_page()
elif page == "Predict Wine Quality":
    prediction_page()
