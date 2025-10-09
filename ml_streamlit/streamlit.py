import streamlit as st
import joblib

# Load the model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app
st.title("MetaBrains Student Test Score Predictor")
st.write("Enter the number of hours studied to predict the test score.")

# User input
hours = st.number_input("Hours studied:", min_value=0.0, step=1.0)

if st.button("Predict"):
    try:
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write(f"Predicted Test Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
