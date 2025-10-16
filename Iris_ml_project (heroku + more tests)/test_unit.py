import pytest
import pandas as pd
from main import model, species_mapping


def test_model_prediction():
    # Arrange: Create a test input DataFrame
    input_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    
    # Act: Perform prediction
    prediction = model.predict(input_df)[0]
    species = species_mapping.get(prediction, "Unknown")
    
    # Assert: Check prediction result
    assert species == "Iris Setosa", "The prediction did not return the expected result."
