# Import necessary libraries and modules
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import os
import pathlib
import sys

# Define the directory paths for the model and source code
model_dir = pathlib.Path(__file__).resolve().parent / 'model'
src = pathlib.Path(__file__).resolve().parent / 'src'
sys.path.append(str(src))

# Load the pre-trained model from the specified path
model_path = model_dir / 'iris_model.joblib'
model = joblib.load(model_path)

# Initialize the FastAPI app
app = FastAPI()

# Define a root endpoint that returns a welcome message
@app.get("/")
def home():
    return "welcome to fastapi application"

# Define a Pydantic model for the input data
class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define an endpoint for making predictions
@app.post("/predict")
def predict_iris(data: Iris):
    """
    Predict the species of an iris flower based on its features.

    Args:
        data (Iris): An instance of the Iris class containing the features of the iris flower.

    Returns:
        dict: A dictionary containing the prediction result with the key 'prediction'.

    Raises:
        KeyError: If any of the required keys ('sepal_length', 'sepal_width', 'petal_length', 'petal_width') are missing in the data.
    """
    # Convert the incoming data to a dictionary format
    data = data.model_dump()
    
    # Extract individual features from the data dictionary
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']

    # Prepare the data in the format expected by the prediction model
    data = {
        'sepal.length': sepal_length,
        'sepal.width': sepal_width,
        'petal.length': petal_length,
        'petal.width': petal_width
    }

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, index=[0]) 

    # Make a prediction using the pre-trained model
    prediction = model.predict(df)[0]

    # Return the prediction result as a dictionary
    return {
        'prediction': prediction
    }

# Run the FastAPI app using Uvicorn when the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)