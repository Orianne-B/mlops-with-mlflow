from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model_uri = "runs:/bf4e479eb6b044979319b154c43acac6/model"
model = mlflow.pyfunc.load_model(model_uri)

# Define the input schema
class PredictionInput(BaseModel):
    sepal_lenght: float
    sepal_width: float
    petal_lenght: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Map old names
    input_df.rename(columns={
        "sepal_lenght": "150",
        "sepal_width": "4",
        "petal_lenght": "setosa",
        "petal_width": "versicolor",
    }, inplace=True)
    
    # Make predictions
    predictions = model.predict(input_df)
    
    # Return the predictions
    return {"predictions": predictions.tolist()}
