from pydantic import BaseModel


# Define the input schema
class PredictionInput(BaseModel):
    sepal_lenght: float
    sepal_width: float
    petal_lenght: float
    petal_width: float
