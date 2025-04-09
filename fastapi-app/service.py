import pandas as pd

import models
from schemas import PredictionInput


def make_predictions(input_data: PredictionInput):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    model = models.get_iris_model()
    # Make predictions
    predictions = model.predict(input_df)

    # Return the predictions
    return predictions.tolist()
