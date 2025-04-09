from fastapi import APIRouter

from schemas import PredictionInput
import service

# Initialize the router
router = APIRouter()


@router.get("/", tags=["Home Page"])
def root():
    """Default page."""
    return {"message": "Welcome to the ML Flow Service!"}


# Define the prediction endpoint
@router.post("/predict")
def predict(input_data: PredictionInput):
    predictions = service.make_predictions(input_data)

    # Return the predictions
    return {"predictions": predictions}
