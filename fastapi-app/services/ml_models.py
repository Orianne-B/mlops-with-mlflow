import mlflow

from core import config

def get_iris_model():
    model_uri = config.MODEL_URI
    return mlflow.pyfunc.load_model(model_uri)
