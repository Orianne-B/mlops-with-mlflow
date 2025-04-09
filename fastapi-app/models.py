import mlflow

from config import Configuration


def get_iris_model():
    model_uri = Configuration.model_uri
    return mlflow.pyfunc.load_model(model_uri)
