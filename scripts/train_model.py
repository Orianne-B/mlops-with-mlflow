import argparse
import uuid
import joblib
import mlflow
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

root_folder = pathlib.Path("__file__").resolve().parent

# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
id = uuid.uuid4().hex
# mlflow.set_experiment(experiment_id=id)
mlflow.sklearn.autolog()


def train_model(
    training_data: str = str(root_folder.joinpath("data", "processed", "train.csv")),
) -> None:
    # Load the dataset
    train_df = pd.read_csv(training_data, sep=",")

    # Extract features and labels
    x_train = train_df.iloc[
        :, 1:-1
    ]  # Features (skip the first column which is an index)
    y_train = train_df.iloc[:, -1]  # Target (last column)

    with mlflow.start_run():
        # Train a Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)
        """        mlflow.sklearn.log_model(
            registered_model_name="mlflow",
            sk_model=model,
            artifact_path="iris_model",
        )"""
        mlflow.sklearn.save_model(model, "iris_model")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--training_data",
        type=str,
        default=str(root_folder.joinpath("data", "processed", "train.csv")),
    )

    args = parser.parse_args()

    train_model(args.training_data)
