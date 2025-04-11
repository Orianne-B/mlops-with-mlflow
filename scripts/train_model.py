"""Train a model using MLflow and scikit-learn."""

import argparse
import mlflow
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog()


def train_model(training_data: str) -> None:
    # Load the dataset
    train_df = pd.read_csv(pathlib.Path(training_data).joinpath("train.csv"), sep=",")

    # Extract features and labels
    x_train = train_df.iloc[
        :, 1:-1
    ]  # Features (skip the first column which is an index)
    y_train = train_df.iloc[:, -1]  # Target (last column)

    with mlflow.start_run():
        # Train a Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)
        mlflow.sklearn.log_model(
            registered_model_name="iris_model",
            sk_model=model,
            artifact_path="iris_model",
        )
        # mlflow.sklearn.save_model(model, "iris_model")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)

    args = parser.parse_args()

    train_model(args.data_folder)
