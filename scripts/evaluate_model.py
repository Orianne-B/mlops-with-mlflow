import argparse
import joblib
import pathlib
import pandas as pd
from sklearn.metrics import accuracy_score


root_folder = pathlib.Path("__file__").resolve().parent


def evaluate_model(
    model_folder: pathlib.Path = root_folder.joinpath("models"),
    model_name: str = "random_forest_model_v1.pkl",
    data_root_folder: pathlib.Path = root_folder.joinpath("data"),
    evaluation_threshold: float = 0.8,
) -> bool:
    # Load the model
    model_path = model_folder.joinpath(model_name)
    model = joblib.load(model_path)

    # Load the test data
    test_df = pd.read_csv(
        data_root_folder.joinpath("processed", "test.csv"), sep=","
    )

    x_test = test_df.iloc[:, 1:-1]
    y_test = test_df.iloc[:, -1]

    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy}")

    if accuracy >= evaluation_threshold:
        return True
    return False

if __name__ == "__main__":
    evaluate_model()
