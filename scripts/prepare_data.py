import argparse
import numpy as np
import pathlib
import pandas as pd

root_folder = pathlib.Path("__file__").resolve().parent


def prepare_data(
    input_data: str = str(root_folder.joinpath("data", "raw", "iris.csv")),
    output_folder: str = root_folder.joinpath("data", "processed"),
) -> None:
    iris_df = pd.read_csv(input_data, sep=",")

    train, validate, test = np.split(
        iris_df.sample(frac=1, random_state=42),
        [int(0.6 * len(iris_df)), int(0.8 * len(iris_df))],
    )

    train.to_csv(pathlib.Path(output_folder).joinpath("train.csv"))
    validate.to_csv(pathlib.Path(output_folder).joinpath("validate.csv"))
    test.to_csv(pathlib.Path(output_folder).joinpath("test.csv"))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--input_data",
        type=str,
        default=str(root_folder.joinpath("data", "raw", "iris.csv")),
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=str(root_folder.joinpath("data", "processed")),
    )

    args = parser.parse_args()

    prepare_data(args.input_data, args.output_folder)
