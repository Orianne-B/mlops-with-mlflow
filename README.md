pyproject2conda yaml -f pyproject.toml >> python_env.yaml

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed-mlflow -P model_folder=/home/administrateur/exo/mlops-with-mlflow/models/random_forest/v1

- ml project
- experimentation 
