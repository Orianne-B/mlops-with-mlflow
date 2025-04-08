pyproject2conda yaml -f pyproject.toml >> python_env.yaml

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed-mlflow --env-manager=local

- ml project
- experimentation 

https://github.com/hanabi70/m2i_formation/tree/mlflow

fondiokim@gmail.com
