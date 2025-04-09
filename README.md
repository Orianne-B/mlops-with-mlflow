pyproject2conda yaml -f pyproject.toml >> python_env.yaml

export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed-mlflow --env-manager=local

mlflow models serve -m runs:/a9199189feba45419a43b65328b903f3/model -p 8081 --env-manager=local

curl http://127.0.0.1:8081/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"150":6.1,"4":3.0,"setosa":4.6,"versicolor":1.4}]}'

https://github.com/hanabi70/m2i_formation/tree/mlflow
