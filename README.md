# Local 

./run-pipeline.sh data/raw/iris.csv data/processed

# MLFLow Project

pyproject2conda yaml -f pyproject.toml >> python_env.yaml

export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
export MLFLOW_EXPERIMENT_NAME="iris_experiment"

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed-mlflow --env-manager=local

# MLFLow Serve

mlflow models serve -m runs:/a9199189feba45419a43b65328b903f3/model -p 8081 --env-manager=local

https://mlflow.org/docs/latest/deployment/deploy-model-locally/

curl http://127.0.0.1:8081/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"150":6.1,"4":3.0,"setosa":4.6,"versicolor":1.4}]}'

Poissible aussi de faire requete en python, cf documentation

# FastAPI

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

uvicorn app:app --host 0.0.0.0 --port 8000

TODO 
- module route ou api
- module service fonction métier (ex chargement du modèle )
- module schéma : modèle de données avec schéma 
- module modèle : connection base de données ex save historique des prédiction
- module core : securité, config, connection bdd
- module config : settings, pydantic, .env 

# Example Antoine

https://github.com/hanabi70/m2i_formation/tree/mlflow
