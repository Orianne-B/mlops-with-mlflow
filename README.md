# Local 

`./run-pipeline.sh data/raw/iris.csv data/processed`

# MLFLow Project

`pyproject2conda yaml -f pyproject.toml >> python_env.yaml`

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:8080"
export MLFLOW_EXPERIMENT_NAME="iris_experiment"

mlflow run https://github.com/Orianne-B/mlops-with-mlflow.git -P input_data=/home/administrateur/exo/mlops-with-mlflow/data/raw/iris.csv -P processed_data_folder=/home/administrateur/exo/mlops-with-mlflow/data/processed --env-manager=local
```

# MLFLow Serve

`mlflow models serve -m runs:/a9199189feba45419a43b65328b903f3/model -p 8081 --env-manager=local`

https://mlflow.org/docs/latest/deployment/deploy-model-locally/

`curl http://127.0.0.1:8081/invocations -H "Content-Type:application/json"  --data '{"inputs": [{"sepal_lenght":6.1,"sepal_width":3.0,"petal_lenght":4.6,"petal_width":1.4}]}'`

Poissible aussi de faire requete en python, cf documentation

# FastAPI

`python3 app.py`

Doc
- lifespan : chose à faire au démarage et à la fin de l'application, gestion des dépendances synchrones

# Docker

```bash
cd fastapi-app
sudo docker build -t mlflow-service:latest .
sudo docker run --env-file .env --name mlflow-container -d mlflow-service 
sudo docker ps
sudo docker inspect <container_name>
```
<IPAddress>:5089

`docker stop mlflow-container`

# Example Antoine

https://github.com/hanabi70/m2i_formation/tree/mlflow
