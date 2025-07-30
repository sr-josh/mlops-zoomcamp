import os
import pickle
import click
import mlflow
import numpy as np

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    
    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            # new_params[param] = int(params[param]) # 입력받은 파라미터 값을 int로 변환해 딕셔너리에 저장.
            
            value = int(params[param])
            if param == "n_estimators":
                value = min(value, 10)
            if param == "max_depth":
                value = min(value, 5)
            new_params[param] = value
        
        rf = RandomForestRegressor(**new_params) # 변환된 파라미터로 랜덤포레스트 모델 객체 생성.
        rf.fit(X_train, y_train) # 학습 데이터로 모델을 학습시킴.
        print("!")
        mlflow.sklearn.log_model(rf, artifact_path="model") # 학습된 모델을 MLflow에 저장함.
        print("?")
        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        print("??")
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        

# 커맨드라인에서 인자를 받아서 전달
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        try:
            train_and_log_model(data_path=data_path, params=run.data.params)
        except Exception as e:
            print(f"Error: {e}")
    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        # run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )[0]
    print(f"Best run ID: {best_run.info.run_id} with RMSE: {best_run.data.metrics['rmse']}")

    # # Register the best model
    mlflow.register_model(
        f"runs:/{best_run.info.run_id}/model",
        "RandomForestRegressor"
    )


if __name__ == '__main__':
    run_register_model()