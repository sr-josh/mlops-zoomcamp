import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hw2")

    best_rmse = float("inf")
    best_split = None

    for mss in [2, 4, 8, 10]:
        with mlflow.start_run():
            # mlflow.autolog()

            rf = RandomForestRegressor(
                max_depth=10,
                min_samples_split=mss,
                random_state=0
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_param("min_samples_split", mss)
            mlflow.log_metric("rmse", rmse)
            # mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models")

if __name__ == '__main__':
    run_train()
