import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

def read_dataframe(year, month):
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationId', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationId'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        params = {
            'max_depth': 8,
            'learning_rate': 0.3505001399438667,
            'reg_alpha': 0.03931781860170088,
            'reg_lambda': 0.3002990958721907,
            'min_child_weight': 4.658981377254529,
            'objective': 'reg:linear',
            'seed': 42
        }
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="models")

        mlflow.xgboost.log_model(booster, artifact_path="model_mlflow", input_example=X_val[0:1])
        # sample이 없으면 WARNING mlflow.models.model: Model logged without a signature and input example. Please set input_example parameter when logging the model to auto infer the model signature.


def run(year, month):
    df_train = read_dataframe(year=2021, month=1)
    df_val = read_dataframe(year=2021, month=2)
        
    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)

if __name__ == "__main__":
    run(year=2021, month=1)