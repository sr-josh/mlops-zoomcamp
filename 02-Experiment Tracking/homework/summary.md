1. Install mlflow with pip or conda

2. Preprocess

load the data
fit a DictVectorizer
save the data

```bash
python preprocessing.py --raw_data_path /data --dest_path ./output
```

3. Training

load the preprocessed dataset
train the model on the training set
calculate the RMSE on the validation set
autologging with MLflow


min_samples_split은 RandomForestRegressor의 하이퍼파라미터로, 노드를 분할하기 위한 최소 샘플수이다.

# 5.429798190124849 10 [v] 
# 5.429994133149121 8
# 5.430374895679978 4
# 5.431162180141208 2

4. 로컬 tracking server를 실행

mlflow.set_tracking_uri("sqlite:///mlflow.db")를 통해 로컬 데이터베이스를 사용하도록 설정
mlflow ui를 실행할 때 backend-store-uri를 지정

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:/workspaces/mlops-zoomcamp/02-Experiment\ Tracking/homework/mlruns \
  --host 0.0.0.0 --port 5000
```


5. 하이퍼 파라미터 튜닝

Now let's try to reduce the validation error by tuning the hyperparameters of the RandomForestRegressor using hyperopt. We have prepared the script hpo.py for this exercise.

Your task is to modify the script hpo.py and make sure that the validation RMSE is logged to the tracking server for each run of the hyperparameter optimization (you will need to add a few lines of code to the objective function) and run the script without passing any parameters.

After that, open UI and explore the runs from the experiment called random-forest-hyperopt to answer the question below.

Note: Don't use autologging for this exercise.

The idea is to just log the information that you need to answer the question below, including:

the list of hyperparameters that are passed to the objective function during the optimization,
the RMSE obtained on the validation set (February 2023 data).
What's the best validation RMSE that you got? 5.335

6. 최적의 모델을 모델 레지스트리에 프로모트

The results from the hyperparameter optimization are quite good. So, we can assume that we are ready to test some of these models in production. In this exercise, you'll promote the best model to the model registry. We have prepared a script called register_model.py, which will check the results from the previous step and select the top 5 runs. After that, it will calculate the RMSE of those models on the test set (March 2023 data) and save the results to a new experiment called random-forest-best-models.

Your task is to update the script register_model.py so that it selects the model with the lowest RMSE on the test set and registers it to the model registry.

Tip 1: you can use the method search_runs from the MlflowClient to get the model with the lowest RMSE,

Tip 2: to register the model you can use the method mlflow.register_model and you will need to pass the right model_uri in the form of a string that looks like this: "runs:/<RUN_ID>/model", and the name of the model (make sure to choose a good one!).

What is the test RMSE of the best model?