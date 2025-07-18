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