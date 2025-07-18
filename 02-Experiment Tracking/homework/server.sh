mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:/workspaces/mlops-zoomcamp/02-Experiment\ Tracking/homework/mlruns \
  --host 0.0.0.0 --port 5000
