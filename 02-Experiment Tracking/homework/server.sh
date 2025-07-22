mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root file:/workspaces/mlops-zoomcamp/02-Experiment\ Tracking/homework/mlruns \
  --host 0.0.0.0 --port 5000

#--backend-store-uri 실험 메타데이터 저장 위치
#--default-artifact-root 모델, 아티팩트 등 파일 저장 위치(로컬 경로 또는 S3, GCS 등)
