# workflow orchestration의 필요성

ML Pipeline is..

1. Ingestion: download the data
2. Transforming: filtering and removing outliers
3. Preparing data for ML
4. Hyper parameter tuning
5. Train the final model
6. Register model to registry

notebook만으로 모든 과정을 진행하는 경우 

중복되는 코드가 많아지고, 재현성이 낮으며 테스트 및 모듈화, 버전 관리, 스케일 확장이 어렵기 때문에

파이썬 스크립트를 이용해 파이프라인을 실행하고

중앙화되고, scalable한 컴퓨팅 리소스에서의 협업 및 스케줄링을 위해 workflow orchestration이 필요

ex) Airflow, Prefect, Mage ... Specific for ML: Kubeflow pipllines, MLflow pipelines


# Turning notebook to python script


```bash
jupyter nbconvert --to=script duration-prediction.ipynb
```