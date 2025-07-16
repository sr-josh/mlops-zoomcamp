## Concepts
ML experiment란 ML 모델을 만드는 과정으로서,
실험에서 이루어지는 각 시도들을 experiment run 이라고 하고
실험과 관련된 파일을 run artifact라고 한다.
experiment metadata

## Experiment tracking이란

ML experiment에서 관련된 모든 정보를 추적하는 프로세스이다.
예를 들면 소스 코드, 환경, 데이터, 모델, 하이퍼파라미터, 메트릭스(평가지표) 등이 있다.

## Why

Reproducibility 동일한 결과를 재현할 수 있고
Organization 수많은 실험의 결과를 체계적으로 관리할 수 있으며
Optimization 최적화에 도움을 준다.

## How about spreadsheets

Error prone 오류가 발생하기 쉽고
No standard format 표준 형식이 존재하지 않아 혼란을 초래할 수 있으며
visibility collaboration 시각화 및 협업에 어려움이 있다

## MLflow

def) An open source platform for the machine learning lifecycle

```Bash
pip install mlflow
mlflow
```
```Bash
# Usage: mlflow [OPTIONS] COMMAND [ARGS]...

# Options:
#   --version  Show the version and exit.
#   --help     Show this message and exit.

# Commands:
#   artifacts    Upload, list, and download artifacts from an MLflow...
#   db           Commands for managing an MLflow tracking database.
#   deployments  Deploy MLflow models to custom targets.
#   doctor       Prints out useful information for debugging issues with MLflow.
#   experiments  Manage experiments.
#   gc           Permanently delete runs in the `deleted` lifecycle stage.
#   models       Deploy MLflow models locally.
#   run          Run an MLflow project from the given URI.
#   runs         Manage runs.
#   sagemaker    Serve models on SageMaker.
#   server       Run the MLflow tracking server.
```

pip로 설치할 수 있는 pythone package이며, 아래와 같이 4가지 주요 모듈을 포함하고 있다.

- Tracking

Parameters, Metrics, Metadata, Artifacts, Models
\+ automatically logs source code, version of the code, start and end time, author 

- Models
- Model Registry
- Projects


```Bash

cd 02-Experiment Tracking
mlflow ui --backend-store-uri sqlite:///mlflow.db #메타데이터를 저장할 위치 지정

```

실험을 진행하는 모듈이 아닌 곳에서 실행하면 트래킹을 할 수 없다


```Bash
conda create -n exp-tracking-env python=3.9.12
conda activate exp-tracking-env
pip install -r requirements.txt
```

https://hyperopt.github.io/hyperopt/getting-started/search_spaces/


## Model Management

Data Sourcing Labeling Versioning

Model Management
{
    Experiment tracking
    [Model Architecture Training Evaluation(Scaling HW)]

    Model Versioning Deployment(also affect to labeling) (Scaling HW)
}

Prediction monitoring

```Python
# 1. mlflow log_artifact
mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models")

# 2. using framework's log_model - 어떤 프레임워크를 사용하든 MLflow Model 형태로 저장한 뒤 다양한 플랫폼에 배포할 수 있다. 
mlflow.xgboost.log_model(booster, artifact_path="model_mlflow")
```