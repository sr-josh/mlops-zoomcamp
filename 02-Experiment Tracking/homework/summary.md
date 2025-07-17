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
