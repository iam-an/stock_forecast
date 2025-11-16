import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.prophet
import joblib
from pathlib import Path

def evaluate_score(test, y_pred_test):
    # test データの実測値
    y_true = test.values
    # Prophet 予測の中心値 (yhat)
    y_pred = y_pred_test.values

    # 精度指標を計算
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def save_artifacts(model, artifact_path):
    joblib.dump(model, Path(artifact_path) / "model.pkl")



def set_mlflow(model, scores, artifacts):
    mlflow.end_run()
    mlflow.set_experiment("try_experiment")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("mlflow.runName", run_id)
        mlflow.log_metric("RMSE", scores)
        mlflow.prophet.log_model(model)
        for i in artifacts:
            mlflow.log_artifact(i)