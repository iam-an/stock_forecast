import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.prophet
import joblib
from pathlib import Path
import json

def evaluate_score(test, y_pred_test)->dict:
    # test データの実測値
    y_true = test.values
    # Prophet 予測の中心値 (yhat)
    y_pred = y_pred_test.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true=test, y_pred=y_pred)
    scores = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return scores

def save_artifacts(model, artifact_path):
    Path(artifact_path).mkdir(exist_ok=True)
    joblib.dump(model, Path(artifact_path) / "model.pkl")

def write_to_json(file, **settings):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)


def set_mlflow(model, scores: dict, artifacts):
    #  uv run mlflow server --backend-store-uri sqlite:///mlflow.db
    mlflow.end_run()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("try_experiment")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("mlflow.runName", run_id)
        for key, value in scores.items():
            mlflow.log_metric(key, value)
        for i in artifacts:
            mlflow.log_artifact(i)