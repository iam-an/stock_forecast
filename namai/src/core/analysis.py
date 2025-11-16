import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.prophet
import joblib
from pathlib import Path

def evaluate_score(test, y_pred_test)->dict:
    # test データの実測値
    y_true = test.values
    # Prophet 予測の中心値 (yhat)
    y_pred = y_pred_test.values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    scores = {"MAE": mae, "RMSE": rmse}
    return scores

def save_artifacts(model, artifact_path):
    joblib.dump(model, Path(artifact_path) / "model.pkl")



def set_mlflow(model, scores: dict, artifacts):
    mlflow.end_run()
    mlflow.set_experiment("try_experiment")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("mlflow.runName", run_id)
        for key, value in scores.items():
            mlflow.log_metric(key, value)
        mlflow.prophet.log_model(model)
        for i in artifacts:
            mlflow.log_artifact(i)