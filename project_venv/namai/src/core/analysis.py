import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_score(test, y_pred_test):
    # test データの実測値
    y_true = test["y"].values
    # Prophet 予測の中心値 (yhat)
    y_pred = y_pred_test["yhat"].values

    # 精度指標を計算
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
