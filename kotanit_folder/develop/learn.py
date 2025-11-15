#%%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


"""model_choose"""
from statsmodels.tsa.arima.model import ARIMA
#model = ARIMA(order=(1, 1, 1))
from sklearn.linear_model import LinearRegression
model = LinearRegression()

def learn_classical_regression():
    from pre_process import pre_process_No1
    datasets, settings = pre_process_No1()

    if settings["stl"]:
        target_cols = ["trend", "seasonal"]
    else:
        target_cols = ["High"]

    for target_col in target_cols:

        model.fit(datasets[f"{target_col}:X_train"], datasets[f"{target_col}:y_train"])

        # 予測
        datasets[f"{target_col}:y_test_pred"]  = model.predict(datasets[f"{target_col}:X_test"])
        datasets[f"{target_col}:y_train_pred"] = model.predict(datasets[f"{target_col}:X_train"])

    return datasets, settings
