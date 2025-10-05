#%%
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def learn_No1():
    from pre_process import pre_process_No1
    X_train, X_test, y_train, y_test = pre_process_No1()

    # モデル作成と学習
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 予測
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    return(y_test, y_train, y_test_pred, y_train_pred)

