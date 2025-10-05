
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from sklearn.model_selection import train_test_split


def pre_process_No1():
    """
    コンセプト：
    データの時系列関係を利用するため、過去n日分のデータから明日の予測を行うこと繰り返す
    """
    from read_API_data import read_datasets
    df, settings = read_datasets() 
    target = np.array(df[f"('High', '{settings["company"]}')"])

    #学習用データセット作成
    n = 10
    X, y = [], []

    for i in range(len(target)-n):
        X.append(target[i:i+n])
        y.append(target[i+n])

    X = np.array(X)
    y = np.array(y)

    # 学習・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return(X_train, X_test, y_train, y_test)
