
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split


#方針としてSTL分解して、トレンドや季節性だけを取り出して予測する方向性にする、最後に合わせるイメージ

from read_API_data import read_datasets
df, settings = read_datasets() 

target = df["High"]
stl = STL(target, period=30)
result = stl.fit()

# 分解結果
df['trend'] = result.trend
df['seasonal'] = result.seasonal
df['resid'] = result.resid




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

