#%%
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split


#方針としてSTL分解して、トレンドや季節性だけを取り出して予測する方向性にする、最後に合わせるイメージ

def stl_decomposition(df, settings):
    target = df["High"]
    np.save(f"models/{settings['company']}_LR.npy", target.to_numpy())
    stl = STL(target, period=settings["stl_period"], robust=True)
    result = stl.fit()

    # 分解結果
    df = df.with_columns([
        pl.Series("High", target),
        pl.Series("trend", result.trend),
        pl.Series("seasonal", result.seasonal),
        pl.Series("resid", result.resid)
    ])

    return df


def pre_process_No1():

    """
    コンセプト：
    データの時系列関係を利用するため、過去in_window日分のデータからout_window日分の予測を行うこと繰り返す
    前処理内容：
    source_data(dfに格納されている予測対象データ)を入力：in_window日分一塊、出力：out_window日分一塊ものをたくさん用意する（一日ずらしで作成）
    学習用とテスト用で分ける（一塊は維持）
    """
    from read_API_data import read_datasets
    df, settings = read_datasets() 
    in_window   = int(settings["in_window"])
    out_window  = int(settings["out_window"])

    datasets = {}
    datasets["target"] = df["High"].to_list()
    if settings["stl"]:
        df = stl_decomposition(df, settings)
        target_cols = ["trend", "seasonal"]
        datasets["resid"] = df["resid"].to_list()
    else:
        target_cols = ["High"]

    for target_col in target_cols:

        # --- データセットの作成 ---
        source_data = df[target_col].to_numpy()

        # 入力(X): in_window営業日
        # 出力(y): out_window営業日
        X = []
        y = []

        # データをスライドさせながら学習用ペアを作成
        # データの長さから、ウィンドウサイズ分を引いた回数だけループ
        for i in range(len(source_data) - in_window - out_window + 1):
            # 入力データ: i番目から i+10番目まで
            window_X = source_data[i : i + in_window]
            
            # 正解データ: 入力の直後から5日分
            window_y = source_data[i + in_window : i + in_window + out_window]
            
            X.append(window_X)
            y.append(window_y)

        X = np.array(X)
        y = np.array(y)

        # 学習・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        datasets[target_col] = source_data.tolist()
        datasets[f"{target_col}:X_train"] = X_train 
        datasets[f"{target_col}:X_test"]  = X_test 
        datasets[f"{target_col}:y_train"] = y_train 
        datasets[f"{target_col}:y_test"]  = y_test

    return datasets, settings

datasets, settings = pre_process_No1()
# %%
