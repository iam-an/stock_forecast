from statsmodels.tsa.seasonal import STL
import pandas as pd
import polars as pl

def use_stl(target: str, df_train: pd.DataFrame, period: int)->pd.DataFrame:
    y = df_train[target]
    stl = STL(y, period=period)
    result = stl.fit()
    df_train_stl = df_train.copy()
    df_train_stl["stl_trend"] = result.trend
    df_train_stl["stl_seasonal"] = result.seasonal
    df_train_stl["stl_resid"] = result.resid

    return df_train_stl