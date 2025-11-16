from statsmodels.tsa.seasonal import STL
import pandas as pd
import polars as pl

def use_stl(target: str, df: pd.DataFrame, period: int)->pd.DataFrame:
    print(df)
    y = df[target]
    stl = STL(y, period=period)
    result = stl.fit()
    df_stl = df.copy()
    df_stl["stl_trend"] = result.trend
    df_stl["stl_seasonal"] = result.seasonal
    df_stl["stl_resid"] = result.resid

    return df_stl