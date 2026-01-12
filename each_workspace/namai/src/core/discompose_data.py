from statsmodels.tsa.seasonal import STL
import polars as pl
from each_workspace.namai.src.utils.pl_pd import convert_pd_pl

@convert_pd_pl
def use_stl(target: str, df: pl.DataFrame, period: int)->pl.DataFrame:
    y = df[target]
    stl = STL(y, period=period)
    result = stl.fit()
    df_stl = df.copy()
    df_stl["stl_trend"] = result.trend
    df_stl["stl_seasonal"] = result.seasonal
    df_stl["stl_resid"] = result.resid

    return df_stl