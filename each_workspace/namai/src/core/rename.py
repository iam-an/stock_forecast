import polars as pl

def rename_for_prophet(df: pl.DataFrame, rename_map: dict)->pl.DataFrame:
    """_summary_
    sanitize df to prohet can use

    Parameters
    ----------
    df : pl.DataFrame
        株価情報の入ったpolars dataframe
    rename_map : dict
        時間と目的変数のrename情報

    Returns
    -------
    pl.DataFrame
        renamed df
    """
    df = df.rename(rename_map)
    assert "ds" in df.columns and "y" in df.columns, "rename_mapに'ds'と'y'の情報を入れてください"
    return df

