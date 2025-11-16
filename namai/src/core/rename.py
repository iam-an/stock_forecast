import polars as pl

def rename_for_prophet(df: pl.DataFrame, rename_map: dict)->pl.DataFrame:
    df = df.rename(rename_map)
    return df