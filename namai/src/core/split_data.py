import polars as pl
import pandas as pd
from typing import Tuple

def split_train_test(df: pl.DataFrame, test_size: float)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    dfをtrain,testに分割

    Parameters
    ----------
    df : pl.DataFrame
        前処理が終わったpl.df
    test_size : float
        テストの割合

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train, testに分割されたdf
    """
    n = df.height
    n_test = int(n * test_size)
    n_train = n - n_test

    train_pd = df.slice(0, n_train).to_pandas()
    test_pd = df.slice(n_train, n_test).to_pandas()

    return train_pd, test_pd

