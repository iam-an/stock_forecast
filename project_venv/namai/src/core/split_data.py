import polars as pl
from sklearn.model_selection import train_test_split

def split_train_test(df):
    df_for_train = df.select(pl.col("Date", "High"))
    df_for_train = df.rename({"Date": "ds", "High": "y"})
    train, test = train_test_split(df_for_train.to_pandas(), test_size=0.2, shuffle=False)
    return train, test