import yfinance as yf
import polars as pl

def get_yfinace(company_name: str, start: str, end: str)->None | pl.DataFrame:
    """_summary_
    - yfinanceから株価情報を取得し、機械学習可能な形に整形する

    Parameters
    ----------
    company_name : str
        取得したい会社の名前
    start : str
        start day that you want get information
    end : str
        end day that you want get information

    Returns
    -------
    pl.DataFrame
        cleaned dataframe
    """
    company = yf.Ticker(company_name) 
    df_company_pd = company.history(start=start, end=end)
    if df_company_pd.empty:
        return None
    # timezoneを削除(時刻なし、日付ありで予測には問題ない)
    df_company_pd.index = df_company_pd.index.tz_localize(None)
    df_company_pd.reset_index(inplace=True)
    df_company_pl = pl.from_pandas(df_company_pd)
    return df_company_pl
