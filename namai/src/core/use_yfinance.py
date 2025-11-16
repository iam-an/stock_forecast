import yfinance as yf
import polars as pl

def get_yfinace(company_name: str, start, end)->pl.DataFrame:
    company = yf.Ticker(company_name) 
    df_company_pd = company.history(start=start, end=end)
    # timezoneを削除(prophetに入らないから)
    df_company_pd.index = df_company_pd.index.tz_localize(None)
    df_company_pd.reset_index(inplace=True)
    df_company_pl = pl.from_pandas(df_company_pd)
    return df_company_pl
