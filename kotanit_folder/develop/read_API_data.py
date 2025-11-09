#%%
import numpy as np
import matplotlib.pyplot as plt
import json
import polars as pl
import yfinance as yf



"""
🔹 yfinance の主なモジュール・関数
yfinance.download(tickers, ...)
複数銘柄の時系列株価データをまとめて取得。
引数: tickers, start, end, period, interval, group_by など。

yfinance.Ticker(symbol)
個別銘柄オブジェクトを生成。
ここから各種情報にアクセスできる。
・history() → 株価データ取得 #pandasのdf
・info → 基本情報（時価総額、業種など）
・financials / quarterly_financials → 財務データ
"""

def read_datasets():
    #settings
    with open("settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)
    company = settings["company"]
    hist_period = settings["hist_period"]
    hist_inter = settings["hist_inter"]

    #銘柄別株価データセット
    stock_dataset = yf.Tickers(company)
    """
    history
    株価データ
    'Open' → 始値（その日の取引開始時の株価）
    'High' → 高値（その日の取引中の最高値）
    'Low' → 安値（その日の取引中の最安値）
    'Close' → 終値（その日の取引終了時の株価）
    出来高とイベント
    'Volume' → 出来高（その日の取引株数）
    'Dividends' → 配当（その日に配当が出た場合は金額が入る。それ以外は 0）
    'Stock Splits' → 株式分割（株式分割があった日には分割比率が入る。それ以外は 0）
    """

    df = pl.DataFrame(stock_dataset.history(period=hist_period, interval=hist_inter))
    df = df.drop(f"('Volume', '{company}')", f"('Dividends', '{company}')", f"('Stock Splits', '{company}')")
    df = df.rename({f"('Open', '{company}')" : "Open"})
    df = df.rename({f"('High', '{company}')" : "High"})
    df = df.rename({f"('Low', '{company}')" : "Low"})
    df = df.rename({f"('Close', '{company}')" : "Close"})
    df = df.drop("Open", "Low", "Close")

    return(df, settings)
