from __future__ import annotations

import pandas as pd
import yfinance as yf


def get_yfinance(company_name: str, start: str, end: str | None) -> pd.DataFrame | None:
    """
    yfinance から株価データを取り出します。

    ここでは後続で使う列だけをきれいに整えて返します。
    """
    ticker = yf.Ticker(company_name)

    # END_DAY が未指定なら、その日までのデータを素直に取りにいきます。
    if end:
        raw_frame = ticker.history(start=start, end=end, auto_adjust=False)
    else:
        raw_frame = ticker.history(start=start, auto_adjust=False)

    if raw_frame.empty:
        return None

    # yfinance の index は timezone を持つことがあるので、まずは素直な日付にそろえます。
    if getattr(raw_frame.index, "tz", None) is not None:
        raw_frame.index = raw_frame.index.tz_localize(None)

    frame = raw_frame.reset_index()

    # 日付列の名前はケースによって揺れるので、ここで吸収しておきます。
    if "Datetime" in frame.columns and "Date" not in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})

    use_columns = [column for column in ["Date", "Open", "High", "Low", "Close", "Volume"] if column in frame.columns]
    frame = frame[use_columns].copy()

    # 欠損が混じったまま特徴量を作ると後で原因が追いづらいので、入口で軽く掃除します。
    frame = frame.dropna(subset=["Date", "High", "Close"]).reset_index(drop=True)
    if "Volume" in frame.columns:
        frame["Volume"] = frame["Volume"].fillna(0.0)
    else:
        frame["Volume"] = 0.0

    frame = frame.sort_values("Date").reset_index(drop=True)
    return frame
