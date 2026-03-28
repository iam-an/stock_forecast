import os
import joblib
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import polars as pl
import yaml
from datetime import timedelta
import jpholiday
import sys
from pathlib import Path
from namai.src.utils.constant import MODELS, COMPANIES

# """settings"""
# company = "NTT"
# pred_range = "1week"
# model_type = "LR"

def load_model(company, pred_range, model_type):
    # select model
    model_path = MODELS / company / f"{pred_range}_{model_type}.joblib"
    try:
        model = joblib.load(model_path)
    except:
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
    assert model is not None, "modelがありません"
    return model


def select_month_model_from_kamomes(company):
    """
    1month だけは kamomes_work へ

    """
    kamomes_root = Path(__file__).resolve().parents[1] / "each_workspace" / "kamomes_work"
    if str(kamomes_root) not in sys.path:
        sys.path.insert(0, str(kamomes_root))

    from src.core.streamlit_month_bridge import select_streamlit_month

    return select_streamlit_month(company=company, history_window=30, forecast_window=30)
def select_models(company, pred_range, model_type):

    # 1month だけは kamomes_work へ
    # 消しやすいように先に条件分岐
    if pred_range == "1month":
        return select_month_model_from_kamomes(company)

    # 期間決定
    if pred_range == "1day":
        hist_period = "1mo"
        hist_inter = "1d"
        window = 7
    elif pred_range == "1week":
        hist_period = "1y"
        hist_inter = "1d"
        window = 14
    elif pred_range == "1month":
        hist_period = "1y"
        hist_inter = "1d"
        window = 30
    elif pred_range == "1year":
        hist_period = "3y"
        hist_inter = "1d"
        window = 90
    # 略称の取得
    with open(COMPANIES / "ticker_master.yaml", "r", encoding="utf-8") as f:
        ticker_master = yaml.safe_load(f)
    ticker = ticker_master[company]
    # common process independ on period
    # data cleaning
    df = yf.download(ticker, period=hist_period, interval=hist_inter, threads=False)
    df = df.reset_index()
    df = pl.from_pandas(df)
    df = df.drop(f"('Volume', '{ticker}')")
    df = df.rename({"('Date', '')" : "Date"})
    df = df.rename({f"('Open', '{ticker}')" : "Open"})
    df = df.rename({f"('High', '{ticker}')" : "High"})
    df = df.rename({f"('Low', '{ticker}')" : "Low"})
    df = df.rename({f"('Close', '{ticker}')" : "Close"})
    df = df.drop("Open", "Low", "Close")
    """make display data"""
    indata = df["High"].to_numpy()
    act_data = indata[-window:]

    if pred_range == "1day" or pred_range == "1week":
        model = load_model(company, pred_range, model_type)

        min_date = df.select(pl.col("Date").min()).to_series()[0]
        df = df.with_columns((df["Date"]-min_date).alias("Times"))
        df = df.with_columns(df["Times"].dt.total_microseconds()/ (24*60*60*1_000_000))

        """use model to predict"""
        pred_data = np.append(indata[-1:] , model(indata))
        act_date = df["Date"][-window:].dt.strftime("%m/%d")
        base_date = df["Date"][-1]
        pred_date = [base_date]

        while len(pred_date) < len(pred_data):
            base_date = base_date + timedelta(days=1)
            if base_date.weekday() < 5 and not jpholiday.is_holiday(base_date):
                pred_date.append(base_date)
        pred_date = [base_date.strftime("%m/%d") for base_date in pred_date]
    
    elif pred_range == "1month" or pred_range == "1year":
        if pred_range == "1month":
            period = 30
        elif pred_range == "1year":
            period = 365
        from namai.src.core.make_model import make_model_prophet
        from namai.src.core.discompose_data import use_stl
        df = df.rename({"Date": "ds"})
        df = df.rename({"High": "y"})
        real_data = df.to_pandas()
        model = make_model_prophet(real_data)
        future = model.make_future_dataframe(periods=period)
        pred_data = model.predict(future)
        last_date = real_data["ds"].max()

        # 営業日 + 未来のみ抽出
        filtered_dates = []
        filtered_values = []

        for i in range(len(pred_data)):
            date = pred_data["ds"].iloc[i]
            if date > last_date:
                if date.weekday() < 5 and not jpholiday.is_holiday(date):
                    filtered_dates.append(date)  # 👈 文字列やめる
                    filtered_values.append(pred_data["yhat"].iloc[i])

        pred_date = filtered_dates
        pred_data = filtered_values

        act_date = df["ds"][-window:].to_pandas().tolist()
        act_data = df["y"][-window:].to_list()
        if len(act_date) > 0 and len(pred_date) > 0:
            pred_date = [act_date[-1]] + pred_date
            pred_data = [act_data[-1]] + pred_data

    return act_date, act_data, pred_date, pred_data
        
