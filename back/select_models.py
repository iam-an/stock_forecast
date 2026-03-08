import os
import joblib
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import polars as pl
import yaml
from datetime import datetime, timedelta
import jpholiday
import sys

"""settings"""
company = "NTT"
pred_range = "1week"
model_type = "LR"

def select_models(company, pred_range, model_type):
    """select model"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = f"{base_path}/models/{company}/{pred_range}_{model_type}.joblib"
    try:
        model = joblib.load(model_path)
    except:
        try:
            model = cloudpickle.load(open(model_path, 'rb'))
        except:
            sys.exit()

    """data import"""
    if pred_range == "1day":
        hist_period = "1mo"
        hist_inter = "1d"
    elif pred_range == "1week":
        hist_period = "1y"
        hist_inter = "1d"
    elif pred_range == "1month":
        hist_period = "1y"
        hist_inter = "1d"
    elif pred_range == "1year":
        hist_period = "3y"
        hist_inter = "1d"

    with open(f"{base_path}/back/ticker_master.yaml", "r", encoding="utf-8") as f:
        ticker_master = yaml.safe_load(f)
    ticker = ticker_master[company]

    if pred_range == "1day" or pred_range == "1week":
        df = yf.download(ticker, period=hist_period, interval=hist_inter)
        df = df.reset_index()
        df = pl.from_pandas(df)
        df = df.drop(f"('Volume', '{ticker}')")
        df = df.rename({"('Date', '')" : "Date"})
        df = df.rename({f"('Open', '{ticker}')" : "Open"})
        df = df.rename({f"('High', '{ticker}')" : "High"})
        df = df.rename({f"('Low', '{ticker}')" : "Low"})
        df = df.rename({f"('Close', '{ticker}')" : "Close"})
        df = df.drop("Open", "Low", "Close")

        min_date = df.select(pl.col("Date").min()).to_series()[0]
        df = df.with_columns((df["Date"]-min_date).alias("Times"))
        df = df.with_columns(df["Times"].dt.total_microseconds()/ (24*60*60*1_000_000))
        indata = df["High"].to_numpy()

        """use model to predict"""
        pred_data = np.append(indata[-1:] , model(indata))

        """make display data"""
        act_data = indata[-15:]
        act_date = df["Date"][-15:].dt.strftime("%m/%d")
        base_date = df["Date"][-1]
        pred_date = [base_date]
        d = base_date
        while len(pred_date) < len(pred_data):
            d = d + timedelta(days=1)
            if d.weekday() < 5 and not jpholiday.is_holiday(d):
                pred_date.append(d)
        pred_date = [d.strftime("%m/%d") for d in pred_date]
    
    elif pred_range == "1month" or pred_range == "1year":
        
        pass

    return act_date, act_data, pred_date, pred_data