#%% -*- coding: utf-8 -*-
import pandas as pd
import yfinance
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

symbol = "2220.T"
ticker = yfinance.Ticker(symbol)
intervals = range(2,25)
score_results = {}
max_score = -999
for interval in intervals:
    try:
        # --- 1. データ取得 ---

        data = ticker.history(period=f"{interval}y")
        data.index = data.index.tz_localize(None)

        # --- 直近1年を評価用に分割 ---
        end_eval = data.index.max()
        start_eval = end_eval - pd.DateOffset(years=1)
        train_data = data[data.index < start_eval]  # 学習用
        eval_data = data[data.index >= start_eval]  # 評価用

        # --- Prophet 用に学習データ整形 ---
        df_train = pd.DataFrame({
            "ds": train_data.index,
            "y": train_data["Close"]
        })

        #df_train["ds"] = pd.to_datetime(df_train["ds"])
        df_train = df_train.dropna(subset=["ds","y"]).reset_index(drop=True)

        # --- モデル構築と学習 --- ここをいろいろ試す
        proph = Prophet()
        proph.fit(df_train)

        # --- 予測データフレーム作成 & 予測実行 ---
        #future = proph.make_future_dataframe(periods=len(eval_data), freq='D')  # 評価期間を日単位で
        future = pd.DataFrame({"ds": eval_data.index})
        #
        forecast = proph.predict(future)

        # --- 直近1年の予測結果のみ抽出 ---
        forecast_eval = forecast[(forecast["ds"] >= eval_data.index.min()) & 
                                (forecast["ds"] <= eval_data.index.max())]

        from sklearn.metrics import r2_score
        pred_data = forecast_eval["yhat"]
        raw_data  = eval_data["Close"]
        #R2スコアが最大(1に近い)になるintervalを探す
        score = r2_score(raw_data, pred_data)
        score_results[interval] = score
        if max_score <= score:
            max_score = score
            optimazed_interval = interval
    except:
        pass
#%%
interval = optimazed_interval
# --- 1. データ取得 ---
#symbol = "7011.T"
#ticker = yfinance.Ticker(symbol)
data = ticker.history(period=f"{interval}y")
data.index = data.index.tz_localize(None)

# --- 直近1年を評価用に分割 ---
end_eval = data.index.max()
start_eval = end_eval - pd.DateOffset(years=1)
train_data = data[data.index < start_eval]  # 学習用
eval_data = data[data.index >= start_eval]  # 評価用
#%%
# --- Prophet 用に学習データ整形 ---
df_train = pd.DataFrame({
    "ds": train_data.index,
    "y": train_data["Close"]
})

#df_train["ds"] = pd.to_datetime(df_train["ds"])
df_train = df_train.dropna(subset=["ds","y"]).reset_index(drop=True)

# --- モデル構築と学習 --- ここをいろいろ試す
proph = Prophet()
proph.fit(df_train)

# --- 予測データフレーム作成 & 予測実行 ---
#future = proph.make_future_dataframe(periods=len(eval_data), freq='D')  # 評価期間を日単位で
future = pd.DataFrame({"ds": eval_data.index})
#
forecast = proph.predict(future)

# 予測結果（yhat）

# --- 直近1年の予測結果のみ抽出 ---
forecast_eval = forecast[(forecast["ds"] >= eval_data.index.min())]

from sklearn.metrics import r2_score
pred_data = forecast_eval["yhat"]
raw_data  = eval_data["Close"]


#%%
# --- 予測結果のグラフ化（学習用gと評価用yで色分け） ---
plt.figure(figsize=(12,6))

# 学習用データ
plt.plot(df_train["ds"], df_train["y"], label="Training Data", color="blue")

# 評価用データ
plt.plot(eval_data.index, eval_data["Close"], label="Evaluation Data", color="orange")
# 予測結果（直近1年のみ）
plt.plot(forecast_eval["ds"], forecast_eval["yhat"], label="Predicted (1y)", color="green", linestyle="--")

plt.xlabel("Date")
plt.ylabel("Price (JPY)")
plt.legend()
plt.gcf().autofmt_xdate()
plt.title("MHI")
plt.show()

# %%
#%%
# --- 予測結果のグラフ化（学習用gと評価用yで色分け） ---
plt.figure(figsize=(12,6))

# 学習用データ
plt.plot(df_train["ds"], df_train["y"], label="Training Data", color="blue")

# 評価用データ
plt.plot(eval_data.index, eval_data["Close"], label="Evaluation Data", color="orange")
# 予測結果（直近1年のみ）
plt.plot(forecast_eval["ds"], forecast_eval["yhat"], label="Predicted (1y)", color="green", linestyle="--")

plt.xlim(2024-11-28 00:00:00, forecast_eval["ds"][-1])
plt.xlabel("Date")
plt.ylabel("Price (JPY)")
plt.legend()
plt.gcf().autofmt_xdate()
plt.title("MHI")
plt.show()