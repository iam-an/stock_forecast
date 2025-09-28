#%%
import yfinance as yf




# %%
# 株価6か月分の取得
import polars as pl
company = yf.Ticker("AAPL") 
df_company_pd = company.history(start='2025-01-01', end='2025-07-01')
# timezoneを削除(prophetに入らないから)
df_company_pd.index = df_company_pd.index.tz_localize(None)
df_company_pd.reset_index(inplace=True)
df_company_pl = pl.from_pandas(df_company_pd)
print(df_company_pl)

# %%
# trainとtestに分割
from sklearn.model_selection import train_test_split
# X = df_company_pl.select(pl.col("High")).to_numpy()
# Y = df_company_pl.drop("High").to_numpy()
# x_train, x_test, y_train, y_test= train_test_split(X,Y, test_size=0.2, shuffle=False)
# print(y_train)
# print(x_train)
df_for_train = df_company_pl.select(pl.col("Date", "High"))
df_for_train = df_company_pl.rename({"Date": "ds", "High": "y"})
train, test = train_test_split(df_for_train.to_pandas(), test_size=0.2, shuffle=False)
print(df_for_train)
# %%
# モデルを構築
from prophet import Prophet
model = Prophet()
model.fit(train)
y_pred_test = model.predict(test)
# %%
# 予測値の取得
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
print(y_pred_test)
print(test)
# test データの実測値
y_true = test["y"].values
# Prophet 予測の中心値 (yhat)
y_pred = y_pred_test["yhat"].values

# 精度指標を計算
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# %%
# 可視化
plt.figure(figsize=(10, 5))
plt.plot(train["ds"], train["y"], label="Train")
plt.plot(test["ds"], y_true, label="Test (Actual)", marker="o")
plt.plot(test["ds"], y_pred, label="Forecast (yhat)", linestyle="--")
plt.fill_between(
    test["ds"],
    y_pred_test["yhat_lower"],
    y_pred_test["yhat_upper"],
    color="lightblue",
    alpha=0.4,
    label="Confidence Interval"
)
plt.legend()
plt.title("Prophet Prediction vs Actual")
plt.show()