# pipeline_tokyo_stock.py
# 日次の東京観測データ（tokyo1-YYYY.csv群）と株価(Close)を結合して
# Prophet に外生変数 (regressor) を与えて予測する一連のパイプライン例。
#
# 前提:
# - 各 tokyo1-YYYY.csv は行が日 (1..31)、列が月 (1..12)。1列目が行ラベル(day)である想定。
# - ファイル名パターン: tokyo1-YYYY.csv が run.py と同じ階層に複数存在。
# - 出力: 学習済モデルの保存、評価指標 JSON、予測プロット。

import glob
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import yfinance as yf
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import matplotlib.pyplot as plt
from typing import Tuple

# ----------------------------
# 設定（必要に応じて編集）
# ----------------------------
CONFIG = {
    "COMPANY": "AAPL",
    "START_DAY": "2018-01-01",
    "END_DAY": "2023-12-31",
    "TOKYO_PATTERN": "tokyo1-*.csv",   # 水のファイルパターン
    "PRED_TARGET": "y",                # Prophet の目的変数列名
    "PRED_FEATURE": "ds",              # Prophet の日付列名
    "TOKYO_COL_NAME": "tokyo_val",     # 結合後の観測値カラム名
    "OUTPUT_DIR": "output",
    "FIG_NAME": "predict.png",
    "TEST_RATIO": 0.2,
    "STL_PERIOD": 365                   # 例: 年周期（日次データなら365）
}

Path(CONFIG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1) tokyo CSV 群を読み込み → long 形式で日付列を作る
# ----------------------------
def load_tokyo_csvs_to_daily(pattern: str) -> pd.DataFrame:
    """
    tokyo1-YYYY.csv の集合を読み込み、year/month/day -> date にして
    日次の long テーブルを返す (pandas.DataFrame)。
    """
    files = sorted(glob.glob(pattern))
    all_rows = []
    for fp in files:
        # ファイル名から年を取り出す (例: tokyo1-2023.csv)
        year = None
        try:
            # 数字4桁をファイル名から抽出
            import re
            m = re.search(r"(\d{4})", Path(fp).name)
            if m:
                year = int(m.group(1))
            else:
                continue
        except Exception:
            continue

        # CSV を読み込む。1列目が day ラベル（ヘッダなしの場合も想定）
        # ヘッダは1行目（列名が月名）を仮定。万が一ヘッダがない場合は適宜修正して。
        df_raw = pd.read_csv(fp, index_col=0)  # index が day
        # index を日（整数）に変換
        df_raw.index = df_raw.index.astype(int)

        # melt: day x month -> rows of (day, month, value)
        df_melt = df_raw.reset_index().melt(id_vars=df_raw.index.name or "index",
                                            var_name="month", value_name="value")
        # 列名が「index」の場合処理
        if df_melt.columns[0] == "index":
            df_melt = df_melt.rename(columns={"index": "day"})

        # month が文字列（例: '1' や 'Jan'）の場合、数値化を試みる
        # ここでは数字文字列を期待
        df_melt["month"] = df_melt["month"].astype(int)
        df_melt["day"] = df_melt["day"].astype(int)

        # 年月日から日付を作成（無効日は NaT になり drop）
        df_melt["year"] = year
        df_melt["date"] = pd.to_datetime(dict(year=df_melt["year"],
                                              month=df_melt["month"],
                                              day=df_melt["day"]),
                                          errors="coerce")
        df_melt = df_melt[["date", "value"]].dropna(subset=["date"]).reset_index(drop=True)
        all_rows.append(df_melt)

    if not all_rows:
        raise RuntimeError("tokyo CSV が見つからないか読み込みに失敗しました。")

    df_all = pd.concat(all_rows, ignore_index=True)
    # 同じ日付が複数ある場合は平均（観測源が複数だった場合に備えて）
    df_all = df_all.groupby("date", as_index=False).agg({"value": "mean"})
    # 日付インデックスで reindex（日付ギャップは NaN）
    df_all = df_all.set_index("date").asfreq("D").reset_index()
    # 欠測は前方/後方埋めで埋める（必要に応じて変更）
    df_all["value"] = df_all["value"].ffill().bfill()
    df_all = df_all.rename(columns={"value": CONFIG["TOKYO_COL_NAME"]})
    return df_all

# ----------------------------
# 2) 株価を取得
# ----------------------------
def fetch_stock_close(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    yfinance から日次の終値(Close)を取得して pandas.DataFrame を返す。
    ds, y の列にして返す（Prophet 用の列名）。
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end)
    # tz を除去
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    # ds を datetime に
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# ----------------------------
# 3) データを結合（date 列で merge）
# ----------------------------
def merge_stock_tokyo(stock_df: pd.DataFrame, tokyo_df: pd.DataFrame, tokyo_col_name: str) -> pd.DataFrame:
    # tokyo: column 'date' -> rename to 'ds' for merge
    tokyo_df = tokyo_df.rename(columns={"date": "ds"})
    merged = pd.merge(stock_df, tokyo_df[["ds", tokyo_col_name]], on="ds", how="left")
    # 欠損埋め（必要なら手法を変える）
    merged[tokyo_col_name] = merged[tokyo_col_name].ffill().bfill()
    return merged

# ----------------------------
# 4) STL 分解（オプション）：ここでは株価 y に STL を適用して特徴量追加
# ----------------------------
def add_stl_features(df: pd.DataFrame, target_col: str = "y", period: int = 365) -> pd.DataFrame:
    """
    df に trend/seasonal/resid 列を追加して返す (pandas.DataFrame)。
    """
    series = df[target_col].fillna(method="ffill").values
    stl = STL(series, period=period)
    res = stl.fit()
    df = df.copy()
    df["stl_trend"] = res.trend
    df["stl_seasonal"] = res.seasonal
    df["stl_resid"] = res.resid
    return df

# ----------------------------
# 5) 学習データ分割（時系列末尾を test とする）
# ----------------------------
def split_train_test_timewise(df: pd.DataFrame, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = max(1, int(n * test_ratio))  # 少なくとも1点は test に
    n_train = n - n_test
    train = df.iloc[:n_train].reset_index(drop=True)
    test  = df.iloc[n_train:].reset_index(drop=True)
    return train, test

# ----------------------------
# 6) Prophet モデル作成（tokyo_val を外生変数として追加）
# ----------------------------
def build_and_fit_prophet_with_regressor(train_df: pd.DataFrame, reg_col: str) -> Prophet:
    """
    Prophet モデルを作成し、reg_col を add_regressor して学習する。
    train_df は ds, y, reg_col を含む pandas.DataFrame。
    """
    model = Prophet()
    model.add_regressor(reg_col, standardize=True)  # regressor を追加
    model.fit(train_df)
    return model

# ----------------------------
# 7) 評価指標
# ----------------------------
def compute_metrics(true: pd.Series, pred: pd.Series) -> dict:
    mae = mean_absolute_error(true.values, pred.values)
    rmse = np.sqrt(mean_squared_error(true.values, pred.values))
    return {"MAE": float(mae), "RMSE": float(rmse)}

# ----------------------------
# 8) 描画（実測 vs 予測 + 予測区間）
# ----------------------------
def plot_result(train_ds, train_y, test_ds, test_y,
                pred_df, reg_col, out_dir, fig_name):
    fig, ax = plt.subplots(figsize=(10, 4))

    # 学習実測
    ax.plot(train_ds, train_y, label="train_actual")
    # テスト実測（連続線）
    ax.plot(np.concatenate([[train_ds.iloc[-1]], test_ds]),
            np.concatenate([[train_y.iloc[-1]], test_y]),
            label="test_actual", color="orange")

    # 予測中心値
    yhat = pred_df["yhat"].values
    ax.plot(np.concatenate([[train_ds.iloc[-1]], test_ds]),
            np.concatenate([[train_y.iloc[-1]], yhat]),
            linestyle="dotted", label="prediction", color="green")

    # 予測区間
    lower = pred_df["yhat_lower"].values
    upper = pred_df["yhat_upper"].values
    x_fill = np.concatenate([[train_ds.iloc[-1]], test_ds])
    lower_all = np.concatenate([[yhat[0]], lower])
    upper_all = np.concatenate([[yhat[0]], upper])
    ax.fill_between(x_fill, lower_all, upper_all, alpha=0.2)

    ax.legend()
    ax.set_title("Actual vs Forecast (with tokyo regressor)")
    plt.savefig(Path(out_dir) / fig_name)
    plt.close(fig)

# ----------------------------
# 9) 保存等ユーティリティ
# ----------------------------
def save_model_and_results(model, metrics: dict, out_dir: str, result_path: str):
    joblib.dump(model, Path(out_dir) / "model.pkl")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, indent=2)

# ----------------------------
# 10) メイン
# ----------------------------
def main():
    cfg = CONFIG
    #%% 1) tokyo データ読み込み
    tokyo_df = load_tokyo_csvs_to_daily(cfg["TOKYO_PATTERN"])

    #%% 2) 株価取得
    stock_df = fetch_stock_close(cfg["COMPANY"], cfg["START_DAY"], cfg["END_DAY"])

    # 3) 結合
    merged = merge_stock_tokyo(stock_df, tokyo_df, cfg["TOKYO_COL_NAME"])

    # 4) STL を y に適用して特徴量追加（任意）
    merged = add_stl_features(merged, target_col="y", period=cfg["STL_PERIOD"])

    # 5) train/test 分割
    train_df, test_df = split_train_test_timewise(merged, cfg["TEST_RATIO"])

    # 6) Prophet 学習（外生変数 regressor を追加）
    reg = cfg["TOKYO_COL_NAME"]
    model = build_and_fit_prophet_with_regressor(train_df[["ds", "y", reg]], reg)

    # 7) 予測 — regressor の値を future に入れる（test の regressor をそのまま使用）
    future = test_df[["ds", reg]].copy().reset_index(drop=True)
    pred = model.predict(future)  # yhat, yhat_lower, yhat_upper を含む

    # 8) 評価
    metrics = compute_metrics(test_df["y"], pred["yhat"])

    # 9) 保存
    save_model_and_results(model, metrics, cfg["OUTPUT_DIR"], Path(cfg["OUTPUT_DIR"]) / "result.json")

    # 10) プロット
    plot_result(train_df["ds"], train_df["y"], test_df["ds"], test_df["y"],
                pred, reg, cfg["OUTPUT_DIR"], cfg["FIG_NAME"])

    print("完了。metrics =", metrics)

if __name__ == "__main__":
    main()
 