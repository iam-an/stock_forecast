from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


# 窓の中にそのまま敷き詰める列です。
# 「値そのもの」と「少しだけ要約した情報」を混ぜると、
# 単純すぎず、でも読みやすさは保ちやすいです。
WINDOW_FEATURE_COLUMNS = [
    "High",
    "Close",
    "log_volume",
    "return_1d",
    "high_close_gap",
    "hl_range",
    "trend",
    "swing",
]

# 最後の 1 行から抜く要約列です。
SUMMARY_FEATURE_COLUMNS = [
    "high_ma_5",
    "high_ma_10",
    "high_ma_20",
    "high_std_5",
    "high_std_10",
    "high_std_20",
    "volume_ma_5",
    "volume_ma_10",
    "volume_ma_20",
    "momentum_5",
    "momentum_10",
]


def prepare_feature_frame(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """
    生の株価データから、学習しやすい特徴量テーブルを作ります。

    kotani っぽい「窓切り前提」の素朴さは残しつつ、
    少しだけ中身を足してオリジナル感を出しています。
    """
    work = frame.copy()

    # まずは扱いやすい基本指標から作ります。
    work["log_volume"] = np.log1p(work["Volume"])
    work["return_1d"] = work["Close"].pct_change().fillna(0.0)

    safe_close = work["Close"].replace(0.0, np.nan)
    work["high_close_gap"] = ((work["High"] - work["Close"]) / safe_close).fillna(0.0)
    work["hl_range"] = ((work["High"] - work["Low"]) / safe_close).fillna(0.0)

    # 短期・中期の雰囲気がわかるように、移動平均とばらつきも持たせます。
    for window in (5, 10, 20):
        work[f"high_ma_{window}"] = work["High"].rolling(window=window, min_periods=window).mean()
        work[f"high_std_{window}"] = (
            work["High"].rolling(window=window, min_periods=window).std().fillna(0.0)
        )
        work[f"volume_ma_{window}"] = work["Volume"].rolling(window=window, min_periods=window).mean()

    work["momentum_5"] = work["Close"].pct_change(periods=5).fillna(0.0)
    work["momentum_10"] = work["Close"].pct_change(periods=10).fillna(0.0)

    use_stl = bool(settings["USE_STL"])
    stl_period = int(settings["STL_PERIOD"])

    # データが短いと STL が不安定なので、そのときだけ素直にフォールバックします。
    if use_stl and len(work) > stl_period * 2:
        result = STL(work["High"], period=stl_period, robust=True).fit()
        work["trend"] = result.trend
        work["seasonal"] = result.seasonal
        work["resid"] = result.resid
        # seasonal と resid をまとめて「揺れ」として持たせます。
        # トレンドと役割を分けるための列です。
        work["swing"] = work["seasonal"] + work["resid"]
    else:
        # STL を使わない場合でも、trend + swing = High になるようにしておきます。
        # ここが崩れると、あとで合算したときに値が二重になってしまいます。
        work["trend"] = work["High"].rolling(window=10, min_periods=1).mean()
        work["seasonal"] = 0.0
        work["resid"] = work["High"] - work["trend"]
        work["swing"] = work["resid"]

    # ここで一度きれいに落としておくと、以降の窓切りがかなり追いやすくなります。
    work = work.dropna().reset_index(drop=True)
    return work


def build_feature_names(input_window: int) -> list[str]:
    """保存した joblib を見たときに意味がわかるよう、特徴量名も残します。"""
    feature_names: list[str] = []

    for column in WINDOW_FEATURE_COLUMNS:
        for lag in range(input_window, 0, -1):
            feature_names.append(f"{column}_lag_{lag}")

    feature_names.extend(SUMMARY_FEATURE_COLUMNS)
    feature_names.extend(
        [
            "window_high_mean",
            "window_high_std",
            "window_close_move",
            "window_volume_move",
        ]
    )
    return feature_names


def build_supervised_dataset(frame: pd.DataFrame, settings: dict) -> dict:
    """
    時系列テーブルを「学習用の窓データ」に変換します。

    kotani に寄せて窓切りを中心にしつつ、
    その窓から少しだけ要約特徴量も足しています。
    """
    input_window = int(settings["INPUT_WINDOW"])
    out_window = int(settings["OUT_WINDOW"])

    X: list[list[float]] = []
    y_total: list[np.ndarray] = []
    y_trend: list[np.ndarray] = []
    y_swing: list[np.ndarray] = []
    meta: list[dict] = []

    for start_idx in range(len(frame) - input_window - out_window + 1):
        end_idx = start_idx + input_window
        future_end_idx = end_idx + out_window

        context = frame.iloc[start_idx:end_idx]
        future = frame.iloc[end_idx:future_end_idx]

        feature_vector: list[float] = []

        # 窓の時系列そのものをフラットに並べます。
        for column in WINDOW_FEATURE_COLUMNS:
            feature_vector.extend(context[column].to_numpy(dtype=float).tolist())

        # 最後の 1 点に集約された特徴量も少しだけ足します。
        last_row = context.iloc[-1]
        for column in SUMMARY_FEATURE_COLUMNS:
            feature_vector.append(float(last_row[column]))

        # 手元で見返したときに意味が取りやすい、素朴な要約値も残しておきます。
        feature_vector.extend(
            [
                float(context["High"].mean()),
                float(context["High"].std(ddof=0)),
                float(context["Close"].iloc[-1] - context["Close"].iloc[0]),
                float(context["Volume"].iloc[-1] - context["Volume"].iloc[0]),
            ]
        )

        X.append(feature_vector)
        y_total.append(future["High"].to_numpy(dtype=float))
        y_trend.append(future["trend"].to_numpy(dtype=float))
        y_swing.append(future["swing"].to_numpy(dtype=float))

        meta.append(
            {
                "as_of_date": context["Date"].iloc[-1],
                "history_dates": context["Date"].tolist(),
                "history_high": context["High"].tolist(),
                "future_dates": future["Date"].tolist(),
                "last_high": float(context["High"].iloc[-1]),
            }
        )

    return {
        "X": np.asarray(X, dtype=float),
        "y_total": np.asarray(y_total, dtype=float),
        "y_trend": np.asarray(y_trend, dtype=float),
        "y_swing": np.asarray(y_swing, dtype=float),
        "meta": meta,
        "feature_names": build_feature_names(input_window),
        "feature_frame": frame,
    }
