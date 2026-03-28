from __future__ import annotations

from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_medium_horizon_models(split_data: dict, settings: dict, company: str) -> dict:
    """
    中期予測用の 2 段モデルを作ります。

    役割分担はかなり単純です。
    - trend_model: 緩やかな流れを追う
    - swing_model: 細かい揺れを補正する
    """
    trend_model = Pipeline(
        steps=[
            # Ridge はスケールの影響を受けやすいので、ここは先に標準化しておきます。
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(settings["TREND_MODEL"]["ALPHA"]))),
        ]
    )

    swing_model = RandomForestRegressor(
        n_estimators=int(settings["SWING_MODEL"]["N_ESTIMATORS"]),
        max_depth=int(settings["SWING_MODEL"]["MAX_DEPTH"]),
        min_samples_leaf=int(settings["SWING_MODEL"]["MIN_SAMPLES_LEAF"]),
        random_state=int(settings["RANDOM_STATE"]),
        n_jobs=-1,
    )

    trend_model.fit(split_data["X_train"], split_data["y_trend_train"])
    swing_model.fit(split_data["X_train"], split_data["y_swing_train"])

    y_trend_pred_train = trend_model.predict(split_data["X_train"])
    y_swing_pred_train = swing_model.predict(split_data["X_train"])
    y_total_pred_train = y_trend_pred_train + y_swing_pred_train

    if split_data["has_test"]:
        y_trend_pred_test = trend_model.predict(split_data["X_test"])
        y_swing_pred_test = swing_model.predict(split_data["X_test"])
        y_total_pred_test = y_trend_pred_test + y_swing_pred_test
    else:
        out_window = split_data["y_total_train"].shape[1]
        y_trend_pred_test = np.empty((0, out_window), dtype=float)
        y_swing_pred_test = np.empty((0, out_window), dtype=float)
        y_total_pred_test = np.empty((0, out_window), dtype=float)

    # namai っぽい「1 ファイルで全体がわかる保存物」にするため、
    # bundle にはモデル本体と設定の両方を入れておきます。
    bundle = {
        "model_name": "kamomes_midterm_hybrid",
        "company": company,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_window": int(settings["INPUT_WINDOW"]),
        "out_window": int(settings["OUT_WINDOW"]),
        "feature_names": split_data["feature_names"],
        "trend_model": trend_model,
        "swing_model": swing_model,
        "settings": settings,
    }

    return {
        "bundle": bundle,
        "trend_model": trend_model,
        "swing_model": swing_model,
        "y_pred_train": y_total_pred_train,
        "y_pred_test": y_total_pred_test,
        "y_trend_pred_test": y_trend_pred_test,
        "y_swing_pred_test": y_swing_pred_test,
        "model_summary": {
            "trend_model": "StandardScaler + Ridge",
            "swing_model": "RandomForestRegressor",
        },
    }
