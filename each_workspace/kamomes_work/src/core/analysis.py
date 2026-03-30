from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.constant import MODELS


def evaluate_score(y_true: np.ndarray, y_pred: np.ndarray, last_values: np.ndarray | None = None) -> dict:
    """
    予測精度をまとめて返します。

    flatten した全体評価に加えて、
    horizon ごとの MAE も残しておくと後で見返しやすいです。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)

    # 株価では 0 除算が怖いので、ごく小さい値で下支えしておきます。
    safe_true = np.maximum(np.abs(y_true_flat), 1e-8)
    mape = float(np.mean(np.abs((y_true_flat - y_pred_flat) / safe_true)) * 100.0)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "HorizonMAE": np.mean(np.abs(y_true - y_pred), axis=0).round(4).tolist(),
    }

    if last_values is not None and len(last_values) > 0:
        last_values = np.asarray(last_values, dtype=float)
        true_direction = np.sign(y_true[:, 0] - last_values)
        pred_direction = np.sign(y_pred[:, 0] - last_values)
        metrics["DirectionAccuracy"] = float(np.mean(true_direction == pred_direction))

    return metrics


def save_artifacts(bundle: dict, company: str) -> dict:
    """
    joblib を namai / kotani の両方っぽく残します。

    - {company}.joblib
      namai のように「ひと目で主役だとわかる」保存物
    - {company}_MID_trend.joblib / {company}_MID_swing.joblib
      kotani のように役割別で追える保存物
    """
    MODELS.mkdir(parents=True, exist_ok=True)

    bundle_path = MODELS / f"{company}.joblib"
    trend_path = MODELS / f"{company}_MID_trend.joblib"
    swing_path = MODELS / f"{company}_MID_swing.joblib"

    joblib.dump(bundle, bundle_path)
    joblib.dump(bundle["trend_model"], trend_path)
    joblib.dump(bundle["swing_model"], swing_path)

    return {
        "bundle": str(bundle_path),
        "trend_model": str(trend_path),
        "swing_model": str(swing_path),
    }


def _json_default(value: Any) -> Any:
    """numpy / pandas 系の値を json に流し込みやすい形へ変換します。"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"JSON に変換できない型です: {type(value)!r}")


def write_to_json(file_path: Path, payload: dict) -> None:
    """結果を読みやすい json で保存します。"""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False, default=_json_default)
