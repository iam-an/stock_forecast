from __future__ import annotations

from pathlib import Path

import cloudpickle
import joblib

from src.utils.constant import APP_MODELS


MONTH_RANGE = "1month"
MONTH_MODEL = "LR"


def _load_model(model_path: Path):
    """
    root 側の load_model と同じ流儀で読みます。

    既存コードとの距離を縮めておくと、
    後で見返した人も挙動を想像しやすくなります。
    """
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as file:
            return cloudpickle.load(file)


def load_streamlit_month_model(company: str):
    """
    1month は UI の model_type 表示に関係なく、
    kamomes 側で管理している `1month_LR.joblib` を正本として読みます。
    """
    model_path = APP_MODELS / company / f"{MONTH_RANGE}_{MONTH_MODEL}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"1month 用モデルが見つかりません: {model_path}")
    return _load_model(model_path)


def select_streamlit_month(company: str, history_window: int = 30, forecast_window: int = 30):
    """
    root の Streamlit から 1month だけ安全に切り離すための橋渡し関数です。

    `model_type` の文言差や root 側の ticker_master には依存せず、
    kamomes 側の定義と保存物だけで表示用データを返します。
    """
    model = load_streamlit_month_model(company)

    if hasattr(model, "build_streamlit_payload"):
        return model.build_streamlit_payload(
            history_window=history_window,
            forecast_window=forecast_window,
        )

    # 念のための後方互換です。
    if hasattr(model, "get_recent_actual_frame") and hasattr(model, "get_filtered_forecast"):
        actual = model.get_recent_actual_frame(window=history_window)
        forecast = model.get_filtered_forecast(periods=forecast_window)

        act_date = actual["ds"].tolist()
        act_data = actual["y"].astype(float).tolist()
        pred_date = forecast["ds"].tolist()
        pred_data = forecast["yhat"].astype(float).tolist()

        if act_date and pred_date:
            pred_date = [act_date[-1]] + pred_date
            pred_data = [act_data[-1]] + pred_data

        return act_date, act_data, pred_date, pred_data

    raise TypeError("読み込んだ 1month モデルが Streamlit 表示用インターフェースに対応していません。")
