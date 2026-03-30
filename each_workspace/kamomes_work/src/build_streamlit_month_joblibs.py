from __future__ import annotations

import json
import sys
from pathlib import Path

import cloudpickle
import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.streamlit_month_compat import build_streamlit_month_model
from src.utils.constant import APP_MODELS, STREAMLIT_EXPORT, STREAMLIT_TARGETS, ensure_workspace_dirs

PRED_RANGE = "1month"
MODEL_TYPE = "LR"


def load_targets() -> dict:
    """1month 用の対象銘柄一覧を読み込みます。"""
    with open(STREAMLIT_TARGETS, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_cloudpickle_model(model, output_path: Path) -> None:
    """
    既存 back/select_models.py は joblib 失敗時に cloudpickle にフォールバックするので、
    ここではその経路でも確実に読める形で保存します。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        cloudpickle.dump(model, file)


def main() -> None:
    ensure_workspace_dirs()
    settings = load_targets()

    manifest: dict[str, dict[str, str]] = {}

    for stock in settings["STOCKS"]:
        display_name = str(stock["DISPLAY_NAME"])
        ticker = str(stock["TICKER"])

        model = build_streamlit_month_model(
            display_name=display_name,
            ticker=ticker,
            forecast_days=int(settings["FORECAST_DAYS"]),
            history_period=str(settings["HISTORY_PERIOD"]),
            interval=str(settings["INTERVAL"]),
            prophet_settings=settings["PROPHET"],
        )

        stock_dir = APP_MODELS / display_name

        # 現行 back/select_models.py の命名規則にそのまま合わせます。
        # model_path = MODELS / company / f"{pred_range}_{model_type}.joblib"
        output_path = stock_dir / f"{PRED_RANGE}_{MODEL_TYPE}.joblib"

        save_cloudpickle_model(model, output_path)

        manifest[display_name] = {
            "ticker": ticker,
            "pred_range": PRED_RANGE,
            "model_type": MODEL_TYPE,
            "output_path": str(output_path),
        }

        print(f"[INFO] saved streamlit month model for {display_name} ({ticker})")

    manifest_path = STREAMLIT_EXPORT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, ensure_ascii=False)

    print(f"[INFO] manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
