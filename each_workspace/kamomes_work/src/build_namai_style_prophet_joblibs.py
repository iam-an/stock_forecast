from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.namai_style_prophet import (
    load_prophet_training_frame,
    make_model_prophet,
    save_namai_style_artifact,
)
from src.utils.constant import OUTPUT, STREAMLIT_TARGETS, ensure_workspace_dirs


def load_targets() -> dict:
    """
    対象銘柄の定義は既存の Streamlit 用設定をそのまま流用します。

    表示名がそのまま保存ファイル名になるので、
    フロントと見比べたときに迷いにくいのが小さな利点です。
    """
    with open(STREAMLIT_TARGETS, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    ensure_workspace_dirs()
    settings = load_targets()

    manifest: dict[str, dict[str, str]] = {}

    for stock in settings["STOCKS"]:
        display_name = str(stock["DISPLAY_NAME"])
        ticker = str(stock["TICKER"])

        print(f"[INFO] building namai-style prophet model for {display_name} ({ticker})")

        # Streamlit の 1year 分岐と同じで、
        # 保存前に一度最新データから Prophet を素直に学習します。
        train = load_prophet_training_frame(
            display_name=display_name,
            ticker=ticker,
            history_period=str(settings["HISTORY_PERIOD"]),
            interval=str(settings["INTERVAL"]),
        )
        model = make_model_prophet(train)

        # namai の save_artifacts と同じノリで {company}.joblib へ保存します。
        output_path = save_namai_style_artifact(model=model, artifact_path=OUTPUT, company=display_name)

        manifest[display_name] = {
            "ticker": ticker,
            "history_period": str(settings["HISTORY_PERIOD"]),
            "interval": str(settings["INTERVAL"]),
            "train_rows": str(len(train)),
            "output_path": str(output_path),
        }

    manifest_path = OUTPUT / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4, ensure_ascii=False)

    print(f"[INFO] manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
