from __future__ import annotations

import sys
from pathlib import Path

import joblib


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils.constant import OUTPUT


def main() -> None:
    test_cases = ["Nvidia", "NTT"]

    for company in test_cases:
        model_path = OUTPUT / f"{company}.joblib"
        model = joblib.load(model_path)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # 未来側だけを 3 件見ると、読み込み確認としてちょうど見やすいです。
        tail = forecast[["ds", "yhat"]].tail(3).copy()
        tail["ds"] = tail["ds"].dt.strftime("%Y-%m-%d")

        print(f"[INFO] {company}: model_path={model_path}")
        print(f"[INFO] {company}: future_rows={len(future)}, forecast_rows={len(forecast)}")
        print(f"[INFO] {company}: forecast_tail={tail.to_dict(orient='records')}")


if __name__ == "__main__":
    main()
