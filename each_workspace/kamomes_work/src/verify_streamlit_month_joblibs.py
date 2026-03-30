from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils.constant import APP_MODELS


def load_like_streamlit(model_path: Path):
    """
    back/select_models.py と同じ流儀で読みます。
    """
    try:
        return joblib.load(model_path), "joblib"
    except Exception:
        import cloudpickle

        with open(model_path, "rb") as file:
            return cloudpickle.load(file), "cloudpickle"


def main() -> None:
    test_cases = [
        ("Nvidia", "1month_LR.joblib"),
        ("NTT", "1month_LR.joblib"),
    ]

    dummy_history = np.linspace(100.0, 120.0, 90)

    for company, file_name in test_cases:
        model_path = APP_MODELS / company / file_name
        model, loader = load_like_streamlit(model_path)

        print(f"[INFO] {company}: loader={loader}, callable={callable(model)}, has_predict={hasattr(model, 'predict')}")

        prediction = model(dummy_history)
        print(f"[INFO] {company}: forecast_length={len(prediction)}, first_three={[round(value, 3) for value in prediction[:3]]}")


if __name__ == "__main__":
    main()
