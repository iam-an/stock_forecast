from __future__ import annotations

import numpy as np


def split_train_test(datasets: dict, test_size: float, full_data: bool = False) -> dict:
    """
    時系列の順番を保ったまま train / test に分けます。

    ここは namai の考え方を踏襲して、未来側を test に回しています。
    """
    X = datasets["X"]
    y_total = datasets["y_total"]
    y_trend = datasets["y_trend"]
    y_swing = datasets["y_swing"]
    meta = datasets["meta"]

    if len(X) == 0:
        raise ValueError("学習用サンプルが 0 件です。設定か取得期間を見直してください。")

    if full_data:
        # full_data 時は評価をせず、全データでモデルだけ作ります。
        empty_X = np.empty((0, X.shape[1]), dtype=float)
        empty_y = np.empty((0, y_total.shape[1]), dtype=float)
        return {
            "X_train": X,
            "X_test": empty_X,
            "y_total_train": y_total,
            "y_total_test": empty_y,
            "y_trend_train": y_trend,
            "y_trend_test": empty_y,
            "y_swing_train": y_swing,
            "y_swing_test": empty_y,
            "meta_train": meta,
            "meta_test": [],
            "last_high_train": np.asarray([item["last_high"] for item in meta], dtype=float),
            "last_high_test": np.asarray([], dtype=float),
            "has_test": False,
            "feature_names": datasets["feature_names"],
        }

    n_total = len(X)
    n_test = max(1, int(round(n_total * test_size)))
    n_train = n_total - n_test

    if n_train < 1:
        raise ValueError("train データが作れませんでした。取得期間を少し長くしてください。")

    return {
        "X_train": X[:n_train],
        "X_test": X[n_train:],
        "y_total_train": y_total[:n_train],
        "y_total_test": y_total[n_train:],
        "y_trend_train": y_trend[:n_train],
        "y_trend_test": y_trend[n_train:],
        "y_swing_train": y_swing[:n_train],
        "y_swing_test": y_swing[n_train:],
        "meta_train": meta[:n_train],
        "meta_test": meta[n_train:],
        "last_high_train": np.asarray([item["last_high"] for item in meta[:n_train]], dtype=float),
        "last_high_test": np.asarray([item["last_high"] for item in meta[n_train:]], dtype=float),
        "has_test": True,
        "feature_names": datasets["feature_names"],
    }
