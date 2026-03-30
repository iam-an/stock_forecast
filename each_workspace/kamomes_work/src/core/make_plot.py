from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_plot_time(
    company: str,
    sample_meta: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_folder: Path,
    run_date: str,
    band_size: np.ndarray | None = None,
) -> Path:
    """
    直近の文脈と、その先の予測を 1 枚で見られる図を作ります。

    product_folder には「銘柄_日付.png」で保存します。
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    history_dates = sample_meta["history_dates"]
    history_high = np.asarray(sample_meta["history_high"], dtype=float)
    future_dates = sample_meta["future_dates"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # まずは予測の土台になる履歴を落ち着いた色で描きます。
    ax.plot(history_dates, history_high, color="tab:blue", linewidth=2.2, label="History")

    # 未来側は、直前の実測値からつながるように描いておくと見やすいです。
    actual_line_x = [history_dates[-1], *future_dates]
    actual_line_y = [history_high[-1], *np.asarray(y_true, dtype=float).tolist()]
    pred_line_y = [history_high[-1], *np.asarray(y_pred, dtype=float).tolist()]

    ax.plot(actual_line_x, actual_line_y, color="tab:orange", linewidth=2.0, marker="o", label="Actual")
    ax.plot(
        actual_line_x,
        pred_line_y,
        color="tab:green",
        linewidth=2.0,
        linestyle="--",
        marker="o",
        label="Prediction",
    )

    # ざっくりした不確実性も見せたいので、平均誤差幅があれば帯で足します。
    if band_size is not None and len(band_size) == len(y_pred):
        band_size = np.asarray(band_size, dtype=float)
        lower = np.asarray(y_pred, dtype=float) - band_size
        upper = np.asarray(y_pred, dtype=float) + band_size
        ax.fill_between(future_dates, lower, upper, color="tab:green", alpha=0.15, label="Error band")

    ax.set_title(f"{company} Mid-Term High Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("High Price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()

    save_path = output_folder / f"{company}_{run_date}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)

    return save_path
