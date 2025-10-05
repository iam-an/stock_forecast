import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def make_plot_time(ds_train, y_train, ds_test, y_test,\
                    y_pred_test, y_pred_high, y_pred_low, output, fig_name):
    # Prophet 予測の中心値 (yhat)
    y_pred = y_pred_test.values
    fig, ax = plt.subplots(figsize=(7,3))
    # numpy配列に変換してから扱う
    ds_train = ds_train.to_numpy()
    y_train = y_train.to_numpy()
    ds_test = ds_test.to_numpy()
    y_test = y_test.to_numpy()

    # Train
    ax.plot(ds_train, y_train, color="tab:blue", label="Train")

    # Test（境目をつなげる）
    ax.plot(
        np.concatenate([[ds_train[-1]], ds_test]),
        np.concatenate([[y_train[-1]], y_test]),
        color="tab:orange", label="Test (Actual)"
    )

    # 予測
    ax.plot(
        np.concatenate([[ds_train[-1]], ds_test]),
        np.concatenate([[y_train[-1]], y_pred]),
        linestyle="dotted", color="tab:green", label="Predict",
    )


    # 予測範囲の描画
    ds_fill = np.concatenate([[ds_train[-1]], ds_test])
    yhat_lower = np.concatenate([[y_pred[0]], y_pred_high.to_numpy()])
    yhat_upper = np.concatenate([[y_pred[0]], y_pred_low.to_numpy()])

    ax.fill_between(
        ds_fill,
        yhat_lower,
        yhat_upper,
        color="lightblue",
        alpha=0.3,
        label="Confidence Interval"
    )

    plt.legend()
    plt.title("Predict vs Actual")
    plt.savefig(Path(output / fig_name))