from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def pred_score(raw_data, pred_data, make_fig):

    # 評価

    """
    R2値 意味
    1.0  完璧なモデル。全ての変動を説明している。
    0.8  データの変動の80%をモデルが説明している。非常に良い。
    0.0  モデルがデータの平均値を予測するのと同じ程度の性能。全く予測できていない。
    ~0.0 モデルが平均値を予測するよりも性能が悪い（モデルとして役に立たない）。
    """
    print("R² score:", r2_score(raw_data, pred_data))

    """「モデルの予測値は、平均して実データから RMSE 円だけズレている」"""
    print("RMSE:", np.sqrt(mean_squared_error(raw_data, pred_data)))
    if make_fig:
        #可視化
        mpl.rcParams["font.family"] = 'DejaVu Sans'
        mpl.rcParams["font.size"] = 11
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        axes = np.ravel(ax)  # 1軸でも複数軸でも平坦化
        for a in axes:
            for side in ['top', 'right', 'left', 'bottom']:
                a.spines[side].set_color('black')
                a.spines[side].set_linewidth(2)
            a.tick_params(direction='in', length=6, width=2, color='black')

        ax[0].set_title(f"y-y_plot")
        ax[0].scatter(raw_data, pred_data, s=1)
        ax[0].axline([min(raw_data),min(raw_data)],[max(raw_data),max(raw_data)])