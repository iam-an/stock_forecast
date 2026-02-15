
#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import warnings
import joblib
warnings.filterwarnings('ignore')
from evaluation import pred_score

#%%
#結果の読み出し
from learn import learn_classical_regression
datasets, settings, models = learn_classical_regression()
in_window  = int(settings["in_window"])
out_window = int(settings["out_window"])
if settings["stl"]:
    target_cols = ["trend", "seasonal"]
else:
    target_cols = ["High"]


#ランダムにcount個testデータセットを選んで、画像表示
count = 5 #結果を確認する数
# リスト内包表記を使って5個の乱数を生成
random_numbers = [random.randint(0, datasets[f"{target_cols[0]}:y_test"].shape[0]-1) for _ in range(5)]
for target_col in target_cols:
    X_test, X_train       = datasets[f"{target_col}:X_test"], datasets[f"{target_col}:X_train"]
    y_test, y_test_pred   = datasets[f"{target_col}:y_test"], datasets[f"{target_col}:y_test_pred"]
    y_train, y_train_pred = datasets[f"{target_col}:y_train"], datasets[f"{target_col}:y_train_pred"]

    for i in random_numbers:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = np.ravel(ax)  # 1軸でも複数軸でも平坦化
        for a in axes:
            for side in ['top', 'right', 'left', 'bottom']:
                a.spines[side].set_color('black')
                a.spines[side].set_linewidth(2)
            a.tick_params(direction='in', length=6, width=2, color='black')

        pred_score(y_test[i], y_test_pred[i], make_fig=False)
        ax.set_title(f"{target_col}:test")
        ax.plot(np.append(X_test[i], y_test[i]), linewidth=5, alpha=0.5, color="black", marker="o", label="raw_data")
        ax.plot(np.arange(in_window-1,in_window+out_window), np.append(X_test[i][-1], y_test_pred[i]), linewidth=2, color="r", marker="o", label="pred_data")
        ax.legend()


# %%

if settings["stl"]:
    y_test_pred_total = np.array(datasets["trend:y_test_pred"]) + np.array(datasets["seasonal:y_test_pred"])
    X_test_total      = np.array(datasets["trend:X_test"]) + np.array(datasets["seasonal:X_test"]) 
    y_test_total      = np.array(datasets["trend:y_test"]) + np.array(datasets["seasonal:y_test"])

    for i in random_numbers:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = np.ravel(ax)  # 1軸でも複数軸でも平坦化
        for a in axes:
            for side in ['top', 'right', 'left', 'bottom']:
                a.spines[side].set_color('black')
                a.spines[side].set_linewidth(2)
            a.tick_params(direction='in', length=6, width=2, color='black')
        
        j = i+datasets[f"{target_cols[0]}:y_train"].shape[0]
        pred_score(datasets["target"][j+in_window:j+in_window+out_window],  y_test_pred_total[i]+np.average(datasets["resid"][j:j+in_window]), make_fig=False)
        
        ax.set_title(f"total")
        ax.plot(datasets["target"][j:j+in_window+out_window], label="raw_data", color="black", alpha=0.5, marker="o", linewidth=5)
        ax.plot(np.arange(in_window-1,in_window+out_window), 
                np.append(datasets["target"][j+in_window-1], y_test_pred_total[i]+np.average(datasets["resid"][j:j+in_window])), 
                linewidth=2, color="r", marker="o", label="pred_data")
        ax.legend()
        def total_model(X):
            return models["trend"].predict(X) + models["seasonal"].predict(X)
        joblib.dump(total_model, f'models/{settings["company"]}_1week_LR.joblib') 

# %%
