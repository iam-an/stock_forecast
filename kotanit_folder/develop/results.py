
#%%
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#%%
from learn import learn_classical_regression
datasets, settings = learn_classical_regression()
if settings["stl"]:
    target_cols = ["trend", "seasonal"]
else:
    target_cols = ["High"]

for target_col in target_cols:
    y_test, y_test_pred   = datasets[f"{target_col}:y_test"], datasets[f"{target_col}:y_test_pred"]
    y_train, y_train_pred = datasets[f"{target_col}:y_train"], datasets[f"{target_col}:y_train_pred"]
    # 評価
    print("R² score:", r2_score(y_test, y_test_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
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

    ax[0].set_title(f"{target_col}:test")
    ax[0].scatter(y_test, y_test_pred, s=1)
    ax[0].axline([min(y_train),min(y_train)],[max(y_train),max(y_train)])

    ax[1].set_title(f"{target_col}:train")
    ax[1].scatter(y_train, y_train_pred, s=1)
    ax[1].axline([min(y_train),min(y_train)],[max(y_train),max(y_train)])

#%%
for target_col in target_cols:
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    axes = np.ravel(ax)  # 1軸でも複数軸でも平坦化
    for a in axes:
        for side in ['top', 'right', 'left', 'bottom']:
            a.spines[side].set_color('black')
            a.spines[side].set_linewidth(2)
        a.tick_params(direction='in', length=6, width=2, color='black')

    ax[0].set_title(f"{target_col}:test")
    ax[0].plot(y_test, linewidth=10, alpha=0.5, color="black")
    ax[0].plot(y_test_pred, linewidth=1, color="r")

    ax[1].set_title(f"{target_col}:train")
    ax[1].plot(y_train, linewidth=10, alpha=0.5, color="black")
    ax[1].plot(y_train_pred, linewidth=1, color="r")





#%%
if settings["stl"]:

    #可視化
    mpl.rcParams["font.family"] = ['DejaVu Sans']
    mpl.rcParams["font.size"] = 11
    fig, ax = plt.subplots(figsize=(7,3))
    ax.spines['top'].set_linewidth(2)
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_linewidth(2)
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('black')
    ax.tick_params(direction='in', length=6, width=2, color='black', bottom=False) 

    ax.set_title("stl_results")
    ax.plot(y_test_pred_all, marker="o", color="r", label="pred")
    ax.plot(y_test_pred_all, marker="o", color="b", label="pred")


# %%
"""STL分解をtest,trainで分割した後にやるほうがいい"""
"""trainデータが経時的になっているものではないように見える"""