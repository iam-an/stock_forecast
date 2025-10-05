
#%%
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
from learn import learn_No1
y_test, y_train, y_test_pred, y_train_pred = learn_No1()
#%%
# 評価

print("R² score:", r2_score(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

#%%
#可視化
#mpl.rcParams["font.family"] = "Arial"
#mpl.rcParams["font.size"] = 17.5
fig, ax = plt.subplots(1,2, figsize=(7,3))
ax.spines['top'].set_linewidth(2)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(2)
ax.spines['right'].set_color('black')
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_color('black')
ax.tick_params(direction='in', length=6, width=2, color='black', bottom=False) 

ax[0].set_title("test")
ax[0].scatter(y_test, y_test_pred, s=1)
ax[0].axline([min(y_train),min(y_train)],[max(y_train),max(y_train)])

ax[1].set_title("train")
ax[1].scatter(y_train, y_train_pred, s=1)
ax[1].axline([min(y_train),min(y_train)],[max(y_train),max(y_train)])
# %%
