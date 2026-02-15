import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from product.models.use_model import load_model

company = "6701.T"
model_type = "LR"
option = "seasonal"

model_path = f"./models/{company}_{model_type}_{option}.joblib"
model = joblib.load(model_path)
X_test = np.load(f"./models/{company}_{model_type}_{option}_X_test.npy")
xdata = X_test[0].reshape(1, -1)
ydata = model.predict(xdata)
ydata = ydata.flatten()
print(ydata)
plt.figure(figsize=(10, 5))
plt.plot(ydata, label='Predicted', color='red')
plt.title(f'{company} {model_type} {option} Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()