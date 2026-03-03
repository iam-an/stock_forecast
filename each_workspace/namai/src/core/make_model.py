from prophet import Prophet
import pandas as pd
import numpy as np

def make_model_prophet(train):
    model = Prophet(growth="linear", changepoint_prior_scale=0.3, n_changepoints=50)
    train["y"] = np.log(train["Close"])
    model.fit(train)
    return model
