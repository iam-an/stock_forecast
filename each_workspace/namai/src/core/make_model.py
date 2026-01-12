from prophet import Prophet
import pandas as pd

def make_model_prophet(train):


    model = Prophet()


    model.fit(train)
    return model
