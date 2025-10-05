from prophet import Prophet
import pandas as pd

def make_model_prophet(train, test, target: str, date_col: str = "Date"):
    train = train.rename(columns={date_col: "ds", target: "y"})
    test = test.rename(columns={date_col: "ds", target: "y"})


    model = Prophet()


    model.fit(train)
    y_pred_test = model.predict(test.drop("y", axis=1))

    return y_pred_test, test, train
