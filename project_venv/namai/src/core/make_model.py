from prophet import Prophet
def make_model_prophet(train, test):
    model = Prophet()
    model.fit(train)
    y_pred_test = model.predict(test)
    return y_pred_test