from core.use_yfinance import get_yfinace
from core.split_data import split_train_test
from core.make_model import make_model_prophet
from core.analysis import evaluate_score
from core.make_plot import make_plot_time
from core.discompose_data import use_stl
from utils.constant import OUTPUT, CONFIG
import json

with open(CONFIG, "r", encoding="utf-8") as f:
    settings = json.load(f)
company = settings["COMPANY"]
start = settings["START_DAY"]
end = settings["END_DAY"]

def main():
    df = get_yfinace(company_name=company, start=start, end=end)
    train, test = split_train_test(df=df, test_size=0.2)
    df_train_stl = use_stl(target = "High", df_train=train, period=24)
    y_pred_test, test, train = make_model_prophet(train=df_train_stl, test=test, target="High", date_col="Date")
    evaluate_score(test=test["y"], y_pred_test=y_pred_test["yhat"])
    make_plot_time(
        ds_train=train["ds"],
        y_train=train["y"],
        ds_test=test["ds"],
        y_test=test["y"],
        y_pred_test=y_pred_test["yhat"],
        y_pred_high=y_pred_test["yhat_upper"],
        y_pred_low=y_pred_test["yhat_lower"],
        output=OUTPUT,
        fig_name="time.png"
        )

if __name__ == "__main__":
    main()

