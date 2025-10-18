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
stl_target = settings["STL_TARGET"]
pred_target = settings["PRED_TARGET"]
pred_feature = settings["PRED_FEATURE"]
average = settings["AVERAGE"]
average_upper = settings["AVERAGE_UPPER"]
average_lower = settings["AVERAGE_LOWER"]
period = settings["PERIOD"]
output_name_png = settings["OUTPUT_NAME_PNG"]

def main():
    df = get_yfinace(company_name=company, start=start, end=end)
    train, test = split_train_test(df=df, test_size=0.2)
    df_train_stl = use_stl(target = stl_target, df_train=train, period=period)
    y_pred_test, test, train = make_model_prophet(train=df_train_stl, test=test, target="High", date_col="Date")
    evaluate_score(test=test[pred_target], y_pred_test=y_pred_test[average])
    make_plot_time(
        ds_train=train[pred_feature],
        y_train=train[pred_target],
        ds_test=test[pred_feature],
        y_test=test[pred_target],
        y_pred_test=y_pred_test[average],
        y_pred_high=y_pred_test[average_upper],
        y_pred_low=y_pred_test[average_lower],
        output=OUTPUT,
        fig_name=output_name_png
        )

if __name__ == "__main__":
    main()

