from core.use_yfinance import get_yfinace
from core.split_data import split_train_test
from core.make_model import make_model_prophet
from core.analysis import evaluate_score, set_mlflow, save_artifacts
from core.make_plot import make_plot_time
from core.discompose_data import use_stl
from core.rename import rename_for_prophet
from utils.constant import OUTPUT, CONFIG
import json
import polars as pl

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
rename_map = settings["RENAME_COLUMNS"]

def main():
    df = get_yfinace(company_name=company, start=start, end=end)
    df = rename_for_prophet(df, rename_map)
    df = df.to_pandas()
    df_stl = use_stl(target = "y", df=df, period=period)
    df_stl = pl.from_pandas(df_stl)
    train, test = split_train_test(df=df_stl, test_size=0.2)
    model_fitted = make_model_prophet(train)
    save_artifacts(model=model_fitted, artifact_path=OUTPUT)
    y_pred_test = model_fitted.predict(test[["ds"]])
    
    scores = evaluate_score(test=test[pred_target], y_pred_test=y_pred_test[average])
    set_mlflow(model=model_fitted, scores=scores, artifacts=[OUTPUT, CONFIG])
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
# to do
# make graph by plotly
# 日付と銘柄をいれる意識
# 本番は作ったmodelをいれる
# try

if __name__ == "__main__":
    main()

