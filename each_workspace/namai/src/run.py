from namai.src.core.use_yfinance import get_yfinace
from namai.src.core.split_data import split_train_test
from namai.src.core.make_model import make_model_prophet
from namai.src.core.analysis import evaluate_score, set_mlflow, save_artifacts, write_to_json
from namai.src.core.make_plot import make_plot_time
from namai.src.core.discompose_data import use_stl
from namai.src.core.rename import rename_for_prophet
from utils.constant import OUTPUT, CONFIG, RESULT
import yaml
import click
import numpy as np

with open(CONFIG, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)
companies = settings["COMPANY"]
start = settings["START_DAY"]
end = settings["END_DAY"]
stl_target = settings["STL_TARGET"]
pred_target = settings["PRED_TARGET"]
pred_feature = settings["PRED_FEATURE"]
average = settings["AVERAGE"]
average_upper = settings["AVERAGE_UPPER"]
average_lower = settings["AVERAGE_LOWER"]
period = settings["PERIOD"]
rename_map = settings["RENAME_COLUMNS"]
print(companies)

@click.command()
@click.option("--full_data", is_flag=True)
def main(full_data):
    for company in companies:
        df = get_yfinace(company_name=company, start=start, end=end)
        if df is None:
            continue
        df = rename_for_prophet(df, rename_map)
        df_stl = use_stl(target = "y", df=df, period=period)
        if full_data:
            train = df_stl.to_pandas()
        else:
            train, test = split_train_test(df=df_stl, test_size=0.2)
        model_fitted = make_model_prophet(train)
        save_artifacts(model=model_fitted, artifact_path=OUTPUT, company=company)
        if not full_data:
            y_pred_test = model_fitted.predict(test)
            y_pred_test["yhat"] = np.exp(y_pred_test["yhat"])
            scores = evaluate_score(test=test[pred_target], y_pred_test=y_pred_test[average])
            write_to_json(RESULT, scores = scores, company= company)
            set_mlflow(model=model_fitted, scores=scores, artifacts=[OUTPUT, CONFIG, RESULT])
            make_plot_time(
                ds_train=train[pred_feature],
                y_train=train[pred_target],
                ds_test=test[pred_feature],
                y_test=test[pred_target],
                y_pred_test=y_pred_test[average],
                y_pred_high=y_pred_test[average_upper],
                y_pred_low=y_pred_test[average_lower],
                output=OUTPUT,
                fig_name=f"{company}_predict.png"
                )
#TODO
# make graph by plotly
# 日付と銘柄をいれる意識
# 本番は作ったmodelをいれる
# try
# loggerの設定
# assert
# tabpfn

if __name__ == "__main__":
    main()

