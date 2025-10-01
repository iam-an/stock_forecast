from namai.src.core.use_yfinance import get_yfinace
from namai.src.core.split_data import split_train_test
from namai.src.core.make_model import make_model_prophet
from namai.src.core.analysis import evaluate_score
from namai.src.core.make_plot import make_plot_time
from namai.src.core.discompose_data import use_stl
from namai.src.utils.constant import OUTPUT

def main():
    df = get_yfinace(company_name="AAPL", start="2025-01-01", end="2025-07-01")
    train, test = split_train_test(df=df, test_size=0.2)
    df_train_stl = use_stl(target = "High", df_train=train, period=24)
    y_pred_test, test, train = make_model_prophet(train=df_train_stl, test=test, target="High", date_col="Date")
    evaluate_score(test=test, y_pred_test=y_pred_test)
    make_plot_time(train=train, test=test, y_pred_test=y_pred_test, output=OUTPUT, fig_name="time.png")

if __name__ == "__main__":
    main()

