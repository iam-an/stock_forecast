#%% -*- coding: utf-8 -*-
"""
株価予測モデル（LSTM, ARIMA, XGBoost）の構築と可視化
複数のモデルを比較し、最適なものを選択
"""
import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


# ======================== Configuration ========================
SYMBOL = "2220.T"
EVAL_PERIOD_YEARS = 1
LOOKBACK = 30  # LSTM用の過去データ日数
TRAIN_COLOR = "blue"
EVAL_COLOR = "orange"
PRED_COLOR = "green"
LSTM_COLOR = "red"
ARIMA_COLOR = "purple"
XGBOOST_COLOR = "brown"


# ======================== Functions ========================
def fetch_stock_data(symbol: str, years: int = 10) -> pd.DataFrame:
    """yfinanceからデータを取得し、タイムゾーンを削除"""
    ticker = yfinance.Ticker(symbol)
    data = ticker.history(period=f"{years}y")
    data.index = data.index.tz_localize(None)
    return data


def split_train_eval(data: pd.DataFrame, eval_years: int = 1) -> tuple:
    """データを学習用と評価用に分割"""
    end_eval = data.index.max()
    start_eval = end_eval - pd.DateOffset(years=eval_years)
    train_data = data[data.index < start_eval]
    eval_data = data[data.index >= start_eval]
    return train_data, eval_data


def evaluate_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """モデルの性能を評価"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = np.mean(np.abs(actual - pred))
    
    # ゼロ除算チェック
    if np.mean(np.abs(actual)) < 1e-10:
        mape = 0.0
    else:
        mape = mean_absolute_percentage_error(actual, pred)
    
    # R²スコア（負になる可能性がある）
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


def build_lstm_model(train_data: np.ndarray, eval_data: np.ndarray, lookback: int = 30):
    """LSTM モデルの構築と学習"""
    if not LSTM_AVAILABLE:
        return None, None
    
    try:
        # スケーリング
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.reshape(-1, 1))
        
        # 学習データの準備
        X_train, y_train = [], []
        for i in range(len(scaled_train) - lookback):
            X_train.append(scaled_train[i:i+lookback])
            y_train.append(scaled_train[i+lookback])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # モデル構築
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # 学習（静かに）
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # 予測
        scaled_eval = scaler.transform(eval_data.reshape(-1, 1))
        combined = np.vstack([scaled_train[-lookback:], scaled_eval])
        
        X_eval = []
        for i in range(len(combined) - lookback):
            X_eval.append(combined[i:i+lookback])
        X_eval = np.array(X_eval)
        
        pred_scaled = model.predict(X_eval, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        
        return pred.flatten(), scaler
        
    except Exception as e:
        print(f"LSTM Error: {e}")
        return None, None


def build_arima_model(train_data: np.ndarray, eval_data: np.ndarray):
    """ARIMA モデルの構築と学習"""
    if not ARIMA_AVAILABLE:
        return None
    
    try:
        # ARIMA(1,1,1) で学習
        model = ARIMA(train_data, order=(1, 1, 1))
        model_fit = model.fit()
        
        # 予測
        forecast = model_fit.get_forecast(steps=len(eval_data))
        pred = forecast.predicted_mean.values
        
        return pred
        
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None


def build_xgboost_model(train_data: np.ndarray, eval_data: np.ndarray, lookback: int = 30):
    """XGBoost モデルの構築と学習（時系列特徴量付き）"""
    try:
        # 特徴量エンジニアリング
        def create_features(data, lookback):
            X, y = [], []
            for i in range(len(data) - lookback):
                features = list(data[i:i+lookback])
                features.extend([
                    np.mean(data[i:i+lookback]),
                    np.std(data[i:i+lookback]),
                    data[i+lookback-1] - data[i]
                ])
                X.append(features)
                y.append(data[i+lookback])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_features(train_data, lookback)
        
        # XGBoost モデル
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # 予測
        X_eval, _ = create_features(np.concatenate([train_data[-lookback:], eval_data]), lookback)
        pred = model.predict(X_eval[-len(eval_data):])
        
        return pred
        
    except Exception as e:
        print(f"XGBoost Error: {e}")
        return None


def build_baseline_model(train_data: np.ndarray, eval_data: np.ndarray, lookback: int = 30):
    """ベースライン：線形回帰 + 過去の移動平均"""
    try:
        # 特徴量：過去30日のトレンド
        def create_features(data, lookback):
            X, y = [], []
            for i in range(len(data) - lookback):
                features = [
                    np.mean(data[i:i+lookback]),
                    np.median(data[i:i+lookback]),
                    data[i+lookback-1],
                    (data[i+lookback-1] - data[i]) / lookback
                ]
                X.append(features)
                y.append(data[i+lookback])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_features(train_data, lookback)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 予測
        X_eval, _ = create_features(np.concatenate([train_data[-lookback:], eval_data]), lookback)
        pred = model.predict(X_eval[-len(eval_data):])
        
        return pred
        
    except Exception as e:
        print(f"Baseline Error: {e}")
        return None


def compare_models(symbol: str, eval_years: int = 1):
    """複数のモデルを構築・評価・比較"""
    # データ取得
    print(f"Fetching data for {symbol}...")
    data = fetch_stock_data(symbol, years=10)
    train_data, eval_data = split_train_eval(data, eval_years)
    
    train_prices = train_data["Close"].values
    eval_prices = eval_data["Close"].values
    
    results = {
        "dates": eval_data.index,
        "actual": eval_prices
    }
    
    print(f"Training data: {len(train_prices)} days | Eval data: {len(eval_prices)} days")
    
    # LSTM
    if LSTM_AVAILABLE:
        print("Building LSTM model...")
        lstm_pred, _ = build_lstm_model(train_prices, eval_prices, LOOKBACK)
        if lstm_pred is not None:
            results["lstm"] = lstm_pred
            results["lstm_metrics"] = evaluate_metrics(eval_prices, lstm_pred)
            print(f"  LSTM R²={results['lstm_metrics']['r2']:.4f}")
    
    # ARIMA
    if ARIMA_AVAILABLE:
        print("Building ARIMA model...")
        arima_pred = build_arima_model(train_prices, eval_prices)
        if arima_pred is not None:
            results["arima"] = arima_pred
            results["arima_metrics"] = evaluate_metrics(eval_prices, arima_pred)
            print(f"  ARIMA R²={results['arima_metrics']['r2']:.4f}")
    
    # XGBoost
    print("Building XGBoost model...")
    xgb_pred = build_xgboost_model(train_prices, eval_prices, LOOKBACK)
    if xgb_pred is not None:
        results["xgboost"] = xgb_pred
        results["xgboost_metrics"] = evaluate_metrics(eval_prices, xgb_pred)
        print(f"  XGBoost R²={results['xgboost_metrics']['r2']:.4f}")
    
    # ベースライン
    print("Building Baseline model...")
    baseline_pred = build_baseline_model(train_prices, eval_prices, LOOKBACK)
    if baseline_pred is not None:
        results["baseline"] = baseline_pred
        results["baseline_metrics"] = evaluate_metrics(eval_prices, baseline_pred)
        print(f"  Baseline R²={results['baseline_metrics']['r2']:.4f}")
    
    # 最適なモデルを特定
    best_model = None
    best_r2 = -np.inf
    for model_name in ["lstm", "arima", "xgboost", "baseline"]:
        if f"{model_name}_metrics" in results:
            if results[f"{model_name}_metrics"]["r2"] > best_r2:
                best_r2 = results[f"{model_name}_metrics"]["r2"]
                best_model = model_name
    
    results["best_model"] = best_model
    results["train_data"] = train_data
    results["eval_data"] = eval_data
    
    return results


def plot_comparison(results: dict):
    """複数モデルの予測結果を比較"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # プロット1: 全期間の予測結果
    ax = axes[0]
    dates = results["dates"]
    actual = results["actual"]
    
    ax.plot(results["train_data"].index, results["train_data"]["Close"], 
            label="Training Data", color=TRAIN_COLOR, linewidth=1.5, alpha=0.7)
    ax.plot(dates, actual, label="Actual", color=EVAL_COLOR, linewidth=2, marker='o', markersize=3)
    
    if "lstm" in results:
        ax.plot(dates, results["lstm"], label="LSTM", color=LSTM_COLOR, 
                linewidth=2, linestyle="--", alpha=0.8)
    if "arima" in results:
        ax.plot(dates, results["arima"], label="ARIMA", color=ARIMA_COLOR, 
                linewidth=2, linestyle="--", alpha=0.8)
    if "xgboost" in results:
        ax.plot(dates, results["xgboost"], label="XGBoost", color=XGBOOST_COLOR, 
                linewidth=2, linestyle="--", alpha=0.8)
    if "baseline" in results:
        ax.plot(dates, results["baseline"], label="Baseline", color="gray", 
                linewidth=1.5, linestyle=":", alpha=0.6)
    
    ax.set_ylabel("Price (JPY)", fontsize=12)
    ax.set_title(f"Stock Price Prediction Comparison - {SYMBOL}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    
    # プロット2: メトリクス比較
    ax = axes[1]
    models = []
    r2_scores = []
    rmse_scores = []
    
    for model_name in ["lstm", "arima", "xgboost", "baseline"]:
        if f"{model_name}_metrics" in results:
            models.append(model_name.upper())
            r2_scores.append(results[f"{model_name}_metrics"]["r2"])
            rmse_scores.append(results[f"{model_name}_metrics"]["rmse"])
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, r2_scores, width, label="R² Score", alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, rmse_scores, width, label="RMSE", alpha=0.8, color="orange")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12, color="blue")
    ax2.set_ylabel("RMSE (JPY)", fontsize=12, color="orange")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Baseline R²=0")
    ax.grid(True, alpha=0.3, axis="y")
    
    # 凡例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    
    plt.tight_layout()
    return fig


def print_summary(results: dict):
    """結果サマリーを表示"""
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    
    for model_name in ["lstm", "arima", "xgboost", "baseline"]:
        if f"{model_name}_metrics" in results:
            metrics = results[f"{model_name}_metrics"]
            best_mark = "✓ BEST" if results["best_model"] == model_name else ""
            print(f"\n{model_name.upper():12} {best_mark}")
            print(f"  R² Score:  {metrics['r2']:8.4f}")
            print(f"  RMSE:      {metrics['rmse']:8.2f} JPY")
            print(f"  MAE:       {metrics['mae']:8.2f} JPY")
            print(f"  MAPE:      {metrics['mape']:8.2f}%")
    
    print("\n" + "="*60)


# ======================== Main Execution ========================
if __name__ == "__main__":
    # 複数モデルの比較
    results = compare_models(SYMBOL, EVAL_PERIOD_YEARS)
    
    # 結果を表示
    print_summary(results)
    
    # グラフを表示
    print("\nGenerating comparison plots...")
    fig = plot_comparison(results)
    plt.show()
