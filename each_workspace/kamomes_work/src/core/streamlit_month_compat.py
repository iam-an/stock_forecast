from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any

import pandas as pd
import yfinance as yf
from prophet import Prophet

try:
    import jpholiday
except ImportError:  # pragma: no cover - 環境差を吸収するための保険です
    jpholiday = None


@dataclass
class StreamlitMonthCompatModel:
    """
    現行 Streamlit で扱いやすい 1month 用の互換モデルです。

    かなり大事なポイントは 2 つあります。

    1. callable であること
       既存の day/week 系は `model(indata)` という呼び方をしているため、
       それに寄せて `__call__` を持たせています。

    2. Prophet っぽいメソッドも持つこと
       month/year 系は Prophet 風の `.make_future_dataframe()` と `.predict()` を
       使う流れに寄っているので、こちらも用意しています。
    """

    display_name: str
    ticker: str
    forecast_days: int
    history_period: str
    interval: str
    growth: str
    changepoint_prior_scale: float
    n_changepoints: int

    def __post_init__(self) -> None:
        # 実行のたびに新しいデータで学習し直す前提なので、
        # モデル本体はその場でキャッシュするだけにしています。
        self._model: Prophet | None = None
        self._train_frame: pd.DataFrame | None = None
        self._last_refresh_at: str | None = None

    def _download_high_frame(self) -> pd.DataFrame:
        """
        yfinance から High 系列を取り出して Prophet 学習向けに整えます。
        """
        raw = pd.DataFrame()
        last_error: Exception | None = None

        # 通信系の一時エラーでアプリ全体が不安定に見えるのは避けたいので、
        # ここだけは少しだけ粘るようにしています。
        for attempt in range(3):
            try:
                raw = yf.download(
                    self.ticker,
                    period=self.history_period,
                    interval=self.interval,
                    threads=False,
                    auto_adjust=False,
                    progress=False,
                )
                if not raw.empty:
                    break
            except Exception as exc:  # pragma: no cover - 外部通信の揺らぎ対策です
                last_error = exc

            if attempt < 2:
                sleep(2)

        if raw.empty:
            if last_error is not None:
                raise ValueError(
                    f"{self.display_name} ({self.ticker}) のデータを取得できませんでした: {last_error}"
                ) from last_error
            raise ValueError(f"{self.display_name} ({self.ticker}) のデータを取得できませんでした。")

        frame = raw.reset_index()

        # download の戻りは MultiIndex 風に見えることがあるので、
        # ここで無理なく平坦化しておきます。
        flattened_columns: list[str] = []
        for column in frame.columns:
            if isinstance(column, tuple):
                left = str(column[0])
                right = str(column[1])
                if right in ("", "None"):
                    flattened_columns.append(left)
                else:
                    flattened_columns.append(f"{left}_{right}")
            else:
                flattened_columns.append(str(column))
        frame.columns = flattened_columns

        date_column = "Date"
        if "Datetime" in frame.columns and "Date" not in frame.columns:
            date_column = "Datetime"

        high_candidates = [
            "High",
            f"High_{self.ticker}",
            f"('High', '{self.ticker}')",
        ]
        high_column = next((column for column in high_candidates if column in frame.columns), None)
        if high_column is None:
            # 一応ここも保険を入れておきます。
            for column in frame.columns:
                if column.startswith("High"):
                    high_column = column
                    break

        if high_column is None or date_column not in frame.columns:
            raise ValueError(
                f"{self.display_name} ({self.ticker}) の High / Date 列を整形できませんでした。"
            )

        train = frame[[date_column, high_column]].copy()
        train = train.rename(columns={date_column: "ds", high_column: "y"})
        train["ds"] = pd.to_datetime(train["ds"]).dt.tz_localize(None)
        train["y"] = pd.to_numeric(train["y"], errors="coerce")
        train = train.dropna().sort_values("ds").reset_index(drop=True)

        if len(train) < 60:
            raise ValueError(
                f"{self.display_name} ({self.ticker}) は 1month 予測に必要な履歴が足りません。"
            )

        return train

    def _is_business_day(self, value: pd.Timestamp) -> bool:
        """
        現行 back/select_models.py の休日判定に寄せています。
        jpholiday がない環境では平日判定だけで回せるようにしています。
        """
        timestamp = pd.Timestamp(value)
        if timestamp.weekday() >= 5:
            return False

        if jpholiday is None:
            return True

        return not jpholiday.is_holiday(timestamp.to_pydatetime())

    def _fit_latest(self) -> None:
        """
        最新データで Prophet を作り直します。

        「joblib を読むだけで即推論」よりは少し重いですが、
        current 1month の Streamlit 実装が live 学習寄りなので、
        ここもその思想に合わせています。
        """
        train = self._download_high_frame()
        model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            n_changepoints=self.n_changepoints,
        )
        model.fit(train)

        self._train_frame = train
        self._model = model
        self._last_refresh_at = datetime.now().isoformat(timespec="seconds")

    def make_future_dataframe(self, periods: int | None = None) -> pd.DataFrame:
        """
        Prophet 互換のメソッドです。
        まだ fit していなければ、その場で最新データから学習します。
        """
        if self._model is None:
            self._fit_latest()

        assert self._model is not None
        return self._model.make_future_dataframe(periods=periods or self.forecast_days)

    def predict(self, future: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Prophet 互換の `.predict()` です。
        future 未指定なら、自前で future dataframe を作ります。
        """
        if self._model is None:
            self._fit_latest()

        assert self._model is not None
        if future is None:
            future = self.make_future_dataframe(self.forecast_days)
        return self._model.predict(future)

    def get_filtered_forecast(self, periods: int | None = None) -> pd.DataFrame:
        """
        Streamlit 表示に使いやすいよう、
        未来側だけを平日ベースで抜き出した forecast を返します。
        """
        periods = periods or self.forecast_days
        forecast = self.predict(self.make_future_dataframe(periods))

        assert self._train_frame is not None
        last_date = self._train_frame["ds"].max()

        filtered_rows: list[dict[str, Any]] = []
        for _, row in forecast.iterrows():
            ds_value = pd.Timestamp(row["ds"])
            if ds_value <= last_date:
                continue
            if not self._is_business_day(ds_value):
                continue

            filtered_rows.append(
                {
                    "ds": ds_value,
                    "yhat": float(row["yhat"]),
                    "yhat_lower": float(row["yhat_lower"]),
                    "yhat_upper": float(row["yhat_upper"]),
                }
            )

            if len(filtered_rows) >= periods:
                break

        return pd.DataFrame(filtered_rows)

    def get_recent_actual_frame(self, window: int = 30) -> pd.DataFrame:
        """
        グラフ左側に出す直近実績を返します。

        予測前に fit がまだ走っていなければ、ここで最新データを読みます。
        """
        if self._model is None:
            self._fit_latest()

        assert self._train_frame is not None
        return self._train_frame.tail(window).copy().reset_index(drop=True)

    def build_streamlit_payload(self, history_window: int = 30, forecast_window: int | None = None):
        """
        root 側 Streamlit がそのまま描画できる 4 点セットを返します。

        返り値:
        - act_date
        - act_data
        - pred_date
        - pred_data
        """
        forecast_window = forecast_window or self.forecast_days
        actual = self.get_recent_actual_frame(window=history_window)
        forecast = self.get_filtered_forecast(periods=forecast_window)

        act_date = actual["ds"].tolist()
        act_data = actual["y"].astype(float).tolist()
        pred_date = forecast["ds"].tolist()
        pred_data = forecast["yhat"].astype(float).tolist()

        if act_date and pred_date:
            pred_date = [act_date[-1]] + pred_date
            pred_data = [act_data[-1]] + pred_data

        return act_date, act_data, pred_date, pred_data

    def __call__(self, indata: Any) -> list[float]:
        """
        day/week 系の `model(indata)` 呼び出しに合わせた入口です。

        `indata` は互換性のため受け取っていますが、
        1month では日付情報込みで最新データを取り直したいので、
        実際の予測は ticker ベースで再取得して行います。
        """
        _ = indata  # この引数は「既存フロント互換」のためだけに残しています。
        filtered = self.get_filtered_forecast(self.forecast_days)
        return filtered["yhat"].astype(float).tolist()


def build_streamlit_month_model(
    display_name: str,
    ticker: str,
    forecast_days: int,
    history_period: str,
    interval: str,
    prophet_settings: dict,
) -> StreamlitMonthCompatModel:
    """
    設定値から 1month 用の互換モデルを組み立てます。
    """
    return StreamlitMonthCompatModel(
        display_name=display_name,
        ticker=ticker,
        forecast_days=forecast_days,
        history_period=history_period,
        interval=interval,
        growth=str(prophet_settings["GROWTH"]),
        changepoint_prior_scale=float(prophet_settings["CHANGEPOINT_PRIOR_SCALE"]),
        n_changepoints=int(prophet_settings["N_CHANGEPOINTS"]),
    )
