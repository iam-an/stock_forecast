from __future__ import annotations

from pathlib import Path
from time import sleep

import joblib
import pandas as pd
import yfinance as yf
from prophet import Prophet


def load_prophet_training_frame(
    display_name: str,
    ticker: str,
    history_period: str,
    interval: str,
) -> pd.DataFrame:
    """
    yfinance から Prophet 学習用の ds / y フレームを作ります。

    namai の 1year 系は Streamlit 側で都度 Prophet を学習しているので、
    ここでも素直に High 系列を取り出して学習フレームへ整えます。
    """
    raw = pd.DataFrame()
    last_error: Exception | None = None

    # yfinance はネットワーク気分で空振りすることがあるので、
    # 少しだけ待ちながら数回試すようにしておきます。
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                period=history_period,
                interval=interval,
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
                f"{display_name} ({ticker}) の株価データを取得できませんでした: {last_error}"
            ) from last_error
        raise ValueError(f"{display_name} ({ticker}) の株価データを取得できませんでした。")

    frame = raw.reset_index()

    # yfinance の列名は環境や銘柄で少し揺れるので、
    # ここで一度ならしておくと後段がかなり読みやすくなります。
    normalized_columns: list[str] = []
    for column in frame.columns:
        if isinstance(column, tuple):
            left = str(column[0])
            right = str(column[1])
            normalized_columns.append(left if right in ("", "None") else f"{left}_{right}")
        else:
            normalized_columns.append(str(column))
    frame.columns = normalized_columns

    date_column = "Date"
    if "Datetime" in frame.columns and "Date" not in frame.columns:
        date_column = "Datetime"

    high_candidates = [
        "High",
        f"High_{ticker}",
        f"('High', '{ticker}')",
    ]
    high_column = next((column for column in high_candidates if column in frame.columns), None)

    if high_column is None:
        for column in frame.columns:
            if column.startswith("High"):
                high_column = column
                break

    if high_column is None or date_column not in frame.columns:
        raise ValueError(f"{display_name} ({ticker}) の Date / High 列を整形できませんでした。")

    train = frame[[date_column, high_column]].copy()
    train = train.rename(columns={date_column: "ds", high_column: "y"})
    train["ds"] = pd.to_datetime(train["ds"]).dt.tz_localize(None)
    train["y"] = pd.to_numeric(train["y"], errors="coerce")
    train = train.dropna().sort_values("ds").reset_index(drop=True)

    if len(train) < 60:
        raise ValueError(f"{display_name} ({ticker}) は Prophet 学習に必要な履歴が足りません。")

    return train


def make_model_prophet(train: pd.DataFrame) -> Prophet:
    """
    namai の make_model_prophet をほぼそのまま踏襲します。

    ここはあえて設定を増やしすぎず、
    「namai の 1year と同じ発想で学習する」ことを優先しています。
    """
    model = Prophet(growth="linear", changepoint_prior_scale=0.3, n_changepoints=50)
    model.fit(train)
    return model


def save_namai_style_artifact(model: Prophet, artifact_path: Path, company: str) -> Path:
    """
    namai と同じく {company}.joblib という見た目で保存します。

    pred_range や model_type をあえてファイル名へ入れないのが、
    namai 流の保存との大きな違いです。
    """
    artifact_path.mkdir(parents=True, exist_ok=True)
    output_path = artifact_path / f"{company}.joblib"
    joblib.dump(model, output_path)
    return output_path
