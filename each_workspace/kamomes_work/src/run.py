from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml


# `python src/run.py` でも `python -m src.run` でも動くようにしておきます。
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.core.analysis import evaluate_score, save_artifacts, write_to_json
from src.core.make_model import train_medium_horizon_models
from src.core.make_plot import make_plot_time
from src.core.pre_process import build_supervised_dataset, prepare_feature_frame
from src.core.split_data import split_train_test
from src.core.use_yfinance import get_yfinance
from src.utils.constant import CONFIG, PRODUCT_FOLDER, RESULT, ensure_workspace_dirs


def load_settings() -> dict:
    """設定ファイルを読み込みます。"""
    with open(CONFIG, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_parser() -> argparse.ArgumentParser:
    """最低限の実行オプションだけ持った引数パーサです。"""
    parser = argparse.ArgumentParser(description="kamomes_work の中期株価予測モデル")
    parser.add_argument(
        "--company",
        action="append",
        help="設定ファイルの COMPANY を上書きしたいときに使います。複数回指定できます。",
    )
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="評価を省いて全データ学習だけしたいときに使います。",
    )
    parser.add_argument(
        "--run_date",
        default=date.today().isoformat(),
        help="出力ファイル名に使う日付です。未指定なら今日の日付を使います。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()

    if args.company:
        settings["COMPANY"] = args.company

    ensure_workspace_dirs()

    results_payload = {
        "run_date": args.run_date,
        "mode": "full_data" if args.full_data else "train_test",
        "settings": settings,
        "results": {},
    }

    companies = settings["COMPANY"]
    input_window = int(settings["INPUT_WINDOW"])
    out_window = int(settings["OUT_WINDOW"])
    stl_period = int(settings["STL_PERIOD"])

    for company in companies:
        print(f"[INFO] {company} のデータを取得しています")

        # 保存する設定を見返したときに迷わないよう、銘柄ごとの設定をここで固めます。
        company_settings = dict(settings)
        company_settings["company"] = company
        company_settings["COMPANY"] = [company]

        frame = get_yfinance(
            company_name=company,
            start=company_settings["START_DAY"],
            end=company_settings["END_DAY"],
        )

        if frame is None:
            print(f"[WARN] {company} は yfinance からデータを取得できませんでした")
            continue

        # 少なすぎるデータで学習すると見た目だけ動いてしまうので、先に足切りします。
        min_required_rows = max(input_window + out_window + 10, stl_period * 2 + out_window)
        if len(frame) < min_required_rows:
            print(f"[WARN] {company} はデータ量が足りないためスキップします")
            continue

        feature_frame = prepare_feature_frame(frame=frame, settings=company_settings)
        datasets = build_supervised_dataset(frame=feature_frame, settings=company_settings)

        if len(datasets["X"]) < 8:
            print(f"[WARN] {company} は学習サンプルが少なすぎるためスキップします")
            continue

        split_data = split_train_test(
            datasets=datasets,
            test_size=float(company_settings["TEST_SIZE"]),
            full_data=args.full_data,
        )

        model_result = train_medium_horizon_models(
            split_data=split_data,
            settings=company_settings,
            company=company,
        )

        artifact_paths = save_artifacts(bundle=model_result["bundle"], company=company)

        company_result = {
            "data_rows": int(len(frame)),
            "feature_rows": int(len(feature_frame)),
            "sample_count": int(len(datasets["X"])),
            "artifacts": artifact_paths,
            "model_summary": model_result["model_summary"],
        }

        if split_data["has_test"]:
            metrics = evaluate_score(
                y_true=split_data["y_total_test"],
                y_pred=model_result["y_pred_test"],
                last_values=split_data["last_high_test"],
            )

            # 最新の test サンプルを 1 枚だけ保存すると、あとで確認しやすいです。
            last_meta = split_data["meta_test"][-1]
            last_y_true = split_data["y_total_test"][-1]
            last_y_pred = model_result["y_pred_test"][-1]
            band_size = np.mean(
                np.abs(split_data["y_total_test"] - model_result["y_pred_test"]),
                axis=0,
            )

            plot_path = make_plot_time(
                company=company,
                sample_meta=last_meta,
                y_true=last_y_true,
                y_pred=last_y_pred,
                output_folder=PRODUCT_FOLDER,
                run_date=args.run_date,
                band_size=band_size,
            )

            company_result.update(
                {
                    "metrics": metrics,
                    "plot_path": str(plot_path),
                    "latest_as_of": last_meta["as_of_date"],
                    "latest_forecast_dates": last_meta["future_dates"],
                    "latest_actual": np.round(last_y_true, 4).tolist(),
                    "latest_prediction": np.round(last_y_pred, 4).tolist(),
                }
            )

        results_payload["results"][company] = company_result
        print(f"[INFO] {company} の保存が完了しました")

    write_to_json(RESULT, results_payload)
    print(f"[INFO] 結果を {RESULT} に保存しました")


if __name__ == "__main__":
    main()
