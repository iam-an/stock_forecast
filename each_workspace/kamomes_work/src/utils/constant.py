from pathlib import Path


# このファイル基準でパスを張っておくと、実行場所が変わっても迷いにくいです。
ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent
WORKSPACE = SRC.parent
APP_ROOT = WORKSPACE.parents[1]

# 設定と生成物の置き場所です。
CONFIG = ROOT / "config.yaml"
MODELS = WORKSPACE / "models"
OUTPUT = WORKSPACE / "output"
PRODUCT_FOLDER = WORKSPACE / "product_folder"
RESULT = WORKSPACE / "result.json"
APP_MODELS = APP_ROOT / "models"
STREAMLIT_EXPORT = WORKSPACE / "streamlit_models"
STREAMLIT_TARGETS = ROOT / "streamlit_month_targets.yaml"


def ensure_workspace_dirs() -> None:
    """保存先フォルダを先に作っておく、小さいけれど大事な前処理です。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PRODUCT_FOLDER.mkdir(parents=True, exist_ok=True)
    APP_MODELS.mkdir(parents=True, exist_ok=True)
    STREAMLIT_EXPORT.mkdir(parents=True, exist_ok=True)
