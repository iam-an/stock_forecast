import pickle
from pathlib import Path
from typing import Any

def save_model(model: Any, file_path: Path)-> None:
    """
    modelを保存する

    Parameters
    ----------
    model : Any
        理屈上はなんでも保存できる
        機械学習モデルを想定
    file_path : Path
        保存するpath
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"sucess to write model as {file_path}")

def load_model(file_path: Path)-> Any:
    """
    load model

    Parameters
    ----------
    file_path : Path
        model が save された Path 

    Returns
    -------
    Any
        model
    """
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print(f"sucess to load model as {file_path}")
    return model