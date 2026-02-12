from functools import wraps
import polars as pl
import pandas as pd

def convert_pd_pl(func):
    """_summary_
    - polarsのdataframeとして受け取った引数をすべてpandasに変換してから関数に渡す
    - 返り値でdataframeを返す場合はpolarsに変換する
    - このdecoratorを使用すれば引数、戻り値はpl.DataFrameであることが保証される

    Parameters
    ----------
    func : _type_
        _description_
    """
    def wrapper(*args, **kwargs):
        # input
        new_args = []
        for a in args:
            if isinstance(a, pl.DataFrame):
                new_args.append(a.to_pandas())
            else:
                new_args.append(a)
        new_kwargs = {}
        for k,v in kwargs.items():
            if isinstance(v, pl.DataFrame):
                new_kwargs[k] = v.to_pandas()
            else:
                new_kwargs[k] = v
        result = func(*new_args, **new_kwargs)

        # return
        #TODO
        # resultが単一のときのみ使用可能
        if isinstance(result, pd.DataFrame):
            return pl.from_pandas(result)
        else:
            return result
    return wrapper