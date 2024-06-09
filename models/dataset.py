import torch
import numpy as np
import pandas as pd
import scipy
from typing import Tuple, List, Dict


def load_data(symbol: str, winsorize: bool = False) -> pd.DataFrame:
    """
    Load the orderbook data for a given symbol and optionally apply winsorization to the data.
    Args:
        symbol (str): The symbol to load data for.
        winsorize (bool): Whether to apply winsorization to the data.
                    Note: Only quantities are winsorized.
    Returns:
        pd.DataFrame: The orderbook data with winsorization applied.
    """
    df = pd.read_parquet(
        f"../../data/WebsocketData/binance_orderbook_{symbol.lower()}_depth_10_2.parquet",
        engine="pyarrow",
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.sort_values(by="datetime", inplace=True)
    df.drop(columns=["E", "T", "timestamp"], inplace=True)
    df.set_index("datetime", inplace=True)
    if winsorize:
        qty_columns = [col for col in df.columns if "qty" in col]
        df[qty_columns] = scipy.stats.mstats.winsorize(
            df[qty_columns].values, limits=(0.05, 0.05)
        )

    ordered_columns = np.array(
        [
            [f"ask_price_{i}", f"ask_qty_{i}", f"bid_price_{i}", f"bid_qty_{i}"]
            for i in range(1, 10 + 1)
        ]
    ).flatten()
    return df[ordered_columns]


def get_window_boundaries(
    df: pd.DataFrame,
    train_interval_ms: int,
    val_interval_ms: int,
    test_interval_ms: int,
    h: int,
) -> List[Dict[str, pd.Timestamp]]:
    resampled = df.resample(f"{val_interval_ms + test_interval_ms + 2*h}ms")
    test_ends = resampled.last().index[1:]
    train_starts = test_ends - pd.Timedelta(
        milliseconds=test_interval_ms + val_interval_ms + train_interval_ms + 2 * h
    )
    window_boundaries = pd.DataFrame(
        {"train_start": train_starts, "test_end": test_ends}
    )
    window_boundaries["train_end"] = window_boundaries.train_start + pd.Timedelta(
        milliseconds=train_interval_ms
    )
    window_boundaries["val_start"] = window_boundaries.train_end + pd.Timedelta(
        milliseconds=h
    )
    window_boundaries["val_end"] = window_boundaries.val_start + pd.Timedelta(
        milliseconds=val_interval_ms
    )
    window_boundaries["test_start"] = window_boundaries.val_end + pd.Timedelta(
        milliseconds=h
    )
    # Re-order columns
    window_boundaries = window_boundaries[
        ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
    ]
    # Remove window_boundaries for which our df doesn't go far enough back
    window_boundaries = window_boundaries.loc[
        window_boundaries.train_start >= df.index.min(), :
    ].reset_index(drop=True)
    return window_boundaries.to_dict("records")


def get_window(
    df: pd.DataFrame, window_boundary: Dict[str, pd.Timestamp], target_col: str = "y"
) -> Tuple[torch.Tensor]:
    """We do this on a per-window basis to be use less memory."""
    wb = window_boundary
    feature_cols = [col for col in df.columns if col != target_col]
    _df = df.loc[(wb["train_start"] <= df.index) & (df.index <= wb["test_end"]), :]
    df_train = _df.loc[
        (wb["train_start"] <= _df.index) & (_df.index <= wb["train_end"]), :
    ]
    X_train = torch.from_numpy(df_train[feature_cols].values)
    y_train = torch.from_numpy(df_train[target_col].values)
    df_val = _df.loc[(wb["val_start"] <= _df.index) & (_df.index <= wb["val_end"]), :]
    X_val = torch.from_numpy(df_val[feature_cols].values)
    y_val = torch.from_numpy(df_val[target_col].values)
    df_test = _df.loc[
        (wb["test_start"] <= _df.index) & (_df.index <= wb["test_end"]), :
    ]
    X_test = torch.from_numpy(df_test[feature_cols].values)
    y_test = torch.from_numpy(df_test[target_col].values)
    return X_train, y_train, X_val, y_val, X_test, y_test


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, T: int = 100):
        n, d = X.shape
        _X = torch.zeros((n - (T - 1), T, d))
        for i in range(T, n + 1):
            _X[i - T] = X[i - T : i, :]

        self.X = torch.unsqueeze(_X, 1)
        self.y = y[T - 1 : n]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
