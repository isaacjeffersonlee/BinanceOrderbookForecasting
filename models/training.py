import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from rich.console import Console
import pickle
import os
from dataset import load_data, get_window, get_window_boundaries
import xgb  # noqa
import deepOF  # noqa
import deepLOB  # noqa
import lr  # noqa
import utils


def train(k, model_str, use_orderflow, console):
    assert model_str in ["xgb", "deepOF", "deepLOB", "lr"]
    train_pipeline = eval(f"{model_str}.train_pipeline")
    console.rule(f"Training model: {model_str}")
    cls_weight = (1.0, 1.0, 1.0)
    if model_str == "deepLOB":
        use_orderflow = False
    elif model_str == "deepOF":
        use_orderflow = True

    learning_rate = 1e-4
    batch_size = 258
    train_interval_ms = 48 * 60 * 60 * 1000
    val_interval_ms = 24 * 60 * 60 * 1000
    test_interval_ms = 24 * 60 * 60 * 1000
    epochs = 5

    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "MATICUSDT")
    for symbol in symbols:
        console.log("Processing symbol: ", symbol)
        # Load and parse data
        console.log("Reading parquet data...")
        df = load_data(symbol, winsorize=False)
        console.log("Parquet data read!")
        console.log("Calculating labels...")
        # Calculate Labels
        mid_price = (df.ask_price_1 + df.bid_price_1) / 2
        m_minus = mid_price.rolling(k, closed="both").mean()
        m_plus = mid_price[::-1].rolling(k, closed="left").mean()[::-1]
        df["y"] = (m_plus - m_minus) / m_minus  # noqa
        df.dropna(inplace=True)
        # df["y"] = utils.discretize(df["y"], ϵ) -> Do this for each window instead
        if use_orderflow:
            orderflow_df = pd.DataFrame()
            console.log("df LOB -> df OF...")
            for j in range(1, 10 + 1):
                ask_orderflow, bid_orderflow = utils.orderflow(
                    ask_price=df[f"ask_price_{j}"],
                    bid_price=df[f"bid_price_{j}"],
                    ask_qty=df[f"ask_qty_{j}"],
                    bid_qty=df[f"bid_qty_{j}"],
                )
                orderflow_df[f"ask_orderflow_{j}"] = ask_orderflow
                orderflow_df[f"bid_orderflow_{j}"] = bid_orderflow

            orderflow_df["y"] = df["y"][1:]
            df = orderflow_df

        console.log("Windowing data...")
        window_boundaries = get_window_boundaries(
            df, train_interval_ms, val_interval_ms, test_interval_ms, h=10000
        )
        console.log(
            f"{len(window_boundaries)} (train, val, test) windows generated for {symbol}!"
        )
        for i, wb in enumerate(tqdm(window_boundaries)):
            save_path = f"val/k={k}/{model_str}/{'OF' if use_orderflow else 'LOB'}/{symbol}_window_{i}"
            console.log(f"Processing window {i+1} / {len(window_boundaries)}...")
            X_train, y_train, X_val, y_val, _, _ = get_window(df, wb)
            # Standardization
            μ = X_train.mean(dim=0).unsqueeze(dim=0)
            σ = X_train.std(dim=0).unsqueeze(dim=0)
            X_train = (X_train - μ) / σ
            X_val = (X_val - μ) / σ

            # Discretize y_train and y_val
            ϵ = 0.5 * (np.abs(np.quantile(y_train, 0.33)) + np.quantile(y_train, 0.66))
            y_train = torch.tensor(utils.discretize(y_train, ϵ))
            y_val = torch.tensor(utils.discretize(y_val, ϵ))
            console.print(f"y_train distribution, with ϵ_{k}_{i}={ϵ}:")
            console.print(pd.Series(y_train).value_counts())
            y_val, y_val_pred = train_pipeline(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                save_path=save_path,
                batch_size=batch_size,
                learning_rate=learning_rate,
                cls_weight=cls_weight,
                epochs=epochs,
                symbol=symbol,
                window_idx=i,
                model_str=model_str,
                k=k,
            )

            if use_orderflow and "OF" not in model_str:
                method_str = "OF"
            elif not use_orderflow and "LOB" not in model_str:
                method_str = "LOB"
            else:
                method_str = ""
            console.rule(f"{model_str}{method_str} k={k} Window {i} Results")
            clfr_str = classification_report(y_val, y_val_pred, digits=4)
            console.print(clfr_str)
            console.rule()

            clfr_dict = classification_report(y_val, y_val_pred, output_dict=True)
            clfr_save_path = f"./classification_reports/{save_path}.pkl"
            if not os.path.exists(clfr_save_path):
                os.makedirs(os.path.dirname(clfr_save_path), exist_ok=True)

            with open(clfr_save_path, "wb") as f:
                pickle.dump(clfr_dict, f)


def main():
    console = Console()
    start_time = pd.Timestamp.now()
    for k in tqdm([4, 10, 50, 200]):
        console.rule(f"Training models for k={k}")
        for model_str, use_orderflow in (
            ("xgb", True),
            ("xgb", False),
            # ("lr", True),
            # ("lr", False),
            # ("deepOF", True),
            # ("deepLOB", False),
        ):
            train(k, model_str, use_orderflow, console)

    end_time = pd.Timestamp.now()
    console.rule("FINISHED TRAINING!")
    console.print(f"Total training time: {end_time - start_time}")


if __name__ == "__main__":
    main()
