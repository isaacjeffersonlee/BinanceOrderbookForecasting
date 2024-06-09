import pandas as pd
import numpy as np
import torch
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


def test(k, model_str, use_orderflow, console):
    ϵ = 1e-15  # Make zero change class as small as possible
    if model_str == "deepLOB":
        use_orderflow = False
    elif model_str == "deepOF":
        use_orderflow = True

    test_pipeline = eval(f"{model_str}.test_pipeline")

    batch_size = 64
    train_interval_ms = 48 * 60 * 60 * 1000
    val_interval_ms = 24 * 60 * 60 * 1000
    test_interval_ms = 24 * 60 * 60 * 1000
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
        # df["y"] = mid_price.pct_change(k).shift(-k)
        df.dropna(inplace=True)
        # df["y"] = utils.discretize(df["y"], ϵ)
        if use_orderflow:
            orderflow_df = pd.DataFrame()
            console.log("df LOB -> df OF...")
            for j in range(1, 10 + 1):
                aOF, bOF = utils.orderflow(
                    ask_price=df[f"ask_price_{j}"],
                    bid_price=df[f"bid_price_{j}"],
                    ask_qty=df[f"ask_qty_{j}"],
                    bid_qty=df[f"bid_qty_{j}"],
                )
                orderflow_df[f"aOF_{j}"] = aOF
                orderflow_df[f"bOF_{j}"] = bOF

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
            console.log(f"Processing window {i+1} / {len(window_boundaries)}...")
            X_train, y_train, _, _, X_test, y_test = get_window(df, wb)
            console.log(f"X_test shape: {X_test.shape}")

            # Standardization
            μ = X_train.mean(dim=0).unsqueeze(dim=0)
            σ = X_train.std(dim=0).unsqueeze(dim=0)
            X_test = (X_test - μ) / σ
            ϵ = 0.5 * (np.abs(np.quantile(y_train, 0.33)) + np.quantile(y_train, 0.66))
            y_test = torch.tensor(utils.discretize(y_test, ϵ))
            console.print(f"y_test distribution, with ϵ_{k}_{i}={ϵ}:")
            console.print(pd.Series(y_test).value_counts())
            save_path = f"test/k={k}/{model_str}/{'OF' if use_orderflow else 'LOB'}/{symbol}_window_{i}"
            y_test, y_test_pred = test_pipeline(
                X_test=X_test, y_test=y_test, save_path=save_path, batch_size=batch_size, k=k
            )
            if use_orderflow and "OF" not in model_str:
                method_str = "OF"
            elif not use_orderflow and "LOB" not in model_str:
                method_str = "LOB"
            else:
                method_str = ""
            console.rule(f"{model_str}{method_str} k={k} Window {i} Results")
            clfr_str = classification_report(y_test, y_test_pred, digits=4)
            console.print(clfr_str)
            console.rule()

            clfr_dict = classification_report(y_test, y_test_pred, output_dict=True)
            clfr_save_path = f"./classification_reports/{save_path}.pkl"
            if not os.path.exists(clfr_save_path):
                os.makedirs(os.path.dirname(clfr_save_path), exist_ok=True)

            with open(clfr_save_path, "wb") as f:
                pickle.dump(clfr_dict, f)


def main():
    console = Console()
    start_time = pd.Timestamp.now()
    for k in (4, 10, 50, 200):
        console.rule(f"Testing models for k={k}")
        for model_str, use_orderflow in (
            ("xgb", True),
            ("xgb", False),
            ("lr", True),
            ("lr", False),
            ("deepOF", True),
            ("deepLOB", False),
        ):
            test(k, model_str, use_orderflow, console)

    end_time = pd.Timestamp.now()
    console.rule("FINISHED TESTING!")
    console.print(f"Total testing time: {end_time - start_time}")


if __name__ == "__main__":
    main()
