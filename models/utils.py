import pandas as pd
import numpy as np


def orderflow(ask_price, bid_price, ask_qty, bid_qty):
    Δask_price, Δbid_price = np.diff(ask_price), np.diff(bid_price)
    Δask_qty, Δbid_qty = np.diff(ask_qty), np.diff(bid_qty)
    aOF = Δask_qty * (Δask_price >= 0).astype(int) + ask_qty[1:] * -np.sign(Δask_price)
    aOF.name = "aOF"
    bOF = Δbid_qty * (Δbid_price <= 0).astype(int) + bid_qty[1:] * np.sign(Δbid_price)
    bOF.name = "bOF"
    return aOF, bOF


def discretize(v: pd.Series, eps: float) -> pd.Series:
    y = np.ones_like(v)
    y[v > eps] = 2
    y[(-eps <= v) & (v <= eps)] = 1
    y[v < -eps] = 0
    return y.astype(int)
