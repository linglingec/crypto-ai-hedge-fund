"""
Rebalancing policies: time-based, threshold-based, and signal-based.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def time_based_schedule(dates: pd.DatetimeIndex, freq: str = "30D") -> pd.DatetimeIndex:
    ts = pd.Series(1, index=dates)
    return ts.resample(freq).last().index.intersection(dates)

def weight_deviation_trigger(target_weights: pd.DataFrame, threshold: float = 0.05) -> pd.DatetimeIndex:
    applied = pd.Series(0, index=target_weights.columns, dtype=float)
    triggers = []
    for dt, w in target_weights.iterrows():
        if (w - applied).abs().sum() > threshold:
            triggers.append(dt); applied = w
    return pd.DatetimeIndex(triggers)

def signal_based_trigger(signal_series: pd.Series, z: float = 1.0) -> pd.DatetimeIndex:
    s = (signal_series - signal_series.rolling(60).mean()) / (signal_series.rolling(60).std() + 1e-12)
    cond = (s.abs() > z)
    return signal_series.index[cond.fillna(False)]
