"""
Baseline SMA crossover strategy for a single pair.
"""
from __future__ import annotations
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 60) -> pd.Series:
    f, s = sma(prices, fast), sma(prices, slow)
    return (f > s).astype(float)  # long/flat
