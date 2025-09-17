"""
Risk Guardian Agent: conformal stops/leverage & volatility-aware scaling.
"""
from __future__ import annotations
import numpy as np, pandas as pd

class RiskGuardian:
    def __init__(self, alpha: float = 0.9, max_leverage: float = 1.0):
        self.alpha = alpha; self.max_leverage = max_leverage

    def conformal_threshold(self, ret_hist: pd.Series) -> float:
        r = ret_hist.dropna().values
        if len(r) < 50: return 0.05
        q = np.quantile(r, 1 - self.alpha)
        return float(abs(q))

    def adjust_positions(self, positions: pd.Series, price_series: pd.Series) -> pd.Series:
        rets = price_series.pct_change().fillna(0.0)
        vol = rets.rolling(20).std().bfill()
        thr = self.conformal_threshold(rets.tail(250))
        scale = (thr / (vol + 1e-12)).clip(0.3, 1.0)
        return (positions * scale).clip(-self.max_leverage, self.max_leverage)
