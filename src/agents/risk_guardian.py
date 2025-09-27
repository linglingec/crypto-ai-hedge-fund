"""
Risk Guardian Agent: conformal stops/leverage & volatility-aware scaling.
"""
from __future__ import annotations
import numpy as np, pandas as pd

class RiskGuardian:
    def __init__(self, alpha: float = 0.9, max_leverage: float = 1.0):
        self.alpha = alpha
        self.max_leverage = max_leverage

    def conformal_threshold(self, ret_hist: pd.Series) -> float:
        r = pd.Series(ret_hist).dropna().values
        if len(r) < 50:
            return 0.05
        q = np.quantile(r, 1 - self.alpha)
        return float(abs(q))

    def adjust_positions(self, positions: pd.Series, price_series: pd.Series) -> pd.Series:
        """
        Scale *all* positions by a single scalar computed from recent volatility of the provided series.
        - positions: vector of weights/leverage (index = assets)
        - price_series: time-indexed series (equity or any representative price) to estimate recent volatility
        """
        rets = price_series.pct_change().dropna()
        if rets.empty:
            return positions.clip(-self.max_leverage, self.max_leverage)

        vol20 = rets.rolling(20).std().dropna()
        if vol20.empty:
            return positions.clip(-self.max_leverage, self.max_leverage)

        thr = self.conformal_threshold(rets.tail(250))
        scale_series = (thr / (vol20 + 1e-12)).clip(0.3, 1.0)
        scale = float(scale_series.iloc[-1])  # используем последнюю оценку
        return (positions * scale).clip(-self.max_leverage, self.max_leverage)
