"""
Vectorized single-asset backtester with fees, slippage, and optional stops.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_equity: float = 100000.0
    allow_short: bool = False
    stop_loss: float | None = None      # e.g., 0.1  = -10% cap per period
    take_profit: float | None = None    # e.g., 0.2  = +20% cap per period

class SingleAssetBacktester:
    def __init__(self, prices: pd.Series, cfg: BacktestConfig):
        self.prices = prices.dropna().astype(float)
        self.cfg = cfg

    def run(self, target_position: pd.Series) -> dict:
        p = self.prices.align(target_position, join="left")[0].ffill()
        pos = target_position.reindex(p.index).fillna(0.0)
        ret = p.pct_change().fillna(0.0)
        trades = pos.diff().abs().fillna(pos.abs())
        cost = trades * (self.cfg.fee_bps + self.cfg.slippage_bps) / 10000.0
        if self.cfg.stop_loss is not None:  ret = ret.clip(lower=-abs(self.cfg.stop_loss))
        if self.cfg.take_profit is not None: ret = ret.clip(upper= abs(self.cfg.take_profit))
        strat_ret = pos.shift().fillna(0.0) * ret - cost
        equity = (1.0 + strat_ret).cumprod() * self.cfg.initial_equity
        return {"prices": p, "position": pos, "returns": strat_ret, "equity": equity, "costs": cost}
