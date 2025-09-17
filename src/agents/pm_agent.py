"""
PM Agent: aggregates views and solves portfolio weights (MV, HRP, simplified BL).
"""
from __future__ import annotations
import pandas as pd
from .risk_guardian import RiskGuardian
from ..portfolio.optimizers import mean_variance_opt, hrp_weights, black_litterman_simple

class PMAgent:
    def __init__(self, method: str = "mv", risk_aversion: float = 10.0, bl_tau: float = 0.05):
        self.method = method; self.risk_aversion = risk_aversion; self.bl_tau = bl_tau
        self.guardian = RiskGuardian(alpha=0.9, max_leverage=1.0)

    def allocate(self, price_wide: pd.DataFrame, views: pd.Series | None = None, view_conf: float = 0.3) -> pd.Series:
        ret = price_wide.pct_change().dropna()
        mu, cov = ret.mean()*252.0, ret.cov()*252.0
        if self.method == "mv":
            w = mean_variance_opt(mu, cov, lmbd=self.risk_aversion)
        elif self.method == "hrp":
            w = hrp_weights(ret)
        elif self.method == "bl":
            if views is None: views = mu * 0.0
            w = black_litterman_simple(mu, cov, views, tau=self.bl_tau, omega="diag", risk_aversion=self.risk_aversion)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return w / (w.abs().sum() + 1e-12)
