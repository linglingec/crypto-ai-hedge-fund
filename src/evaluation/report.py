"""
Reporting helpers to compute and aggregate metrics.
"""
from __future__ import annotations
import pandas as pd
from ..utils.metrics import roi, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, var_cvar, turnover

def summarize_single(equity: pd.Series, returns: pd.Series) -> dict:
    var95, cvar95 = var_cvar(returns, alpha=0.95)
    return {"ROI": roi(equity),
            "Sharpe": sharpe_ratio(equity),
            "Sortino": sortino_ratio(equity),
            "MaxDD": max_drawdown(equity),
            "Calmar": calmar_ratio(equity),
            "VaR95": var95, "CVaR95": cvar95}

def summarize_portfolio(equity: pd.Series, returns: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    res = summarize_single(equity, returns)
    if weights is not None: res["Turnover"] = turnover(weights)
    return res
