"""
Metrics utilities for strategy and portfolio evaluation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
EPS = 1e-12

def roi(equity: pd.Series) -> float:
    equity = equity.dropna()
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) >= 2 else 0.0

def drawdown_curve(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    peak = equity.cummax()
    return equity / (peak + EPS) - 1.0

def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_curve(equity).min())

def calmar_ratio(equity: pd.Series, annual_factor: float = 252.0) -> float:
    ret = equity.pct_change().dropna()
    ann_ret = (1 + ret.mean()) ** annual_factor - 1.0
    mdd = abs(max_drawdown(equity))
    return np.inf if mdd < EPS else float(ann_ret / mdd)

def sharpe_ratio(equity: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ret = equity.pct_change().dropna()
    excess = ret - rf
    return 0.0 if excess.std() < EPS else float((excess.mean()*periods_per_year)/(excess.std()*np.sqrt(periods_per_year)))

def sortino_ratio(equity: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    ret = equity.pct_change().dropna()
    excess = ret - rf
    downside = excess[excess < 0.0]
    dd = downside.std()
    return 0.0 if dd < EPS else float((excess.mean()*periods_per_year)/(dd*np.sqrt(periods_per_year)))

def var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float,float]:
    r = returns.dropna().values
    if len(r) == 0: return 0.0, 0.0
    q = np.quantile(r, 1 - alpha)
    var = -q
    cvar = -r[r <= q].mean() if (r <= q).any() else 0.0
    return float(var), float(cvar)

def turnover(weights: pd.DataFrame) -> float:
    if weights.shape[0] < 2: return 0.0
    delta = weights.diff().abs().dropna()
    return float(delta.sum(axis=1).mean())

def hit_rate(trade_returns: pd.Series) -> float:
    r = trade_returns.dropna()
    return 0.0 if len(r)==0 else float((r>0).mean())
