"""
Rebalancing policies: time-based, threshold-based, and signal-based.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

from .optimizers import mean_variance_opt, hrp_weights

RebalanceMethod = Literal["mv", "hrp"]

@dataclass
class RebalanceConfig:
    method: RebalanceMethod = "mv"
    lookback_days: int = 252
    freq: str = "M"                # pandas offset alias: "M", "W", "Q", etc.
    max_weight: Optional[float] = None  # e.g., 0.1 caps single-name weight at 10%

def _cap_and_norm(weights: pd.Series, max_weight: Optional[float]) -> pd.Series:
    w = weights.copy()
    if max_weight is not None:
        w = w.clip(upper=max_weight)
    s = w.sum()
    # если все склипано к нулю, делаем равные веса
    if not np.isfinite(s) or s <= 1e-12:
        w = pd.Series(1.0 / len(w), index=w.index)
    else:
        w = w / s
    return w

def _optimize(returns_win: pd.DataFrame, method: RebalanceMethod) -> pd.Series:
    if method == "mv":
        mu = returns_win.mean()
        cov = returns_win.cov()
        try:
            w = mean_variance_opt(mu, cov)      # (mu, cov)
        except TypeError:
            try:
                w = mean_variance_opt(returns_win, cov)  # (returns, cov)
            except TypeError:
                w = mean_variance_opt(returns_win)       # (returns)
    elif method == "hrp":
        w = hrp_weights(returns_win)
    else:
        raise ValueError(f"Unknown method: {method}")
    if not isinstance(w, pd.Series):
        w = pd.Series(w, index=returns_win.columns)
    return w.astype(float)

def time_rebalance(
    returns: pd.DataFrame,
    method: RebalanceMethod = "mv",
    lookback_days: int = 252,
    freq: str = "M",
    max_weight: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Periodic rebalancing on a rolling window.

    Args:
        returns: DataFrame of daily percent returns (index = dates, columns = symbols).
        method: "mv" (mean-variance) or "hrp".
        lookback_days: number of prior trading days used to estimate weights.
        freq: pandas offset alias for rebalancing ("M" monthly, "W" weekly, etc.).
        max_weight: optional cap per asset (e.g., 0.1).

    Returns:
        equity_curve: pd.Series of cumulative equity (starting at 1.0)
        last_weights: pd.Series of the final weights used at last rebalance
    """
    ret = returns.sort_index().copy().astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if ret.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    period_end = ret.resample(freq).last().index
    period_end = period_end.intersection(ret.index)
    if len(period_end) == 0:
        period_end = pd.Index([ret.index[-1]])

    equity = pd.Series(index=ret.index, dtype=float)
    equity.iloc[0] = 1.0

    current_w = pd.Series(1.0 / ret.shape[1], index=ret.columns)
    last_weights = current_w.copy()

    prev_reb_date = ret.index[0]

    for rb_date in period_end:
        lb_start = ret.index.get_loc(rb_date)
        win_start_idx = max(0, lb_start - lookback_days)
        hist = ret.iloc[win_start_idx:lb_start]

        if hist.shape[0] >= max(20, int(lookback_days * 0.5)):
            try:
                w_raw = _optimize(hist, method=method)
                w_raw = w_raw.reindex(ret.columns).fillna(0.0)
                current_w = _cap_and_norm(w_raw, max_weight)
            except Exception:
                current_w = _cap_and_norm(current_w, max_weight)

        seg = ret.loc[prev_reb_date:rb_date]
        seg_port = (seg @ current_w).astype(float)
        equity.loc[seg.index] = equity.loc[prev_reb_date] * (1.0 + seg_port).cumprod()

        last_weights = current_w.copy()
        prev_reb_date = rb_date

    if prev_reb_date < ret.index[-1]:
        seg = ret.loc[prev_reb_date:]
        seg_port = (seg @ current_w).astype(float)
        equity.loc[seg.index] = equity.loc[prev_reb_date] * (1.0 + seg_port).cumprod()

    equity.name = "equity"
    last_weights.name = "weights"
    return equity.dropna(), last_weights

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
