"""
Runtime adapters to run PMAgent + RiskGuardian in a rolling, rebalance-based loop.
Works with diverse agent method signatures via duck-typing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

@dataclass
class AgentRunConfig:
    lookback_days: int = 252
    freq: str = "ME"          # month-end
    max_weight: float = 0.10   # per asset cap
    cash_buffer: float = 0.00  # keep in cash (0..1)

@dataclass
class GuardianConfig:
    mdd_limit: float = 0.25      # kill-switch if peak-to-valley drop exceeds (e.g. 0.25=25%)
    var_limit: Optional[float] = None  # daily 95% historical VaR cap (e.g. 0.03)
    vol_target: Optional[float] = None # annualized vol target (e.g. 0.20); scales weights

def _cap_norm(w: pd.Series, max_weight: float, cash_buffer: float) -> pd.Series:
    w = w.clip(lower=0.0, upper=max_weight)
    s = w.sum()
    w = w / s if s > 1e-12 else pd.Series(1.0 / len(w), index=w.index)
    if cash_buffer > 0:
        w = w * (1.0 - cash_buffer)
    return w

def _to_series(x, cols) -> pd.Series:
    if isinstance(x, pd.Series): return x.reindex(cols).fillna(0.0)
    return pd.Series(np.array(x, dtype=float), index=cols)

def _agent_weights(agent, hist: pd.DataFrame, cols) -> pd.Series:
    """
    Try common call shapes:
      propose_weights(hist) or propose_weights(returns=hist) or step(hist) etc.
    Returns pd.Series aligned to 'cols'.
    """
    if hasattr(agent, "propose_weights"):
        try:
            w = agent.propose_weights(hist)
        except TypeError:
            w = agent.propose_weights(returns=hist)
    elif hasattr(agent, "step"):
        try:
            w = agent.step(hist)
        except TypeError:
            w = agent.step(returns=hist)
    else:
        # fallback: simple momentum x vol inverse
        mom = hist.add(1).prod() - 1.0
        ivol = 1.0 / hist.std().replace(0, np.nan)
        w = (mom.clip(lower=0) * ivol).fillna(0)
    return _to_series(w, cols)

def _apply_guardian(guardian, equity: pd.Series, w: pd.Series,
                    hist: pd.DataFrame, gc: GuardianConfig) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    If RiskGuardian present, try to apply; otherwise conservative local checks.
    """
    log = {}
    if guardian is not None:
        try:
            res = guardian.apply(weights=w, equity=equality, returns=hist)  # might not match
        except Exception:
            # try more generic names
            try:
                res = guardian.guard(w, equity, hist)
            except Exception:
                res = None
        if isinstance(res, dict):
            w = _to_series(res.get("weights", w), w.index)
            log.update({k: v for k, v in res.items() if k != "weights"})

    # built-in safety net
    # 1) MDD kill-switch -> move to cash
    if gc.mdd_limit is not None and len(equity) > 2:
        peak = equity.cummax()
        mdd = (equity / peak - 1.0).min()
        if mdd <= -abs(gc.mdd_limit):
            w = w * 0.0
            log["kill_switch_mdd"] = float(mdd)

    # 2) VaR cap (historical daily 95%)
    if gc.var_limit is not None and len(hist) > 60:
        port = (hist @ w).dropna()
        if len(port) > 60:
            var95 = -np.percentile(port.values, 5)
            log["hist_var_95"] = float(var95)
            if var95 > gc.var_limit:
                scale = gc.var_limit / var95
                w = w * float(scale)
                log["var_scale"] = float(scale)

    # 3) Vol targeting (annualized)
    if gc.vol_target is not None and len(hist) > 60:
        port = (hist @ w).dropna()
        ann_vol = port.std() * np.sqrt(252)
        log["ann_vol"] = float(ann_vol)
        if ann_vol > 1e-6:
            scale = gc.vol_target / ann_vol
            w = w * float(scale)
            log["vol_scale"] = float(scale)

    # re-normalize to <=1
    s = w.sum()
    if s > 1.0:
        w = w / s
        log["renorm"] = True
    return w, log

def run_agent_portfolio(returns: pd.DataFrame, agent, guardian=None,
                        cfg: AgentRunConfig = AgentRunConfig(),
                        gcfg: GuardianConfig = GuardianConfig()):
    """
    Rolling monthly (or chosen freq) run: each rebalance compute agent weights on rolling window.
    Returns:
      equity: pd.Series
      last_weights: pd.Series
      logs: list of dict by rebalance date
    """
    ret = returns.sort_index().copy().astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if ret.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    alias = cfg.freq
    period_end = ret.resample(alias).last().index
    period_end = period_end.intersection(ret.index)
    if len(period_end) == 0:
        period_end = pd.Index([ret.index[-1]])

    equity = pd.Series(index=ret.index, dtype=float)
    equity.iloc[0] = 1.0
    w_curr = pd.Series(1.0 / ret.shape[1], index=ret.columns)
    last_w = w_curr.copy()
    logs = []

    prev = ret.index[0]
    for rb in period_end:
        lb_idx = ret.index.get_loc(rb)
        hist = ret.iloc[max(0, lb_idx - cfg.lookback_days):lb_idx]
        if hist.shape[0] >= max(20, int(cfg.lookback_days * 0.5)):
            try:
                w_raw = _agent_weights(agent, hist, ret.columns)
            except Exception:
                # fallback if agent explodes
                mom = hist.add(1).prod() - 1.0
                w_raw = (mom.clip(lower=0)).fillna(0.0)
            w_curr = _cap_norm(w_raw, cfg.max_weight, cfg.cash_buffer)

            equity_so_far = equity.loc[:prev].dropna()
            w_curr, glog = _apply_guardian(guardian, equity_so_far, w_curr, hist, gcfg)
            glog["rebalance_date"] = rb.isoformat()
            logs.append(glog)

        seg = ret.loc[prev:rb]
        port = (seg @ w_curr).astype(float)
        equity.loc[seg.index] = equity.loc[prev] * (1.0 + port).cumprod()
        last_w = w_curr.copy()
        prev = rb

    if prev < ret.index[-1]:
        seg = ret.loc[prev:]
        port = (seg @ w_curr).astype(float)
        equity.loc[seg.index] = equity.loc[prev] * (1.0 + port).cumprod()

    equity.name = "equity"
    last_w.name = "weights"
    return equity.dropna(), last_w, logs
