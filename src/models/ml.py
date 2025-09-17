"""
ML models and features for single-asset prediction.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

@dataclass
class MLConfig:
    target_horizon: int = 1
    lags: int = 20
    random_state: int = 42

def compute_rsi(prices: pd.Series, window: int) -> pd.Series:
    delta = prices.diff()
    up = (delta.where(delta > 0, 0.0)).rolling(window).mean()
    down = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def make_features(prices: pd.Series, lags: int = 20) -> pd.DataFrame:
    px = prices.astype(float); ret = px.pct_change()
    feats = pd.DataFrame(index=px.index)
    feats["ret_1"] = ret
    for k in range(2, lags+1): feats[f"ret_{k}"] = ret.shift(k-1)
    feats["vol_10"] = ret.rolling(10).std(); feats["vol_20"] = ret.rolling(20).std()
    feats["mom_10"] = px.pct_change(10); feats["mom_20"] = px.pct_change(20)
    feats["rsi_14"] = compute_rsi(px, 14)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def make_target(prices: pd.Series, horizon: int = 1) -> pd.Series:
    return prices.shift(-horizon) / prices - 1.0

def train_ml_model(prices: pd.Series, cfg: MLConfig):
    X = make_features(prices, cfg.lags); y = make_target(prices, cfg.target_horizon)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()
    train_idx, test_idx = data.index[:-180], data.index[-180:]
    Xtr, ytr = data.loc[train_idx].drop(columns=["target"]), data.loc[train_idx]["target"]
    Xte, yte = data.loc[test_idx].drop(columns=["target"]), data.loc[test_idx]["target"]
    models = {}
    ridge = Ridge(alpha=1.0, random_state=cfg.random_state).fit(Xtr, ytr)
    pred_ridge = pd.Series(ridge.predict(Xte), index=Xte.index); models["ridge"] = ridge
    gbr = GradientBoostingRegressor(random_state=cfg.random_state).fit(Xtr, ytr)
    pred_gbr = pd.Series(gbr.predict(Xte), index=Xte.index); models["gbr"] = gbr
    return {"models": models, "pred_test": {"ridge": pred_ridge, "gbr": pred_gbr}, "y_test": yte,
            "mse": {"ridge": float(mean_squared_error(yte, pred_ridge)),
                    "gbr": float(mean_squared_error(yte, pred_gbr))}}

def to_positions(pred_ret: pd.Series, threshold: float = 0.0, max_leverage: float = 1.0) -> pd.Series:
    pos = (pred_ret > threshold).astype(float) - (pred_ret < -threshold).astype(float)
    return pos.clip(-1, 1) * max_leverage
