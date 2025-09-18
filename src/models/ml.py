"""
Machine learning models and feature engineering for single-asset prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor

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
    """
    Construct simple technical feature set: lagged returns, rolling volatility,
    momentum windows, and RSI.
    """
    px = prices.astype(float)
    ret = px.pct_change()

    feats = pd.DataFrame(index=px.index)
    feats["ret_1"] = ret
    for k in range(2, lags + 1):
        feats[f"ret_{k}"] = ret.shift(k - 1)

    feats["vol_10"] = ret.rolling(10).std()
    feats["vol_20"] = ret.rolling(20).std()
    feats["mom_10"] = px.pct_change(10)
    feats["mom_20"] = px.pct_change(20)
    feats["rsi_14"] = compute_rsi(px, 14)

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats

def make_target(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Next-period simple return over the given horizon.
    """
    return prices.shift(-horizon) / prices - 1.0

def train_ml_model(prices: pd.Series, cfg: MLConfig) -> Dict[str, object]:
    """
    Train two models: Ridge (linear) and LightGBM (tree boosting).
    LightGBM params mirror the provided classifier settings, adapted for regression.
    Returns:
        dict with fitted models, test predictions, y_test, and MSEs.
    """
    X = make_features(prices, cfg.lags)
    y = make_target(prices, cfg.target_horizon)
    data = pd.concat([X, y.rename("target")], axis=1).dropna()

    # Time-based out-of-sample split (last 180 observations)
    train_idx = data.index[:-180]
    test_idx = data.index[-180:]
    Xtr, ytr = data.loc[train_idx].drop(columns=["target"]), data.loc[train_idx]["target"]
    Xte, yte = data.loc[test_idx].drop(columns=["target"]), data.loc[test_idx]["target"]

    models: Dict[str, object] = {}

    # Ridge baseline
    ridge = Ridge(alpha=1.0, random_state=cfg.random_state)
    ridge.fit(Xtr, ytr)
    pred_ridge = pd.Series(ridge.predict(Xte), index=Xte.index)
    models["ridge"] = ridge

    # LightGBM regressor
    lgbm = LGBMRegressor(
        n_estimators=500,
        boosting_type="gbdt",
        objective="regression",
        metric="l2",
        subsample=0.5,
        subsample_freq=1,
        learning_rate=0.02,
        feature_fraction=0.75,
        max_depth=6,
        lambda_l1=1.0,
        lambda_l2=1.0,
        min_data_in_leaf=50,
        random_state=cfg.random_state,
        n_jobs=8
    )
    lgbm.fit(Xtr, ytr)
    pred_lgbm = pd.Series(lgbm.predict(Xte), index=Xte.index)
    models["gbr"] = lgbm

    return {
        "models": models,
        "pred_test": {"ridge": pred_ridge, "gbr": pred_lgbm},
        "y_test": yte,
        "mse": {
            "ridge": float(mean_squared_error(yte, pred_ridge)),
            "gbr": float(mean_squared_error(yte, pred_lgbm))
        }
    }

def to_positions(pred_ret: pd.Series, threshold: float = 0.0, max_leverage: float = 1.0) -> pd.Series:
    """
    Map predicted returns to a position in [-1, 1] with a deadzone threshold.
    """
    pos = (pred_ret > threshold).astype(float) - (pred_ret < -threshold).astype(float)
    return pos.clip(-1, 1) * max_leverage
