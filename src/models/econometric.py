"""
Econometric models: ARIMA wrapper with graceful fallback.
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_STATS = True
except Exception:
    HAVE_STATS = False

def arima_forecast(prices: pd.Series, order=(1,1,1), horizon: int = 1) -> pd.Series:
    px = prices.dropna().astype(float)
    rets = px.pct_change().dropna()
    pred = pd.Series(index=px.index, dtype=float)
    if not HAVE_STATS:
        pred.iloc[:] = rets.rolling(20).mean().reindex(pred.index).fillna(0.0).values
        return pred
    window = 200
    for i in range(window, len(px)-horizon):
        end = px.index[i]
        y = px.loc[:end]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(y, order=order).fit()
                fc = fit.get_forecast(steps=horizon).predicted_mean.iloc[-1]
            last = y.iloc[-1]; r = float(fc/last - 1.0) if last != 0 else 0.0
        except Exception:
            r = float(rets.loc[:end].tail(20).mean()) if len(rets.loc[:end]) else 0.0
        pred.iloc[i+1] = r
    return pred.fillna(0.0)
