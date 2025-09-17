"""
Optimizers: Mean-Variance, HRP, simplified Black–Litterman.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

def mean_variance_opt(mu: pd.Series, cov: pd.DataFrame, lmbd: float = 10.0) -> pd.Series:
    cols = mu.index
    C = cov.values + np.eye(len(mu))*1e-6
    inv = np.linalg.pinv(C + np.eye(len(mu))*1e-3)
    w = inv @ (mu.values / max(lmbd, 1e-6))
    w = pd.Series(w, index=cols)
    return w / (w.abs().sum() + 1e-12)

def correl_dist(corr: pd.DataFrame) -> np.ndarray:
    return np.sqrt(0.5 * (1 - corr.clip(-1,1)))

def hrp_weights(ret: pd.DataFrame) -> pd.Series:
    corr = ret.corr(); dist = correl_dist(corr)
    Z = linkage(squareform(dist.values, checks=False), method='single')
    order = dendrogram(Z, no_plot=True)['leaves']
    cov = ret.cov(); items = ret.columns[order]
    def alloc(cov, items):
        if len(items)==1: return pd.Series(1.0, index=items)
        split = len(items)//2; left, right = items[:split], items[split:]
        wl, wr = alloc(cov, left), alloc(cov, right)
        varl = wl.values @ cov.loc[left,left].values @ wl.values
        varr = wr.values @ cov.loc[right,right].values @ wr.values
        alpha = 1 - varl/(varl+varr+1e-12)
        return pd.concat([wl*alpha, wr*(1-alpha)])
    w = alloc(cov, list(items))
    return w / w.abs().sum()

def black_litterman_simple(mu: pd.Series, cov: pd.DataFrame, views: pd.Series,
                           tau: float = 0.05, omega: str = "diag", risk_aversion: float = 10.0) -> pd.Series:
    n = len(mu)
    w_eq = pd.Series(1.0/n, index=mu.index)
    pi = risk_aversion * cov.values @ w_eq.values
    pi = pd.Series(pi, index=mu.index)
    Omega = (np.diag(np.diag(cov.values))*tau) if omega=="diag" else (cov.values*tau)
    Cinv = np.linalg.inv(cov.values * tau)
    post = np.linalg.inv(Cinv + np.linalg.inv(Omega)) @ (Cinv @ pi.values + np.linalg.inv(Omega) @ (mu.values + views.values))
    post = pd.Series(post, index=mu.index)
    return mean_variance_opt(post, cov, lmbd=risk_aversion)
