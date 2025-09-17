"""
Data utilities: loaders, synthetic generation, train/test time split.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, List

def load_prices_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["symbol","date"]).reset_index(drop=True)

def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot(index="date", columns="symbol", values="close").sort_index()

def train_test_split_time(df: pd.DataFrame, test_days: int = 180) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = df["date"].drop_duplicates().sort_values() if "date" in df.columns else df.index.to_series().sort_values()
    cutoff = dates.iloc[-test_days]
    if "date" in df.columns:
        return df[df["date"]<=cutoff].copy(), df[df["date"]>cutoff].copy()
    return df[df.index<=cutoff].copy(), df[df.index>cutoff].copy()

def select_symbols(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    return df[df["symbol"].isin(symbols)].copy()

def simulate_prices(n_days=730, symbols=None, seed=42) -> pd.DataFrame:
    """GBM-like with common market factor, idiosyncratic noise, rare jumps."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","MATICUSDT","DOGEUSDT","DOTUSDT","LTCUSDT"]
    n = len(symbols)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    market = rng.normal(0, 0.03, size=n_days)
    betas = rng.normal(1.0, 0.2, size=n)
    idio = rng.uniform(0.02, 0.06, size=n)
    jumps = np.zeros(n_days)
    jump_days = rng.choice(n_days, size=max(3, n_days//60), replace=False)
    jumps[jump_days] = rng.normal(0, 0.15, size=len(jump_days))
    base_prices = np.linspace(30000, 50, n)  # just spaced
    rows = []
    for i, sym in enumerate(symbols):
        r = betas[i]*market + rng.normal(0, idio[i], size=n_days) + 0.2*jumps
        r = np.clip(r, -0.25, 0.25)
        px = [float(base_prices[i])]
        for t in range(1, n_days): px.append(px[-1]*(1+r[t]))
        px = np.array(px)
        close = px; openp = np.concatenate([[px[0]], px[:-1]])
        high = np.maximum(openp, close)*(1+rng.uniform(0,0.01,size=n_days))
        low  = np.minimum(openp, close)*(1-rng.uniform(0,0.01,size=n_days))
        vol  = (1e6/px)*(1+rng.normal(0,0.2,size=n_days)); vol = np.maximum(vol, 1e3)
        rows.append(pd.DataFrame({"date":dates,"symbol":sym,"open":openp,"high":high,"low":low,"close":close,"volume":vol}))
    return pd.concat(rows, ignore_index=True)

def ensure_synthetic_data(root: str | Path):
    root = Path(root)
    (root/"data").mkdir(parents=True, exist_ok=True)
    small_syms = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","XRPUSDT","MATICUSDT","DOGEUSDT","DOTUSDT","LTCUSDT"]
    large_syms = [f"C{i:03d}USDT" for i in range(1,121)]
    small = root/"data"/"prices_small.csv"
    large = root/"data"/"prices_large.csv"
    if not small.exists():
        simulate_prices(730, small_syms, seed=7).to_csv(small, index=False)
    if not large.exists():
        simulate_prices(730, large_syms, seed=13).to_csv(large, index=False)
