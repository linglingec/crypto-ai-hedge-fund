from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import ccxt
import pandas as pd
from dateutil import parser as dateparser
from pathlib import Path

CANON_COLS = ["date", "symbol", "open", "high", "low", "close", "volume"]

@dataclass
class FetchSpec:
    exchange_id: str = "binance"
    symbol: str = "BTC/USDT"         # ccxt формат
    timeframe: str = "1d"
    since: Optional[str] = None      # ISO8601
    until: Optional[str] = None
    limit: int = 1000
    save_path: Optional[str] = None  # если None ⇒ data/real/<...>.csv

def _exchange(spec: FetchSpec):
    if not hasattr(ccxt, spec.exchange_id):
        raise ValueError(f"Unknown exchange: {spec.exchange_id}")
    ex = getattr(ccxt, spec.exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    if spec.timeframe not in ex.timeframes:
        raise ValueError(f"Timeframe {spec.timeframe} not supported by {spec.exchange_id}")
    if spec.symbol not in ex.symbols:
        raise ValueError(f"Symbol {spec.symbol} not found on {spec.exchange_id}")
    return ex

def _to_ms(dt_iso: Optional[str]) -> Optional[int]:
    if dt_iso is None: return None
    return int(dateparser.parse(dt_iso).timestamp() * 1000)

def canon_symbol(ccxt_symbol: str) -> str:
    return ccxt_symbol.replace("/", "").upper()  # "BTC/USDT" -> "BTCUSDT"

def fetch_ohlcv(spec: FetchSpec) -> pd.DataFrame:
    ex = _exchange(spec)
    since_ms = _to_ms(spec.since) or ex.parse8601("2017-01-01T00:00:00Z")
    until_ms = _to_ms(spec.until) or int(time.time() * 1000)

    all_rows = []
    tf_sec = ex.parse_timeframe(spec.timeframe)
    step_ms = tf_sec * 1000 * spec.limit

    while since_ms < until_ms:
        batch = ex.fetch_ohlcv(spec.symbol, timeframe=spec.timeframe, since=since_ms, limit=spec.limit)
        if not batch: break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        since_ms = max(last_ts + 1, since_ms + step_ms)

    if not all_rows:
        return pd.DataFrame(columns=CANON_COLS)

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df["symbol"] = canon_symbol(spec.symbol)
    df = df[CANON_COLS].dropna().sort_values("date").reset_index(drop=True)
    return df

def default_path(spec: FetchSpec) -> Path:
    return Path("data/real") / f"{spec.exchange_id}_{canon_symbol(spec.symbol)}_{spec.timeframe}.csv"

def save_csv(df: pd.DataFrame, spec: FetchSpec) -> str:
    out = Path(spec.save_path) if spec.save_path else default_path(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out, parse_dates=["date"])
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        merged.to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)
    return str(out)
