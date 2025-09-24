from __future__ import annotations
import argparse
from src.data_api.ccxt_ingestor import FetchSpec, fetch_ohlcv, save_csv

def main():
    p = argparse.ArgumentParser(description="Fetch OHLCV via CCXT and save to canonical CSV.")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    spec = FetchSpec(a.exchange, a.symbol, a.timeframe, a.since, a.until, a.limit, a.out)
    df = fetch_ohlcv(spec)
    path = save_csv(df, spec)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    main()
