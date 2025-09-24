"""
CLI utility to fetch OHLCV via CCXT and write CSVs in the project's schema.
Usage example:
    poetry run python -m src.cli.fetch_ccxt --exchange binance --symbol BTC/USDT --timeframe 1d --since 2023-01-01
"""
from __future__ import annotations
import argparse
from src.data_api.ccxt_ingestor import FetchSpec, fetch_ohlcv, save_csv

def main():
    p = argparse.ArgumentParser(description="Fetch OHLCV via CCXT and save to CSV in canonical schema.")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--since", default=None, help="ISO8601, e.g., 2023-01-01")
    p.add_argument("--until", default=None, help="ISO8601, default now")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--out", default=None, help="Optional custom CSV path")
    args = p.parse_args()

    spec = FetchSpec(
        exchange_id=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=args.since,
        until=args.until,
        limit=args.limit,
        save_path=args.out
    )
    df = fetch_ohlcv(spec)
    path = save_csv(df, spec)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    main()
