"""
FastAPI server exposing endpoints to fetch real OHLCV and run quick baseline backtests.
"""
from __future__ import annotations
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import orjson
import ccxt
import pandas as pd

from src.data_api.ccxt_ingestor import FetchSpec, fetch_ohlcv, save_csv
from src.strategies.baseline import sma_crossover
from src.backtest.engine import SingleAssetBacktester, BacktestConfig
from src.evaluation.report import summarize_single

def _json(data) -> JSONResponse:
    return JSONResponse(orjson.dumps(data), media_type="application/json")

app = FastAPI(title="Crypto AI HF API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/exchanges")
def exchanges():
    return {"exchanges": sorted(ccxt.exchanges)}

@app.get("/symbols")
def symbols(exchange: str = Query(..., description="Exchange id, e.g., binance")):
    if not hasattr(ccxt, exchange):
        raise HTTPException(400, f"Unknown exchange: {exchange}")
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ex.load_markets()
    # return top liquid symbols only (spot USDT)
    syms = [s for s in ex.symbols if "/USDT" in s and s.count("/") == 1]
    return {"exchange": exchange, "count": len(syms), "symbols": syms[:2000]}

@app.get("/ohlcv")
def ohlcv(
    exchange: str = Query("binance"),
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("1d"),
    since: Optional[str] = Query(None, description="ISO8601, e.g., 2023-01-01"),
    until: Optional[str] = Query(None, description="ISO8601"),
    limit: int = Query(1000)
):
    spec = FetchSpec(exchange_id=exchange, symbol=symbol, timeframe=timeframe, since=since, until=until, limit=limit)
    try:
        df = fetch_ohlcv(spec)
        path = save_csv(df, spec)
    except Exception as e:
        raise HTTPException(400, f"fetch failed: {e}")
    preview = df.tail(3).to_dict(orient="records")
    return {"saved_to": path, "rows": len(df), "preview_tail": preview}

@app.get("/baseline")
def baseline(
    exchange: str = Query("binance"),
    symbol: str = Query("BTC/USDT"),
    timeframe: str = Query("1d"),
    fast: int = Query(20),
    slow: int = Query(60),
    since: Optional[str] = None,
    until: Optional[str] = None,
):
    spec = FetchSpec(exchange_id=exchange, symbol=symbol, timeframe=timeframe, since=since, until=until)
    try:
        df = fetch_ohlcv(spec)
        # convert to series for backtester
        px = df.set_index("date")["close"].astype(float)
        pos = sma_crossover(px, fast=fast, slow=slow)
        bt = SingleAssetBacktester(px, BacktestConfig(fee_bps=5, slippage_bps=5, initial_equity=100000.0))
        res = bt.run(pos)
        rpt = summarize_single(res["equity"], res["returns"])
    except Exception as e:
        raise HTTPException(400, f"baseline failed: {e}")
    return {"rows": int(df.shape[0]), "metrics": rpt}
