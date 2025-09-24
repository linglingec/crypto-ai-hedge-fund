from __future__ import annotations
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import orjson
import pandas as pd
from pathlib import Path

from src.data_api.ccxt_ingestor import FetchSpec, fetch_ohlcv, save_csv, default_path, canon_symbol
from src.strategies.baseline import sma_crossover
from src.backtest.engine import SingleAssetBacktester, BacktestConfig
from src.evaluation.report import summarize_single, summarize_portfolio
from src.models.ml import MLConfig, train_ml_model, to_positions
from src.models.econometric import arima_forecast
from src.portfolio.optimizers import mean_variance_opt, hrp_weights
from src.portfolio.rebalance import time_rebalance
from src.plots.plotting import plot_equity, plot_drawdown, plot_weights, plot_corr_heatmap, plot_forecast
from src.agents.pm_agent import PMAgent
from src.agents.risk_guardian import RiskGuardian 
from src.agents.runtime import AgentRunConfig, GuardianConfig, run_agent_portfolio

def _json(data) -> JSONResponse:
    return JSONResponse(orjson.dumps(data), media_type="application/json")

app = FastAPI(title="Crypto AI HF API", version="0.2.0")

# -------------------------- Common endpoints --------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ohlcv")
def ohlcv(exchange: str="binance", symbol: str="BTC/USDT", timeframe: str="1d",
          since: Optional[str]=None, until: Optional[str]=None, limit: int=1000, save: bool=True):
    spec = FetchSpec(exchange, symbol, timeframe, since, until, limit)
    df = fetch_ohlcv(spec)
    path = save_csv(df, spec) if save else None
    preview = df.tail(3).to_dict(orient="records")
    return {"saved_to": path, "rows": len(df), "preview_tail": preview}

def _load_close_series(exchange: str, symbol: str, timeframe: str) -> pd.Series:
    path = default_path(FetchSpec(exchange, symbol, timeframe))
    if not Path(path).exists():
        raise HTTPException(404, f"CSV not found: {path}. Call /ohlcv first.")
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["close"].astype(float)
    s.name = canon_symbol(symbol)
    return s

def _load_universe(exchange: str, symbols: List[str], timeframe: str) -> pd.DataFrame:
    series = [_load_close_series(exchange, sym, timeframe) for sym in symbols]
    px = pd.concat(series, axis=1).dropna(how="any")
    return px

# -------------------------- Level 1: Baseline --------------------------
@app.get("/level1/baseline")
def level1_baseline(exchange: str="binance", symbol: str="BTC/USDT", timeframe: str="1d",
                    fast: int=20, slow: int=60):
    px = _load_close_series(exchange, symbol, timeframe)
    pos = sma_crossover(px, fast=fast, slow=slow)
    bt = SingleAssetBacktester(px, BacktestConfig(fee_bps=5, slippage_bps=5, initial_equity=100000.0))
    res = bt.run(pos)
    rpt = summarize_single(res["equity"], res["returns"])

    eq_png = plot_equity(res["equity"], title=f"Equity {canon_symbol(symbol)}")
    dd_png = plot_drawdown(res["equity"])
    return {"metrics": rpt, "equity_png": eq_png, "drawdown_png": dd_png}

# ------------------- Level 2: Econometric/ML/Agent --------------------
@app.get("/level2/ml")
def level2_ml(exchange: str="binance", symbol: str="BTC/USDT", timeframe: str="1d",
              lags: int=20, horizon: int=1, threshold: float=0.0):
    px = _load_close_series(exchange, symbol, timeframe)
    ml_out = train_ml_model(px, MLConfig(target_horizon=horizon, lags=lags))
    yte = ml_out["y_test"]

    # Ridge positions
    pos_ridge = to_positions(ml_out["pred_test"]["ridge"], threshold=threshold)
    bt = SingleAssetBacktester(px.loc[yte.index], BacktestConfig(fee_bps=5, slippage_bps=5, initial_equity=100000.0))
    res_r = bt.run(pos_ridge)

    # LGBM positions
    pos_gbr = to_positions(ml_out["pred_test"]["gbr"], threshold=threshold)
    res_g = bt.run(pos_gbr)

    rpt_r = summarize_single(res_r["equity"], res_r["returns"])
    rpt_g = summarize_single(res_g["equity"], res_g["returns"])

    return {
        "pred_mse": ml_out["mse"],
        "ridge": {"metrics": rpt_r, "equity_png": plot_equity(res_r["equity"], "Equity Ridge")},
        "lgbm":  {"metrics": rpt_g, "equity_png": plot_equity(res_g["equity"], "Equity LGBM")},
    }

@app.get("/level2/econometric/arima")
def level2_econometric_arima(
    exchange: str = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    horizon: int = 1,
    train_days: int = 900,
    threshold: float = 0.0
):
    """
    Single-asset ARIMA forecast with backtest on out-of-sample.
    Uses your arima_forecast adapter; if it returns prices forecast -> convert to returns.
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    px = _load_close_series(exchange, symbol, timeframe)

    # time split: train last N days for out-of-sample evaluation
    if train_days >= len(px):
        raise HTTPException(400, "train_days is too large for the available history.")
    split_idx = -int(train_days)
    train = px.iloc[:split_idx]
    test = px.iloc[split_idx:]

    try:
        pred = arima_forecast(px, horizon=horizon)
    except TypeError:
        pred = arima_forecast(train, horizon=horizon)

    pred = pd.Series(pred, index=px.index[-len(pred):]) if not isinstance(pred, pd.Series) else pred
    pred = pred.loc[test.index]

    ret_true = test.pct_change(horizon).shift(-horizon).dropna()
    if pred.std() > 0 and abs(pred.std() - test.std())/test.std() < 0.5:
        pred_ret = pred.pct_change(horizon).shift(-horizon)
    else:
        pred_ret = pred.copy()

    ytrue, ypred = ret_true.align(pred_ret, join="inner")
    mse = float(mean_squared_error(ytrue, ypred))
    mae = float(mean_absolute_error(ytrue, ypred))

    pos = to_positions(ypred, threshold=threshold)
    bt = SingleAssetBacktester(px.loc[ytrue.index], BacktestConfig(fee_bps=5, slippage_bps=5, initial_equity=100000.0))
    res = bt.run(pos)
    rpt = summarize_single(res["equity"], res["returns"])

    fc_png = plot_forecast(test, pred.reindex(test.index), f"ARIMA Forecast {canon_symbol(symbol)}")

    return {
        "forecast_horizon": horizon,
        "train_days": train_days,
        "mse": mse,
        "mae": mae,
        "metrics_backtest": rpt,
        "forecast_png": fc_png,
        "equity_png": plot_equity(res["equity"], "Equity (ARIMA signals)"),
        "drawdown_png": plot_drawdown(res["equity"])
    }


@app.get("/level2/econometric/var")
def level2_econometric_var(
    exchange: str = "binance",
    symbols: str = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT",
    timeframe: str = "1d",
    lags: int = 3,
    train_days: int = 900,
    max_weight: float = 0.15
):
    """
    Simple VAR-based multi-asset forecast of returns.
    Strategy: positive-mean forecast -> long, negative -> 0; weights normalized and capped.
    """
    import numpy as np
    from statsmodels.tsa.api import VAR

    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    px = _load_universe(exchange, syms, timeframe)
    ret = px.pct_change().dropna()
    if train_days >= len(ret):
        raise HTTPException(400, "train_days is too large for the available history.")

    train = ret.iloc[:-train_days]
    test = ret.iloc[-train_days:]

    model = VAR(train)
    fit = model.fit(lags)

    preds = []
    history = train.copy()
    for t in range(len(test)):
        fc = fit.forecast(y=history.values[-lags:], steps=1)
        preds.append(pd.Series(fc.ravel(), index=ret.columns, name=test.index[t]))
        history = pd.concat([history, test.iloc[[t]]], axis=0)
    pred_df = pd.DataFrame(preds)
    pred_df.index.name = "date"

    pos_pred = pred_df.clip(lower=0.0)
    w_raw = pos_pred.div(pos_pred.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w_capped = w_raw.clip(upper=max_weight)
    w = w_capped.div(w_capped.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    port_ret = (test * w).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()
    rpt = summarize_single(equity, port_ret)

    return {
        "universe": syms,
        "lags": lags,
        "train_days": train_days,
        "metrics": rpt,
        "equity_png": plot_equity(equity, "Equity (VAR signals)"),
        "corr_png": plot_corr_heatmap(train, "Train Corr"),
        "last_weights_png": plot_weights(w.iloc[-1], "Last VAR Weights")
    }

# -------- Level 3: Static portfolio (MV/HRP) on historical data -------
@app.get("/level3/portfolio/static")
def level3_static(exchange: str="binance",
                  symbols: str="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT",
                  timeframe: str="1d"):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    px = _load_universe(exchange, syms, timeframe)
    ret = px.pct_change().dropna()

    mu = ret.mean()
    cov = ret.cov()

    def mv_adapter() -> pd.Series:
        try:
            return mean_variance_opt(mu, cov)
        except TypeError:
            try:
                return mean_variance_opt(ret, cov)
            except TypeError:
                return mean_variance_opt(ret)

    w_mv = mv_adapter()
    w_hrp = hrp_weights(ret)

    eq_mv = (1 + (ret @ w_mv)).cumprod()
    eq_hrp = (1 + (ret @ w_hrp)).cumprod()

    rpt_mv = summarize_single(eq_mv, ret @ w_mv)
    rpt_hrp = summarize_single(eq_hrp, ret @ w_hrp)

    return {
        "mv":  {"weights": w_mv.to_dict(),  "metrics": rpt_mv,
                "equity_png": plot_equity(eq_mv, "Equity MV"),
                "weights_png": plot_weights(w_mv, "MV Weights")},
        "hrp": {"weights": w_hrp.to_dict(), "metrics": rpt_hrp,
                "equity_png": plot_equity(eq_hrp, "Equity HRP"),
                "weights_png": plot_weights(w_hrp, "HRP Weights")},
        "corr_png": plot_corr_heatmap(ret, "Returns Corr"),
    }

# ------------- Level 4: Dynamic rebalancing (time-based) --------------
@app.get("/level4/portfolio/dynamic")
def level4_dynamic(exchange: str="binance",
                   symbols: str="BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT",
                   timeframe: str="1d", rebalance_freq: str="M"):
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    px = _load_universe(exchange, syms, timeframe)
    ret = px.pct_change().dropna()

    eq_curve, weights_last = time_rebalance(
        returns=ret, method="mv", lookback_days=252, freq=rebalance_freq
    )
    rpt = summarize_single(eq_curve, eq_curve.pct_change().fillna(0.0))

    return {
        "metrics": rpt,
        "equity_png": plot_equity(eq_curve, f"Dynamic ({rebalance_freq})"),
        "last_weights_png": plot_weights(weights_last, f"Last Weights ({rebalance_freq})")
    }

# ----------- Level 5: 100+ pairs with dynamic rebalancing -------------
@app.get("/level5/portfolio/expanded")
def level5_expanded(exchange: str="binance", timeframe: str="1d", top_n: int=100,
                    rebalance_freq: str="M"):

    folder = Path("data/real")
    files = list(folder.glob(f"{exchange}_*_{timeframe}.csv"))
    if len(files) == 0:
        raise HTTPException(404, "No CSVs in data/real/. Use /ohlcv to fetch first.")

    vols = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        if {"close","volume"}.issubset(df.columns):
            last = df.tail(90)   # 3 месяца
            dollar = (last["close"] * last["volume"]).sum()
            symbol = f.stem.split("_")[1]
            vols.append((symbol, dollar, f))
    vols.sort(key=lambda x: x[1], reverse=True)
    top = vols[:top_n]

    series = []
    for symbol, _, f in top:
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        s = df["close"].rename(symbol).astype(float)
        series.append(s)
    px = pd.concat(series, axis=1).dropna(how="any")
    ret = px.pct_change().dropna()

    eq_curve, weights_last = time_rebalance(
        returns=ret, method="mv", lookback_days=252, freq=rebalance_freq, max_weight=0.1
    )
    rpt = summarize_single(eq_curve, eq_curve.pct_change().fillna(0.0))

    return {
        "universe": [sym for sym,_,_ in top],
        "metrics": rpt,
        "equity_png": plot_equity(eq_curve, f"Expanded {len(top)} ({rebalance_freq})"),
        "last_weights_png": plot_weights(weights_last, f"Last Weights ({rebalance_freq})")
    }

@app.get("/level5/agent/pm")
def level5_agent_pm(exchange: str="binance",
                    timeframe: str="1d",
                    top_n: int=80,
                    rebalance_freq: str="ME",
                    lookback_days: int=252,
                    max_weight: float=0.10,
                    cash_buffer: float=0.00,
                    mdd_limit: float=0.25,
                    var_limit: float | None = None,
                    vol_target: float | None = None):
    """
    Run PMAgent + RiskGuardian over a large universe with dynamic rebalancing.
    Returns metrics, equity/weights plots, chosen universe and guardian logs.
    """
    from pathlib import Path
    import pandas as pd

    folder = Path("data/real")
    files = list(folder.glob(f"{exchange}_*_{timeframe}.csv"))
    if len(files) == 0:
        raise HTTPException(404, "No CSVs in data/real/. Use /ohlcv or /warmup to fetch first.")

    vols = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"])
        if {"close","volume"}.issubset(df.columns) and len(df) > 30:
            last = df.tail(90)
            dollar = float((last["close"] * last["volume"]).sum())
            symbol = f.stem.split("_")[1]
            vols.append((symbol, dollar, f))
    vols.sort(key=lambda x: x[1], reverse=True)
    top = vols[:top_n]
    if len(top) < 3:
        raise HTTPException(400, "Not enough symbols with data to form a portfolio.")

    series = []
    for symbol, _, f in top:
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")
        s = df["close"].rename(symbol).astype(float)
        series.append(s)
    px = pd.concat(series, axis=1).dropna(how="any")
    if px.shape[1] < 3 or px.shape[0] < 260:
        raise HTTPException(400, "Too little overlapping data. Fetch more history or reduce top_n.")

    ret = px.pct_change().dropna()

    try:
        agent = PMAgent()
    except Exception:
        class _Fallback:
            def propose_weights(self, hist):
                m = hist.add(1).prod() - 1.0
                return (m.clip(lower=0)).fillna(0.0)
        agent = _Fallback()

    try:
        guardian = RiskGuardian(mdd_limit=mdd_limit, var_limit=var_limit, vol_target=vol_target)
    except Exception:
        guardian = None

    cfg = AgentRunConfig(lookback_days=lookback_days, freq=rebalance_freq,
                         max_weight=max_weight, cash_buffer=cash_buffer)
    gcfg = GuardianConfig(mdd_limit=mdd_limit, var_limit=var_limit, vol_target=vol_target)

    equity, w_last, logs = run_agent_portfolio(ret, agent, guardian, cfg, gcfg)

    from src.evaluation.report import summarize_single
    from src.plots.plotting import plot_equity, plot_drawdown, plot_weights

    rpt = summarize_single(equity, equity.pct_change().fillna(0.0))

    return {
        "universe": [sym for sym,_,_ in top],
        "metrics": rpt,
        "equity_png": plot_equity(equity, f"Agent Portfolio ({len(top)} assets)"),
        "drawdown_png": plot_drawdown(equity),
        "last_weights_png": plot_weights(w_last, "Last Agent Weights"),
        "guardian_logs": logs[-12:],
    }

@app.get("/warmup")
def warmup(
    exchange: str = "binance",
    symbols: str = "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT",
    timeframe: str = "1d",
    since: str = "2023-01-01",
    limit: int = 1000
):
    from src.data_api.ccxt_ingestor import FetchSpec, fetch_ohlcv, save_csv
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    saved = []
    for sym in syms:
        spec = FetchSpec(exchange, sym, timeframe, since, None, limit)
        df = fetch_ohlcv(spec)
        path = save_csv(df, spec)
        saved.append({"symbol": sym, "rows": int(df.shape[0]), "path": path})
    return {"count": len(saved), "saved": saved}
