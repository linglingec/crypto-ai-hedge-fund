# Crypto AI Hedge Fund

Agent-based automated cryptocurrency trading and risk management implementation across five levels.
The project is modular, fully reproducible (Poetry/uv/Docker), and driven from a single Jupyter notebook.

This repository ships with synthetic OHLCV data for offline reproducibility. To run on real data, place CSV files with the same schema into `data/`.

## Objectives

1. Baseline strategy for a single cryptocurrency (BTCUSDT).
2. Single-asset strategies using econometric models, machine learning models, and an AI risk agent.
3. Static portfolio management on 5–7 coins (historical data, last 12 months).
4. Dynamic portfolio rebalancing.
5. Dynamic portfolio management on 100+ coins.

All evaluations include performance and risk metrics on out-of-sample data. The code is modular and reproducible.

## Project structure

```
.
├─ README.md
├─ pyproject.toml
├─ Dockerfile
├─ data/                       # Synthetic CSVs (auto-created if missing)
├─ notebooks/
│  └─ Part2_Crypto_HF.ipynb    # Single end-to-end notebook (Levels 1–5)
├─ src/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ data.py               # loaders, synthetic generator, time split
│  │  └─ metrics.py            # ROI, Sharpe, Sortino, MaxDD, Calmar, VaR/CVaR, turnover
│  ├─ backtest/
│  │  └─ engine.py             # vectorized single-asset backtester (fees, slippage, stops)
│  ├─ strategies/
│  │  └─ baseline.py           # SMA crossover baseline
│  ├─ models/
│  │  ├─ econometric.py        # ARIMA wrapper (with fallback)
│  │  └─ ml.py                 # features, Ridge/GBR training, to_positions
│  ├─ agents/
│  │  ├─ risk_guardian.py      # conformal stops/leverage, volatility-aware scaling
│  │  └─ pm_agent.py           # PM agent: MV, HRP, simplified Black–Litterman
│  ├─ portfolio/
│  │  ├─ optimizers.py         # MV, HRP, simplified BL
│  │  └─ rebalance.py          # time-/threshold-/signal-based rebalancing APIs
│  └─ evaluation/
│     └─ report.py             # metric aggregation
└─ tests/
   ├─ test_metrics.py
   └─ test_backtest.py
```

All code comments and docstrings are in English.

## Data schema

CSV columns used in `data/prices_small.csv` and `data/prices_large.csv`:

| column | type     | notes                |
|-------:|----------|----------------------|
| date   | datetime | UTC assumed          |
| symbol | string   | e.g., BTCUSDT        |
| open   | float    |                      |
| high   | float    |                      |
| low    | float    |                      |
| close  | float    | primary price series |
| volume | float    | base-asset volume    |

Synthetic datasets are auto-generated if the files are missing. To use real data, drop CSVs with the same columns into `data/`.

## Installation

### Option A: Poetry
```bash
pip install poetry
poetry install
poetry run jupyter notebook notebooks/Part2_Crypto_HF.ipynb
```

### Option B: uv
```bash
pip install uv
uv pip install -r pyproject.toml
jupyter notebook notebooks/Part2_Crypto_HF.ipynb
```

### Option C: Docker
```bash
docker build -t crypto-hf -f Dockerfile .
docker run -p 8888:8888 crypto-hf
# open http://localhost:8888 and run the notebook
```

## How to run

Open and execute `notebooks/Part2_Crypto_HF.ipynb`.
Artifacts (plots and CSVs) are saved to `notebooks/artifacts/`.

The notebook executes all five levels:

### Level 1 — Baseline (single asset)
- Strategy: SMA(20/60) long/flat on BTCUSDT with fees and slippage.
- Metrics: ROI, Sharpe, Sortino, Max Drawdown, Calmar, VaR(95%)/CVaR(95%).

### Level 2 — Econometric, ML, AI agent (single asset)
- Features: lagged returns, volatility proxies, momentum, RSI.
- Target: next-period return.
- Train/test: time-based split (last 180 days as out-of-sample).
- Models: ARIMA (with fallback), Ridge, Gradient Boosting Regressor.
- AI agent: RiskGuardian scales positions by volatility and conformal threshold.
- Metrics: model MSE; trading metrics as in Level 1.

### Level 3 — Static portfolio (5–7 coins)
- Coins: BTC, ETH, BNB, SOL, ADA, XRP, MATIC.
- Optimizers: Mean–Variance, HRP, simplified Black–Litterman (views from 60D momentum).
- Metrics: portfolio performance and risk metrics.

### Level 4 — Dynamic portfolio rebalancing
- Rebalancing: monthly time-based BL over the last 12 months.
- APIs for threshold- or signal-based rebalancing are provided in `rebalance.py`.

### Level 5 — 100+ coins dynamic management
- Universe: top 100 by dollar volume proxy (price × volume).
- Signals: momentum weighted by inverse volatility for view confidence.
- Risk: liquidity cap (exclude bottom 10%), risk aversion via BL.
- Metrics: portfolio performance and risk metrics.

## Reproducibility

- Deterministic random seeds where applicable.
- Time-based train/test split for out-of-sample evaluation.
- Synthetic data included and generated on demand.
- Poetry or uv lockable environment; Dockerfile provided.
- Single self-contained notebook drives all results.

## Metrics

- Single-asset and portfolio: ROI, Sharpe, Sortino, Max Drawdown, Calmar, VaR(95%), CVaR(95%).
- Portfolio turnover (when applicable).

## Tests

Run unit tests:
```bash
poetry run pytest -q
# or
pytest -q
```

## Notes

- The code is modular and structured to avoid monolithic or spaghetti implementations.
- Replace synthetic CSVs with real data using the same schema to run live-like experiments.
- The econometric, ML, and agent components are minimal and intended for clarity and reproducibility.

## License

For educational use only. No financial advice. Add a license file if required (for example, MIT).
