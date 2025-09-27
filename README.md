# CRYPTO-AI-HEDGE-FUND

This project implements a modular research and trading framework for cryptocurrency hedge fund strategies.  
It combines **classical econometric models**, **machine learning predictors**, and **AI-driven portfolio management agents** with risk controls.

---

## Project Structure

```
CRYPTO-AI-HEDGE-FUND/
├── docker/                # Docker configuration
│   ├── Dockerfile
│   └── entrypoint.sh
├── notebooks/             # Jupyter notebooks for experiments
│   └── run_crypto_hf.ipynb
├── src/                   # Source code
│   ├── agents/            # AI portfolio agents & risk modules
│   │   ├── pm_agent.py
│   │   ├── risk_guardian.py
│   │   └── runtime.py
│   ├── api/               # API server for running strategies
│   │   └── server.py
│   ├── backtest/          # Backtesting engine
│   │   └── engine.py
│   ├── cli/               # CLI utilities
│   │   └── fetch_ccxt.py
│   ├── data_api/          # Data ingestion (CCXT, CSVs)
│   │   └── ccxt_ingestor.py
│   ├── evaluation/        # Evaluation & reporting
│   │   └── report.py
│   ├── models/            # Econometric & ML models
│   │   ├── econometric.py
│   │   └── ml.py
│   ├── plots/             # Visualization
│   │   └── plotting.py
│   ├── portfolio/         # Portfolio optimizers & rebalancing
│   │   ├── optimizers.py
│   │   └── rebalance.py
│   ├── strategies/        # Baseline strategies
│   │   └── baseline.py
│   └── utils/             # Helper functions
│       ├── data.py
│       └── metrics.py
├── tests/                 # Unit tests
│   ├── test_backtest.py
│   └── test_metrics.py
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Running with Docker

The project provides two main ways to run experiments:  
- **Interactive research mode (Jupyter Notebook)**  
- **Production API mode**

### 1) Build the Docker image

```bash
docker build -t crypto-ai-hf -f docker/Dockerfile .
```

### 2) Run in Jupyter Notebook mode

Launch a Jupyter Lab server to explore strategies:

```bash
docker run -it --rm -p 8888:8888 crypto-ai-hf notebook
```

Then open the provided URL in your browser and start with  
`notebooks/run_crypto_hf.ipynb`.

### 3) Run in API mode

Start the API server to query strategies programmatically:

```bash
docker run -it --rm -p 8000:8000 crypto-ai-hf api
```

The API will be available at:  
[http://localhost:8000](http://localhost:8000)  
Interactive docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Example API requests

### Health check

```bash
curl http://localhost:8000/health
```

### Fetch OHLCV data

```bash
curl "http://localhost:8000/ohlcv?exchange=binance&symbol=BTC/USDT&timeframe=1d&since=2021-01-01"
```

### Level 3 — Static portfolio (MV & HRP)

```bash
curl "http://localhost:8000/level3/portfolio/static?exchange=binance&symbols=BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT&timeframe=1d"
```

### Level 4 — Dynamic rebalancing

```bash
curl "http://localhost:8000/level4/portfolio/dynamic?exchange=binance&symbols=BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,ADA/USDT&timeframe=1d&rebalance_freq=ME"
```

### Level 5 — Agent + Guardian on large universe

```bash
curl "http://localhost:8000/level5/agent?exchange=binance&timeframe=1d&since=2021-01-01&target_n=120"
```
---

## Key Components

- **Baseline strategies** – Moving averages (SMA), etc.  
- **Econometric models** – ARIMA, VAR for return forecasting.  
- **Machine Learning** – Ridge regression, LightGBM predictors.  
- **Portfolio optimization** – Mean-Variance (MV), Hierarchical Risk Parity (HRP).  
- **Dynamic rebalancing** – Time-based and adaptive weight updates.  
- **AI Agent (PMAgent)** – Reinforcement-style decision module for portfolio allocation.  
- **Risk Guardian** – Runtime safety net enforcing drawdown and volatility limits.  
- **Backtesting engine** – Single/multi-asset simulations with slippage & fees.

---

## Testing

Run unit tests inside the Docker container:

```bash
docker run -it --rm crypto-ai-hf pytest
```

---

## License

This project is released under the terms of the Apache 2.0 License.
