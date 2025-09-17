import pandas as pd
from src.backtest.engine import SingleAssetBacktester, BacktestConfig

def test_backtester_runs():
    dates = pd.date_range("2022-01-01", periods=10, freq="D")
    prices = pd.Series([100,101,102,103,100,98,99,101,102,103], index=dates)
    cfg = BacktestConfig()
    bt = SingleAssetBacktester(prices, cfg)
    pos = pd.Series(1.0, index=dates)
    res = bt.run(pos)
    assert "equity" in res and len(res["equity"]) == 10
