import pandas as pd
from src.utils.metrics import roi, max_drawdown

def test_roi_mdd():
    eq = pd.Series([100, 110, 105, 120])
    assert round(roi(eq), 4) == 0.2
    assert max_drawdown(eq) <= 0.0
