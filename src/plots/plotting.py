from __future__ import annotations
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def plot_equity(equity: pd.Series, title: str = "Equity") -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    equity.plot(ax=ax)
    ax.set_title(title); ax.grid(True); ax.set_xlabel("Date"); ax.set_ylabel("Equity")
    return fig_to_base64(fig)

def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> str:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig, ax = plt.subplots(figsize=(8, 2.5))
    dd.plot(ax=ax, color="red")
    ax.set_title(title); ax.grid(True); ax.set_xlabel("Date"); ax.set_ylabel("DD")
    return fig_to_base64(fig)

def plot_weights(weights: pd.Series, title: str = "Weights") -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    weights.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title(title); ax.grid(True); ax.set_ylabel("Weight")
    return fig_to_base64(fig)

def plot_corr_heatmap(returns: pd.DataFrame, title: str = "Corr") -> str:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(4 + 0.25 * corr.shape[0], 4))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)), labels=corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig_to_base64(fig)

def plot_forecast(
    actual: pd.Series,
    pred: pd.Series,
    title: str = "ARIMA Forecast"
) -> str:
    """
    Draws actual vs predicted (aligned on pred index).
    Assumes pred.index is a subset/suffix of actual.index.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    actual.plot(ax=ax, linewidth=1.2, label="Actual")
    pred.plot(ax=ax, linewidth=1.2, linestyle="--", label="Forecast")
    ax.legend()
    ax.set_title(title); ax.grid(True); ax.set_xlabel("Date")
    return fig_to_base64(fig)
