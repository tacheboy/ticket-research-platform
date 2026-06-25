"""Pure-Python / numpy technical-indicator math.

Kept dependency-free (no pandas) so the scoring core is deterministic, fast, and
trivially unit-testable in the eval suite. yfinance/pandas live only in the live
data provider.
"""

from __future__ import annotations

import numpy as np


def sma(closes: list[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    return float(np.mean(closes[-window:]))


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Classic Wilder-style RSI on closing prices."""
    if len(closes) < period + 1:
        return None
    diffs = np.diff(np.asarray(closes, dtype=float))
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def macd_histogram(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    """Return the latest MACD histogram value (MACD line - signal line)."""
    if len(closes) < slow + signal:
        return None
    arr = np.asarray(closes, dtype=float)
    macd_line = _ema(arr, fast) - _ema(arr, slow)
    signal_line = _ema(macd_line, signal)
    return float(macd_line[-1] - signal_line[-1])


def pct_return(closes: list[float], lookback_days: int) -> float | None:
    """Simple return over the last ``lookback_days`` trading days."""
    if len(closes) <= lookback_days:
        return None
    past = closes[-(lookback_days + 1)]
    if past == 0:
        return None
    return (closes[-1] - past) / past


def annualized_volatility(closes: list[float], window: int = 90) -> float | None:
    """Annualized stdev of daily log returns over the trailing window."""
    if len(closes) < window + 1:
        if len(closes) < 20:
            return None
        window = len(closes) - 1
    arr = np.asarray(closes[-(window + 1):], dtype=float)
    log_returns = np.diff(np.log(arr))
    return float(np.std(log_returns) * np.sqrt(252))


def max_drawdown(closes: list[float]) -> float | None:
    """Largest peak-to-trough decline over the series (negative number)."""
    if len(closes) < 2:
        return None
    arr = np.asarray(closes, dtype=float)
    running_max = np.maximum.accumulate(arr)
    drawdowns = (arr - running_max) / running_max
    return float(np.min(drawdowns))
