from __future__ import annotations
import numpy as np
import pandas as pd

# --- Lightweight helpers (pure functions, no Streamlit calls!) ---

def _slope(series: pd.Series, lookback: int) -> float:
    """Return simple slope over the last N points (approx trend strength)."""
    if len(series) < max(2, lookback):
        return 0.0
    y = series.iloc[-lookback:].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    # least squares slope
    denom = (x - x.mean()).var() * len(x)
    if denom == 0:
        return 0.0
    slope = ((x - x.mean()) * (y - y.mean())).sum() / denom
    return float(slope)

def detect_trend(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Very simple trend detector based on close and MA200.
    Returns dict: {state, strength, details}
    """
    if df.empty or "close" not in df or "ma200" not in df:
        return {"state": "unknown", "strength": 0.0, "details": "not enough data"}

    close = df["close"]
    ma200 = df["ma200"]

    above_ma = close.iloc[-1] > (ma200.iloc[-1] if not pd.isna(ma200.iloc[-1]) else np.inf)
    slope_close = _slope(close, min(lookback, len(close)))
    slope_ma = _slope(ma200.dropna(), min(lookback, ma200.notna().sum())) if ma200.notna().any() else 0.0

    score = (1.5 if above_ma else -1.5) + 3.0 * np.tanh(5 * slope_close) + 1.0 * np.tanh(5 * slope_ma)
    state = "bullish" if score > 0.6 else "bearish" if score < -0.6 else "neutral"

    return {
        "state": state,
        "strength": float(np.clip(score / 3.5, -1, 1)),
        "details": f"above_ma={above_ma}, slope_close={slope_close:.4f}, slope_ma={slope_ma:.4f}",
    }

def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Naive RSI divergence check: compares swing highs/lows in price vs RSI.
    Returns dict: {type: 'bullish'|'bearish'|'none', confidence}
    """
    if df.empty or "close" not in df or "rsi14" not in df:
        return {"type": "none", "confidence": 0.0}

    sub = df.iloc[-lookback:]
    price = sub["close"].to_numpy(dtype=float)
    rsi = sub["rsi14"].to_numpy(dtype=float)

    # pick last two extrema (very crude)
    p_min_idx = np.argmin(price)
    p_max_idx = np.argmax(price)
    r_min_idx = np.argmin(rsi)
    r_max_idx = np.argmax(rsi)

    bullish = p_min_idx < len(price) - 1 and r_min_idx < len(rsi) - 1 and price[-1] < price[p_min_idx] and rsi[-1] > rsi[r_min_idx]
    bearish = p_max_idx < len(price) - 1 and r_max_idx < len(rsi) - 1 and price[-1] > price[p_max_idx] and rsi[-1] < rsi[r_max_idx]

    if bullish and not bearish:
        return {"type": "bullish", "confidence": 0.6}
    if bearish and not bullish:
        return {"type": "bearish", "confidence": 0.6}
    return {"type": "none", "confidence": 0.0}

def detect_volume_spike(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> dict:
    """
    Detects if the latest volume is a spike compared to rolling mean.
    Returns dict: {spike: bool, ratio}
    """
    if df.empty or "volume" not in df:
        return {"spike": False, "ratio": 0.0}
    v = df["volume"].astype(float)
    if len(v) < window + 1:
        return {"spike": False, "ratio": 0.0}
    mean = v.iloc[-(window+1):-1].mean()
    latest = v.iloc[-1]
    ratio = float(latest / mean) if mean > 0 else 0.0
    return {"spike": ratio >= threshold, "ratio": ratio}
