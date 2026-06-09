# backtest.py
import numpy as np
import pandas as pd

def compute_backtest_trades(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    closes = df["close"].values
    signals = df["signal"].values
    idx = df.index

    for i in range(len(df) - horizon):
        sig = signals[i]
        if sig not in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            continue

        entry = closes[i]
        exit_ = closes[i + horizon]
        direction = 1 if sig in ["BUY", "STRONG BUY"] else -1

        ret_pct = (exit_ - entry) / entry * 100 * direction
        rows.append({
            "entry_time": idx[i],
            "exit_time": idx[i + horizon],
            "signal": sig,
            "entry_price": entry,
            "exit_price": exit_,
            "ret_pct": ret_pct,
            "correct": ret_pct > 0,
        })

    return pd.DataFrame(rows)


def summarize_backtest(df_bt: pd.DataFrame):
    if df_bt.empty:
        return {}

    summary = {
        "total_trades": len(df_bt),
        "overall_avg_return": df_bt["ret_pct"].mean(),
        "overall_hit_rate": df_bt["correct"].mean() * 100,
    }

    per = []
    for sig in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        sub = df_bt[df_bt["signal"] == sig]
        if sub.empty: continue
        per.append({
            "Signal": sig,
            "Trades": len(sub),
            "Avg Return %": sub["ret_pct"].mean(),
            "Hit Rate %": sub["correct"].mean() * 100,
        })

    summary["per_type"] = per
    return summary