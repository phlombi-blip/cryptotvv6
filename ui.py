import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from html import escape  # für sichere Tooltips d

# KI-CoPilot Module
from ai.copilot import ask_copilot

from charts import create_price_rsi_figure, create_signal_history_figure
from email_notifier import send_signal_email

# Optional: Auto-Refresh (falls Paket installiert ist)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ---------------------------------------------------------
# BASIS-KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crypto Live + AI CoPilot",
    layout="wide",
)

# Bitfinex Public API (ohne API-Key)
BITFINEX_BASE_URL = "https://api-pub.bitfinex.com/v2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoTV-V5/1.0; +https://streamlit.io)"
}

# Symbole auf Bitfinex
SYMBOLS = {
    "BTC": "tBTCUSD",
    "ETH": "tETHUSD",
    "XRP": "tXRPUSD",
    "SOL": "tSOLUSD",
    "DOGE": "tDOGE:USD",
}

# Anzeige-Labels → interne Timeframes (Bitfinex: 1m..1D)
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",  # Bitfinex schreibt 1D
}

DEFAULT_TIMEFRAME = "1d"
VALID_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

# Wie viele Jahre Historie sollen ungefähr geladen werden?
YEARS_HISTORY = 3.0


def candles_for_history(interval_internal: str, years: float = YEARS_HISTORY) -> int:
    """Rechnet ungefähr aus, wie viele Kerzen für X Jahre gebraucht werden."""
    candles_per_day_map = {
        "1m": 60 * 24,   # 1440
        "5m": 12 * 24,   # 288
        "15m": 4 * 24,   # 96
        "1h": 24,        # 24
        "4h": 6,         # 6
        "1D": 1,         # 1
    }
    candles_per_day = candles_per_day_map.get(interval_internal, 24)
    return int(candles_per_day * 365 * years)


# ---------------------------------------------------------
# THEME CSS
# ---------------------------------------------------------
DARK_CSS = """
<style>
body, .main {
    background-color: #020617;
}
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
.tv-card {
    background: #020617;
    border-radius: 0.75rem;
    border: 1px solid #1f2933;
    padding: 0.75rem 1rem;
}
.tv-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.signal-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
</style>
"""

LIGHT_CSS = """
<style>
body, .main {
    background-color: #F3F4F6;
}
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
.tv-card {
    background: #FFFFFF;
    border-radius: 0.75rem;
    border: 1px solid #E5E7EB;
    padding: 0.75rem 1rem;
}
.tv-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #6B7280;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.signal-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
</style>
"""


# ---------------------------------------------------------
# API FUNKTIONEN – BITFINEX
# ---------------------------------------------------------
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    timeframe = interval  # z.B. "1m", "1h", "1D"
    key = f"trade:{timeframe}:{symbol}"
    url = f"{BITFINEX_BASE_URL}/candles/{key}/hist"

    params = {"limit": limit, "sort": -1}

    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Candles HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        raw = resp.json()
    except ValueError:
        raise RuntimeError(f"Candles: Ungültige JSON-Antwort: {resp.text[:200]}")

    if not isinstance(raw, list) or len(raw) == 0:
        return pd.DataFrame()

    rows = []
    for c in raw:
        # [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        if len(c) < 6:
            continue
        rows.append(
            {
                "open_time": pd.to_datetime(c[0], unit="ms"),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5]),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("open_time")
    df.sort_index(inplace=True)
    return df


@st.cache_data(ttl=60)
def cached_fetch_klines(symbol: str, interval: str, limit: int = 200):
    """Gecachter Candle-Abruf – reduziert Last & Rate-Limits."""
    return fetch_klines(symbol, interval, limit)


def fetch_ticker_24h(symbol: str):
    url = f"{BITFINEX_BASE_URL}/ticker/{symbol}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Ticker HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        d = resp.json()
    except ValueError:
        raise RuntimeError(f"Ticker: Ungültige JSON-Antwort: {resp.text[:200]}")

    if not isinstance(d, (list, tuple)) or len(d) < 7:
        raise RuntimeError(f"Ticker: Unerwartetes Format: {d}")

    last_price = float(d[6])
    change_pct = float(d[5]) * 100.0
    return last_price, change_pct


# ---------------------------------------------------------
# INDIKATOREN
# ---------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA20/EMA50, MA200, Bollinger 20, RSI14.
    """
    if df.empty:
        return df

    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ma200"] = close.rolling(200).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    df["bb_mid"] = sma20
    df["bb_up"] = sma20 + 2 * std20
    df["bb_lo"] = sma20 - 2 * std20

    df["rsi14"] = compute_rsi(close)

    return df


# ---------------------------------------------------------
# SIGNAL-LOGIK (mit Begründung)
# ---------------------------------------------------------
def _signal_core_with_reason(last, prev):
    """
    Kernlogik:
    - Adaptive Bollinger
    - RSI Trend Confirmation
    - Blow-Off-Top Detector
    Liefert (signal, reason).
    """

    close = last["close"]
    prev_close = prev["close"]

    ema50 = last["ema50"]
    ma200 = last["ma200"]

    rsi_now = last["rsi14"]
    rsi_prev = prev["rsi14"]

    bb_up = last["bb_up"]
    bb_lo = last["bb_lo"]
    bb_mid = last["bb_mid"]

    high = last["high"]
    low = last["low"]
    candle_range = high - low
    upper_wick = high - max(close, last["open"])

    # Adaptive Volatility → passt Bollinger-Sensitivität an
    vol = (bb_up - bb_lo) / bb_mid if bb_mid != 0 else 0
    is_low_vol = vol < 0.06
    is_high_vol = vol > 0.12

    # MA200 fehlt → nicht traden
    if pd.isna(ma200):
        return "HOLD", "MA200 noch nicht verfügbar – zu wenig Historie, daher kein Trade."

    # Nur Long-Trading in Bullen-Trends
    if close < ma200:
        return "HOLD", "Kurs liegt unter MA200 – System handelt nur Long im Bullenmarkt."

    # Blow-Off-Top Detector
    blowoff = (
        candle_range > 0
        and upper_wick > candle_range * 0.45
        and close < prev_close
        and close > bb_up
        and rsi_now > 73
    )

    if blowoff:
        return (
            "STRONG SELL",
            "Blow-Off-Top: langer oberer Docht, Kurs über oberem Bollinger-Band "
            "und RSI > 73 mit Umkehrkerze – hohes Top-Risiko."
        )

    # STRONG BUY – tiefer Dip
    deep_dip = (
        close <= bb_lo
        and rsi_now < 35
        and rsi_now > rsi_prev
    )

    if deep_dip:
        if is_low_vol and close < bb_lo * 0.995:
            return (
                "STRONG BUY",
                "Tiefer Dip: Kurs an/unter unterem Bollinger-Band in ruhiger Phase, "
                "RSI < 35 dreht nach oben – aggressiver Rebound-Einstieg."
            )
        return (
            "STRONG BUY",
            "Tiefer Dip: Kurs am unteren Bollinger-Band, RSI < 35 und steigt wieder – "
            "kräftiges Long-Signal."
        )

    # BUY – normale gesunde Pullbacks
    buy_price_cond = (
        close <= bb_lo * (1.01 if is_high_vol else 1.00)
        or close <= ema50 * 0.96
    )

    buy_rsi_cond = (
        30 < rsi_now <= 48
        and rsi_now > rsi_prev
    )

    if buy_price_cond and buy_rsi_cond:
        return (
            "BUY",
            "Gesunder Pullback: Kurs im Bereich unteres Bollinger-Band bzw. leicht unter EMA50, "
            "RSI zwischen 30 und 48 und dreht nach oben."
        )

    # STRONG SELL – extreme Überhitzung
    strong_sell_cond = (
        close > ema50 * 1.12
        and close > bb_up
        and rsi_now > 80
        and rsi_now < rsi_prev
    )

    if strong_sell_cond:
        return (
            "STRONG SELL",
            "Extreme Überhitzung: Kurs deutlich über EMA50 und oberem Bollinger-Band, "
            "RSI > 80 und fällt bereits – starkes Abverkaufsrisiko."
        )

    # SELL – normale Übertreibung
    sell_cond = (
        close > bb_up
        and rsi_now > 72
        and rsi_now < rsi_prev
    )

    if sell_cond:
        return (
            "SELL",
            "Übertreibung: Kurs über dem oberen Bollinger-Band, RSI > 72 und dreht nach unten – "
            "Gewinnmitnahme / Short-Signal."
        )

    # Nichts erkannt
    return "HOLD", "Keine klare Übertreibung oder Dip – System wartet (HOLD)."


def signal_with_reason(last, prev):
    """Neue Schnittstelle: (signal, reason)."""
    return _signal_core_with_reason(last, prev)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wendet signal_with_reason() an und gibt nur neue Signale aus,
    wenn sich die Richtung ändert → keine gespammten Wiederholungssignale.
    Zusätzlich Spalte 'signal_reason'.
    """
    if df.empty or len(df) < 2:
        df["signal"] = "NO DATA"
        df["signal_reason"] = "Nicht genug Daten für ein Signal."
        return df

    signals = []
    reasons = []
    last_sig = "NO DATA"

    for i in range(len(df)):
        if i == 0:
            signals.append("NO DATA")
            reasons.append("Erste Candle – keine Historie für Signalberechnung.")
            continue

        sig_raw, reason_raw = signal_with_reason(df.iloc[i], df.iloc[i - 1])

        # nur neues Signal, wenn Richtung wechselt
        if sig_raw == last_sig:
            sig_display = "HOLD"
            reason_display = f"Signal '{sig_raw}' besteht weiter – kein neues Signal generiert."
        else:
            sig_display = sig_raw
            reason_display = reason_raw

        signals.append(sig_display)
        reasons.append(reason_display)

        if sig_raw in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            last_sig = sig_raw

    df["signal"] = signals
    df["signal_reason"] = reasons
    return df


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------
def latest_signal(df: pd.DataFrame) -> str:
    if "signal" not in df.columns or df.empty:
        return "NO DATA"
    valid = df[df["signal"].isin(VALID_SIGNALS)]
    return valid["signal"].iloc[-1] if not valid.empty else "NO DATA"


def compute_backtest_trades(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Erzeugt eine Backtest-Tabelle:
    entry_time, exit_time, signal, reason, entry_price, exit_price, ret_pct, correct
    """
    if df.empty or "signal" not in df.columns:
        return pd.DataFrame()

    rows = []
    closes = df["close"].values
    signals = df["signal"].values
    idx = df.index

    has_reason = "signal_reason" in df.columns

    for i in range(len(df) - horizon):
        sig = signals[i]
        if sig not in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            continue

        entry = closes[i]
        exit_ = closes[i + horizon]
        if entry == 0:
            continue

        ret = (exit_ - entry) / entry * 100
        direction = 1 if sig in ["BUY", "STRONG BUY"] else -1
        correct = (np.sign(ret) * direction) > 0
        reason = df["signal_reason"].iloc[i] if has_reason else ""

        rows.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[i + horizon],
                "signal": sig,
                "reason": reason,
                "entry_price": entry,
                "exit_price": exit_,
                "ret_pct": float(ret),
                "correct": bool(correct),
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(df_bt: pd.DataFrame):
    if df_bt.empty:
        return {}

    summary = {
        "total_trades": int(len(df_bt)),
        "overall_avg_return": float(df_bt["ret_pct"].mean()),
        "overall_hit_rate": float(df_bt["correct"].mean() * 100),
    }

    per = []
    for sig in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        sub = df_bt[df_bt["signal"] == sig]
        if sub.empty:
            continue
        per.append(
            {
                "Signal": sig,
                "Trades": len(sub),
                "Avg Return %": float(sub["ret_pct"].mean()),
                "Hit Rate %": float(sub["correct"].mean() * 100),
            }
        )

    summary["per_type"] = per
    return summary


def signal_color(signal: str) -> str:
    return {
        "STRONG BUY": "#00C853",
        "BUY": "#64DD17",
        "HOLD": "#9E9E9E",
        "SELL": "#FF5252",
        "STRONG SELL": "#D50000",
        "NO DATA": "#757575",
    }.get(signal, "#9E9E9E")


# ---------------------------------------------------------
# SESSION STATE INITIALISIERUNG
# ---------------------------------------------------------
def init_state():
    st.session_state.setdefault("selected_symbol", "BTC")
    st.session_state.setdefault("selected_timeframe", DEFAULT_TIMEFRAME)
    st.session_state.setdefault("theme", "Dark")
    st.session_state.setdefault("backtest_horizon", 5)
    st.session_state.setdefault("backtest_trades", pd.DataFrame())
    st.session_state.setdefault("copilot_question", "")
    # Zeitraum-Defaults (nur Platzhalter, Widget steuert diese Keys)
    st.session_state.setdefault("date_from", None)
    st.session_state.setdefault("date_to", None)


# ---------------------------------------------------------
# HAUPT UI / STREAMLIT APP
# ---------------------------------------------------------
def main():
    init_state()

    # Auto-Refresh (TradingView Feel)
    if st_autorefresh is not None:
        st_autorefresh(interval=60 * 1000, key="refresh")

    # -----------------------------------------------------
    # SIDEBAR / NAVIGATION – Reihenfolge: Markt, Zeitraum, Backtest, Theme
    # -----------------------------------------------------
    st.sidebar.title("⚙️ Navigation & Einstellungen")

    # 1) Markt
    st.sidebar.markdown("### Markt")
    symbol_label = st.sidebar.selectbox(
        "Aktives Symbol",
        list(SYMBOLS.keys()),
        index=list(SYMBOLS.keys()).index(st.session_state.selected_symbol),
    )
    st.session_state.selected_symbol = symbol_label

    tf_label = st.sidebar.radio(
        "Timeframe",
        list(TIMEFRAMES.keys()),
        index=list(TIMEFRAMES.keys()).index(st.session_state.selected_timeframe),
    )
    st.session_state.selected_timeframe = tf_label

    # 2) Zeitraum – Widget steuert date_from/date_to in Session State
    st.sidebar.markdown("### Zeitraum")
    today = datetime.utcnow().date()
    default_from = st.session_state.get("date_from") or today
    default_to = st.session_state.get("date_to") or today

    st.sidebar.date_input(
        "📅 Von (Datum)",
        value=default_from,
        key="date_from",
    )
    st.sidebar.date_input(
        "📅 Bis (Datum)",
        value=default_to,
        key="date_to",
    )

    # 3) Backtest
    st.sidebar.markdown("### Backtest")
    horizon = st.sidebar.slider(
        "Halte-Dauer (Kerzen)",
        1,
        20,
        value=st.session_state.backtest_horizon,
    )
    st.session_state.backtest_horizon = horizon

    # 4) Theme
    st.sidebar.markdown("### Theme")
    theme = st.sidebar.radio(
        "Darstellung",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
    )
    st.session_state.theme = theme

    # Theme anwenden
    st.markdown(DARK_CSS if theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)

    # Header Bar
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""
        <div class="tv-card" style="margin-bottom: 0.4rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="tv-title">Crypto Live + AI CoPilot</div>
                    <div style="font-size:1.05rem; font-weight:600;">
                        TradingView Style • Desktop • KI-Unterstützung
                    </div>
                </div>
                <div style="text-align:right; font-size:0.8rem; opacity:0.8;">
                    Datenquelle: Bitfinex Spot<br/>
                    Update: {now}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gemeinsame Basis-Variablen
    symbol = SYMBOLS[symbol_label]
    interval_internal = TIMEFRAMES[tf_label]

    # Layout: Links Markt / Charts, Rechts KI-Copilot
    col_left, col_right = st.columns([4, 1], gap="medium")

    # ---------------------------------------------------------
    # WATCHLIST + CHARTS (LINKS)
    # ---------------------------------------------------------
    with col_left:
        # WATCHLIST
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">Watchlist</div>', unsafe_allow_html=True)

            rows = []
            selected_tf_internal = interval_internal
            limit_watch = candles_for_history(selected_tf_internal, years=YEARS_HISTORY)

            for label, sym in SYMBOLS.items():
                try:
                    price, chg_pct = fetch_ticker_24h(sym)
                    try:
                        df_tmp = cached_fetch_klines(sym, selected_tf_internal, limit=limit_watch)
                        df_tmp = compute_indicators(df_tmp)
                        df_tmp = compute_signals(df_tmp)
                        sig = latest_signal(df_tmp)
                    except Exception:
                        sig = "NO DATA"

                    rows.append(
                        {
                            "Symbol": label,
                            "Price": price,
                            "Change %": chg_pct,
                            "Signal": sig,
                        }
                    )
                except Exception:
                    rows.append(
                        {
                            "Symbol": label,
                            "Price": np.nan,
                            "Change %": np.nan,
                            "Signal": "NO DATA",
                        }
                    )

            df_watch = pd.DataFrame(rows).set_index("Symbol")

            def highlight(row):
                theme_local = st.session_state.theme
                if row.name == st.session_state.selected_symbol:
                    bg = "#111827" if theme_local == "Dark" else "#D1D5DB"
                    fg = "white" if theme_local == "Dark" else "black"
                    return [f"background-color:{bg}; color:{fg}"] * len(row)
                return [""] * len(row)

            styled = df_watch.style.apply(highlight, axis=1).format(
                {"Price": "{:,.2f}", "Change %": "{:+.2f}"}
            )

            st.dataframe(styled, use_container_width=True, height=220)

            st.markdown("</div>", unsafe_allow_html=True)

        # CHART-BEREICH
        st.markdown("")
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)

            st.markdown('<div class="tv-title">Chart</div>', unsafe_allow_html=True)

            # Daten abrufen + Zeitraum auf Basis der Sidebar-Werte
            try:
                limit_main = candles_for_history(interval_internal, years=YEARS_HISTORY)

                # komplette Historie laden
                df_all = cached_fetch_klines(symbol, interval_internal, limit=limit_main)

                date_from = None
                date_to = None
                mask = None

                if not df_all.empty:
                    min_date = df_all.index.min().date()
                    max_date = df_all.index.max().date()

                    # Sidebar-Werte holen (können auch außerhalb des Bereichs liegen)
                    date_from = st.session_state.get("date_from") or min_date
                    date_to = st.session_state.get("date_to") or max_date

                    # Falls out-of-range, einfach lokal clampen – Session State NICHT überschreiben
                    if date_from < min_date:
                        date_from = min_date
                    if date_from > max_date:
                        date_from = max_date

                    if date_to < min_date:
                        date_to = min_date
                    if date_to > max_date:
                        date_to = max_date

                    if date_from > date_to:
                        date_from, date_to = date_to, date_from

                    mask = (df_all.index.date >= date_from) & (df_all.index.date <= date_to)

                # Indikatoren & Signale auf kompletter Historie
                if not df_all.empty:
                    df_all_ind = compute_indicators(df_all.copy())
                    df_all_ind = compute_signals(df_all_ind)

                    if mask is not None:
                        df = df_all_ind.loc[mask]
                    else:
                        df = df_all_ind.copy()
                else:
                    df = pd.DataFrame()

                # Kennzahlen / Stati
                if df.empty:
                    sig = "NO DATA"
                    last_price = 0
                    change_abs = 0
                    change_pct = 0
                    last_time = None
                    signal_reason = ""
                    feed_ok = False
                    error_msg = "Keine Daten im gewählten Zeitraum."
                else:
                    sig = latest_signal(df)
                    last = df.iloc[-1]
                    prev = df.iloc[-2]

                    last_price = last["close"]
                    change_abs = last_price - prev["close"]
                    change_pct = (change_abs / prev["close"]) * 100 if prev["close"] != 0 else 0
                    last_time = df.index[-1]
                    signal_reason = last.get("signal_reason", "")
                    feed_ok = True
                    error_msg = ""

                    # E-Mail Push bei Signalwechsel (Gmail)
                    prev_key = f"last_signal_{symbol_label}_{tf_label}"
                    prev_sig = st.session_state.get(prev_key)

                    if prev_sig != sig and sig in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
                        ok_email, msg_email = send_signal_email(
                            previous_signal=prev_sig,
                            new_signal=sig,
                            symbol=symbol_label,
                            timeframe=tf_label,
                            price=last_price,
                            reason=signal_reason,
                            when=last_time,
                        )
                        if ok_email:
                            st.toast(f"📧 Signalwechsel: {prev_sig or 'NONE'} → {sig} – E-Mail gesendet.")
                        else:
                            # Nur Info im UI, keine Exception
                            st.info(
                                f"Signalwechsel erkannt ({prev_sig or 'NONE'} → {sig}), "
                                f"aber E-Mail konnte nicht gesendet werden: {msg_email}"
                            )

                    st.session_state[prev_key] = sig

            except Exception as e:
                df = pd.DataFrame()
                sig = "NO DATA"
                last_price = 0
                change_abs = 0
                change_pct = 0
                last_time = None
                signal_reason = ""
                feed_ok = False
                error_msg = str(e)

            # Top-Kennzahlen
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.caption("Preis")
                st.markdown(f"**{last_price:,.2f} USD**" if feed_ok else "–")

            with k2:
                st.caption("Change letzte Candle")
                if feed_ok:
                    s = "+" if change_abs >= 0 else "-"
                    st.markdown(f"**{s}{abs(change_abs):.2f} ({s}{abs(change_pct):.2f}%)**")
                else:
                    st.markdown("–")

            with k3:
                st.caption("Signal")
                reason_html = escape(signal_reason, quote=True)
                st.markdown(
                    f'<span class="signal-badge" style="background-color:{signal_color(sig)};" '
                    f'title="{reason_html}">{sig}</span>',
                    unsafe_allow_html=True,
                )

            with k4:
                st.caption("Status")
                if feed_ok:
                    st.markdown("🟢 **Live**")
                    if last_time is not None:
                        st.caption(f"Letzte Candle: {last_time}")
                    if date_from and date_to:
                        st.caption(f"Zeitraum: {date_from} bis {date_to}")
                else:
                    st.markdown("🔴 **Fehler**")
                    st.caption(error_msg[:80])

            st.markdown("---")

            # Gemeinsamer Price+RSI-Chart
            if not df.empty:
                fig_price_rsi = create_price_rsi_figure(df, symbol_label, tf_label, theme)
                st.plotly_chart(fig_price_rsi, use_container_width=True)
            else:
                st.warning("Keine Daten im gewählten Zeitraum – Zeitraum anpassen oder API/Internet prüfen.")

            st.markdown("</div>", unsafe_allow_html=True)

        # Signal-History + Backtest
        st.markdown("")
        col_hist, col_bt = st.columns([3, 2])

        # Signal-History Panel
        with col_hist:
            with st.container():
                st.markdown('<div class="tv-card">', unsafe_allow_html=True)
                st.markdown('<div class="tv-title">Signal History</div>', unsafe_allow_html=True)

                if df.empty:
                    st.info("Keine Signale verfügbar.")
                else:
                    allow = st.multiselect(
                        "Signale anzeigen",
                        ["STRONG BUY", "BUY", "SELL", "STRONG SELL"],
                        default=["STRONG BUY", "BUY", "SELL", "STRONG SELL"],
                    )
                    st.plotly_chart(
                        create_signal_history_figure(df, allow, theme),
                        use_container_width=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        # Backtest Panel
        with col_bt:
            with st.container():
                st.markdown('<div class="tv-card">', unsafe_allow_html=True)
                st.markdown('<div class="tv-title">Backtest</div>', unsafe_allow_html=True)

                if df.empty:
                    st.info("Keine Daten.")
                else:
                    horizon = st.session_state.backtest_horizon
                    st.caption(f"Halte-Dauer: **{horizon} Kerzen**")

                    bt = compute_backtest_trades(df, horizon)
                    st.session_state.backtest_trades = bt

                    stats = summarize_backtest(bt)

                    if not stats:
                        st.info("Keine verwertbaren Trades.")
                    else:
                        st.markdown(f"**Trades gesamt:** {stats['total_trades']}")
                        st.markdown(f"**Ø Return:** {stats['overall_avg_return']:.2f}%")
                        st.markdown(f"**Trefferquote:** {stats['overall_hit_rate']:.1f}%")

                        if stats.get("per_type"):
                            st.markdown("---")
                            st.caption("Pro Signal:")
                            st.table(pd.DataFrame(stats["per_type"]))

                st.markdown("</div>", unsafe_allow_html=True)

        # TRADES LIST – MIT CSV EXPORT
        st.markdown("")
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">Trades List (Backtest)</div>', unsafe_allow_html=True)

            bt = st.session_state.backtest_trades

            if bt.empty:
                st.info("Noch keine Trades.")
            else:
                df_show = bt.copy()
                df_show["entry_time"] = df_show["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
                df_show["exit_time"] = df_show["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
                df_show["ret_pct"] = df_show["ret_pct"].map(lambda x: f"{x:.2f}")
                df_show["correct"] = df_show["correct"].map(lambda x: "✅" if x else "❌")

                cols = [
                    "entry_time",
                    "exit_time",
                    "signal",
                    "reason",
                    "entry_price",
                    "exit_price",
                    "ret_pct",
                    "correct",
                ]
                df_show = df_show[[c for c in cols if c in df_show.columns]]

                st.dataframe(df_show, use_container_width=True, height=220)

                csv = bt.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 CSV Export",
                    csv,
                    file_name=f"trades_{symbol_label}_{tf_label}.csv",
                    mime="text/csv",
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # RECHTS: KI-COPILOT (Auto-Analyse = CoPilot)
    # ---------------------------------------------------------
    with col_right:
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">🤖 KI-CoPilot</div>', unsafe_allow_html=True)

            if df.empty:
                st.info("Keine Marktdaten geladen – bitte zuerst einen gültigen Zeitraum wählen.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # Key für Auto-Analyse im Session-State (pro Symbol + Timeframe)
            auto_key = f"copilot_auto_{symbol_label}_{tf_label}"

            def run_auto_analysis():
                """Startet eine automatische CoPilot-Analyse und speichert das Ergebnis im Session State."""
                with st.spinner("CoPilot analysiert den Chart..."):
                    st.session_state[auto_key] = ask_copilot(
                        question=(
                            "Bitte analysiere den aktuellen Chart anhand von RSI(14), EMA20, EMA50, MA200, "
                            "Bollinger-Bändern und der Candlestick-Struktur (Trend, Pullbacks, Übertreibungen). "
                            "Beschreibe zuerst nüchtern die technische Lage. "
                            "Formuliere danach eine mögliche technische Handelsidee basierend auf der "
                            "Marktpsychologie (z.B. FOMO, Panik, Kapitulation, Rebound), mit grober Einstiegszone, "
                            "Stop-Zone und Zielzone – ohne Beträge, Hebel oder Positionsgrößen. "
                            "Weise am Ende klar darauf hin, dass dies keine Anlageberatung ist, sondern nur ein "
                            "hypothetisches Szenario."
                        ),
                        df=df,
                        symbol=symbol_label,
                        timeframe=tf_label,
                        last_signal=sig,
                    )

            # Beim ersten Aufruf für dieses Symbol/TF automatisch Analyse holen
            if auto_key not in st.session_state:
                run_auto_analysis()

            auto_text = st.session_state.get(auto_key, "Noch keine Analyse verfügbar.")

            # Tabs: links Auto-Analyse (CoPilot), rechts Chat
            tab_auto, tab_chat = st.tabs(["📊 Auto-Analyse (CoPilot)", "💬 KI-Chat"])

            # --- TAB 1: Auto-Analyse / Insights (nur CoPilot) ---
            with tab_auto:
                st.markdown(f"**Automatische KI-Analyse ({symbol_label} – {tf_label})**")
            
                if st.button(
                    "🔄 Analyse aktualisieren",
                    key=f"btn_reanalyse_{symbol_label}_{tf_label}",
                ):
                    run_auto_analysis()
            
                auto_text = st.session_state.get(auto_key, "Noch keine Analyse verfügbar.")
                st.write(auto_text)

            # --- TAB 2: Interaktiver KI-Chat ---
            with tab_chat:
                st.markdown("**Frag den CoPilot** – z.B.:")
                st.caption("„Wie würdest du den aktuellen BTC-Chart interpretieren?“")
                st.caption("„Welche Risiken siehst du im aktuellen Setup?“")

                question = st.text_area(
                    "Deine Frage an den KI-CoPilot",
                    value=st.session_state.get("copilot_question", ""),
                    height=80,
                )
                st.session_state.copilot_question = question

                if st.button(
                    "Antwort vom CoPilot holen",
                    key=f"btn_copilot_chat_{symbol_label}_{tf_label}",
                ):
                    if not question.strip():
                        st.warning("Bitte zuerst eine Frage eingeben.")
                    else:
                        with st.spinner("CoPilot denkt nach..."):
                            answer = ask_copilot(
                                question=question,
                                df=df,
                                symbol=symbol_label,
                                timeframe=tf_label,
                                last_signal=sig,
                            )
                        st.markdown("**Antwort:**")
                        st.write(answer)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------
if __name__ == "__main__":
    main()