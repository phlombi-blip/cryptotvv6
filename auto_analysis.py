# auto_analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import pandas as pd

TrendLabel = Literal["stark bullisch", "bullisch", "seitwärts", "bärisch", "stark bärisch", "unklar"]
RsiLabel = Literal["überkauft", "leicht überkauft", "neutral", "leicht überverkauft", "überverkauft"]
VolLabel = Literal["sehr niedrig", "niedrig", "normal", "hoch", "sehr hoch"]


@dataclass
class AutoAnalysisResult:
    trend: TrendLabel
    rsi: RsiLabel
    rsi_divergence: str | None
    volatility: VolLabel
    bullets: List[str]
    short_summary: str


def _classify_trend(df: pd.DataFrame) -> TrendLabel:
    if df.empty:
        return "unklar"

    last = df.iloc[-1]

    # Wir schauen uns ca. die letzten 80 Kerzen an
    window = min(len(df), 80)
    if window < 30:
        return "unklar"

    sub = df["close"].iloc[-window:]
    x = np.arange(len(sub))

    # Falls wirklich gar keine Bewegung drin ist
    if sub.std() == 0:
        slope = 0.0
    else:
        # lineare Regression für Trendrichtung
        slope, _ = np.polyfit(x, sub.values, 1)

    # normierte Steigung (~% über das Fenster)
    pct_slope = slope / sub.mean() * window

    ema20 = last.get("ema20", np.nan)
    ema50 = last.get("ema50", np.nan)
    ma200 = last.get("ma200", np.nan)

    bullish_stack = ema20 > ema50 > ma200
    bearish_stack = ema20 < ema50 < ma200

    # Kombination aus EMA-Stacking + Steigung
    if bullish_stack and pct_slope > 0.04:
        return "stark bullisch"
    if bullish_stack and pct_slope > 0.015:
        return "bullisch"
    if bearish_stack and pct_slope < -0.04:
        return "stark bärisch"
    if bearish_stack and pct_slope < -0.015:
        return "bärisch"
    if abs(pct_slope) < 0.01:
        return "seitwärts"

    return "unklar"


def _classify_rsi(last_rsi: float) -> RsiLabel:
    if np.isnan(last_rsi):
        return "neutral"
    if last_rsi >= 75:
        return "überkauft"
    if last_rsi >= 65:
        return "leicht überkauft"
    if last_rsi <= 25:
        return "überverkauft"
    if last_rsi <= 35:
        return "leicht überverkauft"
    return "neutral"


def _detect_rsi_divergence(df: pd.DataFrame) -> str | None:
    # Simple Divergenz-Logik auf den letzten ~80 Kerzen
    if df.empty or "rsi14" not in df.columns:
        return None

    window = min(len(df), 80)
    if window < 25:
        return None

    sub = df.iloc[-window:]
    closes = sub["close"].values
    rsis = sub["rsi14"].values

    def local_extrema(series, comp):
        idxs = []
        for i in range(1, len(series) - 1):
            if comp(series[i], series[i - 1]) and comp(series[i], series[i + 1]):
                idxs.append(i)
        return idxs

    highs_price = local_extrema(closes, lambda a, b: a >= b)
    lows_price = local_extrema(closes, lambda a, b: a <= b)
    highs_rsi = local_extrema(rsis, lambda a, b: a >= b)
    lows_rsi = local_extrema(rsis, lambda a, b: a <= b)

    # Bärische Divergenz: Higher High im Preis, Lower High im RSI
    if len(highs_price) >= 2 and len(highs_rsi) >= 2:
        hp1, hp2 = highs_price[-2], highs_price[-1]
        hr1, hr2 = highs_rsi[-2], highs_rsi[-1]
        if closes[hp2] > closes[hp1] and rsis[hr2] < rsis[hr1]:
            return "bärische RSI-Divergenz (Higher High im Preis, Lower High im RSI)"

    # Bullische Divergenz: Lower Low im Preis, Higher Low im RSI
    if len(lows_price) >= 2 and len(lows_rsi) >= 2:
        lp1, lp2 = lows_price[-2], lows_price[-1]
        lr1, lr2 = lows_rsi[-2], lows_rsi[-1]
        if closes[lp2] < closes[lp1] and rsis[lr2] > rsis[lr1]:
            return "bullische RSI-Divergenz (Lower Low im Preis, Higher Low im RSI)"

    return None


def _classify_volatility(df: pd.DataFrame) -> VolLabel:
    if df.empty:
        return "normal"
    if not {"bb_up", "bb_lo"}.issubset(df.columns):
        return "normal"

    sub = df.tail(120)  # ca. letzte 120 Kerzen
    bw = (sub["bb_up"] - sub["bb_lo"]) / sub["close"]
    current = bw.iloc[-1]
    median = bw.median()

    if median == 0 or np.isnan(current) or np.isnan(median):
        return "normal"

    ratio = current / median

    if ratio >= 2.0:
        return "sehr hoch"
    if ratio >= 1.4:
        return "hoch"
    if ratio <= 0.5:
        return "sehr niedrig"
    if ratio <= 0.8:
        return "niedrig"
    return "normal"


def build_auto_analysis(df: pd.DataFrame, symbol_label: str, tf_label: str) -> AutoAnalysisResult:
    if df.empty:
        return AutoAnalysisResult(
            trend="unklar",
            rsi="neutral",
            rsi_divergence=None,
            volatility="normal",
            bullets=["Keine Daten vorhanden – API / Internet prüfen."],
            short_summary="Keine Bewertung möglich, da keine Kursdaten vorliegen.",
        )

    last = df.iloc[-1]
    trend = _classify_trend(df)
    rsi_label = _classify_rsi(last.get("rsi14", np.nan))
    div = _detect_rsi_divergence(df)
    vol = _classify_volatility(df)

    bullets: List[str] = []

    # Trend
    if trend == "stark bullisch":
        bullets.append("Der Markt befindet sich in einem **starken Aufwärtstrend** (EMA-Stacking & steigender Verlauf).")
    elif trend == "bullisch":
        bullets.append("Der Markt zeigt einen **klar bullischen Trend**, aber ohne extremes Momentum.")
    elif trend == "seitwärts":
        bullets.append("Der Markt bewegt sich aktuell **seitwärts** ohne klaren Trend.")
    elif trend == "bärisch":
        bullets.append("Der Markt zeigt einen **bärischen Trend** mit zunehmendem Abwärtsdruck.")
    elif trend == "stark bärisch":
        bullets.append("Der Markt befindet sich in einem **starken Abwärtstrend** (EMA-Stacking nach unten).")
    else:
        bullets.append("Der Trend ist derzeit **unklar / gemischt**.")

    # RSI-Level
    rsi_val = last.get("rsi14", np.nan)
    if not np.isnan(rsi_val):
        if rsi_label == "überkauft":
            bullets.append(f"RSI bei **{rsi_val:.1f}** → klar **überkauft** (hohes Rücksetzer-Risiko).")
        elif rsi_label == "leicht überkauft":
            bullets.append(f"RSI bei **{rsi_val:.1f}** → leicht **überkauft**.")
        elif rsi_label == "überverkauft":
            bullets.append(f"RSI bei **{rsi_val:.1f}** → klar **überverkauft** (Rebound wahrscheinlicher).")
        elif rsi_label == "leicht überverkauft":
            bullets.append(f"RSI bei **{rsi_val:.1f}** → leicht **überverkauft**.")
        else:
            bullets.append(f"RSI bei **{rsi_val:.1f}** → im **neutralen Bereich**.")
    else:
        bullets.append("RSI konnte nicht berechnet werden (zu wenig Daten).")

    # Divergenzen
    if div:
        bullets.append(f"⚠️ **RSI-Divergenz erkannt:** {div}.")
    else:
        bullets.append("Keine eindeutige **RSI-Divergenz** in den letzten Kerzen erkennbar.")

    # Volatilität
    if vol == "sehr hoch":
        bullets.append("Volatilität aktuell **sehr hoch** – starke Ausschläge, erhöhtes Risiko.")
    elif vol == "hoch":
        bullets.append("Volatilität aktuell **überdurchschnittlich hoch**.")
    elif vol == "niedrig":
        bullets.append("Volatilität aktuell **unterdurchschnittlich niedrig** – eher ruhiger Markt.")
    elif vol == "sehr niedrig":
        bullets.append("Volatilität aktuell **sehr niedrig** – Markt in enger Spanne.")
    else:
        bullets.append("Volatilität im **normalen Bereich** – keine Auffälligkeiten.")

    # Kurzfazit aus Trend + RSI
    if trend in ["stark bullisch", "bullisch"] and rsi_label in ["neutral", "leicht überkauft", "leicht überverkauft"]:
        summary = "Leicht bullische Gesamtsituation – Rücksetzer sind möglich, aber Trend bleibt bisher intakt."
    elif trend in ["stark bullisch", "bullisch"] and rsi_label in ["überkauft"]:
        summary = "Bullischer Markt, aber klar überkauft – kurzfristig erhöhte Korrekturgefahr."
    elif trend in ["stark bärisch", "bärisch"] and rsi_label in ["neutral", "leicht überkauft", "leicht überverkauft"]:
        summary = "Eher bärische Gesamtsituation – Erholungen können kurzfristig auftreten, dominanter Trend zeigt nach unten."
    elif trend in ["stark bärisch", "bärisch"] and rsi_label in ["überverkauft"]:
        summary = "Stark bärischer Markt, aber überverkauft – kurzfristiger Bounce möglich, übergeordneter Trend bleibt schwach."
    elif trend == "seitwärts":
        summary = "Seitwärtsmarkt – aktuell weder klar bullisch noch bärisch, eher Range-Trading."
    else:
        summary = "Gemischte Signale – kein eindeutiger Vorteil für Bullen oder Bären."

    return AutoAnalysisResult(
        trend=trend,
        rsi=rsi_label,
        rsi_divergence=div,
        volatility=vol,
        bullets=bullets,
        short_summary=summary,
    )
