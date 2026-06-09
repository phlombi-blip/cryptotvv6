# ai/commentary.py

def market_commentary(symbol, timeframe, trend, rsi_divergence, volatility):
    """
    Generiert eine kompakte, menschlich klingende Marktanalyse.
    Wird vom KI-CoPilot automatisch genutzt.
    """

    txt = []

    # Header
    txt.append(f"**{symbol} ({timeframe}) – Automatische Chart-Analyse**")

    # Trend-Bewertung
    if trend == "up":
        txt.append("• Der Markt zeigt aktuell einen **Aufwärtstrend**.")
    elif trend == "down":
        txt.append("• Der Markt befindet sich in einem **Abwärtstrend**.")
    else:
        txt.append("• Der Markt bewegt sich derzeit **seitwärts** ohne klaren Trend.")

    # RSI-Divergenz
    if rsi_divergence == "bullish":
        txt.append("• Es liegt eine **bullische RSI-Divergenz** vor → mögliches Trend-Reversal nach oben.")
    elif rsi_divergence == "bearish":
        txt.append("• Achtung: **bärische RSI-Divergenz** → Risiko eines Downmoves steigt.")
    else:
        txt.append("• Keine auffällige RSI-Divergenz erkennbar.")

    # Volumen/Volatilität
    if volatility == "spike":
        txt.append("• Ein starker **Volatility Spike** wurde erkannt → Markt wirkt kurzfristig nervös.")
    elif volatility == "calm":
        txt.append("• Die Volatilität ist niedrig → Markt ist ruhig/stabil.")
    else:
        txt.append("• Normale Volatilität – keine Auffälligkeiten.")

    # Zusammenfassung
    txt.append("\n**Kurzfazit:**")
    if trend == "up" and rsi_divergence != "bearish":
        txt.append("Der Markt ist überwiegend positiv – Pullbacks könnten Kaufgelegenheiten sein.")
    elif trend == "down" and rsi_divergence != "bullish":
        txt.append("Momentan eher vorsichtig – Trend weist nach unten.")
    else:
        txt.append("Gemischte Signale – aktuell weder klar bullisch noch bärisch.")

    return "\n".join(txt)
