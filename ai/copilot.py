# ai/copilot.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional

import pandas as pd
import streamlit as st
from groq import Groq


# ---------------------------------------------------------
# Groq Client holen (gecacht)
# ---------------------------------------------------------
@st.cache_resource
def get_groq_client() -> Optional[Groq]:
    """
    Erzeugt einmalig einen Groq-Client und cached ihn für die Session.

    Sucht den API-Key zuerst in Streamlit-Secrets, dann in Umgebungsvariablen.

    Erwartete Secrets-Struktur in .streamlit/secrets.toml:

    [groq]
    api_key = "gsk_..."
    """
    api_key = None

    # 1) Streamlit Secrets bevorzugen
    try:
        api_key = st.secrets["groq"]["api_key"]
    except Exception:
        api_key = None

    # 2) Fallback: Umgebungsvariable
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------
def _looks_like_html_error(text: str) -> bool:
    """
    Erkenne typische HTML-Fehlerseiten (Cloudflare, 500er, etc.),
    damit wir sie nicht roh im UI anzeigen.
    """
    if not text:
        return False

    t = text.strip().lower()

    # Etwas großzügiger prüfen
    return (
        "<!doctype html" in t
        or "<html" in t
        or "cloudflare" in t
        or "cf-error" in t
        or "error code 500" in t
        or "</html>" in t
    )


def _strip_html_tags(text: str) -> str:
    """
    Entfernt normale HTML-Tags aus einer Antwort und konvertiert
    einfache Strukturen in Text/Markdown-ähnliche Form.

    Das ist ein Fallback, falls das Modell trotz Prompt doch HTML schickt.
    """
    if not text:
        return ""

    # Zeilenumbrüche für typische Tags
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = text.replace("</p>", "\n")

    # Listen in Markdown-artige Bullets umwandeln
    text = re.sub(r"<li[^>]*>", "- ", text)
    text = re.sub(r"</li>", "\n", text)
    text = re.sub(r"<(ul|ol)[^>]*>", "", text)
    text = re.sub(r"</(ul|ol)>", "", text)

    # Restliche Tags entfernen
    text = re.sub(r"<[^>]+>", "", text)

    # Mehrfache Leerzeilen reduzieren
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def _compress_df_for_llm(
    df: pd.DataFrame,
    timeframe: str,
    max_override: Optional[int] = None,
    max_chars: int = 2000,
) -> str:
    """
    Reduziert den Chart auf ein kompaktes Text-Preview für den Prompt.

    Schutz vor zu großem Input:
    - Verwendet je nach Timeframe unterschiedlich viele Kerzen (Lookback).
    - Optional kann max_override gesetzt werden, um den Lookback hart zu überschreiben.
    - Zusätzlich wird der erzeugte Text auf max_chars Zeichen begrenzt.
    """
    if df is None or df.empty:
        return "Keine Kursdaten verfügbar."

    # Dynamische Lookbacks pro Timeframe
    lookbacks = {
        "1m": 300,   # ca. ein paar Stunden
        "5m": 300,   # ca. ein Tag
        "15m": 300,  # 2–3 Tage
        "1h": 300,   # ca. 12–13 Tage
        "4h": 250,   # ca. 6–7 Wochen
        "1d": 250,   # ca. 8 Monate
    }

    max_rows = max_override if max_override is not None else lookbacks.get(timeframe, 250)

    # Nur die letzten max_rows Kerzen verwenden
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()

    last = df.iloc[-1]

    # Basiswerte (mit get, falls Spalten fehlen)
    close = float(last.get("close", float("nan")))
    rsi = float(last.get("rsi14", float("nan")))
    ema20 = float(last.get("ema20", float("nan")))
    ema50 = float(last.get("ema50", float("nan")))
    ma200 = float(last.get("ma200", float("nan")))  # MA200 (nicht EMA200)

    bb_mid = float(last.get("bb_mid", float("nan")))
    bb_up = float(last.get("bb_up", float("nan")))
    bb_lo = float(last.get("bb_lo", float("nan")))

    last_signal = str(last.get("signal", "NO DATA"))

    # Grobe Statistik über den betrachteten Zeitraum
    close_min = float(df["close"].min())
    close_max = float(df["close"].max())
    close_change_pct = (
        (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0
        if df["close"].iloc[0] != 0
        else 0.0
    )

    if "rsi14" in df.columns:
        rsi_min = float(df["rsi14"].min())
        rsi_max = float(df["rsi14"].max())
    else:
        rsi_min = float("nan")
        rsi_max = float("nan")

    # Signals-Zusammenfassung
    sig_counts = {}
    if "signal" in df.columns:
        counts = df["signal"].value_counts()
        for sig, cnt in counts.items():
            sig_counts[str(sig)] = int(cnt)

    parts = [
        f"Aktuelle Candle (auf Basis von {len(df)} Kerzen im Timeframe {timeframe}):",
        f"- Close: {close:.2f}",
        f"- RSI14: {rsi:.2f}",
        f"- EMA20: {ema20:.2f}",
        f"- EMA50: {ema50:.2f}",
        f"- MA200: {ma200:.2f}",
        f"- Bollinger: mid={bb_mid:.2f}, upper={bb_up:.2f}, lower={bb_lo:.2f}",
        f"- Letztes Signal: {last_signal}",
        "",
        "Zeitraum-Zusammenfassung:",
        f"- Min/Max Close: {close_min:.2f} / {close_max:.2f}",
        f"- Performance über Zeitraum: {close_change_pct:.2f} %",
        f"- RSI Spannweite: {rsi_min:.2f} – {rsi_max:.2f}",
    ]

    if sig_counts:
        sig_text = ", ".join([f"{k}: {v}" for k, v in sig_counts.items()])
        parts.append(f"- Signal-Häufigkeit: {sig_text}")

    summary = "\n".join(parts)

    # Zusätzliche Sicherheitsbremse: max_chars
    if len(summary) > max_chars:
        summary = summary[: max_chars - 40].rstrip() + "\n…(Chart-Zusammenfassung gekürzt)…"

    return summary


# ---------------------------------------------------------
# Hauptfunktion: CoPilot-Aufruf
# ---------------------------------------------------------
def ask_copilot(
    question: str,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    last_signal: Optional[str] = None,
) -> str:
    """
    Ruft Groq als CoPilot auf.

    Sicherheitsaspekte:
    - Frage wird auf eine maximale Länge begrenzt.
    - DF-Zusammenfassung ist in Kerzen und Zeichen limitiert.
    - Antwort ist immer Text/Markdown, ohne HTML.
    - HTML-Fehlerseiten von Groq/Cloudflare werden erkannt
      und in saubere Meldungen übersetzt.
    """
    if not question or not str(question).strip():
        return "Bitte zuerst eine sinnvolle Frage an den CoPilot eingeben."

    client = get_groq_client()
    if client is None:
        return (
            "❌ KI nicht verfügbar: Kein Groq API-Key gefunden.\n\n"
            "Bitte in Streamlit unter secrets.toml eintragen:\n"
            "[groq]\napi_key = \"DEIN_GROQ_KEY_HIER\"\n"
            "oder die Umgebungsvariable GROQ_API_KEY setzen."
        )

    if last_signal is None:
        last_signal = "NO DATA"

    # Benutzerfrage hart begrenzen
    raw_question = str(question).strip()
    max_question_chars = 1200
    if len(raw_question) > max_question_chars:
        raw_question = raw_question[: max_question_chars - 40].rstrip() + " … (Frage gekürzt)"

    # Kompakte Beschreibung der Marktdaten für den Prompt
    df_summary = _compress_df_for_llm(df, timeframe=timeframe, max_chars=2000)

    # SYSTEM PROMPT — Strategie-basiert & ohne HTML
    system_prompt = (
        "Du bist ein nüchterner, präzise formulierender technischer Analyst für Kryptowährungen.\n"
        "Du arbeitest ausschließlich mit technischer Analyse und verwendest diese Werkzeuge:\n"
        "- RSI(14)\n"
        "- EMA20\n"
        "- EMA50\n"
        "- MA200\n"
        "- Bollinger-Bänder (20)\n"
        "- Candlestick-Struktur\n"
        "- Volumen\n\n"
        "Du kennst zusätzlich die folgende Long-only-Strategie-Logik:\n"
        "- Es wird nur in Long-Richtung gehandelt, wenn der Kurs ÜBER der MA200 liegt.\n"
        "- Ist die MA200 nicht verfügbar oder der Kurs darunter, ist das Ergebnis immer HOLD.\n"
        "- STRONG BUY: starker Dip am unteren Bollinger-Band (Kurs am/unter bb_lo) mit RSI < 35.\n"
        "- BUY: gesunder Pullback, wenn der Kurs nahe/unter dem unteren Band (<= bb_lo * 1.01) "
        "und der RSI zwischen 30 und 48 liegt.\n"
        "- STRONG SELL: Blow-Off-Top, wenn der Kurs über dem oberen Band liegt, der RSI > 73 ist "
        "und der Schlusskurs schwächer ist als die vorherige Kerze.\n"
        "- SELL: normale Übertreibung, wenn der Kurs über dem oberen Band liegt, der RSI > 72 ist "
        "und der RSI gegenüber der vorherigen Kerze fällt.\n"
        "- In allen anderen Fällen: HOLD (kein klares Setup).\n\n"
        "WICHTIG: Du sollst BEIDES tun:\n"
        "1) Eine freie, saubere technische Analyse des Charts geben.\n"
        "2) Die Situation EXPLIZIT im Kontext dieser Strategie-Logik einordnen "
        "(ob sie eher zu STRONG BUY / BUY / SELL / STRONG SELL / HOLD passt).\n\n"
        "Du schreibst prägnant, analytisch und strukturiert und vermeidest Wiederholungen.\n"
        "Die gesamte Antwort soll ungefähr 200–300 Wörter lang sein.\n\n"
        "Du darfst hypothetische technische Handelsideen (Entry-/Stop-/Target-Zonen) formulieren, "
        "musst aber immer klar sagen, dass es KEINE Anlageberatung ist.\n\n"
        "WICHTIG:\n"
        "- Antworte nur in normalem Text oder Markdown.\n"
        "- Verwende KEINE HTML-Tags wie <p>, <ul>, <li>, <div>, <span>, <br>.\n"
    )

    # USER PROMPT — Chartkontext & Strategie-Einordnung
    user_prompt = (
        f"Symbol: {symbol}\n"
        f"Timeframe: {timeframe}\n"
        f"Aktueller Signalscore laut System: {last_signal}\n\n"
        f"Technische Daten (kompakt):\n{df_summary}\n\n"
        f"Benutzerfrage:\n{raw_question}\n\n"
    
        "Strukturiere deine Antwort bitte genau in diese 4 Abschnitte:\n\n"
    
        "#### 1) Kurzfassung (max. 3 Bulletpoints)\n"
        "- Maximal 1 Satz pro Bullet.\n"
        "- Fokus: Trend, Risiko, Chance im aktuellen Setup.\n\n"
    
        "#### 2) Freie technische Analyse des Charts\n"
        "- Beschreibe klar den Trend: EMA20/EMA50 relativ zur MA200.\n"
        "- Bollinger-Bänder (PFLICHT): Lage am oberen/mittleren/unteren Band, Enge/Expansion, Mean-Reversion oder Trendfortsetzung.\n"
        "- Candlesticks (PFLICHT): Mindestens 2 konkrete Muster/Signale (Dochte, Engulfing, Hammer, Shooting Star) – und was sie aussagen.\n"
        "- RSI(14): überkauft/überverkauft, Momentum, Divergenzen.\n"
        "- Unterstützungen/Widerstände: wichtigste Preiszonen.\n\n"
    
        "#### 3) Einordnung im Kontext der Strategie\n"
        "- Ordne das aktuelle Setup einem der fünf Signale zu: STRONG BUY / BUY / SELL / STRONG SELL / HOLD.\n"
        "- Vergleiche diese Einschätzung mit dem Signalscore aus dem System.\n"
        "- Wenn der Kurs unter der MA200 liegt oder die MA200 fehlt, MUSS die Strategie immer HOLD sein.\n\n"
    
        "#### 4) Hypothetische, rein technische Handelsidee (keine Anlageberatung)\n"
        "- WICHTIG: Wenn der Kurs unter der MA200 liegt → KEINE konkrete Einstiegs-, Stop- oder Zielzone formulieren.\n"
        "- Stattdessen: Nur beschreiben, welche Bedingungen erfüllt sein müssten, damit wieder Long-Setups aktiv wären (z.B. Reclaim MA200).\n"
        "- Wenn der Kurs über MA200 liegt, formuliere eine klassische Idee:\n"
        "  * mögliche Einstiegszone\n"
        "  * Stop-Zone\n"
        "  * Zielzone\n"
        "- Schließe IMMER damit ab, dass dies KEINE Anlageberatung ist.\n\n"
    
        "WICHTIG:\n"
        "- Schreibe kompakt und vermeide Wiederholungen.\n"
        "- Antworte nur in Text oder Markdown, ohne HTML-Tags.\n"
    )





    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_completion_tokens=1200,
        )

        content = response.choices[0].message.content or ""
        stripped = content.strip()

        # Falls Groq/Cloudflare uns eine HTML-Seite als "Antwort" schickt:
        if _looks_like_html_error(stripped):
            return (
                "❌ KI Fehler (Groq): Der KI-Dienst hat offenbar eine HTML-Fehlerseite "
                "(Error 5xx, z.B. 500 / Cloudflare) zurückgegeben.\n"
                "Das liegt an der Gegenstelle, nicht an deiner Anfrage. Bitte später erneut versuchen."
            )

        # Falls die Antwort trotzdem HTML/Tags enthält → in Text umwandeln
        if "<" in stripped and ">" in stripped:
            cleaned = _strip_html_tags(stripped)
            if cleaned:
                return cleaned

        return stripped

    except Exception as e:
        msg = str(e).strip()
        lower_msg = msg.lower()

        # 🔴 HTML-Fehlerseiten von Groq/Cloudflare sicher abfangen
        if _looks_like_html_error(lower_msg):
            return (
                "❌ KI Fehler (Groq): Der Server von Groq hat eine interne HTML-Fehlerseite "
                "(Error 5xx, z.B. 500 / Cloudflare) zurückgegeben.\n"
                "Du kannst daran nichts ändern – der Dienst war vermutlich kurzzeitig nicht erreichbar.\n"
                "Bitte versuche es später erneut."
            )

        # Token-/Größenlimit / Rate-Limit
        if (
            "request too large" in lower_msg
            or "tokens per minute" in lower_msg
            or "413" in lower_msg
        ):
            return (
                "❌ KI Fehler (Groq): Die Anfrage war zu groß oder hat ein Token-/Rate-Limit überschritten.\n"
                "Wähle einen kürzeren Zeitraum oder stelle eine einfachere Frage."
            )

        # Wenn sonst HTML drin steckt, strippen wir es und zeigen nur einen gekürzten Text
        if "<" in lower_msg and ">" in lower_msg:
            msg_clean = _strip_html_tags(msg)
        else:
            msg_clean = msg

        if len(msg_clean) > 400:
            msg_clean = msg_clean[:400] + " …"

        return f"❌ KI Fehler (Groq): {msg_clean}"