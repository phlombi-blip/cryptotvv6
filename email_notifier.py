# email_notifier.py
"""
Einfache E-Mail-Benachrichtigung für Signalwechsel (Gmail-freundlich).

Konfiguration bevorzugt über Streamlit-Secrets:

[email]
enabled = true
from_addr = "dein.gmail.account@gmail.com"
to_addr   = "zieladresse@gmail.com"  # kann dieselbe sein
smtp_server = "smtp.gmail.com"
smtp_port   = 465
smtp_user   = "dein.gmail.account@gmail.com"
smtp_password = "APP_PASSWORT"

Alternativ (z.B. lokal) können auch Umgebungsvariablen verwendet werden:

  EMAIL_ENABLED=1
  EMAIL_FROM=dein.gmail.account@gmail.com
  EMAIL_TO=zieladresse@gmail.com
  SMTP_SERVER=smtp.gmail.com
  SMTP_PORT=465
  SMTP_USER=dein.gmail.account@gmail.com
  SMTP_PASSWORD=APP_PASSWORT

Wichtig: Für Gmail brauchst du ein App-Passwort (2FA aktivieren).
"""

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Tuple, Optional

import streamlit as st


def _get_email_settings() -> dict:
    """Liest E-Mail-Settings aus st.secrets oder Umgebungsvariablen."""
    secrets_email = {}
    try:
        # st.secrets kann leer sein – daher defensiv
        secrets_email = st.secrets.get("email", {})
    except Exception:
        secrets_email = {}

    cfg = {
        "enabled": bool(
            secrets_email.get("enabled", False)
            or os.getenv("EMAIL_ENABLED", "0") in ("1", "true", "True")
        ),
        "from_addr": secrets_email.get("from_addr") or os.getenv("EMAIL_FROM"),
        "to_addr": secrets_email.get("to_addr") or os.getenv("EMAIL_TO"),
        "smtp_server": secrets_email.get("smtp_server") or os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(secrets_email.get("smtp_port", os.getenv("SMTP_PORT", "465"))),
        "smtp_user": secrets_email.get("smtp_user") or os.getenv("SMTP_USER"),
        "smtp_password": secrets_email.get("smtp_password") or os.getenv("SMTP_PASSWORD"),
    }

    # Fallback: smtp_user = from_addr, wenn nicht explizit gesetzt
    if not cfg["smtp_user"]:
        cfg["smtp_user"] = cfg["from_addr"]

    return cfg


def send_signal_email(
    previous_signal: Optional[str],
    new_signal: str,
    symbol: str,
    timeframe: str,
    price: float,
    reason: str,
    when,
) -> Tuple[bool, str]:
    """
    Versendet eine E-Mail bei Signalwechsel.

    Gibt (ok: bool, msg: str) zurück.
    """
    cfg = _get_email_settings()

    if not cfg["enabled"]:
        return False, "E-Mail-Benachrichtigungen sind deaktiviert (email.enabled/EMAIL_ENABLED)."

    if not cfg["from_addr"] or not cfg["to_addr"]:
        return False, "Fehlende from_addr/to_addr in [email] oder Env-Variablen."

    if not cfg["smtp_password"]:
        return False, "Kein SMTP/App-Passwort gesetzt (smtp_password/SMTP_PASSWORD)."

    prev_txt = previous_signal or "NONE"
    dt_txt = str(when) if when is not None else "N/A"

    subject = f"[Signalwechsel] {symbol} {timeframe}: {prev_txt} → {new_signal}"
    body = f"""
Es wurde ein neuer Trade-Signalwechsel erkannt.

Symbol: {symbol}
Timeframe: {timeframe}
Zeitpunkt (letzte Candle): {dt_txt}

Vorheriges Signal: {prev_txt}
Neues Signal: {new_signal}

Letzter Preis: {price:,.2f} USD

Begründung des Signals:
{reason or '-'}

Hinweis: Dies ist eine automatische Benachrichtigung aus deiner Streamlit-Chart-App.
Keine Finanzberatung – nur technische Indikation.
""".strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = cfg["to_addr"]
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg["smtp_server"], cfg["smtp_port"], context=context) as server:
            server.login(cfg["smtp_user"], cfg["smtp_password"])
            server.send_message(msg)
        return True, "E-Mail erfolgreich gesendet."
    except Exception as e:
        return False, f"Fehler beim Versand: {e}"
