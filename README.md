# CryptoTV – TradingView Style (Flat Version)

Dieses Projekt ist eine saubere, modulare, TradingView-inspirierte Crypto-Charting-App
für Streamlit – mit Signalen, Bollinger Bands, EMA20/50, MA200, RSI, Backtesting,
Signal-History **und KI-CoPilot (Groq)**.

## Starten

```bash
pip install -r requirements.txt
streamlit run ui.py
```

## KI-CoPilot (Groq)

Der CoPilot verwendet die Groq-API (z.B. Modell `llama-3.3-70b-versatile`).

Du brauchst einen API-Key von https://console.groq.com – danach:

### Lokal

```bash
export GROQ_API_KEY="gsk_..."
# Windows (PowerShell):
# $Env:GROQ_API_KEY="gsk_..."
```

### Streamlit Cloud

In den App-Einstellungen unter **Secrets**:

```toml
GROQ_API_KEY = "gsk_..."
```

## E-Mail Push bei Signalwechsel (Gmail)

Für automatische E-Mail-Benachrichtigungen bei Signalwechsel (BUY/SELL etc.)
kannst du in `secrets.toml` folgenden Block anlegen:

```toml
[email]
enabled = true
from_addr = "dein.gmail.account@gmail.com"
to_addr   = "zieladresse@gmail.com"
smtp_server = "smtp.gmail.com"
smtp_port   = 465
smtp_user   = "dein.gmail.account@gmail.com"
smtp_password = "APP_PASSWORT"
```

Wichtig:
- In Gmail 2FA aktivieren
- Ein **App-Passwort** erstellen
- Dieses als `smtp_password` eintragen

Alternativ kannst du dieselben Werte als Umgebungsvariablen setzen
(`EMAIL_ENABLED`, `EMAIL_FROM`, `EMAIL_TO`, `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`).
