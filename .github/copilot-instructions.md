# Copilot Instructions for CryptoTV Trading Dashboard

## Project Overview
- **CryptoTV** is a modular, TradingView-inspired crypto charting app built with Streamlit.
- Key features: live charting, signal generation, technical indicators (Bollinger Bands, EMA20/50, MA200, RSI), backtesting, and signal history.
- Data is fetched from Bitfinex using the public API (see `api.py`).

## Architecture & Key Files
- `ui.py`: Main Streamlit UI, orchestrates data loading, indicator calculation, signal display, and user interaction. Entry point for `streamlit run ui.py`.
- `api.py`: Handles all data fetching from Bitfinex (candles, tickers). Uses `requests` and pandas. Caching via `@st.cache_data`.
- `signals.py`: Contains the core logic for signal generation and color mapping. Implements project-specific rules for BUY/SELL/HOLD signals.
- `backtest.py`: Provides backtesting logic for signals, including trade simulation and summary statistics.
- `config.py`: Central configuration for symbols, timeframes, signal colors, and themes. All constants and mappings are defined here.
- `app.py`: Minimal script to launch the UI (calls `ui.main()`).

## Developer Workflows
- **Install dependencies:** `pip install -r requirements.txt`
- **Run app:** `streamlit run ui.py`
- **Backtesting:** Use functions in `backtest.py` (integrated in UI or callable directly).
- **Signal logic:** Update or extend in `signals.py` (`compute_signals`, `_signal_core_with_reason`).
- **API integration:** Adjust or extend Bitfinex data fetching in `api.py`.

## Project-Specific Patterns & Conventions
- **Signal values:** Only use values from `config.VALID_SIGNALS` (e.g., `STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL`).
- **Signal colors:** Use `config.SIGNAL_COLORS` and `signals.signal_color()` for UI consistency.
- **Indicators:** Calculated ad-hoc in `ui.py` or via helper functions. Add new indicators in `get_indicator_summary`.
- **Themes:** UI color themes are defined in `config.py` under `THEMES`.
- **DataFrame structure:** Candle data must have columns: `open`, `high`, `low`, `close`, `volume` (see `api.py`).
- **Backtest signals:** Only signals in `VALID_SIGNALS` are considered for trade simulation.

## Integration Points
- **External API:** Bitfinex public endpoints (see `BITFINEX_BASE_URL` in `config.py`).
- **Streamlit:** All UI logic is Streamlit-based. Use `st.cache_data` for expensive data fetches.
- **No database:** All data is in-memory; no persistent storage.

## Examples
- To add a new indicator, extend `get_indicator_summary` in `ui.py` and update the UI display.
- To change signal logic, modify `_signal_core_with_reason` in `signals.py`.
- To add a new symbol or timeframe, update `SYMBOLS` or `TIMEFRAMES` in `config.py`.

---

**For AI agents:**
- Always use project-specific signal values and color conventions.
- When adding new features, follow the modular structure (separate logic, config, and UI).
- Reference and reuse existing helper functions and config constants.
- If unsure about data structure, check `api.py` and `config.py` for canonical formats.
