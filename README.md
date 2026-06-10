# Options Pricing Dashboard

An interactive financial dashboard for pricing and analysing US equity options using the **Black-Scholes (1973)** and **Heston (1993)** stochastic volatility models. Built as part of the WorldQuant University Master of Science in Financial Engineering (MsFE) research project.

🔗 **Live App:** [https://options-pricing.streamlit.app](https://options-pricing.streamlit.app)

---

## Overview

This dashboard loads real options chain data (pre-downloaded from Yahoo Finance) and compares theoretical model prices against actual market prices. It supports paper trading, Greeks analysis, and implied volatility visualisation across multiple expiries.

---

## Features

| Tab | Description |
|-----|-------------|
| 📊 Model vs Market | Compare BS and Heston prices against real bid/ask/mid market prices |
| 📋 Full Chain | Browse the complete options chain with filtering by expiry, type, and moneyness |
| 📈 Charts | IV skew, price comparison, Greeks surface, and candlestick price history |
| 🛒 Trade | Paper trade entry — buy or sell contracts priced via market, BS, or Heston |
| 💼 Portfolio | Track open positions and unrealised P&L |
| 📜 Orders | Full order history log |

---

## Models

### Black-Scholes (1973)
Prices European calls and puts assuming constant volatility. A volatility surface with skew and smile adjustments is applied to better capture market structure:

$$C = S N(d_1) - K e^{-rT} N(d_2)$$

### Heston (1993)
Stochastic volatility model that captures the volatility smile and term structure. Uses characteristic function integration (Gil-Pelaez inversion):

$$dS_t = \mu S_t \, dt + \sqrt{v_t} S_t \, dW_t^S$$
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW_t^v$$

Model parameters (kappa, theta, xi, rho) are pre-calibrated per ticker based on historical volatility regimes.

---

## Greeks

All Greeks are computed under the Black-Scholes framework:

- **Delta** — sensitivity to underlying price
- **Gamma** — rate of change of delta
- **Theta** — time decay (per day)
- **Vega** — sensitivity to implied volatility (per 1% move)

---

## Supported Tickers

| Ticker | Company |
|--------|---------|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| GOOGL | Alphabet Inc. |
| AMZN | Amazon.com Inc. |
| TSLA | Tesla Inc. |
| NVDA | NVIDIA Corporation |
| META | Meta Platforms Inc. |
| SPY | SPDR S&P 500 ETF |
| QQQ | Invesco QQQ Trust |
| JPM | JPMorgan Chase & Co. |

---

## Data

Market data was pre-downloaded from Yahoo Finance using `yfinance` and stored as CSV files in the `data/` directory. The dashboard reads directly from GitHub — no live API calls are made at runtime.

```
data/
├── AAPL_price_history.csv
├── NVDA_price_history.csv
├── ...
└── options/
    ├── AAPL_expiries.csv
    ├── AAPL_calls_2025-01-17.csv
    ├── AAPL_puts_2025-01-17.csv
    └── ...
```

> **Note:** Prices reflect the date the data was downloaded and are not updated in real time.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scipy` | BS/Heston pricing, implied vol solving |
| `plotly` | Interactive charts |
| `yfinance` | Data download (offline, pre-run only) |

---

## Installation (Local)

```bash
# Clone the repo
git clone https://github.com/BMsquared/Options-Pricing.git
cd Options-Pricing

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run options_dashboard.py
```

---

## Project Structure

```
Options-Pricing/
├── options_dashboard.py   # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md
└── data/
    ├── <TICKER>_price_history.csv
    └── options/
        ├── <TICKER>_expiries.csv
        ├── <TICKER>_calls_<YYYY-MM-DD>.csv
        └── <TICKER>_puts_<YYYY-MM-DD>.csv
```

---

## Academic Context

This project was developed as part of the **WorldQuant University MsFE Research Programme**, exploring the empirical performance of classical option pricing models against real market data. Key research questions include:

- How well does Black-Scholes price options relative to the market across moneyness and maturity?
- Does the Heston model reduce pricing error, particularly for deep ITM/OTM options?
- How does the implied volatility skew vary across expiries for high-volatility names (e.g. NVDA, TSLA)?

---

## Disclaimer

This dashboard is for **educational and research purposes only**. The paper trading feature uses simulated cash and does not constitute financial advice. Options trading involves significant risk of loss.

---

## Author

**Muriuki** — Financial Engineer WorldQuant University, MsFE Candidate 
GitHub: [@BMsquared](https://github.com/BMsquared)