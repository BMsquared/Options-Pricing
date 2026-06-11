"""
Options Market Dashboard — Streamlit Version
=============================================
Data loaded from GitHub CSV files (no live yfinance API calls)
Models: Black-Scholes (1973) | Heston (1993)
Compares model prices vs real market prices (bid/ask/last)

Run with:  streamlit run options_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings
import uuid

warnings.filterwarnings("ignore")

from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# GITHUB DATA CONFIG
# ─────────────────────────────────────────────────────────────

GITHUB_BASE = "https://raw.githubusercontent.com/BMsquared/Options-Pricing/main/data"

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Pricing Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 4px 0;
    }
    .metric-title { color: #6b7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    .metric-value { color: #e8eaf0; font-size: 20px; font-weight: 700; margin-top: 2px; }
    .green  { color: #00c896 !important; }
    .red    { color: #ff4d6a !important; }
    .blue   { color: #4f8ef7 !important; }
    .purple { color: #b07eff !important; }
    .section-header {
        color: #4f8ef7; font-size: 15px; font-weight: 700;
        margin: 10px 0 6px 0;
        border-bottom: 1px solid #2d3148;
        padding-bottom: 5px;
    }
    .order-success { background:#0d2e1f; border:1px solid #00c896; border-radius:6px; padding:8px 14px; color:#00c896; font-weight:600; }
    .order-fail    { background:#2e0d14; border:1px solid #ff4d6a; border-radius:6px; padding:8px 14px; color:#ff4d6a; font-weight:600; }
    div[data-testid="stDataFrame"] { border:1px solid #2d3148; border-radius:8px; }

    /* Sticky title — starts after sidebar */
    .sticky-title {
        position: fixed;
        top: 0;
        left: 300px;
        right: 0;
        z-index: 9999;
        background: #0f1117;
        padding: 6px 24px 5px 24px;
        border-bottom: 1px solid #2d3148;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Push main content down so nothing hides under sticky bar */
    .block-container {
        padding-top: 56px !important;
        padding-left: 2rem !important;
    }

    /* Hide Streamlit toolbar and deploy button */
    [data-testid="stToolbar"] { display: none !important; }
    button[kind="header"]      { display: none !important; }
    .stAppDeployButton         { display: none !important; }
    #MainMenu                  { display: none !important; }
    footer                     { display: none !important; }

    /* Keep header transparent but visible so toggle arrow works */
    header {
        background-color: transparent !important;
        z-index: 999998 !important;
    }

    /* Sidebar toggle arrow — always visible even when sidebar is collapsed */
    [data-testid="collapsedControl"] {
        display:    flex !important;
        visibility: visible !important;
        opacity:    1 !important;
        z-index:    999999 !important;
        background: #1a1d27 !important;
        border:     1px solid #2d3148 !important;
        border-radius: 4px !important;
        padding:    4px !important;
    }
    [data-testid="collapsedControl"] svg {
        fill:       #4f8ef7 !important;
        display:    block !important;
        visibility: visible !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sticky-title'>
    <span style='color:#4f8ef7; font-size:20px; font-weight:800; letter-spacing:1px;'>
        ⬡ Options Pricing Dashboard
    </span>
    <span style='color:#6b7280; font-size:11px; font-style:italic; margin-top:1px;'>
        Black-Scholes (1973) & Heston (1993) vs Real Market Prices
    </span>
</div>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#1a1d27",
    font=dict(color="#e8eaf0", size=11),
    xaxis=dict(gridcolor="#2d3148", zerolinecolor="#2d3148", fixedrange=True),
    yaxis=dict(gridcolor="#2d3148", zerolinecolor="#2d3148", fixedrange=True),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="#1a1d27", bordercolor="#2d3148"),
)


POPULAR_TICKERS = ["AAPL","MSFT","GOOGL","TSLA","NVDA","META","SPY"]

HESTON_PARAMS = {
    "AAPL": dict(kappa=2.0, theta_h=0.06, xi=0.40, rho=-0.50),
    "MSFT": dict(kappa=2.0, theta_h=0.05, xi=0.35, rho=-0.50),
    "GOOGL":dict(kappa=2.5, theta_h=0.08, xi=0.45, rho=-0.55),
    "AMZN": dict(kappa=2.5, theta_h=0.09, xi=0.50, rho=-0.55),
    "TSLA": dict(kappa=3.0, theta_h=0.30, xi=0.80, rho=-0.65),
    "NVDA": dict(kappa=3.0, theta_h=0.20, xi=0.70, rho=-0.60),
    "META": dict(kappa=2.5, theta_h=0.10, xi=0.50, rho=-0.50),
    "SPY":  dict(kappa=1.5, theta_h=0.02, xi=0.25, rho=-0.70),
    "QQQ":  dict(kappa=1.5, theta_h=0.03, xi=0.30, rho=-0.65),
    "JPM":  dict(kappa=2.0, theta_h=0.04, xi=0.35, rho=-0.45),
}
DEFAULT_HESTON = dict(kappa=2.0, theta_h=0.06, xi=0.40, rho=-0.50)


# ─────────────────────────────────────────────────────────────
# GITHUB DATA LOADERS
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_price_history(symbol: str) -> pd.DataFrame | None:
    """Load price history CSV from GitHub."""
    url = f"{GITHUB_BASE}/{symbol}_price_history.csv"
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception as e:
        st.warning(f"Could not load price history for {symbol}: {e}")
        return None

@st.cache_data(show_spinner=False)
def load_expiries(symbol: str) -> list:
    """Load available expiry dates from GitHub index CSV."""
    url = f"{GITHUB_BASE}/options/{symbol}_expiries.csv"
    try:
        df = pd.read_csv(url)
        return sorted(df["expiry"].tolist())
    except Exception as e:
        st.warning(f"Could not load expiries for {symbol}: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_option_chain(symbol: str, expiry: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load calls and puts CSVs from GitHub for a given expiry."""
    calls_url = f"{GITHUB_BASE}/options/{symbol}_calls_{expiry}.csv"
    puts_url  = f"{GITHUB_BASE}/options/{symbol}_puts_{expiry}.csv"
    try:
        calls = pd.read_csv(calls_url)
        puts  = pd.read_csv(puts_url)
        return calls, puts
    except Exception as e:
        st.warning(f"Could not load option chain for {symbol} {expiry}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# SPOT & VOL FROM SAVED PRICE HISTORY
# ─────────────────────────────────────────────────────────────

def safe_float(val, default=np.nan):
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except Exception:
        return default

def get_spot_and_vol(symbol: str):
    """Derive spot price and 30-day HV from saved CSV price history."""
    hist = load_price_history(symbol)
    if hist is None or hist.empty:
        return None, None

    if "Close" not in hist.columns:
        # Try case-insensitive match
        close_col = next((c for c in hist.columns if c.lower() == "close"), None)
        if close_col is None:
            return None, None
        hist = hist.rename(columns={close_col: "Close"})

    hist = hist[["Close"]].dropna().sort_index()
    if len(hist) < 5:
        return None, None

    spot = safe_float(hist["Close"].iloc[-1])
    if np.isnan(spot) or spot <= 0:
        return None, None

    returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    if len(returns) < 5:
        vol = 0.25
    else:
        window = min(30, len(returns))
        vol = safe_float(returns.rolling(window).std().iloc[-1] * np.sqrt(252), 0.25)
        if np.isnan(vol) or vol <= 0:
            vol = 0.25

    return spot, vol


# ─────────────────────────────────────────────────────────────
# PRICING MODELS
# ─────────────────────────────────────────────────────────────

def surface_iv(S, K, T, base_vol, skew_slope=-0.10, smile_curve=0.03):
    T = max(T, 1e-4)
    log_m = np.log(K / S) / np.sqrt(T)
    atm   = base_vol * (1.0 + 0.06 * np.sqrt(T))
    return float(np.clip(atm + skew_slope * log_m + smile_curve * log_m**2, 0.05, 2.0))

def bs_price(S, K, T, r, sigma, otype="call"):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0) if otype == "call" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if otype == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def bs_greeks(S, K, T, r, sigma, otype="call"):
    if sigma <= 0 or T <= 0:
        return dict(delta=0, gamma=0, vega=0, theta=0)
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if otype == "call" else norm.cdf(d1) - 1
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * pdf / 100
    if otype == "call":
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta)

def implied_vol(mkt_price, S, K, T, r, otype="call"):
    try:
        intrinsic = max(S - K, 0) if otype == "call" else max(K - S, 0)
        if mkt_price <= intrinsic + 1e-6:
            return np.nan
        return brentq(lambda s: bs_price(S, K, T, r, s, otype) - mkt_price, 1e-4, 5.0, xtol=1e-6)
    except Exception:
        return np.nan

def heston_price(S, K, T, r, v0, kappa, theta_h, xi, rho, otype="call"):
    if T <= 0:
        return max(S - K, 0) if otype == "call" else max(K - S, 0)
    from scipy.integrate import quad

    def char_fn(phi, j):
        u = 0.5 if j == 1 else -0.5
        b = (kappa - rho * xi) if j == 1 else kappa
        a = kappa * theta_h
        d = np.sqrt((rho * xi * phi * 1j - b)**2 - xi**2 * (2 * u * phi * 1j - phi**2))
        g = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
        eg = np.exp(d * T)
        C = (r * phi * 1j * T
             + (a / xi**2) * ((b - rho * xi * phi * 1j + d) * T
             - 2 * np.log((1 - g * eg) / (1 - g))))
        D = ((b - rho * xi * phi * 1j + d) / xi**2 * (1 - eg) / (1 - g * eg))
        return np.exp(C + D * v0 + 1j * phi * np.log(S))

    def integrand(phi, j):
        return (np.exp(-1j * phi * np.log(K)) * char_fn(phi, j) / (1j * phi)).real

    try:
        P1 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 1), 1e-5, 200, limit=150)[0]
        P2 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 2), 1e-5, 200, limit=150)[0]
    except Exception:
        return bs_price(S, K, T, r, np.sqrt(max(v0, 0.01)), otype)

    call = float(max(S * P1 - K * np.exp(-r * T) * P2, 0))
    if otype == "call":
        return call
    return float(max(call - S + K * np.exp(-r * T), 0))

def heston_greeks(S, K, T, r, v0, kappa, theta_h, xi, rho, otype="call", dS=0.5):
    p  = heston_price(S,      K, T,                    r, v0, kappa, theta_h, xi, rho, otype)
    pu = heston_price(S + dS, K, T,                    r, v0, kappa, theta_h, xi, rho, otype)
    pd = heston_price(S - dS, K, T,                    r, v0, kappa, theta_h, xi, rho, otype)
    pt = heston_price(S,      K, max(T - 1/365, 1e-4), r, v0, kappa, theta_h, xi, rho, otype)
    return dict(
        delta=(pu - pd) / (2 * dS),
        gamma=(pu - 2*p + pd) / dS**2,
        theta=(pt - p) / (1/365),
        vega=0.0,
    )


# ─────────────────────────────────────────────────────────────
# BUILD OPTION CHAIN FROM CSV FILES
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_option_chain(symbol: str, r: float, max_expiries: int = 6):
    """
    Load saved CSV option chain data from GitHub and compute
    BS + Heston model prices alongside market prices.
    """
    spot, hv = get_spot_and_vol(symbol)
    if spot is None:
        return None, None, None

    expiries = load_expiries(symbol)
    if not expiries:
        return None, spot, hv

    # Limit to max_expiries nearest future expiries
    today = datetime.date.today()
    valid_exp = []
    for e in expiries:
        try:
            d = datetime.date.fromisoformat(e)
            if (d - today).days >= 1:
                valid_exp.append(e)
        except Exception:
            continue

    selected_exp = valid_exp[:max_expiries]
    if not selected_exp:
        # If all expiries are past (stale data), use them anyway
        selected_exp = expiries[:max_expiries]

    hp = HESTON_PARAMS.get(symbol, DEFAULT_HESTON)
    v0 = hv ** 2
    rows = []

    for exp_str in selected_exp:
        try:
            exp_date = datetime.date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            T = max(dte / 365, 1/365)
        except Exception:
            continue

        calls_df, puts_df = load_option_chain(symbol, exp_str)
        if calls_df is None:
            continue

        for otype, df_raw in [("call", calls_df), ("put", puts_df)]:
            if df_raw is None or df_raw.empty:
                continue

            for _, opt_row in df_raw.iterrows():
                K = safe_float(opt_row.get("strike"))
                if np.isnan(K) or K <= 0:
                    continue

                mkt_last = safe_float(opt_row.get("lastPrice"))
                mkt_bid  = safe_float(opt_row.get("bid"))
                mkt_ask  = safe_float(opt_row.get("ask"))

                if not np.isnan(mkt_bid) and not np.isnan(mkt_ask) and mkt_ask > mkt_bid >= 0:
                    mkt_mid = (mkt_bid + mkt_ask) / 2
                elif not np.isnan(mkt_last) and mkt_last > 0:
                    mkt_mid = mkt_last
                else:
                    continue

                mkt_iv_raw = safe_float(opt_row.get("impliedVolatility"))
                if not np.isnan(mkt_iv_raw) and 0.01 < mkt_iv_raw < 20:
                    mkt_iv = mkt_iv_raw
                else:
                    mkt_iv = implied_vol(mkt_mid, spot, K, T, r, otype)

                sigma_bs = surface_iv(spot, K, T, hv)
                bs_p     = bs_price(spot, K, T, r, sigma_bs, otype)
                g_bs     = bs_greeks(spot, K, T, r, sigma_bs, otype)

                hst_p = heston_price(spot, K, T, r, v0,
                                     hp["kappa"], hp["theta_h"],
                                     hp["xi"],    hp["rho"], otype)

                itm       = (K < spot and otype == "call") or (K > spot and otype == "put")
                moneyness = round(K / spot, 4)
                bs_diff     = round(bs_p  - mkt_mid, 3)
                heston_diff = round(hst_p - mkt_mid, 3)
                bs_vs_heston= round(bs_p  - hst_p,   3)

                rows.append({
                    "Type":           otype.upper(),
                    "Expiry":         exp_str,
                    "DTE":            dte,
                    "Strike":         K,
                    "Moneyness":      moneyness,
                    "ITM":            itm,
                    "Mkt Bid":        round(mkt_bid,  3) if not np.isnan(mkt_bid)  else np.nan,
                    "Mkt Ask":        round(mkt_ask,  3) if not np.isnan(mkt_ask)  else np.nan,
                    "Mkt Mid":        round(mkt_mid,  3),
                    "Mkt IV (%)":     round(mkt_iv * 100, 2) if not np.isnan(mkt_iv) else np.nan,
                    "BS Price":       round(bs_p,   3),
                    "Heston Price":   round(hst_p,  3),
                    "BS IV (%)":      round(sigma_bs * 100, 2),
                    "BS − Mkt":       bs_diff,
                    "Heston − Mkt":   heston_diff,
                    "BS − Heston":    bs_vs_heston,
                    "Delta":          round(g_bs["delta"], 4),
                    "Gamma":          round(g_bs["gamma"], 5),
                    "Theta":          round(g_bs["theta"], 4),
                    "Vega":           round(g_bs["vega"],  4),
                    "Volume":         safe_float(opt_row.get("volume"),       0),
                    "Open Interest":  safe_float(opt_row.get("openInterest"), 0),
                })

    if not rows:
        return None, spot, hv

    df = pd.DataFrame(rows)
    df = df.sort_values(["Expiry", "Type", "Strike"]).reset_index(drop=True)
    return df, spot, hv


# ─────────────────────────────────────────────────────────────
# PORTFOLIO
# ─────────────────────────────────────────────────────────────

def init_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "cash":      100_000.0,
            "start":     100_000.0,
            "positions": {},
            "orders":    [],
        }

def place_order(symbol, otype, strike, expiry, direction, qty, price, model):
    pf   = st.session_state.portfolio
    cost = price * qty * 100 * (1 if direction == "buy" else -1)
    if direction == "buy" and cost > pf["cash"]:
        return False, f"Insufficient cash. Need ${cost:,.2f}, have ${pf['cash']:,.2f}"
    oid = str(uuid.uuid4())[:8].upper()
    pf["orders"].append({
        "order_id": oid, "symbol": symbol, "otype": otype,
        "strike": strike, "expiry": expiry, "direction": direction,
        "qty": qty, "price": price, "model": model,
        "total": price * qty * 100,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    })
    pf["cash"] -= cost
    label  = f"{symbol} {expiry} {strike:.0f} {otype.upper()}"
    signed = qty if direction == "buy" else -qty
    if label in pf["positions"]:
        pos = pf["positions"][label]
        new_qty = pos["qty"] + signed
        if new_qty == 0:
            del pf["positions"][label]
        else:
            total_cost       = pos["avg_price"] * abs(pos["qty"]) + price * qty
            pos["avg_price"] = total_cost / abs(new_qty)
            pos["qty"]       = new_qty
    else:
        pf["positions"][label] = dict(
            symbol=symbol, otype=otype, strike=strike, expiry=expiry,
            qty=signed, avg_price=price, cur_price=price, model=model,
        )
    return True, f"✓ {oid}  {direction.upper()} {qty}× {label} @ ${price:.3f} [{model}]"


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────

def chart_price_comparison(df, otype, spot):
    d = df[df["Type"] == otype].copy().sort_values("Strike")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Strike"], y=d["Mkt Mid"],
        mode="lines+markers", name="Market Mid",
        line=dict(color="#00c896", width=2.5), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=d["Strike"], y=d["BS Price"],
        mode="lines+markers", name="Black-Scholes",
        line=dict(color="#4f8ef7", width=2, dash="dash"), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=d["Strike"], y=d["Heston Price"],
        mode="lines+markers", name="Heston",
        line=dict(color="#b07eff", width=2, dash="dot"), marker=dict(size=4)))
    fig.add_vline(x=spot, line_dash="dot", line_color="#f5a623",
                  annotation_text=f"Spot ${spot:.2f}",
                  annotation_font_color="#f5a623")
    fig.update_layout(title=f"{otype} — Model vs Market Price",
                      xaxis_title="Strike", yaxis_title="Price ($)", **PLOTLY_LAYOUT)
    return fig

def chart_diff(df, otype, spot):
    d = df[df["Type"] == otype].copy().sort_values("Strike")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Strike"], y=d["BS − Mkt"],
        name="BS − Market", marker_color="#4f8ef7", opacity=0.8))
    fig.add_trace(go.Bar(x=d["Strike"], y=d["Heston − Mkt"],
        name="Heston − Market", marker_color="#b07eff", opacity=0.8))
    fig.add_hline(y=0, line_color="#6b7280", line_dash="dash")
    fig.add_vline(x=spot, line_dash="dot", line_color="#f5a623")
    fig.update_layout(title=f"{otype} — Model Overpricing vs Market (Model − Market)",
                      barmode="group", xaxis_title="Strike",
                      yaxis_title="Price Difference ($)", **PLOTLY_LAYOUT)
    return fig

def chart_iv_skew(df, spot):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    calls = df[df["Type"] == "CALL"].dropna(subset=["Mkt IV (%)"])
    for i, (exp, grp) in enumerate(calls.groupby("Expiry")):
        grp = grp.sort_values("Strike")
        fig.add_trace(go.Scatter(
            x=grp["Strike"], y=grp["Mkt IV (%)"],
            mode="lines+markers", name=f"Mkt {exp}",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    bs_exp = calls.groupby("Expiry").first().reset_index()["Expiry"].iloc[0] if len(calls) > 0 else None
    if bs_exp:
        bs_grp = calls[calls["Expiry"] == bs_exp].sort_values("Strike")
        fig.add_trace(go.Scatter(
            x=bs_grp["Strike"], y=bs_grp["BS IV (%)"],
            mode="lines", name="BS Surface IV",
            line=dict(color="#ffffff", width=1.5, dash="longdash"),
        ))
    fig.add_vline(x=spot, line_dash="dot", line_color="#f5a623",
                  annotation_text=f"Spot ${spot:.2f}", annotation_font_color="#f5a623")
    fig.update_layout(title="IV Skew — Market vs BS Surface",
                      xaxis_title="Strike", yaxis_title="Implied Vol (%)", **PLOTLY_LAYOUT)
    return fig

def chart_greeks(df, greek, otype, spot):
    d = df[df["Type"] == otype].copy()
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (exp, grp) in enumerate(d.groupby("Expiry")):
        grp = grp.sort_values("Strike")
        fig.add_trace(go.Scatter(
            x=grp["Strike"], y=grp[greek],
            mode="lines+markers", name=exp,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_vline(x=spot, line_dash="dot", line_color="#f5a623")
    fig.update_layout(title=f"{greek} by Strike — {otype}",
                      xaxis_title="Strike", yaxis_title=greek, **PLOTLY_LAYOUT)
    return fig

def chart_stock_history(symbol: str):
    """Plot price history from saved CSV."""
    hist = load_price_history(symbol)
    if hist is None or hist.empty:
        return None
    if "Close" not in hist.columns:
        close_col = next((c for c in hist.columns if c.lower() == "close"), None)
        if close_col is None:
            return None
        hist = hist.rename(columns={close_col: "Close"})

    # Need OHLC — use Close for all if only Close available
    for col in ["Open", "High", "Low"]:
        if col not in hist.columns:
            alt = next((c for c in hist.columns if c.lower() == col.lower()), None)
            if alt:
                hist = hist.rename(columns={alt: col})
            else:
                hist[col] = hist["Close"]

    hist = hist.dropna(subset=["Close"]).sort_index()
    fig = go.Figure(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],   close=hist["Close"],
        increasing_line_color="#00c896",
        decreasing_line_color="#ff4d6a",
    ))
    fig.update_layout(title=f"{symbol} — Price History (from saved data)",
                      xaxis_rangeslider_visible=False, **PLOTLY_LAYOUT)
    return fig
    
    
def chart_vol_surface(df, spot):
    """
    2D IV smile/skew — BS vs Heston side by side.
    X-axis: Strike  |  Y-axis: Implied Volatility (%)
    One line per expiry, coloured consistently across both panels.
    """
    from plotly.subplots import make_subplots

    calls = df[df["Type"] == "CALL"].copy()
    calls = calls.dropna(subset=["Mkt IV (%)"])
    if calls.empty:
        return None

    expiries = sorted(calls["Expiry"].unique())
    colors   = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Black-Scholes IV Smile", "Heston IV Smile"),
        horizontal_spacing=0.08,
    )

    for i, exp in enumerate(expiries):
        grp   = calls[calls["Expiry"] == exp].sort_values("Strike")
        color = colors[i % len(colors)]
        dte   = grp["DTE"].iloc[0]
        label = f"{exp} ({dte}d)"

        # BS panel (left)
        fig.add_trace(go.Scatter(
            x=grp["Strike"],
            y=grp["BS IV (%)"],
            mode="lines+markers",
            name=label,
            legendgroup=label,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            showlegend=True,
        ), row=1, col=1)

        # Heston panel (right) - back out IV from Heston price
        heston_ivs = []
        for _, r in grp.iterrows():
            T  = max(r["DTE"] / 365, 1/365)
            iv = implied_vol(r["Heston Price"], spot, r["Strike"], T, 0.05, "call")
            heston_ivs.append(iv * 100 if not np.isnan(iv) else np.nan)

        fig.add_trace(go.Scatter(
            x=grp["Strike"],
            y=heston_ivs,
            mode="lines+markers",
            name=label,
            legendgroup=label,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            showlegend=False,
        ), row=1, col=2)

    # Spot line on both panels
    for col_num in [1, 2]:
        fig.add_vline(
            x=spot,
            line_dash="dot",
            line_color="#f5a623",
            annotation_text=f"Spot ${spot:.0f}",
            annotation_font_color="#f5a623",
            annotation_position="top",
            row=1, col=col_num,
        )

    shared_axis = dict(
        gridcolor="#2d3148",
        zerolinecolor="#2d3148",
        fixedrange=True,
    )

    fig.update_layout(
        title="Implied Volatility Smile — Black-Scholes vs Heston (Calls)",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font=dict(color="#e8eaf0", size=11),
        height=480,
        margin=dict(l=50, r=20, t=60, b=50),
        legend=dict(
            bgcolor="#1a1d27",
            bordercolor="#2d3148",
            title=dict(text="Expiry"),
        ),
    )

    fig.update_xaxes(**shared_axis, title_text="Strike ($)")
    fig.update_yaxes(**shared_axis, title_text="Implied Volatility (%)")

    return fig



# ─────────────────────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────────────────────

def style_diff(val):
    if not isinstance(val, (int, float)) or np.isnan(val):
        return ""
    if val >  0.20:  return "color:#ff4d6a; font-weight:600"
    if val >  0.05:  return "color:#f5a623"
    if val < -0.20:  return "color:#4f8ef7; font-weight:600"
    if val < -0.05:  return "color:#7dd3fc"
    return "color:#6b7280"

def style_itm(row):
    if row.get("ITM", False):
        return ["background-color:#1a2e1a"] * len(row)
    return [""] * len(row)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    init_portfolio()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        #st.markdown("## Options Dashboard")
        #st.markdown("*Black-Scholes & Heston (1993) vs Market*")
        st.markdown("---")

        symbol = st.selectbox("Stock", POPULAR_TICKERS)
        # custom = st.text_input("Custom ticker (overrides above):").upper().strip()
        #if custom:
           # symbol = custom
           
        st.markdown("---")
        st.markdown("""
        <div style='background:#1a1d27; border:1px solid #2d3148; border-radius:8px; padding:12px 14px; font-size:11px;'>
            <div style='color:#4f8ef7; font-weight:700; margin-bottom:8px; font-size:12px;'>📖 Glossary</div>
            <div style='color:#6b7280; line-height:2;'>
                <span style='color:#e8eaf0; font-weight:600;'>HV30</span> — 30-day Historical Volatility<br>
                <span style='color:#e8eaf0; font-weight:600;'>IV</span> — Implied Volatility<br>
                <span style='color:#e8eaf0; font-weight:600;'>BS</span> — Black-Scholes Model (1973)<br>
                <span style='color:#e8eaf0; font-weight:600;'>Heston</span> — Heston Stochastic Vol Model (1993)<br>
                <span style='color:#e8eaf0; font-weight:600;'>Mkt Mid</span> — Market Mid Price (Bid+Ask)/2<br>
                <span style='color:#e8eaf0; font-weight:600;'>DTE</span> — Days to Expiry<br>
                <span style='color:#e8eaf0; font-weight:600;'>ITM</span> — In The Money<br>
                <span style='color:#e8eaf0; font-weight:600;'>OTM</span> — Out of The Money<br>
                <span style='color:#e8eaf0; font-weight:600;'>ATM</span> — At The Money<br>
                <span style='color:#e8eaf0; font-weight:600;'>Delta (Δ)</span> — Price sensitivity to spot<br>
                <span style='color:#e8eaf0; font-weight:600;'>Gamma (Γ)</span> — Rate of change of Delta<br>
                <span style='color:#e8eaf0; font-weight:600;'>Theta (Θ)</span> — Time decay per day<br>
                <span style='color:#e8eaf0; font-weight:600;'>Vega (ν)</span> — Sensitivity to volatility<br>
                <span style='color:#e8eaf0; font-weight:600;'>BS − Mkt</span> — BS model price minus market price<br>
                <span style='color:#e8eaf0; font-weight:600;'>Heston − Mkt</span> — Heston price minus market price<br>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        #r_pct = st.slider("Risk-Free Rate (%)", 1.0, 10.0, 5.0, 0.25)
        R = 5 / 100
        max_exp = 4

        st.markdown("---")
        # Data source info
        st.markdown("""
        <div style='background:#1a1d27; border:1px solid #2d3148; border-radius:8px; padding:10px; font-size:11px; color:#6b7280;'>
        📁 <b style='color:#4f8ef7'>Data Source</b><br>
        Loaded from GitHub CSV files.<br>
        Prices reflect download date.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Paper Portfolio")
        pf   = st.session_state.portfolio
        cash = pf["cash"]
        unr  = sum((p["cur_price"] - p["avg_price"]) * p["qty"] * 100
                   for p in pf["positions"].values())
        total_pnl = cash - pf["start"] + unr
        c = "#00c896" if total_pnl >= 0 else "#ff4d6a"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>CASH</div>
            <div class='metric-value blue'>${cash:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>TOTAL P&L</div>
            <div class='metric-value' style='color:{c}'>${total_pnl:+,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Reset Portfolio"):
            st.session_state.portfolio = dict(
                cash=100_000.0, start=100_000.0, positions={}, orders=[])
            st.rerun()

    # ── LOAD DATA ─────────────────────────────────────────────
    with st.spinner(f"Loading {symbol} data from GitHub..."):
        df, spot, hv = build_option_chain(symbol, R, max_exp)

    if df is None or spot is None:
        st.error(
            f"Could not load data for **{symbol}**. "
            "Make sure the CSV files for this ticker exist in your GitHub repo under `data/`. "
            "Check that `data/{symbol}_price_history.csv`, "
            "`data/options/{symbol}_expiries.csv`, and "
            "`data/options/{symbol}_calls_YYYY-MM-DD.csv` are all pushed."
        )
        st.stop()

    expiries = sorted(df["Expiry"].unique())

    # ── HEADER METRICS ────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("SPOT",      f"${spot:.2f}",             "blue"),
        ("HV30",      f"{hv*100:.1f}%",            ""),
        ("EXPIRIES",  str(len(expiries)),           ""),
        ("CONTRACTS", str(len(df)),                ""),
        ("POSITIONS", str(len(pf["positions"])),   ""),
    ]
    for col, (title, val, cls) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value {cls}'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Model vs Market",
        "📋 Full Chain",
        "📈 Charts",
        "🛒 Trade",
        "💼 Portfolio",
        "📜 Orders",
    ])

    # ════════════════════════════════════════════════
    # TAB 1 — MODEL vs MARKET COMPARISON
    # ════════════════════════════════════════════════
    with t1:
        st.markdown("<div class='section-header'>Black-Scholes & Heston vs Real Market Prices</div>",
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            sel_exp   = st.selectbox("Expiry", expiries, key="t1_exp")
        with col2:
            sel_otype = st.selectbox("Type", ["CALL", "PUT"], key="t1_type")

        df_exp = df[(df["Expiry"] == sel_exp) & (df["Type"] == sel_otype)].copy()

        if df_exp.empty:
            st.warning("No data for this selection.")
        else:
            avg_bs_diff     = df_exp["BS − Mkt"].mean()
            avg_heston_diff = df_exp["Heston − Mkt"].mean()
            atm_row = df_exp.iloc[(df_exp["Strike"] - spot).abs().argsort()[:1]]

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("ATM Strike",    f"${atm_row['Strike'].values[0]:.0f}")
            s2.metric("ATM Mkt Mid",   f"${atm_row['Mkt Mid'].values[0]:.3f}")
            s3.metric("Avg BS Bias",   f"${avg_bs_diff:+.3f}",
                      help="Positive = BS overprices vs market on average")
            s4.metric("Avg Heston Bias", f"${avg_heston_diff:+.3f}",
                      help="Positive = Heston overprices vs market on average")

            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(chart_price_comparison(df_exp, sel_otype, spot),
                                width="stretch",
                                key=f"t1_price_cmp_{symbol}_{sel_exp}_{sel_otype}")
            with ch2:
                st.plotly_chart(chart_diff(df_exp, sel_otype, spot),
                                width="stretch",
                                key=f"t1_diff_{symbol}_{sel_exp}_{sel_otype}")

            st.markdown("#### Detailed Comparison Table")
            show_cols = ["Strike", "Moneyness", "Mkt Bid", "Mkt Ask", "Mkt Mid",
                         "Mkt IV (%)", "BS Price", "Heston Price", "BS IV (%)",
                         "BS − Mkt", "Heston − Mkt", "BS − Heston",
                         "Delta", "Gamma", "Theta", "Volume", "Open Interest"]

            styled = df_exp[show_cols].style \
                .map(style_diff, subset=["BS − Mkt", "Heston − Mkt", "BS − Heston"]) \
                .format({
                    "Strike":       "${:.0f}",
                    "Moneyness":    "{:.3f}",
                    "Mkt Bid":      "${:.3f}",
                    "Mkt Ask":      "${:.3f}",
                    "Mkt Mid":      "${:.3f}",
                    "Mkt IV (%)":   "{:.2f}%",
                    "BS Price":     "${:.3f}",
                    "Heston Price": "${:.3f}",
                    "BS IV (%)":    "{:.2f}%",
                    "BS − Mkt":     "${:+.3f}",
                    "Heston − Mkt": "${:+.3f}",
                    "BS − Heston":  "${:+.3f}",
                    "Delta":        "{:.4f}",
                    "Gamma":        "{:.5f}",
                    "Theta":        "{:.4f}",
                    "Volume":       "{:.0f}",
                    "Open Interest":"{:.0f}",
                }, na_rep="—") \
                .set_properties(**{
                    "background-color": "#1a1d27",
                    "color": "#e8eaf0",
                    "border": "1px solid #2d3148",
                })

            st.dataframe(styled, width="stretch", height=400)
            st.caption(
                "🟢 Green = Market  🔵 Blue = Black-Scholes  🟣 Purple = Heston  |  "
                "Diff colour: red = model overprices, blue = model underprices  |  "
                "Threshold: |diff| > $0.20 bold, |diff| > $0.05 amber"
            )

    # ════════════════════════════════════════════════
    # TAB 2 — FULL CHAIN
    # ════════════════════════════════════════════════
    with t2:
        st.markdown("<div class='section-header'>Full Option Chain — All Expiries</div>",
                    unsafe_allow_html=True)

        fa, fb, fc = st.columns(3)
        with fa:
            exp_f  = st.selectbox("Expiry", ["All"] + expiries, key="t2_exp")
        with fb:
            type_f = st.selectbox("Type", ["All","CALL","PUT"], key="t2_type")
        with fc:
            itm_f  = st.selectbox("Moneyness", ["All","ITM","OTM"], key="t2_itm")

        dv = df.copy()
        if exp_f  != "All":  dv = dv[dv["Expiry"] == exp_f]
        if type_f != "All":  dv = dv[dv["Type"]   == type_f]
        if itm_f  == "ITM":  dv = dv[dv["ITM"]]
        if itm_f  == "OTM":  dv = dv[~dv["ITM"]]

        show = ["Type","Expiry","DTE","Strike","Moneyness",
                "Mkt Bid","Mkt Ask","Mkt Mid","Mkt IV (%)",
                "BS Price","Heston Price","BS − Mkt","Heston − Mkt",
                "Delta","Gamma","Theta","Vega","Volume","Open Interest","ITM"]

        styled2 = dv[show].style \
            .apply(style_itm, axis=1) \
            .map(style_diff, subset=["BS − Mkt","Heston − Mkt"]) \
            .format({
                "Strike":       "${:.0f}",
                "Moneyness":    "{:.3f}",
                "Mkt Bid":      "${:.3f}",
                "Mkt Ask":      "${:.3f}",
                "Mkt Mid":      "${:.3f}",
                "Mkt IV (%)":   "{:.2f}%",
                "BS Price":     "${:.3f}",
                "Heston Price": "${:.3f}",
                "BS − Mkt":     "${:+.3f}",
                "Heston − Mkt": "${:+.3f}",
                "Delta":        "{:.4f}",
                "Gamma":        "{:.5f}",
                "Theta":        "{:.4f}",
                "Vega":         "{:.4f}",
                "Volume":       "{:.0f}",
                "Open Interest":"{:.0f}",
            }, na_rep="—") \
            .set_properties(**{
                "background-color": "#1a1d27",
                "color": "#e8eaf0",
                "border": "1px solid #2d3148",
            })

        st.dataframe(styled2, width="stretch", height=520)
        st.caption(f"{len(dv)} contracts  |  ITM rows highlighted green")

# ════════════════════════════════════════════════
    # TAB 3 — CHARTS
    # ════════════════════════════════════════════════
    with t3:
        ca, cb = st.columns(2)
        with ca:
            chart_exp   = st.selectbox("Expiry for price chart", expiries, key="t3_exp")
            chart_otype = st.selectbox("Option Type", ["CALL","PUT"], key="t3_type")
        with cb:
            greek_pick  = st.selectbox("Greek to plot", ["Delta","Gamma","Theta","Vega"], key="t3_greek")

        df_c = df[(df["Expiry"] == chart_exp)].copy()

        # ── Row 1: Price Comparison | IV Skew ─────────────────
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(chart_price_comparison(df_c, chart_otype, spot),
                            width="stretch",
                            key=f"t3_price_cmp_{symbol}_{chart_exp}_{chart_otype}")
        with r2:
            st.plotly_chart(chart_iv_skew(df, spot),
                            width="stretch",
                            key=f"t3_iv_skew_{symbol}_{chart_exp}")

        # ── Row 2: Model Diff | Greeks ─────────────────────────
        r3, r4 = st.columns(2)
        with r3:
            st.plotly_chart(chart_diff(df_c, chart_otype, spot),
                            width="stretch",
                            key=f"t3_diff_{symbol}_{chart_exp}_{chart_otype}")
        with r4:
            st.plotly_chart(chart_greeks(df, greek_pick, chart_otype, spot),
                            width="stretch",
                            key=f"t3_greeks_{symbol}_{chart_exp}_{chart_otype}_{greek_pick}")

        # ── Row 3: Price History ───────────────────────────────
        fig_hist = chart_stock_history(symbol)
        if fig_hist:
            st.plotly_chart(fig_hist, width="stretch", key=f"t3_hist_{symbol}")

        # ── Row 4: Volatility Surface ──────────────────────────
        st.markdown("<div class='section-header'>Implied Volatility Surface — Black-Scholes vs Heston</div>",
                    unsafe_allow_html=True)
        st.caption("Rotate the surface by clicking and dragging. Shows how IV changes across strikes and expiries.")

        fig_surface = chart_vol_surface(df, spot)
        if fig_surface:
            st.plotly_chart(fig_surface, width="stretch", key=f"t3_vol_surface_{symbol}")
        else:
            st.info("Not enough data to render volatility surface. Try loading more expiries.")
            

    # ════════════════════════════════════════════════
    # TAB 4 — TRADE
    # ════════════════════════════════════════════════
    with t4:
        st.markdown("<div class='section-header'>Paper Trade Entry</div>",
                    unsafe_allow_html=True)

        ta, tb, tc, td = st.columns(4)
        with ta:
            tr_exp    = st.selectbox("Expiry",  expiries, key="tr_exp")
        with tb:
            tr_type   = st.selectbox("Type",    ["CALL","PUT"], key="tr_type")
        with tc:
            avail_K   = sorted(df[(df["Expiry"]==tr_exp)&(df["Type"]==tr_type)]["Strike"].unique())
            tr_strike = st.selectbox("Strike",  avail_K, key="tr_strike")
        with td:
            tr_model  = st.selectbox("Price via", ["Mkt Mid","BS Price","Heston Price"], key="tr_model")

        sel = df[(df["Expiry"]==tr_exp)&(df["Type"]==tr_type)&(df["Strike"]==tr_strike)]
        if not sel.empty:
            row = sel.iloc[0]
            price_col = tr_model

            st.markdown("#### Contract Details")
            d1, d2, d3, d4, d5, d6 = st.columns(6)
            d1.metric("Market Mid",   f"${row['Mkt Mid']:.3f}")
            d2.metric("BS Price",     f"${row['BS Price']:.3f}")
            d3.metric("Heston Price", f"${row['Heston Price']:.3f}")
            d4.metric("Mkt IV",       f"{row['Mkt IV (%)']:.2f}%" if not np.isnan(row['Mkt IV (%)']) else "—")
            d5.metric("Delta",        f"{row['Delta']:.4f}")
            d6.metric("DTE",          f"{row['DTE']} days")

            exec_price = row[price_col]
            st.info(f"Executing at **{tr_model}** = **${exec_price:.3f}** per share  →  **${exec_price*100:.2f}** per contract")

            e1, e2, e3 = st.columns(3)
            with e1:
                direction = st.radio("Direction", ["buy","sell"], horizontal=True, key="tr_dir")
            with e2:
                qty = st.number_input("Contracts", 1, 100, 1, key="tr_qty")
            with e3:
                st.metric("Total Cost", f"${exec_price * qty * 100:,.2f}")

            btn_label = f"{'BUY' if direction=='buy' else 'SELL'} {qty}× {tr_type} ${tr_strike:.0f} exp {tr_exp}"
            if st.button(f"🚀 {btn_label}", type="primary"):
                ok, msg = place_order(
                    symbol, tr_type.lower(), tr_strike, tr_exp,
                    direction, qty, exec_price,
                    tr_model.replace(" Price","").replace(" Mid","_Mid")
                )
                if ok:
                    st.markdown(f"<div class='order-success'>{msg}</div>",
                                unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"<div class='order-fail'>{msg}</div>",
                                unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # TAB 5 — PORTFOLIO
    # ════════════════════════════════════════════════
    with t5:
        pf = st.session_state.portfolio
        st.markdown("<div class='section-header'>Open Positions</div>",
                    unsafe_allow_html=True)

        if not pf["positions"]:
            st.info("No open positions yet. Use the Trade tab.")
        else:
            rows = []
            for label, p in pf["positions"].items():
                unr = (p["cur_price"] - p["avg_price"]) * p["qty"] * 100
                mv  = p["cur_price"] * abs(p["qty"]) * 100
                pct = (unr / (p["avg_price"] * abs(p["qty"]) * 100) * 100) if p["avg_price"] > 0 else 0
                rows.append(dict(
                    Label=label, Qty=p["qty"],
                    AvgPx=p["avg_price"], CurPx=p["cur_price"],
                    MktVal=mv, UnrPnL=unr, PnLPct=pct, Model=p["model"],
                ))
            dfp = pd.DataFrame(rows)

            def pnl_color(v):
                if isinstance(v, float):
                    return "color:#00c896" if v >= 0 else "color:#ff4d6a"
                return ""

            st.dataframe(
                dfp.style
                   .map(pnl_color, subset=["UnrPnL","PnLPct"])
                   .format({"AvgPx":"${:.3f}","CurPx":"${:.3f}",
                             "MktVal":"${:,.2f}","UnrPnL":"${:+,.2f}","PnLPct":"{:+.1f}%"})
                   .set_properties(**{"background-color":"#1a1d27","color":"#e8eaf0",
                                      "border":"1px solid #2d3148"}),
                width="stretch",
            )

            tp  = sum(r["UnrPnL"] for r in rows)
            tv  = sum(r["MktVal"] for r in rows)
            col = "#00c896" if tp >= 0 else "#ff4d6a"
            st.markdown(f"""
            <div class='metric-card' style='display:inline-block;margin-right:16px;'>
                <div class='metric-title'>PORTFOLIO VALUE</div>
                <div class='metric-value blue'>${tv:,.2f}</div>
            </div>
            <div class='metric-card' style='display:inline-block;'>
                <div class='metric-title'>UNREALIZED P&L</div>
                <div class='metric-value' style='color:{col}'>${tp:+,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Close a Position")
            to_close = st.selectbox("Position", list(pf["positions"].keys()), key="close_sel")
            if st.button("🔴 Close Position"):
                p = pf["positions"][to_close]
                d = "sell" if p["qty"] > 0 else "buy"
                ok, msg = place_order(p["symbol"], p["otype"], p["strike"], p["expiry"],
                                      d, abs(p["qty"]), p["cur_price"], p["model"])
                if ok:
                    st.success(msg); st.rerun()
                else:
                    st.error(msg)

    # ════════════════════════════════════════════════
    # TAB 6 — ORDER HISTORY
    # ════════════════════════════════════════════════
    with t6:
        pf = st.session_state.portfolio
        st.markdown("<div class='section-header'>Order History</div>",
                    unsafe_allow_html=True)

        if not pf["orders"]:
            st.info("No orders yet.")
        else:
            dfo = pd.DataFrame(list(reversed(pf["orders"])))
            dfo["price"] = dfo["price"].map("${:.3f}".format)
            dfo["total"] = dfo["total"].map("${:,.2f}".format)

            rename = {"order_id":"ID","timestamp":"Time","symbol":"Ticker",
                      "otype":"Type","strike":"Strike","expiry":"Expiry",
                      "direction":"Dir","qty":"Qty","price":"Price",
                      "total":"Total","model":"Model"}

            def dir_color(v):
                if v == "buy":  return "color:#00c896"
                if v == "sell": return "color:#ff4d6a"
                return ""

            st.dataframe(
                dfo.rename(columns=rename)[list(rename.values())].style
                   .map(dir_color, subset=["Dir"])
                   .set_properties(**{"background-color":"#1a1d27","color":"#e8eaf0",
                                      "border":"1px solid #2d3148"}),
                width="stretch", height=500,
            )
            st.caption(f"{len(pf['orders'])} total orders")


if __name__ == "__main__":
    main()