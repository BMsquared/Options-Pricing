"""
Options Market Dashboard — Streamlit Version
=============================================
Live US stock data (Yahoo Finance) + Multi-Model Pricing Engine
Models: Black-Scholes | Heston (1993) | Dupire Local Vol (1994)
Compare model prices vs each other and track paper trades.

Run with:  streamlit run options_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import threading
import time
import warnings
import uuid

warnings.filterwarnings("ignore")

from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Options Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .metric-title { color: #6b7280; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }
    .metric-value { color: #e8eaf0; font-size: 22px; font-weight: 700; margin-top: 2px; }
    .metric-value.green { color: #00c896; }
    .metric-value.red   { color: #ff4d6a; }
    .metric-value.blue  { color: #4f8ef7; }
    .model-badge-bs     { background: #4f8ef7; color: #000; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
    .model-badge-heston { background: #b07eff; color: #000; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
    .model-badge-dupire { background: #ffb830; color: #000; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700; }
    .section-header { color: #4f8ef7; font-size: 16px; font-weight: 700; margin: 12px 0 6px 0; border-bottom: 1px solid #2d3148; padding-bottom: 6px; }
    div[data-testid="stDataFrame"] { border: 1px solid #2d3148; border-radius: 8px; }
    .stButton > button { background: #4f8ef7; color: white; border: none; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #3a7ae6; }
    .order-success { background: #0d2e1f; border: 1px solid #00c896; border-radius: 6px; padding: 8px 14px; color: #00c896; font-weight: 600; }
    .order-fail    { background: #2e0d14; border: 1px solid #ff4d6a; border-radius: 6px; padding: 8px 14px; color: #ff4d6a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION 1 — VOL SURFACE
# ─────────────────────────────────────────────────────────────

def surface_iv(S, K, T, base_vol, skew_slope=-0.10, smile_curve=0.03):
    T = max(T, 1e-4)
    log_m = np.log(K / S) / np.sqrt(T)
    atm   = base_vol * (1.0 + 0.06 * np.sqrt(T))
    skew  = skew_slope * log_m
    smile = smile_curve * log_m ** 2
    return float(np.clip(atm + skew + smile, 0.05, 2.0))


# ─────────────────────────────────────────────────────────────
# SECTION 2 — BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def bs_put(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def bs_price(S, K, T, r, sigma, option_type="call"):
    return bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if sigma <= 0 or T <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2  = d1 - sigma * np.sqrt(T)
    pdf = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        rho   =  K * T * np.exp(-r * T) * norm.cdf(d2)  / 100
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (-(S * pdf * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega  = S * np.sqrt(T) * pdf / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

def implied_vol_bs(market_price, S, K, T, r, option_type="call"):
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, option_type) - market_price,
            1e-4, 5.0, xtol=1e-6
        )
    except Exception:
        return 0.3


# ─────────────────────────────────────────────────────────────
# SECTION 3 — HESTON (1993)
# ─────────────────────────────────────────────────────────────

def heston_char_fn(phi, S, K, T, r, v0, kappa, theta_h, xi, rho, j):
    if j == 1:
        u, b = 0.5, kappa - rho * xi
    else:
        u, b = -0.5, kappa
    a  = kappa * theta_h
    x  = np.log(S)
    d  = np.sqrt((rho * xi * phi * 1j - b)**2
                 - xi**2 * (2 * u * phi * 1j - phi**2))
    g  = (b - rho * xi * phi * 1j + d) / (b - rho * xi * phi * 1j - d)
    eg = np.exp(d * T)
    C  = (r * phi * 1j * T
          + (a / xi**2) * ((b - rho * xi * phi * 1j + d) * T
          - 2 * np.log((1 - g * eg) / (1 - g))))
    D  = ((b - rho * xi * phi * 1j + d) / xi**2
          * (1 - eg) / (1 - g * eg))
    return np.exp(C + D * v0 + 1j * phi * x)

def heston_price(S, K, T, r, v0, kappa, theta_h, xi, rho, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    from scipy.integrate import quad
    def integrand(phi, j):
        cf  = heston_char_fn(phi, S, K, T, r, v0, kappa, theta_h, xi, rho, j)
        val = np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)
        return val.real
    try:
        P1 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 1), 1e-5, 200, limit=200)[0]
        P2 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 2), 1e-5, 200, limit=200)[0]
    except Exception:
        return bs_price(S, K, T, r, np.sqrt(v0), option_type)
    call = S * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return float(max(call, 0))
    else:
        return float(max(call - S + K * np.exp(-r * T), 0))

def heston_greeks(S, K, T, r, v0, kappa, theta_h, xi, rho, option_type="call", dS=0.5):
    p  = heston_price(S,      K, T,                   r, v0, kappa, theta_h, xi, rho, option_type)
    pu = heston_price(S + dS, K, T,                   r, v0, kappa, theta_h, xi, rho, option_type)
    pd = heston_price(S - dS, K, T,                   r, v0, kappa, theta_h, xi, rho, option_type)
    pt = heston_price(S,      K, max(T - 1/365, 1e-4), r, v0, kappa, theta_h, xi, rho, option_type)
    dv = 0.0001
    pvu = heston_price(S, K, T, r, v0 + dv, kappa, theta_h, xi, rho, option_type)
    pvd = heston_price(S, K, T, r, v0 - dv, kappa, theta_h, xi, rho, option_type)
    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2*p + pd) / dS**2
    theta = (pt - p) / (1/365)
    vega  = (pvu - pvd) / (2 * dv) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": 0.0}


# ─────────────────────────────────────────────────────────────
# SECTION 4 — DUPIRE LOCAL VOL (1994)
# ─────────────────────────────────────────────────────────────

def build_local_vol_surface(S, r, base_vol, skew_slope=-0.10, smile_curve=0.03):
    moneyness = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                           1.05, 1.10, 1.15, 1.20, 1.25])
    # Ensure strikes are strictly increasing (round S can cause duplicates)
    raw_strikes = moneyness * S
    # Round to 2 decimal places and deduplicate while preserving order
    seen = set()
    unique_strikes = []
    for k in raw_strikes:
        k_r = round(float(k), 2)
        if k_r not in seen:
            seen.add(k_r)
            unique_strikes.append(k_r)
    strikes  = np.array(sorted(unique_strikes))
    expiries = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.25, 1.5])

    # Recompute moneyness from cleaned strikes
    moneyness_clean = strikes / S
    iv_surf = np.zeros((len(strikes), len(expiries)))
    for i, m in enumerate(moneyness_clean):
        for j, T in enumerate(expiries):
            iv_surf[i, j] = surface_iv(S, m * S, T, base_vol, skew_slope, smile_curve)
    C_surf = np.zeros_like(iv_surf)
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            C_surf[i, j] = bs_call(S, K, T, r, iv_surf[i, j])
    lv_surf = np.zeros_like(iv_surf)
    dK = np.diff(strikes).mean()
    for j in range(len(expiries)):
        T = expiries[j]
        for i in range(1, len(strikes) - 1):
            K = strikes[i]
            if j == 0:
                dCdT = (C_surf[i, j+1] - C_surf[i, j]) / (expiries[j+1] - expiries[j])
            elif j == len(expiries) - 1:
                dCdT = (C_surf[i, j] - C_surf[i, j-1]) / (expiries[j] - expiries[j-1])
            else:
                dCdT = (C_surf[i, j+1] - C_surf[i, j-1]) / (expiries[j+1] - expiries[j-1])
            dCdK   = (C_surf[i+1, j] - C_surf[i-1, j]) / (2 * dK)
            d2CdK2 = (C_surf[i+1, j] - 2*C_surf[i, j] + C_surf[i-1, j]) / dK**2
            num = dCdT + r * K * dCdK
            den = 0.5 * K**2 * d2CdK2
            lv_surf[i, j] = (np.sqrt(np.clip(num / den, 1e-6, 4.0))
                              if den > 1e-10 and num > 0 else iv_surf[i, j])
    lv_surf[0, :]  = lv_surf[1, :]
    lv_surf[-1, :] = lv_surf[-2, :]
    lv_surf = np.clip(lv_surf, 0.01, 2.0)
    spline = RectBivariateSpline(strikes, expiries, lv_surf, kx=3, ky=3)
    def local_vol_fn(spot, t):
        spot = np.clip(spot, strikes[0],  strikes[-1])
        t    = np.clip(t,    expiries[0], expiries[-1])
        return float(np.clip(spline(spot, t), 0.01, 2.0))
    return local_vol_fn, iv_surf, lv_surf, strikes, expiries

def dupire_price_fast(S, K, T, r, local_vol_fn, option_type="call",
                       n_paths=6000, n_steps=50, seed=42):
    np.random.seed(seed)
    dt    = T / n_steps
    paths = np.full(n_paths, float(S))
    for step in range(n_steps):
        t    = step * dt
        Z    = np.random.standard_normal(n_paths)
        sig  = np.array([local_vol_fn(s, max(t, 1/365)) for s in paths])
        paths *= np.exp((r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)
    payoffs = (np.maximum(paths - K, 0) if option_type == "call"
               else np.maximum(K - paths, 0))
    return float(np.exp(-r * T) * np.mean(payoffs))

def dupire_greeks_num(S, K, T, r, local_vol_fn, option_type="call", dS=1.0):
    p  = dupire_price_fast(S,      K, T, r, local_vol_fn, option_type, seed=1)
    pu = dupire_price_fast(S + dS, K, T, r, local_vol_fn, option_type, seed=2)
    pd = dupire_price_fast(S - dS, K, T, r, local_vol_fn, option_type, seed=3)
    pt = dupire_price_fast(S, K, max(T - 7/365, 1e-4), r, local_vol_fn, option_type, seed=4)
    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2*p + pd) / dS**2
    theta = (pt - p) / (7/365) / 365
    return {"delta": delta, "gamma": gamma, "vega": 0.0, "theta": theta, "rho": 0.0}


# ─────────────────────────────────────────────────────────────
# SECTION 5 — UTILITIES
# ─────────────────────────────────────────────────────────────

def historical_vol(ticker_data, window=30):
    df = ticker_data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    closes  = df["Close"].squeeze()
    closes  = closes.dropna().sort_index()
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < window:
        return float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.25
    return float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))

def generate_strikes(S, n=7):
    if S is None or np.isnan(float(S)) or float(S) <= 0:
        return []
    step = max(float(round(S * 0.025, 0)), 1.0)
    atm  = round(float(S) / step) * step
    strikes = sorted(set([atm + (i - n // 2) * step for i in range(n)]))
    return [s for s in strikes if s > 0]

def generate_expiries():
    today = datetime.date.today()
    out   = []
    for months_ahead in [1, 2, 3, 6]:
        d         = today.replace(day=1) + datetime.timedelta(days=32 * months_ahead)
        first_day = d.replace(day=1)
        first_fri = first_day + datetime.timedelta(days=(4 - first_day.weekday()) % 7)
        third_fri = first_fri + datetime.timedelta(weeks=2)
        out.append(third_fri)
    return out

def days_to_expiry(expiry: datetime.date) -> int:
    return max((expiry - datetime.date.today()).days, 0)

def T_from_expiry(expiry: datetime.date) -> float:
    return max(days_to_expiry(expiry) / 365, 1/365)


# ─────────────────────────────────────────────────────────────
# HESTON PARAMS
# ─────────────────────────────────────────────────────────────

HESTON_PARAMS = {
    "AAPL": dict(kappa=2.0, theta_h=0.06, xi=0.4,  rho=-0.5),
    "MSFT": dict(kappa=2.0, theta_h=0.05, xi=0.35, rho=-0.5),
    "GOOGL":dict(kappa=2.5, theta_h=0.08, xi=0.45, rho=-0.55),
    "AMZN": dict(kappa=2.5, theta_h=0.09, xi=0.50, rho=-0.55),
    "TSLA": dict(kappa=3.0, theta_h=0.30, xi=0.80, rho=-0.65),
    "NVDA": dict(kappa=3.0, theta_h=0.20, xi=0.70, rho=-0.60),
    "META": dict(kappa=2.5, theta_h=0.10, xi=0.50, rho=-0.50),
    "SPY":  dict(kappa=1.5, theta_h=0.02, xi=0.25, rho=-0.70),
    "QQQ":  dict(kappa=1.5, theta_h=0.03, xi=0.30, rho=-0.65),
    "JPM":  dict(kappa=2.0, theta_h=0.04, xi=0.35, rho=-0.45),
}
DEFAULT_HESTON = dict(kappa=2.0, theta_h=0.06, xi=0.4, rho=-0.5)
POPULAR_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","SPY","QQQ","JPM"]

R = 0.05  # risk-free rate


# ─────────────────────────────────────────────────────────────
# CORE PRICING FUNCTIONS — used to build comparison rows
# ─────────────────────────────────────────────────────────────

def price_all_models(S, K, T, r, base_vol, symbol, local_vol_fn):
    """Return dict of {model: price} for a single option contract."""
    hp = HESTON_PARAMS.get(symbol, DEFAULT_HESTON)
    v0 = base_vol ** 2

    results = {}
    for otype in ["call", "put"]:
        sigma_bs = surface_iv(S, K, T, base_vol)
        bs  = bs_price(S, K, T, r, sigma_bs, otype)
        hst = heston_price(S, K, T, r, v0, hp["kappa"], hp["theta_h"], hp["xi"], hp["rho"], otype)
        dup = (dupire_price_fast(S, K, T, r, local_vol_fn, otype, n_paths=6000, n_steps=50)
               if local_vol_fn else bs)
        results[otype] = {"BS": bs, "Heston": hst, "Dupire": dup}
    return results


@st.cache_data(ttl=120, show_spinner=False)
def fetch_market_data(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="3mo", interval="1d")
        if hist is None or hist.empty:
            return None, None, None

        # yfinance sometimes returns MultiIndex columns — flatten them
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Drop duplicate column names if any
        hist = hist.loc[:, ~hist.columns.duplicated()]

        # Remove rows where Close is NaN and sort by date
        hist = hist[["Open","High","Low","Close","Volume"]].copy()
        hist = hist.dropna(subset=["Close"]).sort_index()

        if hist.empty or len(hist) < 5:
            return None, None, None

        price = float(hist["Close"].iloc[-1])
        if np.isnan(price) or price <= 0:
            return None, None, None

        vol = historical_vol(hist, window=30)
        if np.isnan(vol) or vol <= 0:
            vol = 0.25  # fallback to 25% if vol calc fails

        return price, vol, hist
    except Exception as e:
        print(f"fetch_market_data error for {symbol}: {e}")
        return None, None, None


@st.cache_data(ttl=120, show_spinner=False)
def build_chain_all_models(symbol: str):
    """Build option chain with ALL three models for the comparison table."""
    price, vol, hist = fetch_market_data(symbol)
    if price is None:
        return None, None, None

    lvf_result = build_local_vol_surface(price, R, vol)
    local_vol_fn = lvf_result[0]

    strikes  = generate_strikes(price, n=7)
    if not strikes:
        return None, None, None
    expiries = generate_expiries()
    hp       = HESTON_PARAMS.get(symbol, DEFAULT_HESTON)
    v0       = vol ** 2

    rows = []
    for expiry in expiries:
        T   = T_from_expiry(expiry)
        dte = days_to_expiry(expiry)
        for K in strikes:
            moneyness = K / price
            for otype in ["call", "put"]:
                sigma_bs = surface_iv(price, K, T, vol)
                bs_p  = bs_price(price, K, T, R, sigma_bs, otype)
                hst_p = heston_price(price, K, T, R, v0,
                                      hp["kappa"], hp["theta_h"],
                                      hp["xi"], hp["rho"], otype)
                dup_p = dupire_price_fast(price, K, T, R, local_vol_fn, otype,
                                           n_paths=5000, n_steps=40)
                # greeks (BS only for chain display — fast)
                g = bs_greeks(price, K, T, R, sigma_bs, otype)
                itm = (K < price and otype == "call") or (K > price and otype == "put")

                rows.append({
                    "Type":       otype.upper(),
                    "Strike":     K,
                    "Expiry":     expiry.strftime("%Y-%m-%d"),
                    "DTE":        dte,
                    "Moneyness":  round(moneyness, 3),
                    "ITM":        itm,
                    "BS Price":   round(bs_p, 3),
                    "Heston Price": round(hst_p, 3),
                    "Dupire Price": round(dup_p, 3),
                    "BS vs Heston": round(bs_p - hst_p, 3),
                    "BS vs Dupire": round(bs_p - dup_p, 3),
                    "Heston vs Dupire": round(hst_p - dup_p, 3),
                    "IV (σ)":     round(sigma_bs * 100, 2),
                    "Delta":      round(g["delta"], 4),
                    "Gamma":      round(g["gamma"], 5),
                    "Theta":      round(g["theta"], 4),
                    "Vega":       round(g["vega"], 4),
                })

    return pd.DataFrame(rows), price, vol


# ─────────────────────────────────────────────────────────────
# PORTFOLIO STATE (session-level)
# ─────────────────────────────────────────────────────────────

def init_portfolio():
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = {
            "cash":      100_000.0,
            "start":     100_000.0,
            "positions": {},   # label -> position dict
            "orders":    [],
        }

def place_order(symbol, otype, strike, expiry, direction, qty, price, model):
    pf    = st.session_state.portfolio
    cost  = price * qty * 100 * (1 if direction == "buy" else -1)
    if direction == "buy" and cost > pf["cash"]:
        return False, f"Insufficient cash. Need ${cost:,.2f}, have ${pf['cash']:,.2f}"

    oid = str(uuid.uuid4())[:8].upper()
    order = {
        "order_id":  oid,
        "symbol":    symbol,
        "otype":     otype,
        "strike":    strike,
        "expiry":    expiry,
        "direction": direction,
        "qty":       qty,
        "price":     price,
        "model":     model,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "total":     price * qty * 100,
    }
    pf["orders"].append(order)
    pf["cash"] -= cost

    label = f"{symbol} {expiry} {strike:.0f} {otype.upper()}"
    signed = qty if direction == "buy" else -qty
    if label in pf["positions"]:
        pos     = pf["positions"][label]
        new_qty = pos["qty"] + signed
        if new_qty == 0:
            del pf["positions"][label]
        else:
            total_cost     = pos["avg_price"] * abs(pos["qty"]) + price * qty
            pos["avg_price"] = total_cost / abs(new_qty)
            pos["qty"]       = new_qty
    else:
        pf["positions"][label] = {
            "symbol":    symbol,
            "otype":     otype,
            "strike":    strike,
            "expiry":    expiry,
            "qty":       signed,
            "avg_price": price,
            "model":     model,
            "cur_price": price,
        }

    return True, f"✓ {oid}  {direction.upper()} {qty}× {label} @ ${price:.2f} [{model}]"


def refresh_position_prices(df_chain):
    """Update current prices in positions using latest chain data."""
    if df_chain is None:
        return
    pf = st.session_state.portfolio
    price_map = {}
    for _, row in df_chain.iterrows():
        lbl = f"{row['Type']} {row['Expiry']} {row['Strike']:.0f}"
        for model in ["BS Price", "Heston Price", "Dupire Price"]:
            price_map[(lbl, model)] = row[model]

    for label, pos in pf["positions"].items():
        parts = label.split()
        lbl   = f"{pos['otype'].upper()} {pos['expiry']} {pos['strike']:.0f}"
        model_col = pos["model"] + " Price"
        if (lbl, model_col) in price_map:
            pos["cur_price"] = price_map[(lbl, model_col)]


# ─────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#1a1d27",
    font=dict(color="#e8eaf0", size=11),
    xaxis=dict(gridcolor="#2d3148", zerolinecolor="#2d3148"),
    yaxis=dict(gridcolor="#2d3148", zerolinecolor="#2d3148"),
    margin=dict(l=50, r=20, t=40, b=40),
)

MODEL_COLORS = {
    "BS Price":     "#4f8ef7",
    "Heston Price": "#b07eff",
    "Dupire Price": "#ffb830",
}

def fig_price_comparison(df_sub, otype, spot):
    """Bar chart comparing three model prices for a fixed expiry."""
    d = df_sub[df_sub["Type"] == otype.upper()].copy()
    fig = go.Figure()
    for col, color in MODEL_COLORS.items():
        fig.add_trace(go.Bar(
            name=col.replace(" Price",""),
            x=d["Strike"],
            y=d[col],
            marker_color=color,
            opacity=0.85,
        ))
    fig.add_vline(x=spot, line_dash="dot", line_color="#00c896",
                  annotation_text=f"Spot ${spot:.0f}", annotation_font_color="#00c896")
    fig.update_layout(
        title=f"{otype.upper()} — Model Price Comparison",
        barmode="group",
        xaxis_title="Strike",
        yaxis_title="Option Price ($)",
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3148"),
        **PLOTLY_TEMPLATE,
    )
    return fig

def fig_iv_skew(df_sub, spot):
    """IV skew by strike for each expiry."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (exp, grp) in enumerate(df_sub[df_sub["Type"]=="CALL"].groupby("Expiry")):
        fig.add_trace(go.Scatter(
            x=grp["Strike"], y=grp["IV (σ)"],
            mode="lines+markers",
            name=exp,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    fig.add_vline(x=spot, line_dash="dot", line_color="#00c896")
    fig.update_layout(
        title="Implied Vol Skew (BS Surface)",
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3148"),
        **PLOTLY_TEMPLATE,
    )
    return fig

def fig_price_diff_heatmap(df_filtered):
    """Heatmap of BS vs Heston price difference."""
    d = df_filtered[df_filtered["Type"] == "CALL"].copy()
    pivot = d.pivot_table(index="Strike", columns="Expiry", values="BS vs Heston")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="BS − Heston ($)"),
    ))
    fig.update_layout(
        title="BS vs Heston Price Difference — CALL",
        xaxis_title="Expiry",
        yaxis_title="Strike",
        **PLOTLY_TEMPLATE,
    )
    return fig

def fig_greeks_by_strike(df_sub, greek, otype):
    d = df_sub[df_sub["Type"] == otype.upper()].copy()
    fig = go.Figure()
    for exp, grp in d.groupby("Expiry"):
        fig.add_trace(go.Scatter(
            x=grp["Strike"], y=grp[greek],
            mode="lines+markers", name=exp,
        ))
    fig.update_layout(
        title=f"{greek} by Strike — {otype.upper()}",
        xaxis_title="Strike",
        yaxis_title=greek,
        **PLOTLY_TEMPLATE,
    )
    return fig

def fig_stock_history(hist, symbol):
    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"],
        low=hist["Low"],  close=hist["Close"],
        name=symbol,
        increasing_line_color="#00c896",
        decreasing_line_color="#ff4d6a",
    ))
    fig.update_layout(
        title=f"{symbol} — 3-Month Price History",
        xaxis_rangeslider_visible=False,
        **PLOTLY_TEMPLATE,
    )
    return fig

def fig_model_spread(df_exp, otype):
    """Line chart of all 3 model prices vs strike for a single expiry."""
    d = df_exp[df_exp["Type"] == otype.upper()].copy().sort_values("Strike")
    fig = go.Figure()
    for col, color in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=d["Strike"], y=d[col],
            mode="lines+markers",
            name=col.replace(" Price",""),
            line=dict(color=color, width=2),
        ))
    fig.update_layout(
        title=f"Model Price Spread — {otype.upper()}",
        xaxis_title="Strike",
        yaxis_title="Price ($)",
        legend=dict(bgcolor="#1a1d27", bordercolor="#2d3148"),
        **PLOTLY_TEMPLATE,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():
    init_portfolio()

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⬡ Options Dashboard")
        st.markdown("---")

        symbol = st.selectbox("Ticker", POPULAR_TICKERS, index=0)
        custom = st.text_input("Or enter ticker:", "").upper().strip()
        if custom:
            symbol = custom

        st.markdown("---")
        st.markdown("### Model Settings")
        r_input = st.slider("Risk-Free Rate (%)", 1.0, 10.0, 5.0, 0.25)
        global R
        R = r_input / 100

        st.markdown("---")
        st.markdown("### Paper Trading")
        cash = st.session_state.portfolio["cash"]
        start = st.session_state.portfolio["start"]
        pnl   = cash - start + sum(
            (p["cur_price"] - p["avg_price"]) * abs(p["qty"]) * 100
            for p in st.session_state.portfolio["positions"].values()
        )
        pnl_col = "green" if pnl >= 0 else "red"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>CASH</div>
            <div class='metric-value blue'>${cash:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>TOTAL P&L</div>
            <div class='metric-value {pnl_col}'>${pnl:+,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Reset Portfolio"):
            st.session_state.portfolio = {
                "cash": 100_000.0, "start": 100_000.0,
                "positions": {}, "orders": []
            }
            st.rerun()

    # ── LOAD DATA ─────────────────────────────────────────────
    with st.spinner(f"Fetching {symbol} data and computing all 3 model chains..."):
        df_chain, spot, vol = build_chain_all_models(symbol)

    if df_chain is None:
        st.error(f"Could not fetch data for **{symbol}**. Please try another ticker.")
        st.stop()

    refresh_position_prices(df_chain)

    # ── HEADER ────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>SPOT PRICE</div>
            <div class='metric-value blue'>${spot:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>HV30 (HIST. VOL)</div>
            <div class='metric-value'>{vol*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>RISK-FREE RATE</div>
            <div class='metric-value'>{R*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        n_pos = len(st.session_state.portfolio["positions"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>OPEN POSITIONS</div>
            <div class='metric-value'>{n_pos}</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        n_ord = len(st.session_state.portfolio["orders"])
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>TOTAL ORDERS</div>
            <div class='metric-value'>{n_ord}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────
    tab_compare, tab_chain, tab_charts, tab_trade, tab_portfolio, tab_history = st.tabs([
        "📊 Model Comparison",
        "📋 Option Chain",
        "📈 Charts & Vol Surface",
        "🛒 Trade",
        "💼 Portfolio",
        "📜 Order History",
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — MODEL COMPARISON (the main differentiator)
    # ════════════════════════════════════════════════════════
    with tab_compare:
        st.markdown("<div class='section-header'>Model Price Comparison — BS vs Heston vs Dupire</div>",
                    unsafe_allow_html=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            expiry_opts = sorted(df_chain["Expiry"].unique())
            sel_exp = st.selectbox("Expiry", expiry_opts, key="cmp_exp")
        with col_f2:
            otype_cmp = st.selectbox("Option Type", ["CALL", "PUT"], key="cmp_otype")
        with col_f3:
            show_diff_only = st.checkbox("Highlight large differences (>$0.10)", value=False)

        df_exp = df_chain[df_chain["Expiry"] == sel_exp].copy()

        # Price comparison bar chart
        st.plotly_chart(fig_price_comparison(df_exp, otype_cmp, spot),
                        use_container_width=True)

        # Model spread line
        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(fig_model_spread(df_exp, otype_cmp),
                            use_container_width=True)
        with col_r:
            st.plotly_chart(fig_price_diff_heatmap(df_chain),
                            use_container_width=True)

        # Summary stats table
        st.markdown("#### Price Difference Summary")
        df_cmp = df_exp[df_exp["Type"] == otype_cmp].copy()

        if show_diff_only:
            df_cmp = df_cmp[
                (df_cmp["BS vs Heston"].abs() > 0.10) |
                (df_cmp["BS vs Dupire"].abs() > 0.10)
            ]

        display_cols = ["Strike", "Moneyness", "DTE",
                        "BS Price", "Heston Price", "Dupire Price",
                        "BS vs Heston", "BS vs Dupire", "Heston vs Dupire", "IV (σ)"]

        def color_diff(val):
            if isinstance(val, float):
                if val > 0.10:   return "color: #ff4d6a"
                if val < -0.10:  return "color: #4f8ef7"
                if abs(val) > 0.05: return "color: #f5a623"
            return ""

        styled = df_cmp[display_cols].style \
            .applymap(color_diff, subset=["BS vs Heston", "BS vs Dupire", "Heston vs Dupire"]) \
            .format({
                "Strike":    "${:.0f}",
                "BS Price":  "${:.3f}",
                "Heston Price": "${:.3f}",
                "Dupire Price": "${:.3f}",
                "BS vs Heston":    "${:+.3f}",
                "BS vs Dupire":    "${:+.3f}",
                "Heston vs Dupire":"${:+.3f}",
                "IV (σ)":    "{:.2f}%",
                "Moneyness": "{:.3f}",
            }) \
            .set_properties(**{"background-color": "#1a1d27", "color": "#e8eaf0", "border": "1px solid #2d3148"})

        st.dataframe(styled, use_container_width=True, height=320)

        # Legend
        st.markdown("""
        <small style='color:#6b7280;'>
        🔵 <b>Black-Scholes</b>: Vol surface σ(K,T) with skew & term structure &nbsp;|&nbsp;
        🟣 <b>Heston</b>: Stochastic vol, mean-reverting variance &nbsp;|&nbsp;
        🟠 <b>Dupire</b>: Local vol σ_L(K,T) via Dupire formula + Monte Carlo &nbsp;|&nbsp;
        Red diff = BS overprices vs other model &nbsp;|&nbsp; Blue diff = BS underprices
        </small>
        """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════
    # TAB 2 — FULL OPTION CHAIN
    # ════════════════════════════════════════════════════════
    with tab_chain:
        st.markdown("<div class='section-header'>Full Option Chain — All Models</div>",
                    unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            exp_chain = st.selectbox("Expiry", ["All"] + sorted(df_chain["Expiry"].unique()), key="chain_exp")
        with col_b:
            type_chain = st.selectbox("Type", ["All","CALL","PUT"], key="chain_type")
        with col_c:
            itm_filter = st.selectbox("Moneyness", ["All","ITM Only","OTM Only"], key="chain_itm")

        df_view = df_chain.copy()
        if exp_chain  != "All":  df_view = df_view[df_view["Expiry"] == exp_chain]
        if type_chain != "All":  df_view = df_view[df_view["Type"]   == type_chain]
        if itm_filter == "ITM Only": df_view = df_view[df_view["ITM"]]
        if itm_filter == "OTM Only": df_view = df_view[~df_view["ITM"]]

        show_cols = ["Type","Strike","Expiry","DTE","BS Price","Heston Price","Dupire Price",
                     "BS vs Heston","BS vs Dupire","IV (σ)","Delta","Gamma","Theta","Vega","ITM"]

        def highlight_itm(row):
            if row["ITM"]:
                return ["background-color: #1a2e1a"] * len(row)
            return [""] * len(row)

        styled_chain = df_view[show_cols].style \
            .apply(highlight_itm, axis=1) \
            .format({
                "Strike":        "${:.0f}",
                "BS Price":      "${:.3f}",
                "Heston Price":  "${:.3f}",
                "Dupire Price":  "${:.3f}",
                "BS vs Heston":  "${:+.3f}",
                "BS vs Dupire":  "${:+.3f}",
                "IV (σ)":        "{:.2f}%",
                "Delta":         "{:.4f}",
                "Gamma":         "{:.5f}",
                "Theta":         "{:.4f}",
                "Vega":          "{:.4f}",
            }) \
            .set_properties(**{"background-color": "#1a1d27", "color": "#e8eaf0", "border": "1px solid #2d3148"})

        st.dataframe(styled_chain, use_container_width=True, height=460)
        st.caption(f"{len(df_view)} contracts shown  |  Green rows = ITM  |  Spot: ${spot:.2f}")


    # ════════════════════════════════════════════════════════
    # TAB 3 — CHARTS & VOL SURFACE
    # ════════════════════════════════════════════════════════
    with tab_charts:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            greek_pick = st.selectbox("Greek to plot", ["Delta","Gamma","Theta","Vega"], key="greek_pick")
        with col_c2:
            otype_chart = st.selectbox("Option Type", ["CALL","PUT"], key="chart_otype")

        _, hist_data = fetch_market_data(symbol)[1], fetch_market_data(symbol)[2]

        row1_l, row1_r = st.columns(2)
        with row1_l:
            st.plotly_chart(fig_iv_skew(df_chain, spot), use_container_width=True)
        with row1_r:
            st.plotly_chart(fig_greeks_by_strike(df_chain, greek_pick, otype_chart),
                            use_container_width=True)

        if hist_data is not None:
            st.plotly_chart(fig_stock_history(hist_data, symbol), use_container_width=True)

        # 3D vol surface
        st.markdown("#### 3D Implied Vol Surface (Black-Scholes)")
        calls = df_chain[df_chain["Type"] == "CALL"].copy()
        pivot_iv = calls.pivot_table(index="Strike", columns="Expiry", values="IV (σ)")
        fig3d = go.Figure(go.Surface(
            z=pivot_iv.values,
            x=pivot_iv.columns.tolist(),
            y=pivot_iv.index.tolist(),
            colorscale="Viridis",
            opacity=0.85,
        ))
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Expiry",
                yaxis_title="Strike",
                zaxis_title="IV (%)",
                bgcolor="#1a1d27",
            ),
            paper_bgcolor="#0f1117",
            font=dict(color="#e8eaf0"),
            margin=dict(l=0, r=0, t=30, b=0),
            height=480,
        )
        st.plotly_chart(fig3d, use_container_width=True)


    # ════════════════════════════════════════════════════════
    # TAB 4 — TRADE ENTRY
    # ════════════════════════════════════════════════════════
    with tab_trade:
        st.markdown("<div class='section-header'>Paper Trade Entry</div>",
                    unsafe_allow_html=True)

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            trade_exp   = st.selectbox("Expiry",    sorted(df_chain["Expiry"].unique()), key="tr_exp")
        with col_t2:
            trade_type  = st.selectbox("Type",      ["CALL","PUT"],            key="tr_type")
        with col_t3:
            avail_strikes = sorted(df_chain[
                (df_chain["Expiry"] == trade_exp) &
                (df_chain["Type"]   == trade_type)
            ]["Strike"].unique())
            trade_strike = st.selectbox("Strike",  avail_strikes,              key="tr_strike")
        with col_t4:
            trade_model  = st.selectbox("Price via Model", ["BS","Heston","Dupire"], key="tr_model")

        row = df_chain[
            (df_chain["Expiry"] == trade_exp) &
            (df_chain["Type"]   == trade_type) &
            (df_chain["Strike"] == trade_strike)
        ]

        if not row.empty:
            row = row.iloc[0]
            model_col = f"{trade_model} Price"
            mid_price = row[model_col]

            st.markdown("#### Selected Contract")
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("BS Price",     f"${row['BS Price']:.3f}")
            mc2.metric("Heston Price", f"${row['Heston Price']:.3f}")
            mc3.metric("Dupire Price", f"${row['Dupire Price']:.3f}")
            mc4.metric("IV",           f"{row['IV (σ)']:.2f}%")
            mc5.metric("Delta",        f"{row['Delta']:.4f}")
            mc6.metric("DTE",          f"{row['DTE']} days")

            st.markdown(f"**Pricing at:** <span class='model-badge-{'bs' if trade_model=='BS' else 'heston' if trade_model=='Heston' else 'dupire'}'>{trade_model}</span> = **${mid_price:.3f}** per share (×100/contract = **${mid_price*100:.2f}**/contract)", unsafe_allow_html=True)

            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                direction = st.radio("Direction", ["buy","sell"], horizontal=True, key="tr_dir")
            with col_d2:
                qty = st.number_input("Contracts", min_value=1, max_value=100, value=1, key="tr_qty")
            with col_d3:
                cost_display = mid_price * qty * 100
                st.metric("Total Cost", f"${cost_display:,.2f}")

            if st.button(f"🚀 {'BUY' if direction=='buy' else 'SELL'} {qty}× {trade_type} ${trade_strike:.0f} {trade_exp}", type="primary"):
                ok, msg = place_order(
                    symbol, trade_type.lower(), trade_strike, trade_exp,
                    direction, qty, mid_price, trade_model
                )
                if ok:
                    st.markdown(f"<div class='order-success'>{msg}</div>", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"<div class='order-fail'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.warning("No data found for this selection.")


    # ════════════════════════════════════════════════════════
    # TAB 5 — PORTFOLIO
    # ════════════════════════════════════════════════════════
    with tab_portfolio:
        pf = st.session_state.portfolio
        st.markdown("<div class='section-header'>Open Positions</div>", unsafe_allow_html=True)

        if not pf["positions"]:
            st.info("No open positions. Use the Trade tab to enter positions.")
        else:
            pos_rows = []
            for label, p in pf["positions"].items():
                unr_pnl = (p["cur_price"] - p["avg_price"]) * p["qty"] * 100
                mkt_val  = p["cur_price"] * abs(p["qty"]) * 100
                pnl_pct  = (unr_pnl / (p["avg_price"] * abs(p["qty"]) * 100)) * 100 if p["avg_price"] > 0 else 0
                pos_rows.append({
                    "Label":      label,
                    "Qty":        p["qty"],
                    "Avg Price":  p["avg_price"],
                    "Cur Price":  p["cur_price"],
                    "Mkt Value":  mkt_val,
                    "Unr. P&L":   unr_pnl,
                    "P&L %":      pnl_pct,
                    "Model":      p["model"],
                })
            df_pos = pd.DataFrame(pos_rows)

            def color_pnl(val):
                if isinstance(val, float):
                    return "color: #00c896" if val >= 0 else "color: #ff4d6a"
                return ""

            styled_pos = df_pos.style \
                .applymap(color_pnl, subset=["Unr. P&L","P&L %"]) \
                .format({
                    "Avg Price": "${:.3f}",
                    "Cur Price": "${:.3f}",
                    "Mkt Value": "${:,.2f}",
                    "Unr. P&L":  "${:+,.2f}",
                    "P&L %":     "{:+.1f}%",
                }) \
                .set_properties(**{"background-color":"#1a1d27","color":"#e8eaf0","border":"1px solid #2d3148"})

            st.dataframe(styled_pos, use_container_width=True)

            total_pnl = sum(r["Unr. P&L"] for r in pos_rows)
            total_val = sum(r["Mkt Value"] for r in pos_rows)
            pnl_col   = "#00c896" if total_pnl >= 0 else "#ff4d6a"

            st.markdown(f"""
            <div class='metric-card' style='display:inline-block; margin-right:16px;'>
                <div class='metric-title'>PORTFOLIO MKT VALUE</div>
                <div class='metric-value blue'>${total_val:,.2f}</div>
            </div>
            <div class='metric-card' style='display:inline-block;'>
                <div class='metric-title'>TOTAL UNREALIZED P&L</div>
                <div class='metric-value' style='color:{pnl_col}'>${total_pnl:+,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # Close position
            st.markdown("#### Close a Position")
            close_label = st.selectbox("Select position to close",
                                        list(pf["positions"].keys()), key="close_pos")
            if st.button("🔴 Close Position"):
                pos = pf["positions"][close_label]
                direction = "sell" if pos["qty"] > 0 else "buy"
                ok, msg = place_order(
                    pos["symbol"], pos["otype"], pos["strike"], pos["expiry"],
                    direction, abs(pos["qty"]), pos["cur_price"], pos["model"]
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)


    # ════════════════════════════════════════════════════════
    # TAB 6 — ORDER HISTORY
    # ════════════════════════════════════════════════════════
    with tab_portfolio:
        pass  # already handled above

    with tab_history:
        pf = st.session_state.portfolio
        st.markdown("<div class='section-header'>Order History</div>", unsafe_allow_html=True)

        if not pf["orders"]:
            st.info("No orders placed yet.")
        else:
            df_orders = pd.DataFrame(reversed(pf["orders"]))
            df_orders["Total ($)"] = df_orders["total"].map("${:,.2f}".format)
            df_orders["Price"]     = df_orders["price"].map("${:.3f}".format)

            show_ord = ["order_id","timestamp","symbol","otype","strike","expiry",
                        "direction","qty","Price","Total ($)","model"]
            rename_map = {
                "order_id":"Order ID","timestamp":"Time","symbol":"Symbol",
                "otype":"Type","strike":"Strike","expiry":"Expiry",
                "direction":"Dir","qty":"Qty","model":"Model",
            }

            def color_dir(val):
                if val == "buy":  return "color: #00c896"
                if val == "sell": return "color: #ff4d6a"
                return ""

            styled_ord = df_orders[show_ord].rename(columns=rename_map).style \
                .applymap(color_dir, subset=["Dir"]) \
                .set_properties(**{"background-color":"#1a1d27","color":"#e8eaf0","border":"1px solid #2d3148"})

            st.dataframe(styled_ord, use_container_width=True, height=480)
            st.caption(f"{len(pf['orders'])} total orders")


if __name__ == "__main__":
    main()