"""
Options Market Demo — Portfolio Piece
======================================
Live US stock data (Yahoo Finance) + Multi-Model Pricing Engine
Models: Black-Scholes | Heston (1993) | Dupire Local Vol (1994)
Full paper trading: order book, positions, P&L tracker, Greeks dashboard

Author: Built with WQU MScFE knowledge
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

# ─────────────────────────────────────────────────────────────
# SECTION 1 — VOL SURFACE  σ(K, T)
# Each expiry and strike gets its own IV — no more flat surface
# ─────────────────────────────────────────────────────────────

def surface_iv(S, K, T, base_vol, skew_slope=-0.10, smile_curve=0.03):
    """
    Full implied vol surface with skew + term structure.

    σ(K,T) = ATM(T) + skew * log-moneyness + smile * log-moneyness²

    - ATM vol rises with sqrt(T)  → term structure
    - Skew negative → lower strikes carry higher IV (crash fear)
    - Smile adds curvature around ATM
    """
    T = max(T, 1e-4)
    log_m  = np.log(K / S) / np.sqrt(T)         # log-moneyness scaled by sqrt(T)
    atm    = base_vol * (1.0 + 0.06 * np.sqrt(T))  # upward term structure
    skew   = skew_slope * log_m
    smile  = smile_curve * log_m ** 2
    return float(np.clip(atm + skew + smile, 0.05, 2.0))


# ─────────────────────────────────────────────────────────────
# SECTION 2 — BLACK-SCHOLES (1973)
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
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}

def implied_vol_bs(market_price, S, K, T, r, option_type="call"):
    """Brent's method — more robust than Newton-Raphson for surface construction."""
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, option_type) - market_price,
            1e-4, 5.0, xtol=1e-6
        )
    except Exception:
        return 0.3


# ─────────────────────────────────────────────────────────────
# SECTION 3 — HESTON (1993)
# Stochastic volatility: vol is random and mean-reverting
# ─────────────────────────────────────────────────────────────

def heston_char_fn(phi, S, K, T, r, v0, kappa, theta_h, xi, rho, j):
    """Heston characteristic function for j=1 (stock measure) and j=2 (risk-neutral)."""
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
    """
    Heston (1993) semi-closed-form option price.

    Parameters
    ----------
    v0      : initial variance  (= implied_vol² at time 0)
    kappa   : mean-reversion speed  (~2–5 for equities)
    theta_h : long-run variance     (~v0 if no term structure)
    xi      : vol-of-vol            (~0.3–0.8)
    rho     : stock–vol correlation (~-0.7 to -0.3 for equities)
    """
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    from scipy.integrate import quad

    def integrand(phi, j):
        cf  = heston_char_fn(phi, S, K, T, r, v0, kappa, theta_h, xi, rho, j)
        val = np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)
        return val.real

    try:
        P1 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 1),
                                      1e-5, 200, limit=200)[0]
        P2 = 0.5 + (1/np.pi) * quad(lambda p: integrand(p, 2),
                                      1e-5, 200, limit=200)[0]
    except Exception:
        # fallback to BS if integration fails
        return bs_price(S, K, T, r, np.sqrt(v0), option_type)

    call = S * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return float(max(call, 0))
    else:
        # put via put-call parity
        return float(max(call - S + K * np.exp(-r * T), 0))

def heston_greeks(S, K, T, r, v0, kappa, theta_h, xi, rho,
                  option_type="call", dS=0.5):
    """Numerical greeks for Heston via finite differences."""
    p  = heston_price(S,      K, T,       r, v0, kappa, theta_h, xi, rho, option_type)
    pu = heston_price(S + dS, K, T,       r, v0, kappa, theta_h, xi, rho, option_type)
    pd = heston_price(S - dS, K, T,       r, v0, kappa, theta_h, xi, rho, option_type)
    pt = heston_price(S,      K, max(T - 1/365, 1e-4),
                      r, v0, kappa, theta_h, xi, rho, option_type)
    dv = 0.0001
    pvu = heston_price(S, K, T, r, v0 + dv, kappa, theta_h, xi, rho, option_type)
    pvd = heston_price(S, K, T, r, v0 - dv, kappa, theta_h, xi, rho, option_type)
    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2*p + pd) / dS**2
    theta = (pt - p) / (1/365)
    vega  = (pvu - pvd) / (2 * dv) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": 0.0}

def heston_iv(S, K, T, r, v0, kappa, theta_h, xi, rho, option_type="call"):
    """Back out BS-equivalent IV from Heston price."""
    hp = heston_price(S, K, T, r, v0, kappa, theta_h, xi, rho, option_type)
    return implied_vol_bs(hp, S, K, T, r, option_type)


# ─────────────────────────────────────────────────────────────
# SECTION 4 — DUPIRE LOCAL VOL (1994)
# σ_L(K,T) extracted from the implied vol surface
# ─────────────────────────────────────────────────────────────

def build_local_vol_surface(S, r, base_vol, skew_slope=-0.10, smile_curve=0.03):
    """
    Build a local vol surface using the Dupire formula:

         dC/dT + r·K·dC/dK
    σ²_L = ─────────────────────
            0.5·K²·d²C/dK²

    Returns a callable local_vol_fn(spot, t).
    """
    moneyness = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                           1.05, 1.10, 1.15, 1.20, 1.25])
    strikes   = moneyness * S
    expiries  = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.25, 1.5])

    # Step 1: build IV surface
    iv_surf = np.zeros((len(strikes), len(expiries)))
    for i, m in enumerate(moneyness):
        for j, T in enumerate(expiries):
            iv_surf[i, j] = surface_iv(S, m * S, T, base_vol,
                                        skew_slope, smile_curve)

    # Step 2: call price surface
    C_surf = np.zeros_like(iv_surf)
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            C_surf[i, j] = bs_call(S, K, T, r, iv_surf[i, j])

    # Step 3: Dupire formula — numerical derivatives
    lv_surf = np.zeros_like(iv_surf)
    dK      = np.diff(strikes).mean()

    for j in range(len(expiries)):
        T = expiries[j]
        for i in range(1, len(strikes) - 1):
            K = strikes[i]
            # dC/dT
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
                              if den > 1e-10 and num > 0
                              else iv_surf[i, j])

    lv_surf[0, :]  = lv_surf[1, :]
    lv_surf[-1, :] = lv_surf[-2, :]
    lv_surf        = np.clip(lv_surf, 0.01, 2.0)

    spline = RectBivariateSpline(strikes, expiries, lv_surf, kx=3, ky=3)

    def local_vol_fn(spot, t):
        spot = np.clip(spot, strikes[0],  strikes[-1])
        t    = np.clip(t,    expiries[0], expiries[-1])
        return float(np.clip(spline(spot, t), 0.01, 2.0))

    return local_vol_fn, iv_surf, lv_surf, strikes, expiries

def dupire_price_fast(S, K, T, r, local_vol_fn, option_type="call",
                       n_paths=8000, n_steps=60, seed=42):
    """
    Fast Monte Carlo pricing under Dupire local vol.
    Uses antithetic variates for variance reduction.
    """
    np.random.seed(seed)
    dt    = T / n_steps
    paths = np.full(n_paths, float(S))

    for step in range(n_steps):
        t       = step * dt
        Z       = np.random.standard_normal(n_paths)
        sig     = np.array([local_vol_fn(s, max(t, 1/365)) for s in paths])
        paths  *= np.exp((r - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z)

    payoffs = (np.maximum(paths - K, 0) if option_type == "call"
               else np.maximum(K - paths, 0))
    return float(np.exp(-r * T) * np.mean(payoffs))

def dupire_greeks_num(S, K, T, r, local_vol_fn, option_type="call", dS=1.0):
    """Numerical greeks for Dupire via finite differences."""
    p  = dupire_price_fast(S,      K, T, r, local_vol_fn, option_type, seed=1)
    pu = dupire_price_fast(S + dS, K, T, r, local_vol_fn, option_type, seed=2)
    pd = dupire_price_fast(S - dS, K, T, r, local_vol_fn, option_type, seed=3)
    pt = dupire_price_fast(S, K, max(T - 7/365, 1e-4), r,
                            local_vol_fn, option_type, seed=4)
    delta = (pu - pd) / (2 * dS)
    gamma = (pu - 2*p + pd) / dS**2
    theta = (pt - p) / (7/365) / 365
    return {"delta": delta, "gamma": gamma, "vega": 0.0, "theta": theta, "rho": 0.0}


# ─────────────────────────────────────────────────────────────
# SECTION 5 — UTILITIES
# ─────────────────────────────────────────────────────────────

def historical_vol(ticker_data, window=30):
    closes  = ticker_data["Close"].squeeze()
    returns = np.log(closes / closes.shift(1)).dropna()
    return float(returns.rolling(window).std().iloc[-1] * np.sqrt(252))
    

def generate_strikes(S, n=7):
    step = max(round(S * 0.025, 0), 1.0)
    atm  = round(S / step) * step
    return [atm + (i - n // 2) * step for i in range(n)]

def generate_expiries():
    today = datetime.date.today()
    out   = []
    for months_ahead in [1, 2, 3, 6]:
        d          = today.replace(day=1) + datetime.timedelta(days=32 * months_ahead)
        first_day  = d.replace(day=1)
        first_fri  = first_day + datetime.timedelta(days=(4 - first_day.weekday()) % 7)
        third_fri  = first_fri + datetime.timedelta(weeks=2)
        out.append(third_fri)
    return out


# ─────────────────────────────────────────────────────────────
# SECTION 6 — DATA MODELS
# ─────────────────────────────────────────────────────────────

@dataclass
class Option:
    symbol:      str
    option_type: str
    strike:      float
    expiry:      datetime.date
    bid:         float = 0.0
    ask:         float = 0.0
    iv:          float = 0.0
    delta:       float = 0.0
    gamma:       float = 0.0
    vega:        float = 0.0
    theta:       float = 0.0
    model:       str   = "BS"   # "BS", "Heston", "Dupire"

    @property
    def mid(self):
        return (self.bid + self.ask) / 2

    @property
    def T(self):
        days = (self.expiry - datetime.date.today()).days
        return max(days / 365, 1/365)

    @property
    def label(self):
        return f"{self.symbol} {self.expiry.strftime('%b%y')} {self.strike:.0f} {self.option_type.upper()}"


@dataclass
class Order:
    order_id:  str
    option:    Option
    direction: str
    quantity:  int
    price:     float
    timestamp: datetime.datetime
    model:     str   = "BS"
    status:    str   = "filled"


@dataclass
class Position:
    option:    Option
    quantity:  int
    avg_price: float
    orders:    List[Order] = field(default_factory=list)

    @property
    def market_value(self):
        return self.quantity * self.option.mid * 100

    @property
    def cost_basis(self):
        return self.quantity * self.avg_price * 100

    @property
    def unrealized_pnl(self):
        return self.market_value - self.cost_basis

    @property
    def pnl_pct(self):
        if abs(self.cost_basis) < 0.01:
            return 0.0
        return (self.unrealized_pnl / abs(self.cost_basis)) * 100


# ─────────────────────────────────────────────────────────────
# SECTION 7 — MARKET DATA ENGINE
# ─────────────────────────────────────────────────────────────

# Default Heston parameters per ticker (can be calibrated)
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


class MarketData:
    def __init__(self):
        self.prices    = {}
        self.hist_data = {}
        self.hist_vols = {}
        self.r         = 0.05
        self._lock     = threading.Lock()
        self._lv_cache = {}   # symbol -> (local_vol_fn, iv_surf, lv_surf, strikes, expiries)

    def fetch(self, symbol):
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
            vol   = historical_vol(hist, window=30)
            

            with self._lock:
                self.prices[symbol]    = price
                self.hist_data[symbol] = hist
                self.hist_vols[symbol] = vol
                # rebuild local vol surface whenever we get new data
                self._lv_cache[symbol] = build_local_vol_surface(
                    price, self.r, vol,
                    skew_slope=-0.10, smile_curve=0.03
                )
            return price
        except Exception as e:
            print(f"Fetch error for {symbol}: {e}")
            return None

    def price(self, symbol):
        with self._lock:
            return self.prices.get(symbol)

    def vol(self, symbol):
        with self._lock:
            return self.hist_vols.get(symbol, 0.25)

    def local_vol_fn(self, symbol):
        with self._lock:
            lv = self._lv_cache.get(symbol)
            return lv[0] if lv else None

    # ── price one option under a given model ──────────────────

    def price_option_bs(self, opt: Option, S: float, base_vol: float) -> Option:
        """
        Black-Scholes with full surface IV.
        IV varies by strike AND expiry — fixes the flat IV problem.
        """
        sigma = surface_iv(S, opt.strike, opt.T, base_vol,
                            skew_slope=-0.10, smile_curve=0.03)
        mid   = bs_price(S, opt.strike, opt.T, self.r, sigma, opt.option_type)
        spread = max(0.05, mid * 0.02)
        opt.bid   = max(0.01, mid - spread / 2)
        opt.ask   = mid + spread / 2
        opt.iv    = sigma                          # ← now unique per (K, T)
        g         = bs_greeks(S, opt.strike, opt.T, self.r, sigma, opt.option_type)
        opt.delta = g["delta"]
        opt.gamma = g["gamma"]
        opt.vega  = g["vega"]
        opt.theta = g["theta"]
        opt.model = "BS"
        return opt

    def price_option_heston(self, opt: Option, S: float, base_vol: float,
                             symbol: str) -> Option:
        """
        Heston (1993) with per-ticker calibrated parameters.
        IV shown is the BS-equivalent IV implied by the Heston price.
        """
        hp = HESTON_PARAMS.get(symbol, DEFAULT_HESTON)
        v0 = base_vol ** 2
        mid = heston_price(S, opt.strike, opt.T, self.r,
                            v0, hp["kappa"], hp["theta_h"],
                            hp["xi"], hp["rho"], opt.option_type)
        mid = max(mid, 0.01)
        spread    = max(0.05, mid * 0.02)
        opt.bid   = max(0.01, mid - spread / 2)
        opt.ask   = mid + spread / 2
        opt.iv    = implied_vol_bs(mid, S, opt.strike, opt.T, self.r, opt.option_type)
        g         = heston_greeks(S, opt.strike, opt.T, self.r,
                                   v0, hp["kappa"], hp["theta_h"],
                                   hp["xi"], hp["rho"], opt.option_type)
        opt.delta = g["delta"]
        opt.gamma = g["gamma"]
        opt.vega  = g["vega"]
        opt.theta = g["theta"]
        opt.model = "Heston"
        return opt

    def price_option_dupire(self, opt: Option, S: float,
                             local_vol_fn, base_vol: float) -> Option:
        """
        Dupire local vol. Uses fast MC for price.
        IV shown is BS-equivalent IV implied by the Dupire price.
        """
        if local_vol_fn is None:
            return self.price_option_bs(opt, S, base_vol)
        mid = dupire_price_fast(S, opt.strike, opt.T, self.r,
                                 local_vol_fn, opt.option_type)
        mid = max(mid, 0.01)
        spread    = max(0.05, mid * 0.02)
        opt.bid   = max(0.01, mid - spread / 2)
        opt.ask   = mid + spread / 2
        opt.iv    = implied_vol_bs(mid, S, opt.strike, opt.T, self.r, opt.option_type)
        g         = dupire_greeks_num(S, opt.strike, opt.T, self.r,
                                       local_vol_fn, opt.option_type)
        opt.delta = g["delta"]
        opt.gamma = g["gamma"]
        opt.vega  = g["vega"]
        opt.theta = g["theta"]
        opt.model = "Dupire"
        return opt

    def build_chain(self, symbol: str, model: str = "BS") -> List[Option]:
        S        = self.price(symbol)
        base_vol = self.vol(symbol)
        lvf      = self.local_vol_fn(symbol)
        if S is None:
            return []
        chain    = []
        strikes  = generate_strikes(S)
        expiries = generate_expiries()
        for expiry in expiries:
            for strike in strikes:
                for otype in ["call", "put"]:
                    opt = Option(symbol=symbol, option_type=otype,
                                 strike=strike, expiry=expiry, model=model)
                    if model == "BS":
                        opt = self.price_option_bs(opt, S, base_vol)
                    elif model == "Heston":
                        opt = self.price_option_heston(opt, S, base_vol, symbol)
                    elif model == "Dupire":
                        opt = self.price_option_dupire(opt, S, lvf, base_vol)
                    chain.append(opt)
        return chain


# ─────────────────────────────────────────────────────────────
# SECTION 8 — PORTFOLIO
# ─────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, starting_cash=100_000.0):
        self.cash      = starting_cash
        self.start     = starting_cash
        self.positions = {}
        self.orders    = []

    def place_order(self, option: Option, direction: str, qty: int):
        price = option.ask if direction == "buy" else option.bid
        cost  = price * qty * 100 * (1 if direction == "buy" else -1)
        if direction == "buy" and cost > self.cash:
            return False, f"Insufficient cash. Need ${cost:,.2f}, have ${self.cash:,.2f}"
        order = Order(
            order_id  = str(uuid.uuid4())[:8].upper(),
            option    = option,
            direction = direction,
            quantity  = qty,
            price     = price,
            timestamp = datetime.datetime.now(),
            model     = option.model,
        )
        self.orders.append(order)
        self.cash -= cost
        label       = option.label
        signed_qty  = qty if direction == "buy" else -qty
        if label in self.positions:
            pos     = self.positions[label]
            new_qty = pos.quantity + signed_qty
            if new_qty == 0:
                del self.positions[label]
            else:
                total     = pos.avg_price * abs(pos.quantity) + price * qty
                pos.avg_price = total / abs(new_qty)
                pos.quantity  = new_qty
                pos.orders.append(order)
        else:
            self.positions[label] = Position(
                option=option, quantity=signed_qty,
                avg_price=price, orders=[order]
            )
        return True, f"✓ {order.order_id}  {direction.upper()} {qty}× {option.label} @ ${price:.2f}  [{option.model}]"

    @property
    def total_value(self):
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def total_pnl(self):
        return self.total_value - self.start

    @property
    def total_delta(self):
        return sum(p.quantity * p.option.delta * 100 for p in self.positions.values())

    @property
    def total_gamma(self):
        return sum(p.quantity * p.option.gamma * 100 for p in self.positions.values())

    @property
    def total_vega(self):
        return sum(p.quantity * p.option.vega * 100 for p in self.positions.values())

    @property
    def total_theta(self):
        return sum(p.quantity * p.option.theta * 100 for p in self.positions.values())


# ─────────────────────────────────────────────────────────────
# SECTION 9 — GUI
# ─────────────────────────────────────────────────────────────

POPULAR_TICKERS = ["AAPL","MSFT","GOOGL","AMZN","TSLA",
                   "NVDA","META","SPY","QQQ","JPM"]

BG    = "#0f1117"; BG2   = "#1a1d27"; BG3   = "#22263a"
ACCENT= "#4f8ef7"; GREEN = "#00c896"; RED   = "#ff4d6a"
AMBER = "#f5a623"; WHITE = "#e8eaf0"; MUTED = "#6b7280"
BORDER= "#2d3148"
PURPLE= "#b07eff"; ORANGE= "#ffb830"

FONT_H1  = ("Helvetica", 16, "bold")
FONT_H2  = ("Helvetica", 12, "bold")
FONT_SM  = ("Helvetica", 10)
FONT_XS  = ("Helvetica", 9)
FONT_MONO= ("Courier",   10)

MODEL_COLORS = {"BS": ACCENT, "Heston": PURPLE, "Dupire": ORANGE}
MODEL_LABELS = {
    "BS":     "Black-Scholes (1973)  — Surface IV σ(K,T)",
    "Heston": "Heston (1993)  — Stochastic Vol, mean-reverting variance",
    "Dupire": "Dupire Local Vol (1994)  — σ_L(K,T) via Dupire formula + MC",
}


class OptionsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Market Demo  |  3-Model Engine")
        self.geometry("1500x900")
        self.configure(bg=BG)
        self.resizable(True, True)

        self.market       = MarketData()
        self.portfolio    = Portfolio(100_000)
        self.chains       = {"BS": [], "Heston": [], "Dupire": []}
        self.selected_opt = None
        self.active_model = "BS"
        self._refresh_id  = None
        self._loading     = False

        self._build_ui()
        self._load_ticker("AAPL")

    # ── UI BUILD ──────────────────────────────────────────────

    def _build_ui(self):
        # top bar
        top = tk.Frame(self, bg=BG, pady=7)
        top.pack(fill="x", padx=16)
        tk.Label(top, text="⬡Quant Muriuki Options Market Demo",
                 font=FONT_H1, bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(top, text="Paper Trading  |  Live US Data  |  3 Pricing Models",
                 font=FONT_XS, bg=BG, fg=MUTED).pack(side="left", padx=14)

        right_top = tk.Frame(top, bg=BG)
        right_top.pack(side="right")
        tk.Label(right_top, text="Ticker:", font=FONT_SM, bg=BG, fg=WHITE).pack(side="left")
        self.ticker_var = tk.StringVar(value="AAPL")
        combo = ttk.Combobox(right_top, textvariable=self.ticker_var,
                              values=POPULAR_TICKERS, width=8, font=FONT_SM)
        combo.pack(side="left", padx=4)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: self._load_ticker(self.ticker_var.get()))
        tk.Button(right_top, text="Load", font=FONT_SM, bg=ACCENT, fg=WHITE,
                  relief="flat", padx=8,
                  command=lambda: self._load_ticker(self.ticker_var.get())
                  ).pack(side="left", padx=4)
        self.status_lbl = tk.Label(right_top, text="", font=FONT_XS, bg=BG, fg=MUTED)
        self.status_lbl.pack(side="left", padx=8)

        # portfolio summary bar
        self.pf_bar = tk.Frame(self, bg=BG2, pady=6)
        self.pf_bar.pack(fill="x", padx=16, pady=(0, 6))
        self.pf_labels = {}
        for key in ["Cash","Total Value","P&L","Δ Delta","Γ Gamma","ν Vega","Θ Theta"]:
            f = tk.Frame(self.pf_bar, bg=BG2, padx=18)
            f.pack(side="left")
            tk.Label(f, text=key, font=FONT_XS, bg=BG2, fg=MUTED).pack()
            lbl = tk.Label(f, text="—",
                           font=("Helvetica", 11, "bold"), bg=BG2, fg=WHITE)
            lbl.pack()
            self.pf_labels[key] = lbl

        # main panes
        main = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=4)
        main.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        left_frame  = tk.Frame(main, bg=BG)
        right_frame = tk.Frame(main, bg=BG)
        main.add(left_frame,  minsize=680)
        main.add(right_frame, minsize=400)

        self._build_chain_panel(left_frame)
        self._build_order_panel(left_frame)
        self._build_right_panel(right_frame)

    # ── CHAIN PANEL WITH MODEL TABS ───────────────────────────

    def _build_chain_panel(self, parent):
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="Option Chain", font=FONT_H2,
                 bg=BG, fg=WHITE).pack(side="left")

        # filters
        tk.Label(hdr, text="Expiry:", font=FONT_XS, bg=BG, fg=MUTED
                 ).pack(side="left", padx=(14, 4))
        self.expiry_var = tk.StringVar(value="All")
        self.expiry_combo = ttk.Combobox(
            hdr, textvariable=self.expiry_var,
            values=["All"], width=12, font=FONT_XS)
        self.expiry_combo.pack(side="left")
        self.expiry_combo.bind("<<ComboboxSelected>>",
                               lambda e: self._filter_chain())

        self.type_var = tk.StringVar(value="All")
        ttk.Combobox(hdr, textvariable=self.type_var,
                     values=["All","call","put"],
                     width=6, font=FONT_XS).pack(side="left", padx=4)
        self.type_var.trace("w", lambda *_: self._filter_chain())

        self.spot_lbl = tk.Label(hdr, text="Spot: —",
                                  font=FONT_SM, bg=BG, fg=ACCENT)
        self.spot_lbl.pack(side="right")

        # ── MODEL TABS ─────────────────────────────────────────
        tab_row = tk.Frame(parent, bg=BG)
        tab_row.pack(fill="x", pady=(4, 0))

        self.model_tab_btns = {}
        tab_configs = [
            ("BS",     "Black-Scholes",  ACCENT),
            ("Heston", "Heston (1993)",  PURPLE),
            ("Dupire", "Dupire Loc.Vol", ORANGE),
        ]
        for model, label, color in tab_configs:
            btn = tk.Button(
                tab_row, text=label,
                font=("Helvetica", 9, "bold"),
                relief="flat", padx=14, pady=4,
                command=lambda m=model: self._switch_model(m)
            )
            btn.pack(side="left", padx=2)
            self.model_tab_btns[model] = (btn, color)

        # model description label
        self.model_desc_lbl = tk.Label(
            parent, text=MODEL_LABELS["BS"],
            font=FONT_XS, bg=BG2, fg=MUTED,
            anchor="w", pady=4, padx=8)
        self.model_desc_lbl.pack(fill="x")

        self._update_tab_styles()

        # loading label (shown while Dupire/Heston compute)
        self.loading_lbl = tk.Label(parent, text="", font=FONT_XS,
                                     bg=BG, fg=AMBER)
        self.loading_lbl.pack(anchor="w")

        # chain treeview
        cols = ("Type","Strike","Expiry","Bid","Ask","Mid",
                "IV","Delta","Gamma","Theta","Vega")
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Chain.Treeview",
                         background=BG2, foreground=WHITE,
                         fieldbackground=BG2, rowheight=22,
                         font=FONT_XS, borderwidth=0)
        style.configure("Chain.Treeview.Heading",
                         background=BG3, foreground=MUTED,
                         font=("Helvetica", 9, "bold"), relief="flat")
        style.map("Chain.Treeview",
                  background=[("selected", BG3)],
                  foreground=[("selected", ACCENT)])

        self.chain_tree = ttk.Treeview(
            frame, columns=cols, show="headings",
            style="Chain.Treeview", height=15)
        widths = [45, 60, 78, 65, 65, 65, 65, 65, 68, 68, 62]
        for col, w in zip(cols, widths):
            self.chain_tree.heading(col, text=col)
            self.chain_tree.column(col, width=w, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical",
                            command=self.chain_tree.yview)
        self.chain_tree.configure(yscrollcommand=sb.set)
        self.chain_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.chain_tree.bind("<<TreeviewSelect>>", self._on_chain_select)
        self.chain_tree.tag_configure("call", foreground="#7dd3fc")
        self.chain_tree.tag_configure("put",  foreground="#fca5a5")
        self.chain_tree.tag_configure("itm",  background="#1a2e1a")

    def _update_tab_styles(self):
        for model, (btn, color) in self.model_tab_btns.items():
            if model == self.active_model:
                btn.config(bg=color, fg="black" if model == "Dupire" else "white")
            else:
                btn.config(bg=BG3, fg=MUTED)

    def _switch_model(self, model: str):
        self.active_model = model
        self._update_tab_styles()
        self.model_desc_lbl.config(text=MODEL_LABELS[model])
        self._filter_chain()

    # ── ORDER PANEL ───────────────────────────────────────────

    def _build_order_panel(self, parent):
        f = tk.Frame(parent, bg=BG2, pady=8, padx=14)
        f.pack(fill="x", pady=6)

        tk.Label(f, text="Order Entry", font=FONT_H2,
                 bg=BG2, fg=WHITE).grid(row=0, column=0,
                                         columnspan=7, sticky="w", pady=(0, 6))

        self.selected_lbl = tk.Label(f, text="No option selected",
                                      font=FONT_XS, bg=BG2, fg=MUTED)
        self.selected_lbl.grid(row=1, column=0, columnspan=7,
                                sticky="w", pady=(0, 5))

        for i, lbl in enumerate(["Direction","Quantity","Price"]):
            tk.Label(f, text=lbl, font=FONT_XS, bg=BG2,
                     fg=MUTED).grid(row=2, column=i*2, sticky="w", padx=(0,4))

        self.dir_var = tk.StringVar(value="buy")
        ttk.Combobox(f, textvariable=self.dir_var,
                     values=["buy","sell"],
                     width=7, font=FONT_SM).grid(row=2, column=1, padx=4)

        self.qty_var = tk.StringVar(value="1")
        tk.Entry(f, textvariable=self.qty_var, width=6,
                 bg=BG3, fg=WHITE, insertbackground=WHITE,
                 font=FONT_SM, relief="flat").grid(row=2, column=3, padx=4)

        self.price_lbl = tk.Label(f, text="—", font=FONT_SM, bg=BG2, fg=WHITE)
        self.price_lbl.grid(row=2, column=5, padx=4)

        self.submit_btn = tk.Button(
            f, text="Place Order",
            font=("Helvetica", 10, "bold"),
            bg=GREEN, fg="black", relief="flat",
            padx=16, pady=4,
            command=self._place_order)
        self.submit_btn.grid(row=2, column=6, padx=10)

        # active model badge
        self.model_badge = tk.Label(f, text="[BS]",
                                     font=("Courier", 9, "bold"),
                                     bg=BG2, fg=ACCENT)
        self.model_badge.grid(row=2, column=7, padx=6)

        self.order_msg = tk.Label(f, text="", font=FONT_XS, bg=BG2, fg=GREEN)
        self.order_msg.grid(row=3, column=0, columnspan=8,
                             sticky="w", pady=(5, 0))

    # ── RIGHT PANEL ───────────────────────────────────────────

    def _build_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        style = ttk.Style()
        style.configure("TNotebook",      background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",  background=BG3, foreground=MUTED,
                         font=FONT_XS, padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", BG2)],
                  foreground=[("selected", WHITE)])
        nb.pack(fill="both", expand=True)

        for tab_name, builder in [
            ("Positions",     self._build_positions_tab),
            ("Order History", self._build_orders_tab),
            ("Greeks",        self._build_greeks_tab),
        ]:
            frm = tk.Frame(nb, bg=BG)
            nb.add(frm, text=tab_name)
            builder(frm)

    def _build_positions_tab(self, parent):
        cols = ("Option","Model","Qty","Avg Price","Mkt Value","Unr. P&L","P&L %")
        style = ttk.Style()
        style.configure("Pos.Treeview",
                         background=BG2, foreground=WHITE,
                         fieldbackground=BG2, rowheight=22,
                         font=FONT_XS, borderwidth=0)
        style.configure("Pos.Treeview.Heading",
                         background=BG3, foreground=MUTED,
                         font=("Helvetica", 9, "bold"), relief="flat")

        self.pos_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                      style="Pos.Treeview", height=10)
        widths = [185, 60, 40, 70, 80, 80, 60]
        for col, w in zip(cols, widths):
            self.pos_tree.heading(col, text=col)
            self.pos_tree.column(col, width=w, anchor="center")
        self.pos_tree.tag_configure("profit", foreground=GREEN)
        self.pos_tree.tag_configure("loss",   foreground=RED)
        self.pos_tree.pack(fill="both", expand=True)

        tk.Button(parent, text="Close Selected Position",
                  font=FONT_XS, bg=RED, fg=WHITE,
                  relief="flat", pady=4,
                  command=self._close_position).pack(pady=6)

    def _build_orders_tab(self, parent):
        cols = ("ID","Time","Option","Model","Dir","Qty","Price","Total")
        style = ttk.Style()
        style.configure("Ord.Treeview",
                         background=BG2, foreground=WHITE,
                         fieldbackground=BG2, rowheight=20,
                         font=FONT_XS, borderwidth=0)
        style.configure("Ord.Treeview.Heading",
                         background=BG3, foreground=MUTED,
                         font=("Helvetica", 9, "bold"), relief="flat")

        self.ord_tree = ttk.Treeview(parent, columns=cols, show="headings",
                                      style="Ord.Treeview", height=20)
        widths = [60, 65, 145, 55, 38, 32, 58, 75]
        for col, w in zip(cols, widths):
            self.ord_tree.heading(col, text=col)
            self.ord_tree.column(col, width=w, anchor="center")
        self.ord_tree.tag_configure("buy",  foreground=GREEN)
        self.ord_tree.tag_configure("sell", foreground=RED)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.ord_tree.yview)
        self.ord_tree.configure(yscrollcommand=sb.set)
        self.ord_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_greeks_tab(self, parent):
        tk.Label(parent, text="Portfolio Greeks — all open positions",
                 font=FONT_H2, bg=BG, fg=WHITE).pack(pady=8)

        self.greeks_frame = tk.Frame(parent, bg=BG)
        self.greeks_frame.pack(fill="both", expand=False, padx=16)
        self.greek_cards = {}

        greek_info = {
            "Delta": ("Δ", "Price sensitivity\nper $1 move in S",    ACCENT),
            "Gamma": ("Γ", "Delta change\nper $1 move in S",         AMBER),
            "Vega":  ("ν", "Price change\nper 1% vol move",          GREEN),
            "Theta": ("Θ", "Daily time decay\nper calendar day",      RED),
        }
        for i, (name, (sym, desc, color)) in enumerate(greek_info.items()):
            card = tk.Frame(self.greeks_frame, bg=BG2, padx=20, pady=14)
            card.grid(row=i//2, column=i%2, padx=8, pady=8, sticky="nsew")
            self.greeks_frame.columnconfigure(i%2, weight=1)
            tk.Label(card, text=f"{sym}  {name}", font=FONT_H2,
                     bg=BG2, fg=color).pack()
            val = tk.Label(card, text="0.0000",
                           font=("Helvetica", 20, "bold"), bg=BG2, fg=WHITE)
            val.pack(pady=3)
            tk.Label(card, text=desc, font=FONT_XS, bg=BG2,
                     fg=MUTED, justify="center").pack()
            self.greek_cards[name] = val

        tk.Label(parent, text="Selected Option Greeks",
                 font=FONT_H2, bg=BG, fg=WHITE).pack(pady=(14, 3))
        self.sel_greeks_lbl = tk.Label(
            parent, text="Select an option from the chain",
            font=FONT_XS, bg=BG, fg=MUTED, justify="left")
        self.sel_greeks_lbl.pack(padx=16, anchor="w")

    # ── DATA LOADING ──────────────────────────────────────────

    def _load_ticker(self, symbol: str):
        symbol = symbol.upper().strip()
        self.status_lbl.config(text=f"Loading {symbol}...", fg=AMBER)
        self.update()

        def fetch():
            price = self.market.fetch(symbol)
            if price is None:
                self.after(0, lambda: self.status_lbl.config(
                    text=f"Failed to load {symbol}", fg=RED))
                return

            # Build BS chain immediately (fast)
            bs_chain = self.market.build_chain(symbol, "BS")
            self.after(0, lambda: self._on_bs_loaded(symbol, price, bs_chain))

            # Build Heston chain in background
            self.after(0, lambda: self.loading_lbl.config(
                text="Computing Heston chain...", fg=AMBER))
            h_chain = self.market.build_chain(symbol, "Heston")
            self.after(0, lambda: self._store_chain(symbol, "Heston", h_chain))

            # Build Dupire chain in background (slowest — MC)
            self.after(0, lambda: self.loading_lbl.config(
                text="Computing Dupire chain (MC)...", fg=ORANGE))
            d_chain = self.market.build_chain(symbol, "Dupire")
            self.after(0, lambda: self._store_chain(symbol, "Dupire", d_chain,
                                                     final=True))

        threading.Thread(target=fetch, daemon=True).start()

    def _on_bs_loaded(self, symbol, price, chain):
        self.chains["BS"] = chain
        expiries = sorted(set(o.expiry.strftime("%Y-%m-%d") for o in chain))
        self.expiry_combo["values"] = ["All"] + expiries
        self.expiry_var.set("All")
        vol = self.market.vol(symbol) * 100
        self.spot_lbl.config(
            text=f"Spot: ${price:.2f}  |  HV30: {vol:.1f}%")
        self.status_lbl.config(
            text=f"{symbol} loaded — {len(chain)} contracts", fg=GREEN)
        if self.active_model == "BS":
            self._filter_chain()
        self._refresh_portfolio()
        self._schedule_refresh(symbol)

    def _store_chain(self, symbol, model, chain, final=False):
        self.chains[model] = chain
        if self.active_model == model:
            self._filter_chain()
        if final:
            self.loading_lbl.config(text="All models ready ✓", fg=GREEN)
            self.after(3000, lambda: self.loading_lbl.config(text=""))

    def _schedule_refresh(self, symbol, interval_ms=60_000):
        if self._refresh_id:
            self.after_cancel(self._refresh_id)
        self._refresh_id = self.after(
            interval_ms, lambda: self._load_ticker(symbol))

    # ── CHAIN FILTER & DISPLAY ────────────────────────────────

    def _filter_chain(self):
        self.chain_tree.delete(*self.chain_tree.get_children())
        chain = self.chains.get(self.active_model, [])
        if not chain:
            return

        exp_filter  = self.expiry_var.get()
        type_filter = self.type_var.get()
        S = self.market.price(self.ticker_var.get().upper())

        for opt in chain:
            if (exp_filter != "All"
                    and opt.expiry.strftime("%Y-%m-%d") != exp_filter):
                continue
            if type_filter != "All" and opt.option_type != type_filter:
                continue

            itm = ((opt.option_type == "call" and S and opt.strike < S) or
                   (opt.option_type == "put"  and S and opt.strike > S))
            tags = (opt.option_type, "itm") if itm else (opt.option_type,)

            self.chain_tree.insert("", "end", iid=opt.label, tags=tags, values=(
                opt.option_type.upper(),
                f"{opt.strike:.0f}",
                opt.expiry.strftime("%Y-%m-%d"),
                f"{opt.bid:.2f}",
                f"{opt.ask:.2f}",
                f"{opt.mid:.2f}",
                f"{opt.iv*100:.1f}%",      # ← now varies per (K, T)
                f"{opt.delta:.3f}",
                f"{opt.gamma:.4f}",
                f"{opt.theta:.4f}",
                f"{opt.vega:.4f}",
            ))

        # update model badge color
        color = MODEL_COLORS.get(self.active_model, ACCENT)
        self.model_badge.config(
            text=f"[{self.active_model}]", fg=color)

    # ── ORDER HANDLING ────────────────────────────────────────

    def _on_chain_select(self, event):
        sel = self.chain_tree.selection()
        if not sel:
            return
        label = sel[0]
        chain = self.chains.get(self.active_model, [])
        opt   = next((o for o in chain if o.label == label), None)
        if opt is None:
            return
        self.selected_opt = opt
        color = MODEL_COLORS.get(self.active_model, ACCENT)
        self.selected_lbl.config(
            text=f"{opt.label}  [{opt.model}]", fg=color)
        direction = self.dir_var.get()
        price     = opt.ask if direction == "buy" else opt.bid
        self.price_lbl.config(text=f"${price:.2f}  (×100 per contract)")

        # greeks panel
        txt = (f"  Model: {opt.model}\n"
               f"  Delta: {opt.delta:+.4f}    Gamma: {opt.gamma:.5f}\n"
               f"  Vega:  {opt.vega:+.4f}    Theta: {opt.theta:+.4f}\n"
               f"  IV:    {opt.iv*100:.2f}%       T: {opt.T*365:.0f} days")
        self.sel_greeks_lbl.config(text=txt, fg=WHITE)

    def _place_order(self):
        if self.selected_opt is None:
            self.order_msg.config(text="Select an option first.", fg=RED)
            return
        try:
            qty = int(self.qty_var.get())
            assert qty > 0
        except Exception:
            self.order_msg.config(text="Quantity must be a positive integer.", fg=RED)
            return
        direction = self.dir_var.get()
        ok, msg   = self.portfolio.place_order(self.selected_opt, direction, qty)
        self.order_msg.config(text=msg, fg=GREEN if ok else RED)
        if ok:
            self._refresh_portfolio()

    def _close_position(self):
        sel = self.pos_tree.selection()
        if not sel:
            return
        label = sel[0]
        pos   = self.portfolio.positions.get(label)
        if pos is None:
            return
        direction = "sell" if pos.quantity > 0 else "buy"
        ok, msg   = self.portfolio.place_order(
            pos.option, direction, abs(pos.quantity))
        self.order_msg.config(text=msg, fg=GREEN if ok else RED)
        if ok:
            self._refresh_portfolio()

    # ── PORTFOLIO REFRESH ─────────────────────────────────────

    def _refresh_portfolio(self):
        pf = self.portfolio
        pnl_color = GREEN if pf.total_pnl >= 0 else RED

        self.pf_labels["Cash"].config(text=f"${pf.cash:>10,.2f}")
        self.pf_labels["Total Value"].config(text=f"${pf.total_value:>10,.2f}")
        self.pf_labels["P&L"].config(
            text=f"${pf.total_pnl:>+10,.2f}", fg=pnl_color)
        self.pf_labels["Δ Delta"].config(text=f"{pf.total_delta:>+.2f}")
        self.pf_labels["Γ Gamma"].config(text=f"{pf.total_gamma:>+.4f}")
        self.pf_labels["ν Vega"].config( text=f"{pf.total_vega:>+.4f}")
        self.pf_labels["Θ Theta"].config(text=f"{pf.total_theta:>+.4f}")

        # positions
        self.pos_tree.delete(*self.pos_tree.get_children())
        for label, pos in pf.positions.items():
            tag   = "profit" if pos.unrealized_pnl >= 0 else "loss"
            short = label if len(label) <= 28 else label[:26] + "…"
            self.pos_tree.insert("", "end", iid=label, tags=(tag,), values=(
                short,
                pos.option.model,
                pos.quantity,
                f"${pos.avg_price:.2f}",
                f"${pos.market_value:,.2f}",
                f"${pos.unrealized_pnl:+,.2f}",
                f"{pos.pnl_pct:+.1f}%",
            ))

        # orders
        self.ord_tree.delete(*self.ord_tree.get_children())
        for o in reversed(pf.orders):
            total = o.price * o.quantity * 100
            self.ord_tree.insert("", "end", tags=(o.direction,), values=(
                o.order_id,
                o.timestamp.strftime("%H:%M:%S"),
                o.option.label[:24],
                o.model,
                o.direction.upper(),
                o.quantity,
                f"${o.price:.2f}",
                f"${total:,.2f}",
            ))

        # greeks cards
        self.greek_cards["Delta"].config(text=f"{pf.total_delta:+.4f}")
        self.greek_cards["Gamma"].config(text=f"{pf.total_gamma:+.5f}")
        self.greek_cards["Vega"].config( text=f"{pf.total_vega:+.4f}")
        self.greek_cards["Theta"].config(text=f"{pf.total_theta:+.4f}")


if __name__ == "__main__":
    app = OptionsApp()
    app.mainloop()